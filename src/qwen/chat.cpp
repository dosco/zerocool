#include "qwen/chat.hpp"
#include "qwen/chat_client.hpp"
#include <ftxui/component/component.hpp>
#include <ftxui/component/component_options.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <format>
#include <cmath>
#include <mutex>
#include <optional>
#include <print>
#include <sstream>
#include <thread>
#include <mach-o/dyld.h>
#include <unistd.h>
namespace freellm::qwen {
namespace {
using namespace ftxui;
using namespace freellm::chat;
struct Settings {std::string url;std::vector<std::string> child;int max_tokens=256;double temperature=0;bool thinking=false;};
Settings settings(int argc,char** argv) {
    Settings s;bool memory=false,model=false,prepared=false;
    std::string artifact="q4-control";
    for(int i=2;i<argc;++i) {
        std::string key=argv[i];
        if(key=="--thinking") {s.thinking=true;continue;}
        if(i+1==argc)throw std::invalid_argument("missing value for "+key);
        std::string value=argv[++i];
        if(key=="--connect") {s.url=endpoint(value);continue;}
        if(key=="--max-tokens") {s.max_tokens=std::stoi(value);if(s.max_tokens<1 || s.max_tokens>8192)throw std::invalid_argument("max-tokens must be 1..8192");continue;}
        if(key=="--temperature") {s.temperature=std::stod(value);if(!std::isfinite(s.temperature)||s.temperature<0)throw std::invalid_argument("invalid temperature");continue;}
        if(key!="--model" && key!="--artifact" && key!="--prepared" && key!="--memory-gb" && key!="--context")
            throw std::invalid_argument("unsupported chat option: "+key);
        model|=key=="--model";prepared|=key=="--prepared";memory|=key=="--memory-gb";
        if(key=="--artifact")artifact=value;
        s.child.push_back(key);s.child.push_back(value);
    }
    if(!s.url.empty()) {if(!s.child.empty())throw std::invalid_argument("connect cannot change the server's model or memory settings");}
    else {
        if(!model) {
            s.child.push_back("--model");
            s.child.push_back(artifact=="mixed-4_8bit"?".cache/qwen-mixed-reference":".cache/models/qwen38-flash-next");
        }
        if(!prepared) {s.child.push_back("--prepared");s.child.push_back(".cache/prepared/q4-records-v1");}
        if(!memory) {s.child.push_back("--memory-gb");s.child.push_back("12");}
    }
    return s;
}
std::string executable() {
    uint32_t size=0;_NSGetExecutablePath(nullptr,&size);std::string path(size,'\0');
    if(_NSGetExecutablePath(path.data(),&size))throw std::runtime_error("cannot locate engine executable");
    path.resize(std::strlen(path.c_str()));return std::filesystem::canonical(path).string();
}
struct Entry {std::string role,text,reasoning;bool interrupted=false;};
std::string metric(const Json& data,const char* name) {
    return data.contains(name) && !data[name].is_null()?data[name].dump():"—";
}
class Chat {
public:
    explicit Chat(Settings settings):settings_(std::move(settings)),screen_(ScreenInteractive::Fullscreen()) {
        screen_.ForceHandleCtrlC(false);
    }
    int run() {
        InputOption option;option.multiline=true;option.placeholder="Type a message. Enter: newline · Ctrl+D: send";
        input_=Input(&input_text_,option);
        auto component=Renderer(input_,[&]{return render();});
        component=CatchEvent(component,[&](Event event){return key(event);});
        struct PasteMode {
            PasteMode(){std::print("\x1b[?2004h");std::fflush(stdout);}
            ~PasteMode(){std::print("\x1b[?2004l");std::fflush(stdout);}
        } paste_mode;
        try {poller_=std::thread([&]{poll();});sender_=std::thread([&]{send();});screen_.Loop(component);} catch(...) {finish();throw;}
        finish();return 0;
    }
private:
    Settings settings_;ScreenInteractive screen_;Component input_;
    std::mutex mutex_;std::condition_variable changed_;
    std::atomic<bool> stop_=false,cancel_=false;
    std::thread poller_,sender_;std::unique_ptr<ChildServer> child_;
    std::string url_,instance_,input_text_,notice_="Starting local engine…",restore_input_;
    Json status_=Json::object(),history_=Json::array();
    std::vector<Entry> transcript_;
    struct Work {bool reset=false;std::string prompt;Json history;size_t assistant=0;};
    std::optional<Work> work_;
    bool busy_=false,details_=false,help_=false,pasting_=false,paste_blocked_=false;int scroll_=0;
    std::string pasted_;
    size_t transcript_bytes_=0;
    void wake(){screen_.PostEvent(Event::Custom);}
    void finish() {
        stop_=true;cancel_=true;changed_.notify_all();
        if(sender_.joinable())sender_.join();if(poller_.joinable())poller_.join();
        if(child_) {std::println(stderr,"Closing local engine; waiting for outstanding work to drain…");child_.reset();}
    }
    void poll() {
        try {
            std::string url=settings_.url;
            if(url.empty()) {child_=std::make_unique<ChildServer>(executable(),settings_.child);url=child_->await_endpoint(stop_);}
            std::string instance=child_?child_->instance():"";
            {std::lock_guard lock(mutex_);url_=url;instance_=instance;notice_="Connecting…";}
            wake();
            while(!stop_) {
                try {
                    auto reply=http(url+"/freellm/status","GET","",stop_,{},3000,instance);
                    if(reply.status==409)throw std::runtime_error("server instance changed; quit and reconnect explicitly");
                    if(reply.status!=200)throw std::runtime_error("server does not provide FreeLLM status");
                    auto status=Json::parse(reply.body);
                    if(status.value("version",0)!=1)throw std::runtime_error("unsupported FreeLLM status version");
                    const auto observed=status.at("instance_id").get<std::string>();
                    if(observed.empty() || (!instance.empty() && observed!=instance))throw std::runtime_error("server instance changed; quit and reconnect explicitly");
                    instance=observed;
                    {std::lock_guard lock(mutex_);instance_=instance;status_=std::move(status);
                        if(status_.value("phase","")=="failed")notice_=status_.value("error","engine failed");
                        else if(notice_=="Connecting…" || notice_.starts_with("Connection:"))notice_="Ctrl+D send · Ctrl+C stop · Ctrl+Q quit · F2 details · /help";
                    }
                } catch(const std::exception& e) {std::lock_guard lock(mutex_);status_["phase"]="disconnected";notice_="Connection: "+std::string(e.what());}
                wake();std::unique_lock lock(mutex_);changed_.wait_for(lock,std::chrono::seconds(1),[&]{return stop_.load();});
            }
        } catch(const std::exception& e) {std::lock_guard lock(mutex_);notice_=e.what();status_["phase"]="failed";wake();}
    }
    void send() {
        while(!stop_) {
            Work work;std::string url,model,instance;
            {
                std::unique_lock lock(mutex_);changed_.wait(lock,[&]{return stop_ || work_.has_value();});if(stop_)return;
                work=std::move(*work_);work_.reset();url=url_;model=status_.value("model","");instance=instance_;
            }
            try {
                if(work.reset) {
                    auto response=http(url+"/freellm/session/reset","POST","{}",cancel_,{},3000,instance);
                    if(response.status!=200)throw std::runtime_error(Json::parse(response.body)["error"].value("message","reset failed"));
                    std::lock_guard lock(mutex_);history_=Json::array();transcript_.clear();transcript_bytes_=0;notice_="New conversation";
                } else {
                    auto messages=work.history;messages.push_back({{"role","user"},{"content",work.prompt}});
                    Json body={{"model",model},{"messages",messages},{"stream",true},{"stream_options",{{"include_usage",true}}},
                        {"max_tokens",settings_.max_tokens},{"temperature",settings_.temperature},{"enable_thinking",settings_.thinking}};
                    bool finished=false;std::string reason;
                    auto response=http(url+"/v1/chat/completions","POST",body.dump(),cancel_,[&](const Json& event) {
                        const auto& choices=event.at("choices");if(choices.empty())return;
                        const auto& choice=choices.at(0);const auto& delta=choice.at("delta");
                        std::lock_guard lock(mutex_);auto& entry=transcript_.at(work.assistant);
                        auto append=[&](const char* key,std::string& destination) {
                            if(!delta.contains(key) || delta[key].is_null())return;
                            const auto value=delta[key].get<std::string>();
                            if(transcript_bytes_+value.size()>1024*1024)throw std::runtime_error("Transcript reached 1MiB. Save it and use /new before continuing.");
                            destination+=value;transcript_bytes_+=value.size();
                        };
                        append("content",entry.text);append("reasoning_content",entry.reasoning);
                        if(delta.contains("tool_calls"))throw std::runtime_error("chat does not execute tools; connect a coding client");
                        if(!choice.at("finish_reason").is_null()) {finished=true;reason=choice["finish_reason"].get<std::string>();}
                        wake();
                    },0,instance);
                    if(response.status!=200)throw std::runtime_error(Json::parse(response.body)["error"].value("message","request rejected"));
                    if(!finished || cancel_)throw std::runtime_error("response interrupted");
                    std::lock_guard lock(mutex_);const auto& entry=transcript_.at(work.assistant);
                    messages.push_back({{"role","assistant"},{"content",entry.text}});
                    if(settings_.thinking)messages.back()["reasoning_content"]=entry.reasoning;
                    history_=std::move(messages);notice_=reason=="length"?"Output limit reached. Continue or increase --max-tokens.":"Ready";
                }
            } catch(const std::exception& e) {
                std::lock_guard lock(mutex_);notice_=cancel_?"Stopped. Waiting for the engine to drain; the prompt is ready to retry.":e.what();
                if(!work.reset) {transcript_.at(work.assistant).interrupted=true;restore_input_=work.prompt;}
            }
            {std::lock_guard lock(mutex_);busy_=false;}
            wake();
        }
    }
    bool key(const Event& event) {
        if(event==Event::Custom) {
            std::lock_guard lock(mutex_);if(!restore_input_.empty()) {input_text_=std::move(restore_input_);restore_input_.clear();}return true;
        }
        if(event.input()=="\x1b[200~") {std::lock_guard lock(mutex_);pasting_=true;paste_blocked_=busy_;pasted_.clear();return true;}
        if(pasting_) {
            if(event.input()=="\x1b[201~") {
                pasting_=false;
                if(paste_blocked_) {std::lock_guard lock(mutex_);notice_="Paste ignored during an active response; stop it before editing.";}
                else if(input_text_.size()+pasted_.size()<=256*1024)input_->OnEvent(Event::Character(pasted_));
                else {std::lock_guard lock(mutex_);notice_="Paste exceeds 256KiB; shorten it before sending.";}
                pasted_.clear();
            } else if(pasted_.size()<=256*1024)pasted_+=event==Event::Return?"\n":terminal_text(event.input());
            return true;
        }
        if(event==Event::CtrlQ) {stop_=true;cancel_=true;changed_.notify_all();screen_.ExitLoopClosure()();return true;}
        std::lock_guard lock(mutex_);
        if(event==Event::CtrlC) {if(busy_) {cancel_=true;notice_="Cancelling…";}return true;}
        if(event==Event::CtrlU) {if(!busy_)input_text_.clear();return true;}
        if(event==Event::F2) {details_=!details_;return true;}
        if(event==Event::PageUp) {scroll_+=10;return true;}
        if(event==Event::PageDown) {scroll_=std::max(0,scroll_-10);return true;}
        if(event!=Event::CtrlD)return busy_;
        if(input_text_=="/help") {help_=!help_;input_text_.clear();return true;}
        if(input_text_.starts_with("/save ")) {
            try {
                Json entries=Json::array();for(const auto& e:transcript_)entries.push_back({{"role",e.role},{"content",e.text},{"reasoning_content",e.reasoning},{"interrupted",e.interrupted}});
                save_transcript(input_text_.substr(6),{{"format","freellm-transcript-v1"},{"model",status_.value("model","")},{"entries",entries}});
                notice_="Transcript saved";input_text_.clear();
            } catch(const std::exception& e) {notice_=e.what();}
            return true;
        }
        if(busy_ || status_.value("phase","")!="ready") {notice_="Wait until the engine is ready.";return true;}
        if(input_text_.empty())return true;
        if(input_text_.size()>256*1024) {notice_="Input exceeds 256KiB; shorten it before sending.";return true;}
        Work work;work.reset=input_text_=="/new";
        if(!work.reset) {
            if(transcript_.size()>=2048 || transcript_bytes_+input_text_.size()>768*1024) {notice_="Transcript is full. Use /save PATH and /new; history is never silently removed.";return true;}
            work.prompt=input_text_;work.history=history_;
            transcript_bytes_+=input_text_.size();
            transcript_.push_back({"you",input_text_,"",false});transcript_.push_back({"assistant","","",false});work.assistant=transcript_.size()-1;
        }
        cancel_=false;busy_=true;work_=std::move(work);input_text_.clear();scroll_=0;notice_="Sending…";changed_.notify_all();return true;
    }
    Element render() {
        std::lock_guard lock(mutex_);Elements lines;
        // Build DOM only for a window around the scroll position. Export and
        // request history retain every byte; long displayed lines are clipped.
        auto line_count=[](const std::string& value) {return value.empty()?size_t(0):size_t(std::count(value.begin(),value.end(),'\n'))+(value.back()!='\n');};
        size_t count=0;
        for(const auto& e:transcript_)count+=2+line_count(e.text)+(e.reasoning.empty()?0:1+line_count(e.reasoning));
        scroll_=std::clamp(scroll_,0,int(count?count-1:0));
        const size_t end=count-size_t(scroll_),begin=end>256?end-256:0;
        size_t at=0;
        auto emit=[&](const std::string& raw,Decorator style,bool preformatted=false) {
            if(at>=begin && at<end) {
                auto value=terminal_text(raw);
                if(value.size()>4096) {value.resize(4096);while(!value.empty() && (static_cast<unsigned char>(value.back())&0xc0)==0x80)value.pop_back();if(!value.empty() && static_cast<unsigned char>(value.back())>=0xc0)value.pop_back();value+=" … (full line in /save)";}
                lines.push_back((preformatted?text(value):paragraph(value))|style);
            }
            ++at;
        };
        for(const auto& e:transcript_) {
            emit(e.role+(e.interrupted?" · interrupted":""),bold|color(e.role=="you"?Color::Cyan:Color::Green));
            std::string line;
            if(!e.reasoning.empty()) {emit("Reasoning",dim);std::istringstream reasoning(e.reasoning);while(std::getline(reasoning,line))emit(line,dim);}
            std::istringstream stream(e.text);bool code=false;
            while(std::getline(stream,line)) {
                if(line.starts_with("```")) {code=!code;emit(code?"── code ──":"──────────",dim);}
                else emit(code?"  "+line:line,color(code?Color(Color::GrayLight):Color(Color::Default)),code);
            }
            emit("",dim);
        }
        if(lines.empty())lines.push_back(paragraph("FreeLLM · local chat. The model stays in the engine process. Type /help and press Ctrl+D for controls.")|dim);
        lines.back()=lines.back()|focus;
        std::string footer=status_.value("phase","starting")+" · "+metric(status_,"completed_prompt_tokens")+"/"+metric(status_,"prompt_tokens")+" input";
        const auto prompt=status_.contains("prompt_tokens") && status_["prompt_tokens"].is_number()?status_["prompt_tokens"].get<int>():0;
        const auto output=status_.contains("output_tokens") && status_["output_tokens"].is_number()?status_["output_tokens"].get<int>():0;
        const auto rate=status_.contains("generation_tokens_per_second") && status_["generation_tokens_per_second"].is_number()?
            std::format("{:.2f}",status_["generation_tokens_per_second"].get<double>()):"—";
        footer+=" · context "+(status_.contains("prompt_tokens") && status_["prompt_tokens"].is_number()?std::to_string(prompt+output):"—")+"/"+metric(status_,"context_limit")+" · "+rate+" tok/s";
        if(status_.contains("process") && status_["process"].is_object() && status_["process"].contains("physical_footprint_bytes") && !status_["process"]["physical_footprint_bytes"].is_null())
            footer+=" · "+std::format("{:.2f} GiB",status_["process"]["physical_footprint_bytes"].get<double>()/(1024*1024*1024));
        else footer+=" · memory —";
        Elements view={text("FreeLLM  "+status_.value("model","")+"  "+url_)|bold,separator(),vbox(std::move(lines))|vscroll_indicator|yframe|flex,separator(),paragraph(terminal_text(footer))};
        if(details_)view.push_back(paragraph("build "+status_.value("build","—")+" · reused "+metric(status_,"reused_tokens")+" · matched "+metric(status_,"matched_prefix_tokens")+" · memory sample "+metric(status_,"memory_sample_ns"))|dim);
        if(help_)view.push_back(paragraph("Enter newline · Ctrl+D send · Ctrl+C cancel · Ctrl+U clear input · Ctrl+Q quit · PageUp/PageDown scroll · F2 details · /new reset · /save PATH export. Tool execution and automatic history compaction belong to coding clients.")|dim);
        view.push_back(paragraph(terminal_text(notice_))|color(Color::Yellow));view.push_back(separator());
        view.push_back(input_->Render()|size(HEIGHT,GREATER_THAN,3)|size(HEIGHT,LESS_THAN,8));
        return vbox(std::move(view));
    }
};
}
int chat_main(int argc,char** argv) {
    if(!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO))throw std::invalid_argument("chat requires an interactive terminal; use run or serve for pipes");
    Chat chat(settings(argc,argv));return chat.run();
}
} // namespace freellm::qwen
