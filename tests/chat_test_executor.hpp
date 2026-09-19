#pragma once
#include "qwen/server.hpp"
#include <chrono>
#include <mutex>
#include <set>
#include <thread>
namespace freellm::qwen::chat_test {
using namespace std::chrono_literals;
struct Control {
    std::atomic<bool> loading=false,fail_load=false,hold=false,draining=false,hold_drain=false,fail_drain=false,cancelled=false;
    std::atomic<int> resets=0,destroyed=0;
    std::mutex mutex;std::set<std::thread::id> owners;
    void record(){std::lock_guard lock(mutex);owners.insert(std::this_thread::get_id());}
};
class TestExecutor final:public ChatExecutor {
    std::shared_ptr<Control> control;
public:
    explicit TestExecutor(std::shared_ptr<Control> c):control(c){control->record();}
    ~TestExecutor(){control->record();++control->destroyed;}
    void initialize(const ChatObserver& progress) override {
        control->record();progress({{"phase","loading_model"},{"test_executor",true}});
        while(control->loading)std::this_thread::sleep_for(1ms);
        if(control->fail_load)throw std::runtime_error("test startup failure");
    }
    ChatResponse generate(const ChatRequest& request,const ChatObserver& delta,const ChatObserver& progress,const std::atomic<bool>& cancel) override {
        control->record();auto text=request.messages.back().at("content").get<std::string>();
        if(text=="invalid")throw std::invalid_argument("test invalid prompt");
        if(delta)delta({{"role","assistant"}});
        progress({{"phase","prefill"},{"prompt_tokens",8},{"completed_prompt_tokens",4},{"reused_tokens",0},{"matched_prefix_tokens",0}});
        while(control->hold && !cancel)std::this_thread::sleep_for(1ms);
        if(cancel) {control->cancelled=true;throw std::runtime_error("generation cancelled");}
        if(text=="slow") {
            if(delta)delta({{"content","Starting…"}});
            for(int i=0;i<500 && !cancel;++i)std::this_thread::sleep_for(2ms);
            if(cancel)throw std::runtime_error("generation cancelled");
        }
        if(text=="overflow" && delta)delta({{"content",std::string(300*1024,'x')}});
        if(request.thinking && delta)delta({{"reasoning_content","Check café 🦉 first."}});
        if(delta)delta({{"content","OK 🦉"}});
        progress({{"phase","generating"},{"output_tokens",2}});
        Json message={{"role","assistant"},{"content","OK 🦉"}};
        if(text=="tool")message["tool_calls"]=Json::array({{{"id","old"},{"type","function"},{"function",{{"name","read_file"},{"arguments","{\"path\":\"a.cpp\"}"}}}}});
        return {message,{{"prompt_tokens",8},{"completion_tokens",2},{"total_tokens",10}},Json::object(),text=="tool"?"tool_calls":"stop"};
    }
    void clear() override {control->record();++control->resets;}
    void recover() override {clear();}
    void drain() override {
        control->record();control->draining=true;
        while(control->hold_drain)std::this_thread::sleep_for(1ms);
        control->draining=false;
        if(control->fail_drain)throw std::runtime_error("test drain failure");
    }
};
}
