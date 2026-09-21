#include "engine/chat_client.hpp"
#include <curl/curl.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <poll.h>
#include <spawn.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
namespace zerocool::chat {
namespace {
void need(bool value,const std::string& error) {if(!value)throw std::runtime_error(error);}
}
std::string endpoint(std::string value) {
    while(!value.empty() && value.back()=='/')value.pop_back();
    constexpr std::string_view prefix="http://127.0.0.1:";
    need(value.starts_with(prefix),"connect must be http://127.0.0.1:PORT");
    auto port=value.substr(prefix.size());
    need(!port.empty() && port.size()<=5 && port.find_first_not_of("0123456789")==std::string::npos,"invalid loopback port");
    const int n=std::stoi(port);need(n>0 && n<=65535,"invalid loopback port");return value;
}
std::string terminal_text(const std::string& value) {
    // Never pass C0/DEL or encoded C1 controls from model output to the terminal.
    std::string out;
    for(size_t i=0;i<value.size();++i) {
        auto c=static_cast<unsigned char>(value[i]);
        if(c==0xc2 && i+1<value.size()) {
            auto next=static_cast<unsigned char>(value[i+1]);
            if(next>=0x80 && next<=0x9f) {out+='?';++i;continue;}
        }
        if(c=='\t')out+="    ";
        else if(c=='\n' || (c>=32 && c!=127))out+=value[i];
        else out+='?';
    }
    return out;
}
void save_transcript(const std::filesystem::path& path,const Json& transcript) {
    const auto body=transcript.dump(2,' ',false,Json::error_handler_t::replace)+"\n";
    int fd=open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
    need(fd>=0,"cannot create transcript (existing files are never overwritten): "+std::string(std::strerror(errno)));
    size_t at=0;
    while(at<body.size()) {
        auto n=write(fd,body.data()+at,body.size()-at);
        if(n<0 && errno==EINTR)continue;
        if(n<=0) {close(fd);throw std::runtime_error("transcript write failed; partial file retained");}
        at+=size_t(n);
    }
    const int result=close(fd);need(result==0,"transcript close failed");
}
void EventStream::line(std::string value) {
    if(!value.empty() && value.back()=='\r')value.pop_back();
    if(value.empty()) {
        if(data_.empty())return;
        if(data_.back()=='\n')data_.pop_back();
        need(!done_,"data after stream completion");
        if(data_=="[DONE]")done_=true;
        else {
            auto event=Json::parse(data_);if(event.contains("error"))throw std::runtime_error(event["error"].value("message","inference failed"));
            observer_(event);
        }
        data_.clear();return;
    }
    if(value.starts_with("data:")) {
        value.erase(0,5);if(value.starts_with(' '))value.erase(0,1);
        need(data_.size()+value.size()<256*1024,"stream event exceeds bound");data_+=value+'\n';
    }
}
void EventStream::feed(std::string_view bytes) {
    pending_.append(bytes);
    for(size_t n;(n=pending_.find('\n'))!=std::string::npos;) {
        auto value=pending_.substr(0,n);pending_.erase(0,n+1);line(std::move(value));
    }
    need(pending_.size()<256*1024,"stream line exceeds bound");
}
void EventStream::finish() const {need(done_ && pending_.empty() && data_.empty(),"incomplete event stream");}
HttpResult http(const std::string& url,const std::string& method,const std::string& body,
                const std::atomic<bool>& cancel,const std::function<void(const Json&)>& event,long timeout_ms,const std::string& instance) {
    static std::once_flag initialized;
    std::call_once(initialized,[]{need(curl_global_init(CURL_GLOBAL_DEFAULT)==CURLE_OK,"curl initialization failed");});
    using Easy=std::unique_ptr<CURL,decltype(&curl_easy_cleanup)>;
    using Multi=std::unique_ptr<CURLM,decltype(&curl_multi_cleanup)>;
    Easy easy(curl_easy_init(),curl_easy_cleanup);Multi multi(curl_multi_init(),curl_multi_cleanup);
    need(easy && multi,"cannot create HTTP client");
    struct Transfer {CURL* easy;HttpResult result;EventStream parser;bool stream;std::exception_ptr error;};
    Transfer t{easy.get(),{},EventStream(event),bool(event),{}};
    char error[CURL_ERROR_SIZE]={};
    curl_easy_setopt(easy.get(),CURLOPT_URL,url.c_str());curl_easy_setopt(easy.get(),CURLOPT_NOSIGNAL,1L);
    curl_easy_setopt(easy.get(),CURLOPT_NOPROXY,"*");curl_easy_setopt(easy.get(),CURLOPT_CONNECTTIMEOUT_MS,2000L);
    curl_easy_setopt(easy.get(),CURLOPT_TIMEOUT_MS,timeout_ms);curl_easy_setopt(easy.get(),CURLOPT_ERRORBUFFER,error);
    curl_easy_setopt(easy.get(),CURLOPT_WRITEFUNCTION,+[](char* p,size_t size,size_t n,void* raw)->size_t {
        auto& t=*static_cast<Transfer*>(raw);const auto bytes=size*n;
        try {
            curl_easy_getinfo(t.easy,CURLINFO_RESPONSE_CODE,&t.result.status);
            if(t.stream && t.result.status==200)t.parser.feed({p,bytes});
            else {need(t.result.body.size()+bytes<=4*1024*1024,"HTTP response exceeds bound");t.result.body.append(p,bytes);}
            return bytes;
        } catch(...) {t.error=std::current_exception();return 0;}
    });
    curl_easy_setopt(easy.get(),CURLOPT_WRITEDATA,&t);
    std::unique_ptr<curl_slist,decltype(&curl_slist_free_all)> headers(curl_slist_append(nullptr,"Content-Type: application/json"),curl_slist_free_all);
    if(!instance.empty()) {
        need(instance.size()<=128 && instance.find_first_not_of("0123456789-")==std::string::npos,"invalid server instance identity");
        auto* extended=curl_slist_append(headers.get(),("X-ZeroCool-Instance-Id: "+instance).c_str());
        need(extended,"cannot allocate HTTP headers");headers.release();headers.reset(extended);
    }
    curl_easy_setopt(easy.get(),CURLOPT_HTTPHEADER,headers.get());
    if(method=="POST") {
        curl_easy_setopt(easy.get(),CURLOPT_POST,1L);curl_easy_setopt(easy.get(),CURLOPT_HTTPHEADER,headers.get());
        curl_easy_setopt(easy.get(),CURLOPT_POSTFIELDS,body.c_str());curl_easy_setopt(easy.get(),CURLOPT_POSTFIELDSIZE_LARGE,curl_off_t(body.size()));
    } else need(method=="GET","unsupported HTTP method");
    need(curl_multi_add_handle(multi.get(),easy.get())==CURLM_OK,"cannot schedule HTTP request");
    struct Remove {CURLM* m;CURL* e;~Remove(){curl_multi_remove_handle(m,e);}} remove{multi.get(),easy.get()};
    int running=0;
    do {
        need(!cancel.load(),"request cancelled");
        need(curl_multi_perform(multi.get(),&running)==CURLM_OK,"HTTP progress failed");
        if(running) {int fds;need(curl_multi_poll(multi.get(),nullptr,0,50,&fds)==CURLM_OK,"HTTP poll failed");}
    } while(running);
    if(t.error)std::rethrow_exception(t.error);
    int left;CURLMsg* msg;CURLcode result=CURLE_FAILED_INIT;
    while((msg=curl_multi_info_read(multi.get(),&left)))if(msg->msg==CURLMSG_DONE)result=msg->data.result;
    need(result==CURLE_OK,error[0]?error:curl_easy_strerror(result));
    curl_easy_getinfo(easy.get(),CURLINFO_RESPONSE_CODE,&t.result.status);
    if(t.stream && t.result.status==200)t.parser.finish();
    return t.result;
}
ChildServer::ChildServer(const std::string& executable,const std::vector<std::string>& arguments) {
    int pair[2];need(socketpair(AF_UNIX,SOCK_STREAM,0,pair)==0,"cannot create child control channel");
    for(int fd:pair)fcntl(fd,F_SETFD,FD_CLOEXEC);
    constexpr int inherited=198;
    posix_spawn_file_actions_t actions;posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions,pair[1],inherited);
    posix_spawn_file_actions_addopen(&actions,STDIN_FILENO,"/dev/null",O_RDONLY,0);
    posix_spawn_file_actions_addopen(&actions,STDOUT_FILENO,"/dev/null",O_WRONLY,0);
    posix_spawn_file_actions_addopen(&actions,STDERR_FILENO,"/dev/null",O_WRONLY,0);
    std::vector<std::string> values{executable,"serve","--port","0","--control-fd",std::to_string(inherited)};
    values.insert(values.end(),arguments.begin(),arguments.end());
    std::vector<char*> argv;for(auto& value:values)argv.push_back(value.data());argv.push_back(nullptr);
    const int error=posix_spawn(&pid_,executable.c_str(),&actions,nullptr,argv.data(),environ);
    posix_spawn_file_actions_destroy(&actions);close(pair[1]);
    if(error) {close(pair[0]);pid_=-1;throw std::runtime_error("cannot start engine: "+std::string(std::strerror(error)));}
    control_=pair[0];
}
std::string ChildServer::await_endpoint(const std::atomic<bool>& cancel) {
    std::string message;const auto start=std::chrono::steady_clock::now();
    while(message.find('\n')==std::string::npos) {
        need(!cancel.load(),"startup cancelled");
        need(std::chrono::steady_clock::now()-start<std::chrono::seconds(15),"engine listener did not start");
        pollfd p{control_,POLLIN,0};if(poll(&p,1,50)<=0)continue;
        char bytes[1024];auto n=read(control_,bytes,sizeof(bytes));need(n>0,"engine exited before opening its listener");
        message.append(bytes,size_t(n));need(message.size()<=4096,"invalid child handshake");
    }
    const auto handshake=Json::parse(message.substr(0,message.find('\n')));
    instance_=handshake.at("instance_id").get<std::string>();
    need(!instance_.empty(),"missing child instance identity");
    return endpoint(handshake.at("endpoint").get<std::string>());
}
void ChildServer::shutdown() {
    if(control_>=0) {close(control_);control_=-1;}
    if(pid_>0) {while(waitpid(pid_,nullptr,0)<0 && errno==EINTR){}pid_=-1;}
}
ChildServer::~ChildServer(){shutdown();}
} // namespace zerocool::chat
