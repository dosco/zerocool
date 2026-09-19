#include "qwen/server.hpp"
#include <condition_variable>
#include <deque>
#include <set>
#include <thread>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstring>
#include <mutex>
#include <print>
#include <stdexcept>
#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace freellm::qwen {
namespace {
struct Socket { int fd; ~Socket(){if(fd>=0)close(fd);} };
bool write_all(int fd,const std::string& s) {
    size_t at=0;
    while(at<s.size()) {
        const auto n=send(fd,s.data()+at,s.size()-at,0);
        if(n<0 && errno==EINTR) continue;
        if(n<=0) return false;
        at+=size_t(n);
    }
    return true;
}
std::string dump(const Json& j) {return j.dump(-1,' ',false,Json::error_handler_t::replace);}
const char* phrase(int status) {
    switch(status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 403: return "Forbidden";
        case 404: return "Not Found";
        case 409: return "Conflict";
        case 415: return "Unsupported Media Type";
        case 429: return "Too Many Requests";
        case 503: return "Service Unavailable";
        default: return "Error";
    }
}
void response(int fd,int status,const Json& value,const std::string& request_id={}) {
    const auto body=dump(value);
    write_all(fd,"HTTP/1.1 "+std::to_string(status)+" "+phrase(status)+"\r\nContent-Type: application/json\r\nContent-Length: "+
        std::to_string(body.size())+"\r\nConnection: close\r\n"+
        (request_id.empty()?std::string():"X-Request-Id: "+request_id+"\r\n")+"\r\n"+body);
}
struct Request {std::string method,path; Json body;std::string instance,host,origin,content_type;};
std::string trimmed(const std::string& value) {
    const auto begin=value.find_first_not_of(" \t");
    if(begin==std::string::npos) return {};
    return value.substr(begin,value.find_last_not_of(" \t")-begin+1);
}
// A browser can reach this listener. Only a loopback Host is served, which
// also refuses a rebound DNS name that resolves to 127.0.0.1.
bool loopback_host(const std::string& value) {
    auto host=value;
    if(!host.empty() && host.front()=='[') {
        const auto close=host.find(']');
        if(close==std::string::npos) return false;
        host=host.substr(1,close-1);
    } else if(const auto colon=host.rfind(':');colon!=std::string::npos &&
              host.find_first_not_of("0123456789",colon+1)==std::string::npos) host=host.substr(0,colon);
    std::transform(host.begin(),host.end(),host.begin(),[](unsigned char c){return char(std::tolower(c));});
    return host=="127.0.0.1" || host=="localhost" || host=="::1";
}
Request read_request(int fd) {
    constexpr size_t header_limit=16384,body_limit=4*MiB;
    std::string data; std::array<char,8192> buf;
    size_t split=std::string::npos;
    while((split=data.find("\r\n\r\n"))==std::string::npos) {
        auto n=recv(fd,buf.data(),buf.size(),0);
        if(n<=0) throw std::runtime_error("incomplete HTTP header");
        data.append(buf.data(),size_t(n));
        if(data.size()>header_limit+buf.size()) throw std::runtime_error("HTTP header too large");
    }
    if(split>header_limit) throw std::runtime_error("HTTP header too large");
    const auto first=data.find("\r\n"),space=data.find(' '),space2=data.find(' ',space+1);
    if(space==std::string::npos || space2==std::string::npos || space2>first) throw std::runtime_error("invalid request line");
    Request r{data.substr(0,space),data.substr(space+1,space2-space-1),Json::object(),{},{},{},{}};
    size_t length=0; bool has_length=false;
    for(size_t at=first+2;at<split;) {
        auto end=data.find("\r\n",at),colon=data.find(':',at);
        if(colon>=end) throw std::runtime_error("invalid HTTP header");
        auto key=data.substr(at,colon-at),value=data.substr(colon+1,end-colon-1);
        std::transform(key.begin(),key.end(),key.begin(),[](unsigned char c){return char(std::tolower(c));});
        if(key=="transfer-encoding") throw std::runtime_error("chunked request bodies are unsupported");
        if(key=="x-freellm-instance-id") {
            if(!r.instance.empty())throw std::runtime_error("duplicate server instance header");
            r.instance=trimmed(value);
            if(r.instance.empty())throw std::runtime_error("empty server instance header");
        }
        if(key=="host") r.host=trimmed(value);
        if(key=="origin") r.origin=trimmed(value);
        if(key=="content-type") {
            r.content_type=trimmed(value);
            std::transform(r.content_type.begin(),r.content_type.end(),r.content_type.begin(),
                [](unsigned char c){return char(std::tolower(c));});
        }
        if(key=="content-length") {
            if(has_length) throw std::runtime_error("duplicate content length");
            has_length=true;
            auto begin=value.find_first_not_of(" \t"),last=value.find_last_not_of(" \t");
            if(begin==std::string::npos) throw std::runtime_error("invalid content length");
            value=value.substr(begin,last-begin+1);
            if(value.find_first_not_of("0123456789")!=std::string::npos) throw std::runtime_error("invalid content length");
            length=std::stoull(value);
            if(length>body_limit) throw std::runtime_error("request body exceeds 4MiB");
        }
        at=end+2;
    }
    const auto body_at=split+4;
    while(data.size()-body_at<length) {
        auto n=recv(fd,buf.data(),std::min(buf.size(),length-(data.size()-body_at)),0);
        if(n<=0) throw std::runtime_error("incomplete request body");
        data.append(buf.data(),size_t(n));
    }
    if(length) r.body=Json::parse(data.substr(body_at,length));
    return r;
}
}
std::unique_ptr<ChatExecutor> make_native_chat(const Options& options);
namespace {
constexpr size_t StreamLimit=256*1024;
struct Job {
    std::string id;
    ChatRequest request;
    bool reset=false,stream=false,include_usage=false;
    int64_t created=std::time(nullptr);
    std::atomic<bool> cancel=false;
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::string> events;
    size_t bytes=0;
    bool done=false;
    int status=200;
    Json result;
    void push(std::string item) {
        std::lock_guard lock(mutex);
        if(cancel || item.size()>StreamLimit-bytes) {cancel=true;throw std::runtime_error("client disconnected or streaming queue full");}
        bytes+=item.size();events.push_back(std::move(item));changed.notify_all();
    }
};
struct Coordinator {
    std::mutex mutex;
    std::condition_variable changed;
    std::shared_ptr<Job> pending,active;
    bool stopping=false,ready=false,failed=false;
    uint64_t sequence=0;
    Json status;
    void progress(const Json& update) {
        std::lock_guard lock(mutex);
        // Cancellation is observable immediately, even if a layer is finishing.
        for(auto i=update.begin();i!=update.end();++i)
            if(i.key()!="phase" || !active || !active->cancel) status[i.key()]=i.value();
        status["updated_ns"]=monotonic_ns();
    }
    void cancel(const std::shared_ptr<Job>& job) {
        job->cancel=true;
        std::lock_guard lock(mutex);
        if(active==job) {status["phase"]="cancelling";status["updated_ns"]=monotonic_ns();}
        job->changed.notify_all();
    }
    void shutdown() {
        std::lock_guard lock(mutex);stopping=true;
        if(active) active->cancel=true;
        changed.notify_all();
    }
};
Json error_body(const std::string& message,const char* type="inference_error") {
    return {{"error",{{"message",message},{"type",type}}}};
}
Json chunk(const Job& job,const std::string& model,const Json& delta,const Json& reason=Json()) {
    Json result={{"id",job.id},{"object","chat.completion.chunk"},{"created",job.created},{"model",model},
        {"choices",Json::array({{{"index",0},{"delta",delta},{"finish_reason",reason}}})}};
    if(job.include_usage)result["usage"]=nullptr;
    return result;
}
void inference(Coordinator& c,const Options& options,const ChatExecutorFactory& factory) {
    std::unique_ptr<ChatExecutor> executor;
    try {
        executor=factory?factory():make_native_chat(options);
        executor->initialize([&](const Json& event){c.progress(event);});
        std::lock_guard lock(c.mutex);c.ready=true;c.status["phase"]="ready";c.changed.notify_all();
    } catch(const std::exception& e) {
        // A listener reporting failed startup must not retain a partially loaded
        // model indefinitely. Cleanup stays on the inference owner.
        if(executor) {try {executor->drain();executor->clear();} catch(...) {}executor.reset();}
        std::lock_guard lock(c.mutex);c.failed=true;c.status["phase"]="failed";c.status["error"]=e.what();
        c.changed.notify_all();
    }
    for(;;) {
        std::shared_ptr<Job> job;
        {
            std::unique_lock lock(c.mutex);c.changed.wait(lock,[&]{return c.stopping || bool(c.pending);});
            if(!c.pending) break;
            job=std::exchange(c.pending,{});
        }
        Json result;int http_status=200;std::string finish_event,usage_event;
        try {
            if(job->cancel) throw std::runtime_error("generation cancelled");
            if(job->reset) {executor->clear();result={{"status","ready"}};}
            else {
                ChatObserver delta;
                if(job->stream) delta=[&](const Json& value){job->push("data: "+dump(chunk(*job,api_model_id(options.artifact),value))+"\n\n");};
                auto answer=executor->generate(job->request,delta,[&](const Json& event){c.progress(event);},job->cancel);
                if(job->cancel) throw std::runtime_error("generation cancelled");
                if(answer.message.contains("tool_calls")) {
                    auto calls=answer.message["tool_calls"];
                    for(size_t i=0;i<calls.size();++i) {calls[i]["id"]=job->id+"-"+std::to_string(i);calls[i]["index"]=i;}
                    if(delta) delta({{"tool_calls",calls}});
                    for(auto& call:calls) call.erase("index");
                    answer.message["tool_calls"]=std::move(calls);
                }
                result={{"id",job->id},{"object","chat.completion"},{"created",job->created},{"model",api_model_id(options.artifact)},
                    {"choices",Json::array({{{"index",0},{"message",answer.message},{"finish_reason",answer.finish_reason}}})},
                    {"usage",answer.usage},{"freellm",answer.diagnostics}};
                if(job->stream) {
                    finish_event="data: "+dump(chunk(*job,api_model_id(options.artifact),Json::object(),answer.finish_reason))+"\n\n";
                    if(job->include_usage) {
                        auto usage=chunk(*job,api_model_id(options.artifact),Json::object());
                        usage["choices"]=Json::array();usage["usage"]=answer.usage;
                        usage_event="data: "+dump(usage)+"\n\n";
                    }
                }
            }
        } catch(const std::exception& e) {
            http_status=400;result=error_body(e.what());
        }
        // Do not admit another generation until every outstanding user has drained.
        {std::lock_guard lock(c.mutex);c.status["phase"]="draining";}
        try {
            executor->drain();
            // Success is only observable after draining succeeds. Clients often
            // stop reading at finish_reason, without waiting for [DONE].
            if(http_status==200) {
                try {
                    if(job->cancel)throw std::runtime_error("generation cancelled");
                    if(!finish_event.empty())job->push(std::move(finish_event)+usage_event);
                } catch(const std::exception& e) {http_status=400;result=error_body(e.what());}
            }
            if(http_status!=200 || job->cancel)executor->recover();
        }
        catch(const std::exception& e) {
            http_status=500;result=error_body(e.what());
            std::lock_guard lock(c.mutex);c.failed=true;c.ready=false;c.status["error"]=e.what();
        }
        {
            std::lock_guard lock(c.mutex);
            c.status["last_request_id"]=job->id;c.status["last_request_cancelled"]=job->cancel.load();
            c.status["phase"]=c.failed?"failed":"ready";c.status["active_request_id"]=nullptr;
            c.status["busy"]=false;c.status["updated_ns"]=monotonic_ns();
            if(http_status!=200)c.status["last_error"]=result["error"]["message"];
            c.active.reset();
        }
        {
            std::lock_guard lock(job->mutex);job->result=std::move(result);job->status=http_status;job->done=true;job->changed.notify_all();
        }
    }
    if(executor) {try {executor->drain();executor->clear();} catch(...) {}}
    // Executor and all Metal owners are destroyed here, on their owning worker.
}
bool disconnected(int fd) {
    pollfd p{fd,POLLIN,0};poll(&p,1,0);
    if(p.revents&(POLLHUP|POLLERR|POLLNVAL))return true;
    if(p.revents&POLLIN) {char byte;return recv(fd,&byte,1,MSG_PEEK|MSG_DONTWAIT)==0;}
    return false;
}
void handle(int fd,Coordinator& c,const Options& options) {
    bool headers=false;std::shared_ptr<Job> job;
    try {
        auto request=read_request(fd);
        // Cross-origin and non-loopback requests are refused before any work,
        // so a page in the user's browser cannot drive the local engine.
        if(!request.origin.empty()) {
            response(fd,403,error_body("cross-origin requests are not accepted","forbidden"));return;
        }
        if(!request.host.empty() && !loopback_host(request.host)) {
            response(fd,403,error_body("unexpected Host header; this listener serves loopback only","forbidden"));return;
        }
        if(request.method=="POST" && !request.content_type.starts_with("application/json")) {
            response(fd,415,error_body("POST requires Content-Type: application/json","invalid_request_error"));return;
        }
        if(!request.instance.empty()) {
            bool matches;
            {std::lock_guard lock(c.mutex);matches=request.instance==c.status.at("instance_id").get<std::string>();}
            if(!matches) {response(fd,409,error_body("server instance changed; reconnect explicitly","server_instance_changed"));return;}
        }
        if(request.method=="GET" && (request.path=="/health" || request.path=="/freellm/status")) {
            Json status;int code=200;
            {std::lock_guard lock(c.mutex);status=c.status;code=c.failed || !c.ready?503:200;}
            if(request.path=="/health")response(fd,code,{{"status",code==200?"ready":status.value("phase","loading_model")},{"phase",status["phase"]}});
            else response(fd,200,status);
            return;
        }
        if(request.method=="GET" && request.path=="/v1/models") {
            response(fd,200,{{"object","list"},{"data",Json::array({{{"id",api_model_id(options.artifact)},
                {"object","model"},{"created",0},{"owned_by","local"}}})}});return;
        }
        const bool reset=request.method=="POST" && request.path=="/freellm/session/reset";
        if(!reset && (request.method!="POST" || request.path!="/v1/chat/completions")) {response(fd,404,error_body("unknown endpoint"));return;}
        job=std::make_shared<Job>();job->reset=reset;
        if(!reset) {
            job->request=parse_chat_request(request.body,options);
            if(request.body.contains("stream") && !request.body["stream"].is_null()) {
                if(!request.body["stream"].is_boolean()) throw std::invalid_argument("stream must be a boolean");
                job->stream=request.body["stream"].get<bool>();
            }
            if(request.body.contains("stream_options") && !request.body["stream_options"].is_null()) {
                const auto& stream_options=request.body["stream_options"];
                if(!stream_options.is_object()) throw std::invalid_argument("stream_options must be an object");
                if(stream_options.contains("include_usage") && !stream_options["include_usage"].is_null()) {
                    if(!stream_options["include_usage"].is_boolean()) throw std::invalid_argument("stream_options.include_usage must be a boolean");
                    job->include_usage=stream_options["include_usage"].get<bool>();
                }
            }
        }
        int rejected=0;std::string why;
        {
            std::lock_guard lock(c.mutex);
            if(c.stopping || c.failed || !c.ready) {rejected=503;why=c.failed?"engine startup or drain failed":"engine is loading or stopping";}
            else if(c.active) {rejected=reset?409:429;why="one conversation is already active; wait for it to finish draining";}
            else {
                job->id="chatcmpl-freellm-"+std::to_string(job->created)+"-"+std::to_string(++c.sequence);
                c.active=c.pending=job;c.status["busy"]=true;c.status["active_request_id"]=job->id;
                c.status["phase"]=reset?"resetting":"preparing";
                for(auto key:{"prompt_tokens","completed_prompt_tokens","reused_tokens","matched_prefix_tokens","output_tokens","elapsed_ms","time_to_first_token_ms","generation_tokens_per_second","last_error"})c.status[key]=nullptr;
                c.changed.notify_all();
            }
        }
        if(rejected) {response(fd,rejected,error_body(why,"engine_unavailable"));return;}
        auto heartbeat=std::chrono::steady_clock::now();
        for(;;) {
            if(disconnected(fd)) {c.cancel(job);return;}
            std::string event;bool done;int status;Json result;
            {
                std::unique_lock lock(job->mutex);
                job->changed.wait_for(lock,std::chrono::milliseconds(50),[&]{return job->done || !job->events.empty();});
                if(!job->events.empty()) {event=std::move(job->events.front());job->events.pop_front();job->bytes-=event.size();}
                done=job->done && job->events.empty();status=job->status;
                if(done)result=job->result;
            }
            if(!event.empty()) {
                if(!headers) {
                    if(!write_all(fd,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nX-Request-Id: "+job->id+"\r\n\r\n")) {c.cancel(job);return;}
                    headers=true;
                }
                if(!write_all(fd,event)) {c.cancel(job);return;}
            }
            if(done) {
                if(headers) {
                    if(status!=200)write_all(fd,"data: "+dump(result)+"\n\n");
                    write_all(fd,"data: [DONE]\n\n");
                } else response(fd,status,result,job->id);
                return;
            }
            if(headers && std::chrono::steady_clock::now()-heartbeat>=std::chrono::seconds(5)) {
                if(!write_all(fd,": heartbeat\n\n")) {c.cancel(job);return;}
                heartbeat=std::chrono::steady_clock::now();
            }
        }
    } catch(const std::exception& e) {
        if(job) {std::lock_guard lock(c.mutex);if(c.active==job)job->cancel=true;}
        if(!headers)response(fd,400,error_body(e.what(),"invalid_request_error"));
    }
}
}
void serve(const Options& options,uint16_t port,const std::atomic<bool>* stop,int control_fd,ChatExecutorFactory factory) {
    Socket listener{socket(AF_INET,SOCK_STREAM,0)};
    if(listener.fd<0) throw std::runtime_error("cannot create server socket");
    int one=1;setsockopt(listener.fd,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));
    sockaddr_in address{};address.sin_family=AF_INET;address.sin_port=htons(port);address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    if(bind(listener.fd,reinterpret_cast<sockaddr*>(&address),sizeof(address)) || listen(listener.fd,8))
        throw std::runtime_error("cannot listen on localhost:"+std::to_string(port)+": "+std::strerror(errno));
    socklen_t size=sizeof(address);if(getsockname(listener.fd,reinterpret_cast<sockaddr*>(&address),&size))throw std::runtime_error("cannot read listener address");
    const auto endpoint="http://127.0.0.1:"+std::to_string(ntohs(address.sin_port));
    const auto instance=std::to_string(getpid())+"-"+std::to_string(monotonic_ns());
    std::println(stderr,"FreeLLM listening on {} (one active conversation)",endpoint);
    if(control_fd>=0) {
        setsockopt(control_fd,SOL_SOCKET,SO_NOSIGPIPE,&one,sizeof(one));
        if(!write_all(control_fd,dump({{"endpoint",endpoint},{"instance_id",instance}})+"\n"))return;
    }
    Coordinator coordinator;
    coordinator.status={{"version",1},{"instance_id",instance},{"phase","loading_model"},{"busy",false},{"active_request_id",nullptr},
        {"model",api_model_id(options.artifact)},{"artifact_revision",artifact_revision(options.artifact)},
        {"build",build_fingerprint()},{"context_limit",options.context},{"requested_memory_bytes",options.memory},
        {"memory_plan",nullptr},{"process",nullptr},{"memory_sample_ns",nullptr},{"updated_ns",monotonic_ns()}};
    std::thread worker([&]{inference(coordinator,options,factory);});
    std::mutex sockets_mutex;std::condition_variable sockets_changed;
    std::deque<int> sockets;std::set<int> active_sockets;bool done=false;
    std::vector<std::thread> handlers;
    auto cleanup=[&] {
        coordinator.shutdown();
        {
            std::lock_guard lock(sockets_mutex);done=true;
            for(int fd:sockets)close(fd);sockets.clear();
            for(int fd:active_sockets)shutdown(fd,SHUT_RDWR);
            sockets_changed.notify_all();
        }
        for(auto& thread:handlers)thread.join();
        worker.join();
    };
    try {
    for(int i=0;i<4;++i) handlers.emplace_back([&] {
        for(;;) {
            int fd;
            {
                std::unique_lock lock(sockets_mutex);sockets_changed.wait(lock,[&]{return done || !sockets.empty();});
                if(done && sockets.empty())return;
                fd=sockets.front();sockets.pop_front();active_sockets.insert(fd);
            }
            Socket client{fd};handle(fd,coordinator,options);
            {std::lock_guard lock(sockets_mutex);active_sockets.erase(fd);}
        }
    });
    auto sample=std::chrono::steady_clock::time_point::min();
    while(!stop || !stop->load()) {
        const auto now=std::chrono::steady_clock::now();
        if(sample==std::chrono::steady_clock::time_point::min() || now-sample>=std::chrono::seconds(1)) {
            auto process=process_memory();const auto timestamp=monotonic_ns();
            {std::lock_guard lock(coordinator.mutex);coordinator.status["process"]=std::move(process);coordinator.status["memory_sample_ns"]=timestamp;}
            sample=now;
        }
        pollfd incoming[2]={{listener.fd,POLLIN,0},{control_fd,POLLIN,0}};
        if(poll(incoming,control_fd>=0?2:1,50)<0) {if(errno==EINTR)continue;break;}
        if(control_fd>=0 && incoming[1].revents)break;
        if(!(incoming[0].revents&POLLIN))continue;
        const int fd=accept(listener.fd,nullptr,nullptr);if(fd<0)continue;
        setsockopt(fd,SOL_SOCKET,SO_NOSIGPIPE,&one,sizeof(one));
        timeval timeout{2,0};setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));
        bool queued=false;
        {std::lock_guard lock(sockets_mutex);if(sockets.size()<8) {sockets.push_back(fd);queued=true;sockets_changed.notify_one();}}
        if(!queued) {response(fd,503,error_body("too many pending connections","engine_unavailable"));close(fd);}
    }
    } catch(...) {cleanup();throw;}
    cleanup();
}
} // namespace freellm::qwen
