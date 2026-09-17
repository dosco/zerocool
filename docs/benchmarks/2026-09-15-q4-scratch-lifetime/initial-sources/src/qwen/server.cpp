#include "qwen/session.hpp"
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
void response(int fd,int status,const Json& value) {
    const auto body=dump(value);
    write_all(fd,"HTTP/1.1 "+std::to_string(status)+" Response\r\nContent-Type: application/json\r\nContent-Length: "+
        std::to_string(body.size())+"\r\nConnection: close\r\n\r\n"+body);
}
struct Request {std::string method,path; Json body;};
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
    Request r{data.substr(0,space),data.substr(space+1,space2-space-1),Json::object()};
    size_t length=0; bool has_length=false;
    for(size_t at=first+2;at<split;) {
        auto end=data.find("\r\n",at),colon=data.find(':',at);
        if(colon>=end) throw std::runtime_error("invalid HTTP header");
        auto key=data.substr(at,colon-at),value=data.substr(colon+1,end-colon-1);
        std::transform(key.begin(),key.end(),key.begin(),[](unsigned char c){return char(std::tolower(c));});
        if(key=="transfer-encoding") throw std::runtime_error("chunked request bodies are unsupported");
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
void serve(Model& model,Tokenizer& tokenizer,const Options& options,uint16_t port,const std::atomic<bool>* stop) {
    Socket listener{socket(AF_INET,SOCK_STREAM,0)};
    if(listener.fd<0) throw std::runtime_error("cannot create server socket");
    int one=1; setsockopt(listener.fd,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));
    sockaddr_in address{}; address.sin_family=AF_INET; address.sin_port=htons(port);
    address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    if(bind(listener.fd,reinterpret_cast<sockaddr*>(&address),sizeof(address)) || listen(listener.fd,8))
        throw std::runtime_error("cannot listen on localhost:"+std::to_string(port)+": "+std::strerror(errno));
    std::println(stderr,"FreeLLM listening on http://127.0.0.1:{} (one active conversation)",port);
    Session session(model,tokenizer);
    const std::string model_id=model.checkpoint().model_id();
    uint64_t request_id=0;
    while(!stop || !stop->load()) {
        pollfd incoming{listener.fd,POLLIN,0};
        if(poll(&incoming,1,100)<=0) continue;
        Socket client{accept(listener.fd,nullptr,nullptr)};
        if(client.fd<0) {if(errno==EINTR)continue;throw std::runtime_error("accept failed");}
        setsockopt(client.fd,SOL_SOCKET,SO_NOSIGPIPE,&one,sizeof(one));
        timeval timeout{30,0}; setsockopt(client.fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
        setsockopt(client.fd,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));
        bool streaming=false,headers=false;
        std::atomic<bool> cancelled{false},finished{false};
        std::mutex writer; std::thread monitor;
        try {
            auto r=read_request(client.fd);
            if(r.method=="GET" && r.path=="/v1/models") {
                response(client.fd,200,{{"object","list"},{"data",Json::array({{{"id",model_id},{"object","model"},{"owned_by","local"}}})}});continue;
            }
            if(r.method=="GET" && r.path=="/health") {response(client.fd,200,{{"status","ready"}});continue;}
            if(r.method!="POST" || r.path!="/v1/chat/completions") {response(client.fd,404,{{"error",{{"message","unknown endpoint"}}}});continue;}
            auto& j=r.body;
            if(j.value("model",model_id)!=model_id) throw std::invalid_argument("unknown model");
            if(j.value("n",1)!=1) throw std::invalid_argument("only n=1 is supported");
            for(const auto* key:{"response_format","logit_bias","logprobs","stop"})
                if(j.contains(key) && !j[key].is_null()) throw std::invalid_argument(std::string(key)+" is not supported");
            if(j.value("frequency_penalty",0.0)!=0 || j.value("presence_penalty",0.0)!=0) throw std::invalid_argument("sampling penalties are unsupported");
            Options o=options;
            o.max_tokens=j.value("max_completion_tokens",j.value("max_tokens",options.max_tokens));
            o.temperature=j.value("temperature",options.temperature);o.top_p=j.value("top_p",options.top_p);
            o.top_k=j.value("top_k",options.top_k);o.seed=j.value("seed",options.seed);
            if(o.max_tokens<1 || o.max_tokens>options.context || !std::isfinite(o.temperature) || o.temperature<0 ||
               !std::isfinite(o.top_p) || o.top_p<=0 || o.top_p>1 || o.top_k<0) throw std::invalid_argument("invalid generation parameters");
            auto tools=j.value("tools",Json::array());
            if(j.contains("tool_choice")) {
                if(j["tool_choice"]=="none") tools=Json::array();
                else if(j["tool_choice"]!="auto") throw std::invalid_argument("tool_choice supports auto or none");
            }
            bool thinking=j.value("enable_thinking",false);
            if(j.contains("chat_template_kwargs")) thinking=j["chat_template_kwargs"].value("enable_thinking",thinking);
            auto prompt=tokenizer.encode(tokenizer.render(j.at("messages"),tools,thinking,j.value("reasoning_effort",std::string("xhigh"))));
            if(prompt.size()+uint64_t(o.max_tokens)>uint64_t(options.context)) throw std::invalid_argument("prompt plus max_tokens exceeds context");
            streaming=j.value("stream",false);
            const std::string id="chatcmpl-freellm-"+std::to_string(++request_id);
            auto event=[&](const Json& delta,const Json& reason=Json()) {
                Json item={{"id",id},{"object","chat.completion.chunk"},{"model",model_id},
                    {"choices",Json::array({{{"index",0},{"delta",delta},{"finish_reason",reason}}})}};
                std::lock_guard lock(writer);
                if(!write_all(client.fd,"data: "+dump(item)+"\n\n")) cancelled=true;
            };
            if(streaming) {
                headers=write_all(client.fd,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n");
                if(!headers) throw std::runtime_error("client disconnected");
                event({{"role","assistant"}});
            }
            monitor=std::thread([&]{
                auto heartbeat=std::chrono::steady_clock::now();
                while(!finished) {
                    if(stop && stop->load()) {cancelled=true;break;}
                    pollfd p{client.fd,POLLIN,0};poll(&p,1,100);
                    if(p.revents&(POLLHUP|POLLERR|POLLNVAL)) {cancelled=true;break;}
                    if(p.revents&POLLIN) {
                        char c;const auto n=recv(client.fd,&c,1,MSG_PEEK|MSG_DONTWAIT);
                        if(n==0) {cancelled=true;break;}
                    }
                    if(streaming && std::chrono::steady_clock::now()-heartbeat>std::chrono::seconds(10)) {
                        std::lock_guard lock(writer);
                        if(!write_all(client.fd,": prefill\n\n")) {cancelled=true;break;}
                        heartbeat=std::chrono::steady_clock::now();
                    }
                }
            });
            StreamParser parser(thinking);
            auto result=session.generate(prompt,o,[&](int token){
                const std::array<int,1> ids={token}; const auto bytes=tokenizer.decode(ids);
                if(streaming) for(const auto& delta:parser.push(bytes)) event(delta);
            },&cancelled);
            finished=true;monitor.join();
            auto message=parse_output(result.text,tools,thinking);
            auto reason=message.contains("tool_calls")?std::string("tool_calls"):result.finish_reason;
            if(streaming) {
                for(const auto& delta:parser.push("",true)) event(delta);
                if(message.contains("tool_calls")) {
                    auto calls=message["tool_calls"];
                    for(size_t i=0;i<calls.size();++i) {calls[i]["index"]=i;calls[i]["id"]=id+"-"+std::to_string(i);}
                    event({{"tool_calls",calls}});
                }
                event(Json::object(),reason);
                std::lock_guard lock(writer);write_all(client.fd,"data: [DONE]\n\n");
            } else {
                if(message.contains("tool_calls")) for(size_t i=0;i<message["tool_calls"].size();++i)
                    message["tool_calls"][i]["id"]=id+"-"+std::to_string(i);
                auto diagnostics=result.json();
                diagnostics["after"]=model.stats();
                response(client.fd,200,{{"id",id},{"object","chat.completion"},{"model",model_id},
                    {"choices",Json::array({{{"index",0},{"message",message},{"finish_reason",reason}}})},
                    {"usage",{{"prompt_tokens",result.prompt_tokens},{"completion_tokens",result.tokens.size()},
                        {"total_tokens",result.prompt_tokens+result.tokens.size()},
                        {"prompt_tokens_details",{{"cached_tokens",result.reused_tokens}}}}},
                    {"freellm",std::move(diagnostics)}});
            }
        } catch(const std::exception& error) {
            finished=true;if(monitor.joinable())monitor.join();
            Json failure={{"error",{{"message",error.what()},{"type","inference_error"}}}};
            if(headers) write_all(client.fd,"data: "+dump(failure)+"\n\ndata: [DONE]\n\n");
            else response(client.fd,400,failure);
        }
    }
}
} // namespace freellm::qwen
