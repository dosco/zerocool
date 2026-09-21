#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "engine/server.hpp"
#include "chat_test_executor.hpp"
#include "engine/chat_client.hpp"
#include <chrono>
#include <arpa/inet.h>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <poll.h>
#include <set>
#include <thread>
#include <sys/socket.h>
#include <unistd.h>
using namespace zerocool::engine;
namespace client=zerocool::chat;
using namespace std::chrono_literals;
namespace {
using zerocool::engine::chat_test::Control;
using zerocool::engine::chat_test::TestExecutor;
struct Server {
    std::shared_ptr<Control> control=std::make_shared<Control>();
    int pair[2]={-1,-1};std::thread thread;std::string url;std::atomic<bool> stop=false;
    explicit Server(bool loading=false,bool fail=false) {
        control->loading=loading;control->fail_load=fail;
        if(socketpair(AF_UNIX,SOCK_STREAM,0,pair))throw std::runtime_error("socketpair failed");
        thread=std::thread([&]{try {serve(Options{},0,&stop,pair[1],[&]{return std::make_unique<TestExecutor>(control);});} catch(...) {shutdown(pair[1],SHUT_RDWR);}});
        pollfd fd{pair[0],POLLIN,0};char bytes[4096];
        auto n=poll(&fd,1,5000)>0?read(pair[0],bytes,sizeof(bytes)):0;
        if(n<=0) {stop=true;control->loading=false;thread.join();close(pair[0]);close(pair[1]);throw std::runtime_error("test listener unavailable");}
        url=client::Json::parse(std::string(bytes,size_t(n))).at("endpoint");
        if(!loading && !fail)wait_phase("ready");
    }
    ~Server(){control->loading=false;control->hold=false;control->hold_drain=false;stop=true;thread.join();close(pair[0]);close(pair[1]);}
    client::HttpResult get(const std::string& path){return client::http(url+path,"GET","",stop);}
    Json status(){return Json::parse(get("/zerocool/status").body);}
    void wait_phase(const std::string& phase) {
        for(int i=0;i<200;++i) {if(status()["phase"]==phase)return;std::this_thread::sleep_for(5ms);}
        throw std::runtime_error("phase did not reach "+phase);
    }
    Json request(std::string text="hello",bool stream=false) {return {{"model",api_model_id(Artifact::Q4)},{"messages",Json::array({{{"role","user"},{"content",text}}})},{"max_tokens",8},{"stream",stream}};}
    client::HttpResult post(const Json& body) {return client::http(url+"/v1/chat/completions","POST",body.dump(),stop);}
};
}
TEST_CASE("event stream preserves every byte boundary and detects incomplete output") {
    const std::string data=": heartbeat\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"café 🦉\"}}]}\r\n\r\ndata: [DONE]\n\n";
    for(size_t cut=0;cut<=data.size();++cut) {
        int calls=0;client::EventStream parser([&](const auto& j){CHECK(j["choices"][0]["delta"]["content"]=="café 🦉");++calls;});
        parser.feed(std::string_view(data).substr(0,cut));parser.feed(std::string_view(data).substr(cut));parser.finish();CHECK(calls==1);
    }
    client::EventStream parser([](const auto&){});parser.feed("data: {}\n\n");CHECK_THROWS(parser.finish());
    CHECK_THROWS(parser.feed("data: {\"error\":{\"message\":\"bad\"}}\n\n"));
    client::EventStream complete([](const auto&){});complete.feed("data: [DONE]\n\n");CHECK_THROWS(complete.feed("data: {}\n\n"));
}
TEST_CASE("terminal text cannot issue escapes and saves never overwrite") {
    CHECK(client::terminal_text("a\x1b[2J\t\x7f\xc2\x9b\n🦉")=="a?[2J    ??\n🦉");
    CHECK_THROWS(client::endpoint("https://example.com"));CHECK_THROWS(client::endpoint("http://127.0.0.1:8080/path"));
    CHECK(client::endpoint("http://127.0.0.1:8080/")=="http://127.0.0.1:8080");
    auto path=std::filesystem::temp_directory_path()/("zerocool-chat-"+std::to_string(monotonic_ns())+".json");
    client::save_transcript(path,{{"content","private"}});CHECK_THROWS(client::save_transcript(path,{{"content","overwrite"}}));
    std::ifstream f(path);client::Json j;f>>j;CHECK(j["content"]=="private");std::filesystem::remove(path);
}
TEST_CASE("listener remains responsive during loading and startup failure") {
    Server s(true);CHECK(s.get("/health").status==503);CHECK(s.get("/v1/models").status==200);
    CHECK(s.post(s.request()).status==503);s.control->loading=false;s.wait_phase("ready");CHECK(s.get("/health").status==200);
    Server failed(false,true);failed.wait_phase("failed");CHECK(failed.get("/health").status==503);
    CHECK(failed.status()["error"]=="test startup failure");CHECK(failed.post(failed.request()).status==503);
    CHECK(failed.control->destroyed==1);CHECK(failed.control->resets==1);
}
TEST_CASE("chat envelopes usage and tool streams use one owning inference thread") {
    Server s;
    auto plain=Json::parse(s.post(s.request()).body);CHECK(plain["choices"][0]["message"]["content"]=="OK 🦉");CHECK(plain["created"].get<int64_t>()>0);
    auto body=s.request("tool",true);body["stream_options"]={{"include_usage",true}};
    std::vector<client::Json> events;
    auto result=client::http(s.url+"/v1/chat/completions","POST",body.dump(),s.stop,[&](const auto& e){events.push_back(e);});
    CHECK(result.status==200);REQUIRE(events.size()==5);CHECK(events[2]["choices"][0]["delta"]["tool_calls"][0]["index"]==0);
    CHECK(events[3]["choices"][0]["finish_reason"]=="tool_calls");CHECK(events.back()["choices"].empty());CHECK(events.back()["usage"]["total_tokens"]==10);
    CHECK(events[0]["usage"].is_null());
    body=s.request("think",true);body["enable_thinking"]=true;events.clear();
    client::http(s.url+"/v1/chat/completions","POST",body.dump(),s.stop,[&](const auto& e){events.push_back(e);});
    CHECK(events[1]["choices"][0]["delta"]["reasoning_content"]=="Check café 🦉 first.");
    CHECK_FALSE(events.back().contains("usage"));
    CHECK(s.control->owners.size()==1);
    auto reset=client::http(s.url+"/zerocool/session/reset","POST","{}",s.stop);CHECK(reset.status==200);CHECK(s.control->resets==1);
}
TEST_CASE("HTTP accepts fragmented Unicode requests and rejects ambiguous lengths") {
    Server s;
    auto raw=[&](const std::string& request) {
        int fd=socket(AF_INET,SOCK_STREAM,0);REQUIRE(fd>=0);
        sockaddr_in address{};address.sin_family=AF_INET;address.sin_port=htons(std::stoi(s.url.substr(s.url.rfind(':')+1)));address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
        REQUIRE(connect(fd,reinterpret_cast<sockaddr*>(&address),sizeof(address))==0);
        timeval timeout{2,0};setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
        for(char byte:request)REQUIRE(send(fd,&byte,1,0)==1);
        std::string response;char bytes[4096];ssize_t n;
        while((n=recv(fd,bytes,sizeof(bytes),0))>0)response.append(bytes,size_t(n));
        close(fd);return response;
    };
    const auto body=s.request("café 🦉").dump();
    const auto json="Content-Type: application/json\r\n";
    const auto response=raw("POST /v1/chat/completions HTTP/1.1\r\n"+std::string(json)+"Content-Length: "+std::to_string(body.size())+"\r\n\r\n"+body);
    CHECK(response.starts_with("HTTP/1.1 200 OK"));CHECK(response.find("OK 🦉")!=std::string::npos);
    CHECK(response.find("X-Request-Id: chatcmpl-zerocool-")!=std::string::npos);
    CHECK(raw("POST /v1/chat/completions HTTP/1.1\r\n"+std::string(json)+"Content-Length: 0\r\nContent-Length: 0\r\n\r\n").starts_with("HTTP/1.1 400 Bad Request"));
    // A page in a browser must not be able to drive or reset the local engine.
    CHECK(raw("POST /v1/chat/completions HTTP/1.1\r\nContent-Length: "+std::to_string(body.size())+"\r\n\r\n"+body)
        .starts_with("HTTP/1.1 415 Unsupported Media Type"));
    CHECK(raw("POST /v1/chat/completions HTTP/1.1\r\n"+std::string(json)+"Origin: https://example.com\r\nContent-Length: "+
        std::to_string(body.size())+"\r\n\r\n"+body).starts_with("HTTP/1.1 403 Forbidden"));
    CHECK(raw("GET /zerocool/status HTTP/1.1\r\nHost: attacker.example\r\n\r\n").starts_with("HTTP/1.1 403 Forbidden"));
    CHECK(raw("GET /zerocool/status HTTP/1.1\r\nHost: 127.0.0.1:1234\r\n\r\n").starts_with("HTTP/1.1 200 OK"));
    CHECK(raw("GET /zerocool/status HTTP/1.1\r\nHOST: localhost\r\n\r\n").starts_with("HTTP/1.1 200 OK"));
    CHECK(raw("POST /zerocool/session/reset HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}").starts_with("HTTP/1.1 415"));
}
TEST_CASE("request parameters are rejected by type rather than coerced") {
    const Options options;
    auto body=[&](Json extra) {
        Json request={{"messages",Json::array({{{"role","user"},{"content","hi"}}})}};
        for(auto it=extra.begin();it!=extra.end();++it) request[it.key()]=it.value();
        return request;
    };
    CHECK(parse_chat_request(body({}),options).options.max_tokens==options.max_tokens);
    CHECK(parse_chat_request(body({{"max_tokens",16}}),options).options.max_tokens==16);
    CHECK(parse_chat_request(body({{"max_completion_tokens",7},{"max_tokens",16}}),options).options.max_tokens==7);
    CHECK_THROWS(parse_chat_request(body({{"max_tokens",1.9}}),options));
    CHECK_THROWS(parse_chat_request(body({{"max_tokens",1e300}}),options));
    CHECK_THROWS(parse_chat_request(body({{"max_tokens","8"}}),options));
    CHECK_THROWS(parse_chat_request(body({{"temperature","hot"}}),options));
    CHECK_THROWS(parse_chat_request(body({{"seed",-1}}),options));
    CHECK_THROWS(parse_chat_request(body({{"n",2}}),options));
    CHECK_THROWS(parse_chat_request(body({{"enable_thinking","yes"}}),options));
    CHECK_THROWS(parse_chat_request(body({{"model","other-model"}}),options));
    CHECK_THROWS(parse_chat_request(Json::array(),options));
    // Null is absent, not a value: a client that always sends the field works.
    CHECK(parse_chat_request(body({{"max_tokens",nullptr},{"stop",nullptr}}),options).options.max_tokens==options.max_tokens);
    // Text parts are accepted; other modalities still fail explicitly.
    Json parts={{"messages",Json::array({{{"role","user"},{"content",Json::array({{{"type","text"},{"text","hi"}}})}}})}};
    CHECK(parse_chat_request(parts,options).messages.size()==1);
    parts["messages"][0]["content"][0]["type"]="image_url";
    CHECK_THROWS(parse_chat_request(parts,options));
}
TEST_CASE("busy rejection cancellation and draining precede reuse") {
    Server s;s.control->hold=true;s.control->hold_drain=true;
    std::atomic<bool> cancel=false;std::thread request([&]{try {client::http(s.url+"/v1/chat/completions","POST",s.request("hello",true).dump(),cancel,[](const auto&) {},0);}catch(...) {}});
    s.wait_phase("prefill");CHECK(s.post(s.request()).status==429);
    CHECK(client::http(s.url+"/zerocool/session/reset","POST","{}",s.stop).status==409);
    const auto start=std::chrono::steady_clock::now();cancel=true;request.join();
    CHECK(std::chrono::steady_clock::now()-start<500ms);s.wait_phase("draining");CHECK(s.control->cancelled);
    CHECK(s.post(s.request()).status==429);s.control->hold=false;s.control->hold_drain=false;s.wait_phase("ready");
    CHECK(s.post(s.request()).status==200);CHECK(s.control->resets>=1);
}
TEST_CASE("unsupported requests and bounded stream failures recover") {
    Server s;auto body=s.request();body["stop"]="end";CHECK(s.post(body).status==400);
    CHECK(s.post(s.request("invalid")).status==400);s.wait_phase("ready");
    CHECK_THROWS(client::http(s.url+"/v1/chat/completions","POST",s.request("overflow",true).dump(),s.stop,[](const auto&){}));
    s.wait_phase("ready");CHECK(s.post(s.request()).status==200);
}
TEST_CASE("stream success waits for drain and is absent after drain failure") {
    Server s;s.control->hold_drain=true;
    std::atomic<int> terminal_events=0;
    std::atomic<bool> returned=false,failed=false;
    auto body=s.request("hello",true);body["stream_options"]={{"include_usage",true}};
    std::thread request([&] {
        try {
            client::http(s.url+"/v1/chat/completions","POST",body.dump(),s.stop,[&](const auto& event) {
                const auto& choices=event.at("choices");
                if(choices.empty() || !choices[0]["finish_reason"].is_null())++terminal_events;
            },0);
        } catch(...) {failed=true;}
        returned=true;
    });
    s.wait_phase("draining");std::this_thread::sleep_for(100ms);
    CHECK(terminal_events==0);CHECK_FALSE(returned);
    s.control->fail_drain=true;s.control->hold_drain=false;request.join();
    CHECK(failed);CHECK(terminal_events==0);s.wait_phase("failed");
    CHECK(s.get("/health").status==503);CHECK(s.post(s.request()).status==503);
}
TEST_CASE("a pinned client cannot use a different server instance") {
    Server a,b;
    const auto first=a.status().at("instance_id").get<std::string>();
    const auto second=b.status().at("instance_id").get<std::string>();
    CHECK_FALSE(first.empty());CHECK(first!=second);
    CHECK(client::http(a.url+"/zerocool/status","GET","",a.stop,{},3000,first).status==200);
    CHECK(client::http(b.url+"/zerocool/status","GET","",b.stop,{},3000,first).status==409);
    CHECK(client::http(b.url+"/zerocool/session/reset","POST","{}",b.stop,{},3000,first).status==409);
    CHECK(client::http(b.url+"/v1/chat/completions","POST",b.request().dump(),b.stop,{},3000,first).status==409);
    CHECK(b.control->resets==0);CHECK_FALSE(b.status()["busy"].get<bool>());
    CHECK(b.post(b.request()).status==200); // Ordinary OpenAI clients need no extension header.
}
