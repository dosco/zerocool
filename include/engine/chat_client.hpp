#pragma once
#include <nlohmann/json.hpp>
#include <atomic>
#include <functional>
#include <filesystem>
#include <string>
#include <vector>
#include <sys/types.h>

namespace zerocool::chat {
using Json=nlohmann::json;
std::string endpoint(std::string value);
std::string terminal_text(const std::string& value);
void save_transcript(const std::filesystem::path&,const Json& transcript);
class EventStream {
public:
    explicit EventStream(std::function<void(const Json&)> observer):observer_(std::move(observer)){}
    void feed(std::string_view bytes);
    void finish() const;
    bool done() const {return done_;}
private:
    std::string pending_,data_;
    bool done_=false;
    std::function<void(const Json&)> observer_;
    void line(std::string value);
};
struct HttpResult {long status=0;std::string body;};
HttpResult http(const std::string& url,const std::string& method,const std::string& body,
                const std::atomic<bool>& cancel,const std::function<void(const Json&)>& event={},long timeout_ms=3000,
                const std::string& instance={});
class ChildServer {
public:
    ChildServer(const std::string& executable,const std::vector<std::string>& arguments);
    ~ChildServer();
    ChildServer(const ChildServer&)=delete;
    ChildServer& operator=(const ChildServer&)=delete;
    std::string await_endpoint(const std::atomic<bool>& cancel);
    void shutdown();
    pid_t pid() const {return pid_;}
    const std::string& instance() const {return instance_;}
private:
    int control_=-1;
    pid_t pid_=-1;
    std::string instance_;
};
} // namespace zerocool::chat
