#pragma once
#include "engine/session.hpp"

namespace zerocool::engine {
using ChatObserver=std::function<void(const Json&)>;
struct ChatRequest {
    Options options;
    Json messages,tools;
    bool thinking=false;
    std::string reasoning_effort;
};
struct ChatResponse {
    Json message,usage,diagnostics;
    std::string finish_reason;
};
// A narrow transport test seam. Production always constructs the native executor.
// Its complete lifecycle runs on the single inference worker.
class ChatExecutor {
public:
    virtual ~ChatExecutor()=default;
    virtual void initialize(const ChatObserver& progress)=0;
    virtual ChatResponse generate(const ChatRequest&,const ChatObserver& delta,
                                  const ChatObserver& progress,const std::atomic<bool>& cancel)=0;
    virtual void clear()=0;
    // Discard retained history only when it can no longer be continued, so a
    // rejected or cancelled request does not force a full prompt replay.
    virtual void recover()=0;
    virtual void drain()=0;
};
using ChatExecutorFactory=std::function<std::unique_ptr<ChatExecutor>()>;
ChatRequest parse_chat_request(const Json&,const Options&);
std::string api_model_id(Artifact);
// control_fd is a private socket inherited from the TUI. EOF requests shutdown.
// The listener is available before factory/initialize runs, including on failure.
void serve(const Options&,uint16_t port,const std::atomic<bool>* stop=nullptr,int control_fd=-1,
           ChatExecutorFactory factory={});
} // namespace zerocool::engine
