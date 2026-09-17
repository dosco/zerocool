#include "qwen/pressure.hpp"
#include <dispatch/dispatch.h>
#include <stdexcept>

namespace freellm::qwen {
struct PressureMonitor::Impl {
    std::shared_ptr<PressureInbox> inbox=std::make_shared<PressureInbox>();
    dispatch_queue_t queue=dispatch_queue_create("freellm.memory-pressure",DISPATCH_QUEUE_SERIAL);
    dispatch_source_t source=nullptr;
};
PressureMonitor::PressureMonitor():impl_(std::make_unique<Impl>()) {
    auto& p=*impl_;
    p.source=dispatch_source_create(DISPATCH_SOURCE_TYPE_MEMORYPRESSURE,0,
        DISPATCH_MEMORYPRESSURE_NORMAL|DISPATCH_MEMORYPRESSURE_WARN|DISPATCH_MEMORYPRESSURE_CRITICAL,p.queue);
    if(!p.source) throw std::runtime_error("macOS memory pressure notifications unavailable");
    const auto inbox=p.inbox;__weak dispatch_source_t source=p.source;
    dispatch_source_set_event_handler(p.source,^{
        const auto strong=source;
        if(!strong) return;
        const auto flags=dispatch_source_get_data(strong);
        if(flags&DISPATCH_MEMORYPRESSURE_NORMAL) inbox->publish(0);
        if(flags&DISPATCH_MEMORYPRESSURE_WARN) inbox->publish(1);
        if(flags&DISPATCH_MEMORYPRESSURE_CRITICAL) inbox->publish(2);
    });
    dispatch_resume(p.source);
}
PressureMonitor::~PressureMonitor() {
    dispatch_source_cancel(impl_->source);
    // Finish any executing callback before dropping the queue/monitor. The
    // callback owns only the inbox, never a model or a GPU resource.
    dispatch_sync(impl_->queue,^{});
}
PressureInbox& PressureMonitor::inbox() {return *impl_->inbox;}
} // namespace freellm::qwen
