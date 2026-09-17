#pragma once
#include "qwen/storage.hpp"
#include <algorithm>
#include <optional>

namespace freellm::qwen {
// The OS callback only publishes bits. All policy and resource changes run on
// the inference coordinator. Coalescing retains the strongest pending event.
class PressureInbox {
public:
    void publish(unsigned level) noexcept {
        if(level>2) return;
        counts_[level].fetch_add(1,std::memory_order_relaxed);
        pending_.fetch_or(1u<<level,std::memory_order_release);
    }
    unsigned take() {return pending_.exchange(0,std::memory_order_acquire);}
    std::array<uint64_t,3> counts() const {
        return {counts_[0].load(),counts_[1].load(),counts_[2].load()};
    }
private:
    std::atomic<unsigned> pending_{0};
    std::array<std::atomic<uint64_t>,3> counts_{};
};
struct PressureDecision { unsigned level; size_t before,requested; };
class PressurePolicy {
public:
    std::optional<PressureDecision> poll(unsigned events,uint64_t now,size_t capacity,bool shrink) {
        pending_|=events;
        if(!pending_) return {};
        const unsigned level=pending_&4?2:pending_&2?1:0;
        if(shrink && level==1 && last_warning_ && now-*last_warning_<1000000000) return {};
        pending_=0;
        size_t target=capacity;
        if(shrink && capacity>32) {
            if(level==2) target=std::max<size_t>(32,capacity/2);
            else if(level==1) {target=capacity>420?capacity-388:32;last_warning_=now;}
        }
        return PressureDecision{level,capacity,target};
    }
private:
    unsigned pending_=0;
    std::optional<uint64_t> last_warning_;
};
class PressureMonitor {
public:
    PressureMonitor();
    ~PressureMonitor();
    PressureInbox& inbox();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace freellm::qwen
