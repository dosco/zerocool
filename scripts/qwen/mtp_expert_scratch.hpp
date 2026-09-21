// Developer-only experiment. Never included by the production build.
#pragma once
#include "engine/pipeline.hpp"
#include <chrono>
#include <thread>

namespace zerocool::engine::mtp_scratch {
constexpr uint64_t PoolBytes=MiB, ReserveBytes=2*PoolBytes;
inline thread_local bool enabled=false;
inline thread_local Metal* target=nullptr;
inline thread_local Metal* retained_group_owner=nullptr;
inline thread_local uint64_t forwards=0,groups=0,transitions=0;
// Test-only callback gate; both normal timing arms leave it disabled.
inline std::atomic<bool> hold_completions=false;
inline std::atomic<uint64_t> held_callbacks=0;
inline void completion_gate() {
    if(!hold_completions.load(std::memory_order_acquire)) return;
    ++held_callbacks;
    while(hold_completions.load(std::memory_order_acquire))
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
}
struct ForwardScope {
    explicit ForwardScope(Metal& gpu,bool active) {
        if(target) throw std::logic_error("nested expert scratch forward");
        if(active) {target=&gpu;++forwards;}
    }
    ~ForwardScope() {target=nullptr;}
};
inline Json counters() {
    return {{"enabled",enabled},{"forwards",forwards},{"groups",groups},
        {"transitions",transitions},{"pool_capacity_bytes",PoolBytes},{"total_reserve_bytes",ReserveBytes}};
}
}
