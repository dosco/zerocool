// Developer benchmark only: deterministic cache seeding before timed decode.
#pragma once
#include "engine/pipeline.hpp"

namespace zerocool::engine::mtp_fixed_priming {
inline thread_local bool active=false;
inline thread_local uint64_t calls=0,batches=0,experts=0,peak_leases=0;
struct Scope {
    Scope() {
        if(active) throw std::logic_error("nested fixed draft priming");
        calls=batches=experts=peak_leases=0;active=true;
    }
    ~Scope() {active=false;}
    Scope(const Scope&)=delete;
    Scope& operator=(const Scope&)=delete;
};
inline Json stats() {
    return {{"policy","fixed-lease-batches-32"},{"active",active},{"calls",calls},
        {"batches",batches},{"experts",experts},{"peak_leases",peak_leases}};
}
template<class Encode>
void execute(std::span<const ExpertKey> keys,ExpertCache& cache,ReadPool& reads,
             Metal& gpu,size_t group,Encode&& encode,const std::atomic<bool>* cancel) {
    if(!active) {execute_experts(keys,cache,reads,gpu,group,encode,cancel);return;}
    if(!group || group>8 || cache.capacity()<32) throw std::logic_error("invalid fixed priming geometry");
    // Reuse the existing bounded scheduler, including its cancellation and
    // drain-before-release path. Completion order cannot change victim choice
    // because all leases in a batch stay pinned until its final GPU use ends.
    execute_experts_batched(keys,cache,reads,gpu,encode,cancel);
    ++calls;batches+=(keys.size()+31)/32;experts+=keys.size();
    peak_leases=std::max<uint64_t>(peak_leases,std::min<size_t>(32,keys.size()));
}
}
