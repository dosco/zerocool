// Isolated developer trial; no production class or buffer layout changes.
#pragma once
#include "qwen/pipeline.hpp"
#include <stdexcept>

namespace freellm::qwen::mtp_direct {
inline thread_local bool enabled=false,in_verifier=false;
inline thread_local uint64_t forwards=0,eligible=0,direct_writes=0;
struct Scope {
    explicit Scope(bool active) {
        if(in_verifier) throw std::logic_error("nested direct-output verifier");
        in_verifier=active;if(active) ++forwards;
    }
    ~Scope() {in_verifier=false;}
    Scope(const Scope&)=delete;
    Scope& operator=(const Scope&)=delete;
};
inline bool select(uint32_t tokens,uint32_t rows) {
    if(!in_verifier || (tokens!=4 && tokens!=8) || rows!=1) return false;
    ++eligible;if(enabled) ++direct_writes;return enabled;
}
inline Json counters() {
    return {{"enabled",enabled},{"active",in_verifier},{"forwards",forwards},
        {"eligible_calls",eligible},{"direct_writes",direct_writes},
        {"avoided_copy_bytes",direct_writes*Hidden*4}};
}
}
