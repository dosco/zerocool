// Isolated developer recovery journal. No production state or kernel changes.
#pragma once
#include "engine/model.hpp"
#include <deque>
#include <cstring>

namespace zerocool::engine {
inline thread_local uint64_t recovery_target_forward_calls=0;
struct TargetRecoveryAccess {
    static uint64_t reads(const Model& model) {return model.cache_->stats().bytes;}
};
}

namespace zerocool::engine::mtp_recovery {
constexpr uint64_t ReserveBytes=16*MiB;
inline uint64_t charged(uint64_t n) {return (n+16383)/16384*16384;}
inline void require(bool v,const char* s) {if(!v) throw std::runtime_error(s);}
inline void cancelled(const std::atomic<bool>* flag) {if(flag && flag->load()) throw std::runtime_error("target recovery cancelled");}
struct Entry {
    Buf convolution,normalized,a,b,alog,dt;
    uint32_t ad=0,dd=0;
    bool seen=false;
};
struct Journal {
    std::array<Entry,Layers> entries;
    Buf ple;
    std::array<std::array<Buf,4>,2> outputs;
    uint32_t offset=0;
    std::array<int,4> ids{};
    bool active=false,complete=false,ple_seen=false;
    uint64_t fixed_bytes=0,captures=0,repairs=0,peak_groups=0;
    int test_fail_after_layer=-1; // Fixture-only; normal inference leaves disabled.
    explicit Journal(Metal& gpu) {
        auto alloc=[&](uint64_t n){fixed_bytes+=charged(n);return gpu.zeros(n/4,AllocationClass::Workspace);};
        for(int l=0;l<Layers;++l) if((l+1)%4) {
            auto& e=entries[l];e.convolution=alloc(4*10240*4);e.normalized=alloc(4*10240*4);
            e.a=alloc(4*48*4);e.b=alloc(4*48*4);
        }
        ple=alloc(4*Hyper*4);
        for(auto& group:outputs) for(auto& out:group) out=alloc(4*6144*4);
        require(bound_bytes()<=ReserveBytes,"target journal exceeds 16MiB incremental bound");
    }
    uint64_t bound_bytes() const {
        // Fixed journal/output ownership plus at most two four-layer groups of
        // replaced convolution owners, PLE replacement, and bounded metadata.
        return fixed_bytes+8*charged(3*10240*4)+charged(9*Hyper*4)+charged(sizeof(Journal)+16384);
    }
    void begin(std::span<const int> tokens,uint32_t at) {
        require(!active && tokens.size()==4 && at<=8192-4,"invalid recovery capture scope");
        for(auto id:tokens) require(id>=0 && id<Vocab,"invalid journal token");
        std::copy(tokens.begin(),tokens.end(),ids.begin());offset=at;active=true;complete=false;ple_seen=false;
        for(auto& e:entries) {e.seen=false;e.alog.reset();e.dt.reset();}
    }
    void finish() {
        require(active && ple_seen,"missing PLE journal input");
        for(int l=0;l<Layers;++l) if((l+1)%4) require(entries[l].seen,"missing GDN journal input");
        active=false;complete=true;++captures;
    }
    void abort() noexcept {active=false;complete=false;}
    void capture(Metal& gpu,int layer,const Buf& convolution,const Buf& normalized,const Buf& a,const Buf& b,
                 const Buf& alog,const Buf& dt,uint32_t ad,uint32_t dd) {
        require(active && layer>=0 && layer<Layers && (layer+1)%4,"unexpected GDN capture");
        auto& e=entries[layer];require(!e.seen,"duplicate GDN capture");
        require(convolution && normalized && a && b && alog && dt && convolution->bytes==4*10240*4 &&
            normalized->bytes==convolution->bytes && a->bytes==4*48*4 && b->bytes==a->bytes && ad<=2 && dd<=2 &&
            alog->bytes==48*(ad==1?4:2) && dt->bytes==48*(dd==1?4:2),
            "invalid journal input geometry");
        gpu.copy(convolution,0,e.convolution,0,convolution->bytes);gpu.copy(normalized,0,e.normalized,0,normalized->bytes);
        gpu.copy(a,0,e.a,0,a->bytes);gpu.copy(b,0,e.b,0,b->bytes);
        e.alog=alog;e.dt=dt;e.ad=ad;e.dd=dd;e.seen=true;
    }
    void capture_ple(Metal& gpu,const Buf& x) {
        require(active && !ple_seen && x && x->bytes==ple->bytes,"invalid PLE capture");
        gpu.copy(x,0,ple,0,ple->bytes);ple_seen=true;
    }
    void validate(const State& state,uint32_t keep) const {
        require(complete && !active && ple_seen && keep>=1 && keep<=4 && state.artifact==Artifact::Mixed,
            "incomplete or incompatible recovery journal");
        for(int l=0;l<Layers;++l) {
            const auto& s=state.layers[l];require(s.position==offset+4,"journal position differs");
            if((l+1)%4) {
                const auto& e=entries[l];require(e.seen && e.alog && e.dt && e.ad<=2 && e.dd<=2 &&
                    e.alog->bytes==48*(e.ad==1?4:2) && e.dt->bytes==48*(e.dd==1?4:2) &&
                    e.convolution && e.convolution->bytes==4*10240*4 && e.normalized && e.normalized->bytes==4*10240*4 &&
                    e.a && e.a->bytes==4*48*4 && e.b && e.b->bytes==4*48*4 &&
                    s.conv && s.conv->bytes==3*10240*4 && s.recurrence && s.recurrence->bytes==48*128*128*4,
                    "recovery state geometry differs");
            } else {
                require(s.keys && s.values && s.index && (offset+4ull)*512*4<=s.keys->bytes &&
                    s.keys->bytes==s.values->bytes && (offset+4ull)*128*4<=s.index->bytes,"attention recovery bounds");
            }
        }
        require(ple && ple->bytes==4*Hyper*4 && state.valid && state.tokens==offset+4 && state.layers[1].ple_conv &&
            state.layers[1].ple_conv->bytes==9*Hyper*4,"invalid pre-recovery state");
    }
    void apply(Metal& gpu,State& state,uint32_t keep,const std::atomic<bool>* cancel=nullptr) {
        require(complete && !active && !state.valid && keep>=1 && keep<4 && state.tokens==offset,"invalid recovery transaction");
        std::deque<std::shared_ptr<Metal::Completion>> groups;size_t sequence=0,rows=0;
        auto submit=[&]{if(auto c=gpu.submit()) {groups.push_back(c);peak_groups=std::max<uint64_t>(peak_groups,groups.size());}++sequence;rows=0;};
        try {
            cancelled(cancel);
            auto next=gpu.allocate(9*Hyper*4,AllocationClass::State);
            gpu.dispatch("conv_update",{{ple},{state.layers[1].ple_conv},{next}},{Hyper,keep,9},9*Hyper);
            state.layers[1].ple_conv=std::move(next);
            for(int l=0;l<Layers;++l) if((l+1)%4) {
                if(rows==0 && groups.size()==2) {gpu.wait(groups.front());groups.pop_front();}
                cancelled(cancel);auto& s=state.layers[l];const auto& e=entries[l];
                auto convolution=gpu.allocate(3*10240*4,AllocationClass::State);
                gpu.dispatch("conv_update",{{e.convolution},{s.conv},{convolution}},{10240,keep,3},3*10240);
                s.conv=std::move(convolution);
                gpu.dispatch("gdn_scan",{{e.normalized},{e.a},{e.b},{e.alog},{e.dt},{s.recurrence},{outputs[sequence%2][rows]}},
                    {keep,e.ad,e.dd},32,128,48,32,4,1);
                if(l==test_fail_after_layer) throw std::runtime_error("injected recovery encoding failure");
                if(++rows==4) submit();
            }
            if(rows) submit();
            for(auto& c:groups) gpu.wait(c);
            cancelled(cancel);++repairs;
        } catch(...) {try {gpu.finish();} catch(...) {} throw;}
    }
    Json stats() const {return {{"reserved_bytes",ReserveBytes},{"fixed_allocation_bytes",fixed_bytes},
        {"peak_incremental_bound_bytes",bound_bytes()},{"captures",captures},{"repairs",repairs},
        {"peak_gpu_groups",peak_groups},{"active",active},{"complete",complete}};}
};
inline thread_local Journal* capturing=nullptr;
struct Capture {
    Journal* journal;
    explicit Capture(Journal* j,std::span<const int> ids,uint32_t at):journal(j) {
        require(!capturing,"nested target journal");if(j){j->begin(ids,at);capturing=j;}
    }
    void finish(){if(journal){journal->finish();capturing=nullptr;journal=nullptr;}}
    ~Capture(){if(journal) journal->abort();capturing=nullptr;}
};
} // namespace zerocool::engine::mtp_recovery
