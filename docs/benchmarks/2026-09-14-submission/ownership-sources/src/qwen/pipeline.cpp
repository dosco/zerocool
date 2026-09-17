#include "qwen/pipeline.hpp"
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace freellm::qwen {
void encode_expert_rows(Metal& gpu,const Buf& record,const Buf& input,const Buf& output,
                        std::span<const int> positions,uint32_t tokens,uint32_t chunk,
                        bool direct,int layer,uint32_t offset,const std::atomic<bool>* cancel) {
    if(!chunk || chunk>256 || !tokens || tokens>1024 || !input || !output ||
       input->bytes<uint64_t(tokens)*Hidden*4 || output->bytes<uint64_t(tokens)*TopK*Hidden*4)
        throw std::invalid_argument("invalid expert row encoding geometry");
    for(auto p:positions) if(p<0 || uint64_t(p)>=uint64_t(tokens)*TopK) throw std::invalid_argument("expert destination outside input");
    auto ints=[&](std::span<const int> values) {auto b=gpu.allocate(values.size_bytes());std::memcpy(b->data,values.data(),values.size_bytes());return b;};
    for(size_t at=0;at<positions.size();at+=chunk) {
        if(cancel && cancel->load()) throw std::runtime_error("generation cancelled");
        const auto pos=positions.subspan(at,std::min<size_t>(chunk,positions.size()-at));
        const auto n=uint32_t(pos.size());gpu.label("routed_expert",layer,n,offset);
        std::vector<int> rows;rows.reserve(n);for(auto p:pos) rows.push_back(p/TopK);
        auto rowsbuf=tokens==1?Buf{}:ints(rows);
        auto activated=gpu.gated_linear(expert_linear(record,0),expert_linear(record,1),input,n,rowsbuf);
        if(tokens==1 && direct) gpu.linear_into(expert_linear(record,2),activated,1,{output,uint64_t(pos[0])*Hidden*4});
        else {
            auto posbuf=ints(pos);auto down=gpu.linear(expert_linear(record,2),activated,n);
            gpu.dispatch("scatter_experts",{{down},{posbuf},{output}},{n},Hidden,n);
        }
    }
}

Json execute_experts_batched(std::span<const ExpertKey> selected,ExpertCache& cache,
                            ReadPool& reads,Metal& gpu,
                            const std::function<void(ExpertKey,const Buf&)>& encode,
                            const std::atomic<bool>* cancel) {
    const auto start=monotonic_ns();const auto batch=std::min<size_t>(32,cache.capacity());
    uint64_t hits=0,joins=0,misses=0;
    for(size_t at=0;at<selected.size();at+=batch) {
        std::vector<ExpertCache::Lease> leases;
        try {
            const auto end=std::min(at+batch,selected.size());
            for(size_t i=at;i<end;++i) {
                if(cancel && cancel->load()) throw std::runtime_error("generation cancelled");
                leases.push_back(cache.acquire(selected[i]));
                const std::string_view kind=leases.back().acquisition();
                if(kind=="ready_hit") ++hits;else if(kind=="loading_join") ++joins;else ++misses;
            }
            std::vector<bool> done(leases.size(),false);
            for(size_t i=0;i<leases.size();++i) if(leases[i].ready()) {
                if(cancel && cancel->load()) throw std::runtime_error("generation cancelled");
                encode(leases[i].key(),leases[i].wait());done[i]=true;
            }
            gpu.submit();
            for(size_t i=0;i<leases.size();++i) if(!done[i]) {
                if(cancel && cancel->load()) throw std::runtime_error("generation cancelled");
                const auto& record=leases[i].wait();
                if(cancel && cancel->load()) throw std::runtime_error("generation cancelled");
                encode(leases[i].key(),record);
            }
            gpu.finish();
        } catch(...) {
            const auto failure=std::current_exception();try {gpu.finish();} catch(...) {}
            reads.drain();std::rethrow_exception(failure);
        }
    }
    return {{"duration_ns",monotonic_ns()-start},{"ready_hits",hits},{"loading_joins",joins},{"new_misses",misses},
            {"coordinator_wait_ns",nullptr},{"gpu_execution_sum_ns",nullptr},
            {"records",Json::array()},{"schedule","batched_control"}};
}
struct ExpertTail::Impl {
    struct Work {
        ExpertCache::Lease lease;
        uint64_t admitted=0, encoded=0;
    };
    struct Group {
        std::shared_ptr<Metal::Completion> completion;
        std::vector<Work> work;
    };
    Metal& gpu; ReadPool& reads; const bool detailed;
    const uint64_t start=monotonic_ns();
    std::vector<Work> waiting;std::deque<Group> groups;Group encoding;
    size_t next=0,leased=0,peak_leases=0,peak_groups=0,group_sequence=0;
    uint64_t waits=0,wait_ns=0,queue_ns=0,read_ns=0,last_read=start,ready_delay=0;
    uint64_t ready_hits=0,joins=0,misses=0,gpu_ns=0,ready_gpu_ns=0;
    Json records=Json::array();
    Impl(Metal& g,ReadPool& r,bool d):gpu(g),reads(r),detailed(d) {}
    ~Impl() {
        // A failed read/encode, abandoned tail, or failed reap cannot release a
        // cache slot while its GPU user is outstanding.
        if(!waiting.empty() || !groups.empty() || !encoding.work.empty()) {
            try {gpu.finish();} catch(...) {} reads.drain();
        }
    }
    bool reap() {
        bool progress=false;
        while(!groups.empty() && groups.front().completion->done.load(std::memory_order_acquire)) {
            auto& group=groups.front();const auto& c=*group.completion;
            gpu_ns+=uint64_t(std::max(0.0,c.gpu_end-c.gpu_start)*1e9);
            for(const auto& w:group.work) {
                const auto ready=std::max(w.admitted,w.lease.timing().completed_ns);
                const auto gpu_start=uint64_t(c.gpu_start*1e9);
                if(gpu_start>=ready) ready_gpu_ns+=gpu_start-ready;
                if(!detailed) continue;
                const auto& r=w.lease.timing();
                records.push_back({{"expert",w.lease.key().expert},{"acquisition",w.lease.acquisition()},
                    {"admitted_ns",w.admitted},{"read_queued_ns",r.queued_ns},
                    {"read_started_ns",r.started_ns},{"read_completed_ns",r.completed_ns},
                    {"encoded_ns",w.encoded},{"submitted_ns",c.submitted_ns},
                    {"gpu_start_ns",uint64_t(c.gpu_start*1e9)},
                    {"gpu_end_ns",uint64_t(c.gpu_end*1e9)},{"released_ns",monotonic_ns()}});
            }
            leased-=group.work.size();groups.pop_front();progress=true;
        }
        return progress;
    }
    Json report() {
        return {{"duration_ns",monotonic_ns()-start},{"ready_hits",ready_hits},{"loading_joins",joins},
            {"new_misses",misses},{"read_queue_sum_ns",queue_ns},{"read_service_sum_ns",read_ns},
            {"last_required_read_ns",last_read-start},{"ready_to_encode_sum_ns",ready_delay},
            {"ready_to_gpu_sum_ns",ready_gpu_ns},{"gpu_execution_sum_ns",gpu_ns},{"coordinator_wait_ns",wait_ns},{"completion_waits",waits},
            {"peak_leases",peak_leases},{"peak_gpu_groups",peak_groups},{"records",std::move(records)}};
    }
};
ExpertTail::ExpertTail() = default;
ExpertTail::~ExpertTail() = default;
bool ExpertTail::pending() const {return bool(impl_);}
Json ExpertTail::finish() {
    if(!impl_) throw std::logic_error("no pending expert tail");
    auto state=std::move(impl_);
    const auto before=monotonic_ns();state->gpu.finish();
    state->wait_ns+=monotonic_ns()-before;
    state->reap();
    if(!state->groups.empty()) throw std::logic_error("expert tail still has GPU users after drain");
    return state->report();
}
Json execute_experts(std::span<const ExpertKey> selected, ExpertCache& cache,
                     ReadPool& reads, Metal& gpu, size_t group_size,
                     const std::function<void(ExpertKey,const Buf&)>& encode,
                     const std::atomic<bool>* cancel, bool detailed,const EncodeReadyGroup& encode_group,ExpertTail* tail) {
    if(group_size!=1 && group_size!=2 && group_size!=4 && group_size!=8)
        throw std::invalid_argument("expert group size must be 1, 2, 4, or 8");
    if(tail && tail->pending()) throw std::logic_error("previous expert tail must drain before admission");
    auto state=std::make_unique<ExpertTail::Impl>(gpu,reads,detailed);
    auto& waiting=state->waiting;auto& groups=state->groups;auto& encoding=state->encoding;
    auto& next=state->next;auto& leased=state->leased;auto& peak_leases=state->peak_leases;
    auto& peak_groups=state->peak_groups;auto& group_sequence=state->group_sequence;
    auto& ready_hits=state->ready_hits;auto& joins=state->joins;auto& misses=state->misses;
    auto& queue_ns=state->queue_ns;auto& read_ns=state->read_ns;auto& last_read=state->last_read;
    auto& ready_delay=state->ready_delay;auto& waits=state->waits;auto& wait_ns=state->wait_ns;
    const auto events=reads.events();gpu.completion_events(events);
    const auto window=std::min<size_t>(32,cache.capacity());
    try {
        while(next<selected.size() || !waiting.empty() || !groups.empty()) {
            const auto ticket=events->ticket();
            if(cancel && cancel->load()) throw std::runtime_error("generation cancelled");
            gpu.reap();bool progress=state->reap();
            while(next<selected.size() && leased<window) {
                if(cancel && cancel->load()) throw std::runtime_error("generation cancelled");
                waiting.push_back({cache.acquire(selected[next++]),monotonic_ns(),0});++leased;
                peak_leases=std::max(peak_leases,leased);progress=true;
            }
            if(groups.size()<2) {
                auto& group=encoding;
                for(size_t i=0;i<waiting.size() && group.work.size()<group_size;) {
                    if(cancel && cancel->load()) throw std::runtime_error("generation cancelled");
                    if(!waiting[i].lease.ready()) {++i;continue;}
                    auto work=std::move(waiting[i]);waiting.erase(waiting.begin()+i);
                    const auto& record=work.lease.wait();const auto& r=work.lease.timing();
                    const std::string_view kind=work.lease.acquisition();
                    if(kind=="ready_hit") ++ready_hits;
                    else if(kind=="loading_join") ++joins;
                    else {++misses;queue_ns+=r.started_ns-r.queued_ns;read_ns+=r.completed_ns-r.started_ns;}
                    last_read=std::max(last_read,r.completed_ns);
                    work.encoded=monotonic_ns();ready_delay+=work.encoded-std::max(work.admitted,r.completed_ns);
                    // Encoding owns leases outside the try scope: even a later
                    // failed read must drain previous encodes before releasing them.
                    group.work.push_back(std::move(work));
                    if(!encode_group) encode(group.work.back().lease.key(),record);
                }
                if(encode_group && !group.work.empty()) {
                    std::vector<ReadyExpert> ready;
                    for(const auto& w:group.work) ready.push_back({w.lease.key(),w.lease.wait()});
                    encode_group(ready,group_sequence%2);
                }
                // Flush shared work while the first window is loading.
                group.completion=gpu.submit();
                if(group.completion) {
                    ++group_sequence;
                    groups.push_back(std::move(group));encoding={};peak_groups=std::max(peak_groups,groups.size());progress=true;
                }
            }
            if(tail && next==selected.size() && waiting.empty() && !groups.empty()) {
                // All selected contributions are queued in their original
                // destinations. The caller may encode dependent work on this
                // same queue while this bounded tail still owns the leases.
                if(cancel && cancel->load()) throw std::runtime_error("generation cancelled");
                tail->impl_=std::move(state);return nullptr;
            }
            if(!progress && (!waiting.empty() || !groups.empty())) {
                const auto before=monotonic_ns();events->wait(ticket);wait_ns+=monotonic_ns()-before;++waits;
            }
        }
    } catch(...) {
        const auto failure=std::current_exception();
        try {gpu.finish();} catch(...) {} reads.drain();std::rethrow_exception(failure);
    }
    return state->report();
}
}
