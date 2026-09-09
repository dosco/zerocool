#include "qwen/model.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <cstring>

namespace freellm::qwen {
namespace {
constexpr std::array fields={&LayerState::conv,&LayerState::recurrence,&LayerState::keys,
    &LayerState::values,&LayerState::index,&LayerState::ple_conv};
std::string digest(const void* data,size_t bytes) {
    unsigned char result[CC_SHA256_DIGEST_LENGTH];CC_SHA256(data,CC_LONG(bytes),result);
    std::string hex;constexpr char digits[]="0123456789abcdef";
    for(auto c:result) {hex+=digits[c>>4];hex+=digits[c&15];}return hex;
}
}
State snapshot_state(Metal& gpu,const State& state) {
    if(!state.valid) throw std::invalid_argument("cannot snapshot invalid session state");
    gpu.finish();State result=state;
    for(size_t i=0;i<Layers;++i) for(auto field:fields) if(const auto& b=state.layers[i].*field) {
        auto copy=gpu.allocate(b->bytes,AllocationClass::Snapshot);
        std::memcpy(copy->data,b->data,b->bytes);result.layers[i].*field=std::move(copy);
    }
    return result;
}
void restore_state(Metal& gpu,const State& snapshot,State& state) {
    if(!snapshot.valid || snapshot.artifact!=state.artifact) throw std::invalid_argument("invalid state restoration");
    gpu.finish();
    // Validate the entire geometry before modifying any destination.
    for(size_t i=0;i<Layers;++i) for(auto field:fields) {
        const auto& src=snapshot.layers[i].*field;const auto& dst=state.layers[i].*field;
        if(bool(src)!=bool(dst) || (src && (src->bytes!=dst->bytes || src->metal==dst->metal)))
            throw std::invalid_argument("aliased or mismatched snapshot buffers");
    }
    state.valid=false;
    for(size_t i=0;i<Layers;++i) {
        for(auto field:fields) if(const auto& src=snapshot.layers[i].*field) {
            const auto& dst=state.layers[i].*field;std::memcpy(dst->data,src->data,src->bytes);
        }
        state.layers[i].position=snapshot.layers[i].position;
    }
    state.tokens=snapshot.tokens;state.history=snapshot.history;state.valid=snapshot.valid;
}
Json state_digest(const State& state) {
    Json layers=Json::array();
    for(const auto& layer:state.layers) {
        Json buffers=Json::array();
        for(auto field:fields) {const auto& b=layer.*field;buffers.push_back(b?Json{{"bytes",b->bytes},{"sha256",digest(b->data,b->bytes)}}:Json(nullptr));}
        layers.push_back({{"position",layer.position},{"buffers",buffers}});
    }
    return {{"artifact_revision",artifact_revision(state.artifact)},{"valid",state.valid},{"tokens",state.tokens},
        {"history",state.history},{"layers",layers}};
}
Json Model::cached_token_replay(std::span<const int> tokens,int repetitions,const std::atomic<bool>* cancel) {
    if(options_.cached_compare && (!options_.kernels.q8_decode_rows || options_.kernels.profile || repetitions<5))
        throw std::invalid_argument("cached comparison needs an unprofiled Q8 candidate and five pairs");
    if(!options_.cached_token_replay || tokens.size()<2 || tokens.size()>size_t(options_.context) ||
       repetitions<1 || repetitions>20 || options_.probe_layers!=Layers || options_.diagnostic_stream_trunk || plan_.slots!=480)
        throw std::invalid_argument("cached replay requires a full model, 480 slots, and prompt plus continuation token");
    for(auto id:tokens) if(id<0 || id>=Vocab) throw std::invalid_argument("invalid replay token");
    const auto check_cancel=[&]{if(cancel && cancel->load()) throw std::runtime_error("cached replay cancelled");};
    check_cancel();
    struct Reset { Options& target;Options saved;~Reset(){target=std::move(saved);} } reset{options_,options_};
    const auto requested=options_;const auto setup_start=monotonic_ns();
    options_.decode_path="reference";options_.prefill_pipeline="serial";options_.audit_routes=false;gpu_.configure({});
    auto state=make_state();prepare_pipelines();
    const auto prompt=tokens.first(tokens.size()-1),token=tokens.last(1);
    for(size_t at=0;at<prompt.size();) {
        const auto n=std::min<size_t>(input_limit(),prompt.size()-at);forward(prompt.subspan(at,n),state,false,cancel);at+=n;
    }
    auto snapshot=snapshot_state(gpu_,state);
    options_.audit_routes=true;
    const auto reference=forward(token,state,true,cancel);const auto expected_state=state_digest(state),expected_routes=route_identity();
    std::vector<ExpertKey> selected;
    for(int l=0;l<Layers;++l) {
        const auto& row=route_history_[l];
        for(size_t k=row.size()-TopK;k<row.size();++k) selected.push_back({uint32_t(l),uint32_t(row[k])});
    }
    restore_state(gpu_,snapshot,state);reset_expert_cache();
    const auto preload_start=monotonic_ns();
    for(auto key:selected) {check_cancel();auto lease=cache_->acquire(key);lease.wait();}
    std::array<float,Hidden> ng{};ngrams_->embedding(token,state.history,ng);
    const auto preload_ns=monotonic_ns()-preload_start;
    options_=requested;options_.audit_routes=true;gpu_.configure(options_.kernels);
    const auto setup_ns=monotonic_ns()-setup_start;
    Json runs=Json::array();
    try {
        for(int rep=-1;rep<repetitions;++rep) {
          const int variants=options_.cached_compare?2:1;
          for(int index=0;index<variants;++index) {
            auto kernels=requested.kernels;
            const bool baseline=variants==2 && ((index+(rep>=0?rep%2:0))%2==0);
            if(baseline) kernels.q8_decode_rows=0;
            gpu_.configure(kernels);
            check_cancel();
            const auto restore_start=monotonic_ns();restore_state(gpu_,snapshot,state);
            const auto restore_ns=monotonic_ns()-restore_start;
            for(auto& row:route_history_) row.clear();
            phase(rep<0?"cached_warmup":"cached_replay");
            const auto before=stats();const auto start=monotonic_ns();
            const auto output=forward(token,state,true,cancel);const auto elapsed=monotonic_ns()-start;
            const auto after=stats();
            const bool exact=reference.size()==output.size() && std::memcmp(reference.data(),output.data(),output.size()*sizeof(float))==0 &&
                expected_state==state_digest(state) && expected_routes==route_identity();
            const auto reads=[&](const Json& s){return s.at("checkpoint_application_read_bytes").get<uint64_t>()+
                (s.at("prepared").is_null()?0:s.at("prepared").at("application_read_bytes").get<uint64_t>());};
            const auto hits=after["expert_cache"]["ready_hits"].get<uint64_t>()-before["expert_cache"]["ready_hits"].get<uint64_t>();
            const bool cached=hits==480 && before["expert_cache"]["misses"]==after["expert_cache"]["misses"] &&
                before["ngram_misses"]==after["ngram_misses"] && reads(before)==reads(after);
            if(!exact || !cached) throw std::runtime_error("cached replay failed exact output/state/routes or zero-read checks");
            if(rep>=0) runs.push_back({{"repetition",rep},{"forward_ns",elapsed},{"restore_ns",restore_ns},
                {"variant",baseline?"control":"candidate"},
                {"exact",exact},{"ready_hits",hits},{"application_read_bytes",reads(after)-reads(before)},
                {"before",before},{"after",after}});
          }
        }
    } catch(...) {options_=requested;throw;}
    options_=requested;gpu_.configure(requested.kernels);
    return {{"kind","cached_full_token_replay"},{"normal_request_latency_qualified",false},{"passed",true},
        {"profiling_enabled",requested.kernels.profile},
        {"paired_comparison",requested.cached_compare},
        {"artifact_revision",checkpoint_.revision()},{"prompt_tokens",prompt.size()},{"continuation_token",token[0]},
        {"input_sha256",digest(tokens.data(),tokens.size_bytes())},{"logits_sha256",digest(reference.data(),reference.size()*4)},
        {"setup_ns",setup_ns},{"preload_ns",preload_ns},{"snapshot_bytes",plan_.snapshot},{"runs",runs},
        {"note","Full forward with explicit original routes preloaded. Restore, hashes, preload, and setup are outside timing; all routers still execute. No normal-request performance claim."}};
}
} // namespace freellm::qwen
