#include "streamed_embeddings.hpp"
#include "mtp_target_recovery.hpp"
#include "mtp_direct_output.hpp"
#include "mtp_expert_scratch.hpp"
#include "mtp_draft.hpp"
#include "qwen/model.hpp"
#include "qwen/pipeline.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <CommonCrypto/CommonDigest.h>
#include "qwen/route_trace.hpp"

namespace freellm::qwen {
namespace {
void cancelled(const std::atomic<bool>* flag) {
    if(flag && flag->load()) throw std::runtime_error("generation cancelled");
}
Buf ints(Metal& gpu,std::span<const int> values) {
    auto b=gpu.allocate(values.size_bytes()); std::memcpy(b->data,values.data(),values.size_bytes()); return b;
}
}
void Options::validate_decode_scratch() const {
    if(decode_scratch!="none" && decode_scratch!="reuse") throw std::invalid_argument("decode scratch must be none or reuse");
    if(decode_scratch=="reuse" && (!completion_pipeline || expert_tail!="wait" || decode_path!="reference" ||
       prefill_pipeline!="serial" || phase_memory!="fixed" || cached_token_replay || diagnostic_stream_trunk || kernels.gdn!="original"))
        throw std::invalid_argument("decode scratch reuse requires resident normal execution, original GDN, wait tail and serial fixed memory");
}
void Options::validate_decode_submission() const {
    if(decode_submission!="immediate" && decode_submission!="coalesced") throw std::invalid_argument("decode submission must be immediate or coalesced");
    if(decode_submission=="coalesced" && (!completion_pipeline || expert_tail!="wait" || decode_path!="reference" || cached_token_replay))
        throw std::invalid_argument("coalesced decode requires normal completion pipeline, wait tail and reference experts");
}
Model::Model(Options options) : options_(std::move(options)),checkpoint_(options_.model,true,options_.artifact),
    reads_(options_.io_workers),
    prepared_(options_.prepared.empty()?nullptr:std::make_shared<PreparedArtifact>(options_.prepared,checkpoint_)),
    store_(checkpoint_,prepared_) {
    options_.validate_decode_scratch();
    options_.validate_decode_submission();
    if(options_.memory_pressure_policy!="observe" && options_.memory_pressure_policy!="shrink")
        throw std::invalid_argument("memory pressure policy must be observe or shrink");
    pressure_monitor_=std::make_unique<PressureMonitor>();
    if(options_.expert_tail!="wait" && options_.expert_tail!="overlap")
        throw std::invalid_argument("expert tail must be wait or overlap");
    if(options_.expert_tail=="overlap" && (!options_.completion_pipeline || options_.cached_token_replay))
        throw std::invalid_argument("expert tail overlap requires normal completion-pipeline execution");
    expert_tail_=std::make_unique<ExpertTail>();
    options_.kernels.artifact_revision=checkpoint_.revision();
    gpu_.buffer_diagnostics(options_.decode_diagnostics);
    gpu_.configure(options_.kernels);
    (void)parse_cache_policy(options_.cache_policy);
    if(options_.sparse_selection!="cpu" && options_.sparse_selection!="gpu") throw std::invalid_argument("sparse selection must be cpu or gpu");
    if(options_.decode_path!="reference" && options_.decode_path!="direct" && options_.decode_path!="grouped")
        throw std::invalid_argument("invalid decode path");
    if(options_.prefill_pipeline!="serial" && options_.prefill_pipeline!="double")
        throw std::invalid_argument("invalid prefill pipeline");
    if(options_.phase_memory!="fixed" && options_.phase_memory!="reclaim") throw std::invalid_argument("invalid phase memory policy");
    if(options_.phase_memory=="reclaim" && options_.prefill_pipeline!="double") throw std::invalid_argument("phase memory reclamation requires double prefill");
    if(options_.decode_path!="reference" && !options_.completion_pipeline)
        throw std::invalid_argument("decode candidates require the completion pipeline");
    if(options_.cached_token_replay && (options_.probe_layers!=Layers || options_.diagnostic_stream_trunk || options_.expert_slots!=480 ||
       options_.memory!=12*GiB || options_.context!=8192))
        throw std::invalid_argument("cached replay requires all layers, resident trunk, 480 slots, 12GiB and 8192 context");
    if(options_.short_append<1 || options_.short_append>256) throw std::invalid_argument("short-append threshold must be 1..256");
    if(options_.memory>22*GiB || !options_.memory) throw std::invalid_argument("memory must be >0 and <=22GiB");
    if(options_.probe_layers<1 || options_.probe_layers>Layers) throw std::invalid_argument("probe layers must be 1..48");
    if(options_.ready_group!=1 && options_.ready_group!=2 && options_.ready_group!=4 && options_.ready_group!=8)
        throw std::invalid_argument("ready group must be 1, 2, 4, or 8");
    gpu_.completion_events(reads_.events());
    const auto available=available_memory();
    if(available<=GiB+GiB/2) throw std::runtime_error("insufficient currently available memory");
    const auto admitted=std::min(options_.memory,available-GiB-GiB/2);
    const auto resident_bytes=options_.diagnostic_stream_trunk?checkpoint_.diagnostic_resident_bytes(options_.probe_layers):checkpoint_.resident_bytes(options_.probe_layers);
    plan_=MemoryPlan::make(admitted,gpu_.physical(),gpu_.recommended(),embedding_rows::planned_resident(checkpoint_,resident_bytes),options_.context,options_.chunk,options_.panel,options_.probe_layers,options_.kernels.scratch_bytes(options_.chunk),options_.prefill_pipeline=="double",
        options_.decode_path=="grouped"?2*((uint64_t(options_.ready_group)*Intermediate*4+16383)/16384)*16384:0,options_.cached_token_replay);
    plan_.cap_experts(options_.expert_slots);
    prompt_plan_=plan_;generation_plan_=options_.phase_memory=="reclaim"?plan_.without_prompt_workspaces(options_.expert_slots):plan_;
    if(options_.phase_memory=="reclaim") plan_=generation_plan_;
    if(options_.cached_token_replay && plan_.limit!=12*GiB)
        throw std::runtime_error("the fixed 12GiB cached replay budget is not currently admitted");
    if(!options_.route_trace.empty()) {
        if(plan_.limit!=options_.memory)
            throw std::runtime_error("memory admission requires at least the requested route capture budget");
        if(options_.probe_layers!=Layers || options_.diagnostic_stream_trunk || options_.cached_token_replay)
            throw std::invalid_argument("route capture requires normal full-model inference");
        route_trace_=std::make_unique<RouteTrace>(options_.route_trace,Json{{"build",build_fingerprint()},
            {"artifact_revision",checkpoint_.revision()},{"budget_bytes",plan_.limit},
            {"prepared_manifest_sha256",prepared_?prepared_->inspect().at("manifest_sha256"):Json(nullptr)},
            {"device",gpu_.statistics().at("device")},{"physical_bytes",gpu_.physical()},
            {"expert_payload_bytes",ExpertBytes},{"expert_stride_bytes",ExpertStride}});
    }
    // Reserve includes CPU bookkeeping, tokenizer, and driver allocations.
    gpu_.budget(plan_.limit-plan_.ngram-plan_.reserve);
    gpu_.residency(options_.residency);
    sparse_status_=gpu_.zeros(1,AllocationClass::State);
    resident_=std::make_unique<Resident>(checkpoint_,gpu_,options_.probe_layers,options_.diagnostic_stream_trunk);
    cache_=std::make_unique<ExpertCache>(plan_.slots,[this](uint64_t n){return gpu_.allocate(n,AllocationClass::Expert);},reads_,
        [this](ExpertKey k,const Buf& b){store_.read(k,b);},ExpertStride,parse_cache_policy(options_.cache_policy));
    if(options_.decode_path=="grouped") for(auto& scratch:expert_scratch_) scratch=gpu_.allocate(uint64_t(options_.ready_group)*Intermediate*4);
    ngrams_=std::make_unique<NgramStore>(checkpoint_,reads_,plan_.ngram,prepared_);
}
void Model::check_sparse_status() const {
    if(*reinterpret_cast<const uint32_t*>(sparse_status_->data)) throw std::runtime_error("invalid sparse score");
}
Model::~Model() { try { gpu_.finish(); } catch(...) {} reads_.drain();expert_tail_.reset(); }
void Model::diagnostic_drain() {gpu_.finish();reads_.drain();finish_expert_tail();}
void Model::pressure_boundary(int layer) {
    auto action=pressure_policy_.poll(pressure_monitor_->inbox().take(),monotonic_ns(),cache_->capacity(),
                                      options_.memory_pressure_policy=="shrink");
    if(!action) return;
    const auto started=monotonic_ns();const auto occupied=cache_->occupancy();
    if(action->requested<action->before) {
        gpu_.finish();reads_.drain();finish_expert_tail();
        cache_->resize(action->requested);gpu_.reap();
        plan_.slots=cache_->capacity();plan_.experts=plan_.slots*ExpertStride;++pressure_resizes_;
    }
    if(pressure_events_.size()==256) pressure_events_.erase(pressure_events_.begin());
    pressure_events_.push_back({{"sequence",++pressure_event_count_},{"level",action->level},{"phase",phase_},
        {"layer",layer},{"monotonic_ns",started},{"before_slots",action->before},{"requested_slots",action->requested},
        {"achieved_slots",cache_->capacity()},{"released_bytes",(occupied-cache_->occupancy())*ExpertStride},
        {"drain_duration_ns",monotonic_ns()-started}});
}
void Model::observe_memory(const char* event,int layer,uint32_t tokens,uint32_t offset) const {
    if(options_.memory_observer) options_.memory_observer(
        {{"event",event},{"phase",phase_},{"layer",layer},{"tokens",tokens},{"offset",offset}},gpu_);
}
Json Model::gpu_reference(const std::atomic<bool>* cancel) {
    if(options_.artifact!=Artifact::Mixed || options_.diagnostic_stream_trunk ||
       options_.prefill_pipeline!="serial" || options_.phase_memory!="fixed")
        throw std::invalid_argument("GPU reference requires resident mixed weights and serial fixed memory");
    gpu_.finish();reads_.drain();
    const auto before=stats();
    const auto layer=resident_->linear("model.layers.0.linear_attn.in_proj_z");
    if(layer.input!=2560 || layer.output!=6144) throw std::runtime_error("GPU reference matrix changed");
    const auto host_before=host_conditions();
    auto result=gpu_.reference_probe(layer,cancel);
    const auto after=stats();
    for(const auto* key:{"expert_cache","ngram_hits","ngram_misses","checkpoint_application_read_bytes","memory_plan"})
        if(before.at(key)!=after.at(key)) throw std::logic_error("GPU reference changed model cache or plan");
    result["mode"]="resident-q8-v1";result["host_before"]=host_before;result["host_after"]=host_conditions();
    result["memory_plan"]=plan_.json();result["cache_unchanged"]=true;
    return result;
}
void Model::transition_memory(bool prompt) {
    if(options_.phase_memory!="reclaim" || prompt_memory_==prompt) return;
    const auto started=monotonic_ns();
    const auto before=plan_.json(),pools_before=gpu_.statistics()["scratch_pools"];
    const auto evictions=cache_->stats().evictions;
    gpu_.finish();reads_.drain();
    auto next=prompt?prompt_plan_:generation_plan_;
    // Never reverse an emergency pressure reduction through phase transitions.
    if(pressure_resizes_) {next.slots=std::min(next.slots,cache_->capacity());next.experts=next.slots*ExpertStride;}
    if(prompt) {cache_->resize(next.slots);gpu_.finish();}
    else {gpu_.release_scratch();cache_->resize(next.slots);}
    plan_=next;prompt_memory_=prompt;++transition_count_;
    if(memory_transitions_.size()==256) memory_transitions_.erase(memory_transitions_.begin());
    memory_transitions_.push_back({{"sequence",transition_count_},{"reason",prompt?"ingest_panel":"ingest_complete"},
        {"before",before},{"after",plan_.json()},{"duration_ns",monotonic_ns()-started},
        {"survivors",cache_->occupancy()},{"evictions",cache_->stats().evictions-evictions},{"scratch_before",pools_before},
        {"scratch_after",gpu_.statistics()["scratch_pools"]},{"live_buffer_bytes",gpu_.allocated()}});
}
void Model::prepare_ingest(size_t remaining_tokens) {
    if(ingest_active_) throw std::logic_error("ingestion already active");
    if(remaining_tokens && plan_.panel_tokens && remaining_tokens>size_t(options_.chunk)) transition_memory(true);
    ingest_active_=true;
}
void Model::finish_ingest() {
    if(!ingest_active_) return;
    ingest_active_=false;transition_memory(false);
}
std::vector<float> Model::forward(std::span<const int> ids,State& state,bool logits,const std::atomic<bool>* cancel) {
    ++recovery_target_forward_calls;
    // Developer perfect-draft verifier: existing multi-token kernel selection.
    // Fix priming lease admission/release order so independent processes begin
    // from identical CLOCK state. Restore the live completion-driven schedule
    // on every exit, including cancellation; measured decode is unchanged.
    struct RestoreCompletionPipeline {
        bool& target; bool saved;
        ~RestoreCompletionPipeline() { target=saved; }
    } restore_completion{options_.completion_pipeline,options_.completion_pipeline};
    if(phase_=="prefill") options_.completion_pipeline=false;
    auto verifier_kernels=options_.kernels;
    if(phase_=="decode") {
        if(ids.size()!=1 && ids.size()!=2 && ids.size()!=4 && ids.size()!=8)
            throw std::invalid_argument("perfect-draft decode block must contain 1, 2, 4, or 8 tokens");
        verifier_kernels.token_tile=std::min(4u,uint32_t(ids.size()));
    }
    gpu_.configure(verifier_kernels);
    const bool owned=!ingest_active_,reuse=options_.decode_scratch=="reuse" && ids.size()==1;
    const bool group_reuse=mtp_scratch::enabled && ids.size()==4 && phase_=="decode";
    if(group_reuse && (options_.decode_scratch!="reuse" || !options_.completion_pipeline ||
        options_.decode_path!="reference" || options_.expert_tail!="wait" || plan_.scratch<mtp_scratch::ReserveBytes))
        throw std::logic_error("unsupported expert scratch configuration");
    mtp_scratch::ForwardScope group_scope(gpu_,group_reuse);
    mtp_direct::Scope direct_scope((ids.size()==4 || ids.size()==8) && phase_=="decode");
    bool completed=false,scratch_started=false;
    try {
        if(ids.size()>1) observe_memory("forward_begin",-1,uint32_t(ids.size()),state.tokens);
        // Multi-token ingestion accounts no retained decode workspace. Its
        // release is included in that append/prefill's ordinary wall time.
        const bool was_group=mtp_scratch::retained_group_owner==&gpu_;
        if((options_.decode_scratch=="reuse" && !reuse && !group_reuse) || was_group!=group_reuse) {
            gpu_.release_scratch();
            if(was_group!=group_reuse) ++mtp_scratch::transitions;
        }
        mtp_scratch::retained_group_owner=group_reuse?&gpu_:nullptr;
        if(owned) prepare_ingest(ids.size());
        if(route_trace_) route_trace_->begin(state.trace_session_id,state.tokens,ids,phase_,cache_->capacity());
        if(reuse) {
            // This consumes the existing temporary allowance; expert capacity
            // and the total admitted allocation do not change.
            gpu_.begin_scratch(0,std::min<uint64_t>(128*MiB,plan_.scratch));scratch_started=true;++decode_scratch_passes_;
        }
        auto result=forward_impl(ids,state,logits,cancel);completed=true;
        if(scratch_started) gpu_.end_scratch(); // forward_impl drained GPU and ngram users
        if(owned) finish_ingest();
        if(route_trace_) route_trace_->commit(state.tokens);
        if(ids.size()>1) observe_memory("forward_end",-1,uint32_t(ids.size()),state.tokens-uint32_t(ids.size()));
        return result;
    } catch(...) {
        const auto error=std::current_exception();
        if(completed && options_.memory_observer) state.valid=false;
        if(scratch_started || group_reuse) {
            mtp_scratch::retained_group_owner=nullptr;
            if(completed) state.valid=false;
            // forward_impl drains I/O before unwinding. Never recycle storage
            // still referenced by a command, including failed/cancelled work.
            try {gpu_.release_scratch();} catch(...) {}
        }
        if(route_trace_) {state.valid=false;try {route_trace_->abort();} catch(...) {}}
        if(owned) {if(completed) state.valid=false;try {finish_ingest();} catch(...) {}}
        std::rethrow_exception(error);
    }
}
std::vector<std::byte> sparse_mask(std::span<const float> scores,uint32_t tokens,uint32_t offset,uint32_t length) {
    const uint32_t blocks=length/4;
    if(!tokens || offset+uint64_t(tokens)!=length || length>8192 || scores.size()!=uint64_t(tokens)*blocks)
        throw std::invalid_argument("invalid sparse score geometry");
    std::vector<std::byte> mask(uint64_t(tokens)*length,std::byte{0});
    std::vector<uint32_t> order(blocks);
    for(uint32_t t=0;t<tokens;++t) {
        std::iota(order.begin(),order.end(),0);
        const auto row=scores.subspan(t*blocks,blocks);
        for(auto v:row) if(std::isnan(v) || v==INFINITY) throw std::runtime_error("invalid sparse score");
        const uint32_t count=std::min<uint32_t>(512,blocks);
        std::partial_sort(order.begin(),order.begin()+count,order.end(),[&](auto a,auto b){return row[a]==row[b]?a<b:row[a]>row[b];});
        for(uint32_t j=0;j<count;++j) if(std::isfinite(row[order[j]]) && order[j]*4+3<=offset+t)
            for(uint32_t d=0;d<4;++d) mask[uint64_t(t)*length+order[j]*4+d]=std::byte{1};
        for(uint32_t pos=((offset+t+1)/4)*4;pos<=offset+t;++pos) mask[uint64_t(t)*length+pos]=std::byte{1};
    }
    return mask;
}
State Model::make_state() {
    for(auto& routes:route_history_) routes.clear();
    State s;s.artifact=options_.artifact;
    if(route_trace_) s.trace_session_id=++trace_session_sequence_;
    for(int l=0;l<options_.probe_layers;++l) {
        auto& st=s.layers[l];
        if((l+1)%4) {
            st.conv=gpu_.zeros(3*10240,AllocationClass::State);
            st.recurrence=gpu_.zeros(48*128*128,AllocationClass::State);
        } else {
            st.keys=gpu_.zeros(uint64_t(options_.context)*512,AllocationClass::State);
            st.values=gpu_.zeros(uint64_t(options_.context)*512,AllocationClass::State);
            st.index=gpu_.zeros(uint64_t(options_.context)*128,AllocationClass::State);
        }
        if(l==1) st.ple_conv=gpu_.zeros(9*Hyper,AllocationClass::State);
    }
    s.valid=true;return s;
}
StateUpdate::StateUpdate(State& state,std::span<const int> tokens)
    : state_(state),next_tokens_(state.tokens),next_history_(state.history) {
    if(!state.valid) throw std::invalid_argument("session state is invalid; rebuild before inference");
    if(tokens.empty() || tokens.size()>8192 || state.tokens>8192-tokens.size())
        throw std::invalid_argument("state update exceeds context");
    for(auto token:tokens) {
        if(token<0 || token>=Vocab) throw std::out_of_range("state token outside vocabulary");
        next_history_={next_history_[1],token};
    }
    next_tokens_+=uint32_t(tokens.size());state_.valid=false;
}
void StateUpdate::commit() {
    if(committed_) throw std::logic_error("state update already committed");
    state_.tokens=next_tokens_;state_.history=next_history_;state_.valid=true;committed_=true;
}
void Model::trace(const std::string& name,const Buf& buffer) {
    if(options_.trace_dir.empty() || !buffer) return;
    check_sparse_status();
    // Caller has already completed this command group.
    const auto dir=options_.trace_dir/("step_"+std::to_string(trace_offset_));
    std::filesystem::create_directories(dir);
    std::ofstream f(dir/(name+".bin"),std::ios::binary);
    f.write(reinterpret_cast<const char*>(buffer->data),std::streamsize(buffer->bytes));
    if(!f) throw std::runtime_error("cannot write diagnostic tensor: "+name);
}
void Model::capture_sparse(const Buf& q,const Buf& keys,const Buf& values,const Buf& qg,
                           const Buf& index_scores,uint32_t tokens,uint32_t offset,int layer) {
    // Capture only the first generated token or retained append at the requested
    // context. Prompt microchunks cannot fill this bounded diagnostic collection.
    if((layer!=3 && layer!=47) || !((phase_=="decode" && tokens==1) || (phase_=="append" && tokens==128))) return;
    for(const auto& c:sparse_captures_) if(c["layer"]==layer && c["phase"]==phase_) return;
    const uint32_t length=offset+tokens;
    const uint64_t bytes=q->bytes+uint64_t(length)*512*8+qg->bytes+index_scores->bytes;
    if(sparse_captures_.size()>=16 || bytes>256*MiB-sparse_capture_bytes_) throw std::runtime_error("sparse capture limit exceeded");
    gpu_.finish();check_sparse_status();
    if(sparse_captures_.empty() && std::filesystem::exists(options_.sparse_capture/"manifest.json"))
        throw std::runtime_error("sparse capture destination already exists");
    std::filesystem::create_directories(options_.sparse_capture);
    Json tensors=Json::object();
    auto save=[&](const char* name,const Buf& b,uint64_t size) {
        const auto filename=std::to_string(sparse_captures_.size())+"-"+name+".f32";
        unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256(b->data,CC_LONG(size),digest);
        std::string hex;constexpr char digits[]="0123456789abcdef";
        for(auto v:digest) {hex+=digits[v>>4];hex+=digits[v&15];}
        std::ofstream f(options_.sparse_capture/filename,std::ios::binary);f.write(reinterpret_cast<const char*>(b->data),std::streamsize(size));
        if(!f) throw std::runtime_error("cannot write sparse capture");
        tensors[name]={{"file",filename},{"bytes",size},{"sha256",hex}};
    };
    save("q",q,q->bytes);save("keys",keys,uint64_t(length)*512*4);save("values",values,uint64_t(length)*512*4);
    save("qg",qg,qg->bytes);save("index_scores",index_scores,index_scores->bytes);
    sparse_capture_bytes_+=bytes;
    sparse_captures_.push_back({{"layer",layer},{"phase",phase_},{"tokens",tokens},{"offset",offset},{"length",length},{"tensors",tensors}});
    Json manifest={{"kind","sparse_attention_fixture_v1"},{"build_fingerprint",gpu_.statistics()["build_fingerprint"]},
        {"artifact_revision",checkpoint_.revision()},{"bytes",sparse_capture_bytes_},{"cases",sparse_captures_}};
    std::ofstream f(options_.sparse_capture/"manifest.json");f<<manifest.dump(2)<<'\n';
    if(!f) throw std::runtime_error("cannot write sparse manifest");
}
Buf Model::norm(const Buf& x,const std::string& weight,uint32_t width,uint32_t group,uint32_t tokens,bool grouped) {
    if(x->bytes<uint64_t(tokens)*width*4 || !group || width%group) throw std::invalid_argument("normalization shape");
    auto out=gpu_.allocate(uint64_t(tokens)*width*4);
    gpu_.dispatch("norm",{{x},{resident_->at(weight)},{out}},
        {group,width,tokens,resident_->dtype(weight),uint32_t(grouped)},tokens*(width/group)*32);
    return out;
}
Buf Model::unary(const Buf& x,uint32_t op) {
    auto out=gpu_.allocate(x->bytes);
    gpu_.dispatch("unary",{{x},{out}},{uint32_t(x->bytes/4),op},uint32_t(x->bytes/4)); return out;
}
Buf Model::binary(const Buf& x,const Buf& y,uint32_t op) {
    if(x->bytes!=y->bytes) throw std::invalid_argument("binary shape mismatch");
    auto out=gpu_.allocate(x->bytes);
    gpu_.dispatch("binary",{{x},{y},{out}},{uint32_t(x->bytes/4),op},uint32_t(x->bytes/4)); return out;
}
std::pair<Buf,Buf> Model::hyper(const Buf& x,const std::string& base,uint32_t tokens,bool inject) {
    auto n=norm(x,base+".hc_norm.weight",Hyper,Hidden,tokens,true);
    auto projected=gpu_.linear(resident_->linear(base+".input_mix_weight_down"),n,tokens);
    auto low=unary(projected,2);
    auto mix=unary(gpu_.linear(resident_->linear(base+".input_mix_weight_up"),low,tokens),1);
    auto out=gpu_.allocate(uint64_t(tokens)*Hidden*4);
    gpu_.dispatch("hc_mix",{{n},{mix},{out}},{tokens},Hidden,tokens);
    Buf injection;
    if(inject) injection=unary(gpu_.linear(resident_->linear(base+".block_inject_weight"),n,tokens),3);
    if(!options_.trace_dir.empty()) {
        gpu_.finish();trace(base+".norm",n);trace(base+".projected",projected);trace(base+".low",low);
        trace(base+".mix",mix);trace(base+".inject",injection);
    }
    return {out,injection};
}
Buf Model::conv(const Buf& x,Buf& state,const std::string& weight,uint32_t width,uint32_t tokens,uint32_t dilation) {
    auto out=gpu_.allocate(uint64_t(tokens)*width*4);
    gpu_.dispatch("conv",{{x},{state},{resident_->at(weight)},{out}},
        {width,tokens,4,dilation,resident_->dtype(weight)},width,tokens);
    auto next=gpu_.allocate(uint64_t(3*dilation)*width*4,AllocationClass::State);
    gpu_.dispatch("conv_update",{{x},{state},{next}},{width,tokens,3*dilation},width*3*dilation);
    state=std::move(next); return out;
}
Buf Model::gdn(const Buf& x,LayerState& state,int layer,uint32_t tokens) {
    gpu_.label("gdn",layer,tokens,state.position);
    const auto b="model.layers."+std::to_string(layer)+".linear_attn";
    auto qkv=gpu_.linear(resident_->linear(b+".in_proj_qkv"),x,tokens);
    auto z=gpu_.linear(resident_->linear(b+".in_proj_z"),x,tokens);
    auto beta=gpu_.linear(resident_->linear(b+".in_proj_b"),x,tokens);
    auto a=gpu_.linear(resident_->linear(b+".in_proj_a"),x,tokens);
    if(!options_.trace_dir.empty()) {gpu_.finish();trace("qkv_"+std::to_string(layer),qkv);}
    auto convolved=conv(qkv,state.conv,b+".conv1d.weight",10240,tokens,1);
    auto normalized=gpu_.allocate(convolved->bytes);
    gpu_.dispatch("gdn_qk",{{convolved},{normalized}},{tokens},32*32,tokens);
    if(mtp_recovery::capturing) mtp_recovery::capturing->capture(gpu_,layer,qkv,normalized,a,beta,
        resident_->at(b+".A_log"),resident_->at(b+".dt_bias"),resident_->dtype(b+".A_log"),resident_->dtype(b+".dt_bias"));
    auto y=gpu_.gdn_scan(normalized,a,beta,resident_->at(b+".A_log"),resident_->at(b+".dt_bias"),
        state.recurrence,tokens,resident_->dtype(b+".A_log"),resident_->dtype(b+".dt_bias"));
    auto gated=gpu_.allocate(y->bytes);
    gpu_.dispatch("gdn_gate",{{y},{z},{resident_->at(b+".norm.weight")},{gated}},
        {tokens,resident_->dtype(b+".norm.weight")},32*48,tokens);
    if(!options_.trace_dir.empty()) {
        gpu_.finish();trace(b+".conv",convolved);trace(b+".qk",normalized);
        trace(b+".a",a);trace(b+".b",beta);trace(b+".z",z);
        trace(b+".y",y);trace(b+".gated",gated);
    }
    return gpu_.linear(resident_->linear(b+".out_proj"),gated,tokens);
}
Buf Model::attention(const Buf& x,LayerState& state,int layer,uint32_t tokens,uint32_t offset) {
    gpu_.label("attention",layer,tokens,offset);
    const auto b="model.layers."+std::to_string(layer)+".self_attn";
    const uint32_t length=offset+tokens,blocks=length/4;
    auto index_qk=gpu_.linear(resident_->linear(b+".indexer.index_qk_proj"),x,tokens);
    gpu_.dispatch("index_store",{{index_qk},{state.index}},{tokens,offset},128,tokens);
    auto qg=gpu_.linear(resident_->linear(b+".q_proj"),x,tokens);
    auto rawk=gpu_.linear(resident_->linear(b+".k_proj"),x,tokens);
    auto v=gpu_.linear(resident_->linear(b+".v_proj"),x,tokens);
    auto q=gpu_.allocate(uint64_t(tokens)*6144*4),k=gpu_.allocate(uint64_t(tokens)*512*4);
    gpu_.dispatch("norm_rope",{{qg},{resident_->at(b+".q_norm.weight")},{q}},
        {256,24,12288,512,offset,tokens,resident_->dtype(b+".q_norm.weight")},32*24,tokens);
    gpu_.dispatch("norm_rope",{{rawk},{resident_->at(b+".k_norm.weight")},{k}},
        {256,2,512,256,offset,tokens,resident_->dtype(b+".k_norm.weight")},32*2,tokens);
    gpu_.dispatch("kv_store",{{k},{state.keys}},{512,tokens,offset},512,tokens);
    gpu_.dispatch("kv_store",{{v},{state.values}},{512,tokens,offset},512,tokens);
    auto mask=gpu_.allocate(std::max<uint64_t>(4,uint64_t(tokens)*length));
    const bool sparse=length>2048;
    if(sparse) {
        auto iq=gpu_.allocate(uint64_t(tokens)*512*4),pooled=gpu_.allocate(uint64_t(blocks)*128*4);
        gpu_.dispatch("norm_rope",{{index_qk},{resident_->at(b+".indexer.q_layernorm.weight")},{iq}},
            {128,4,640,128,offset,tokens,resident_->dtype(b+".indexer.q_layernorm.weight")},32*4,tokens);
        gpu_.dispatch("index_pool",{{state.index},{resident_->at(b+".indexer.k_layernorm.weight")},{pooled}},
            {blocks,resident_->dtype(b+".indexer.k_layernorm.weight")},32*blocks);
        auto scores=gpu_.allocate(uint64_t(tokens)*blocks*4);
        gpu_.dispatch("index_scores",{{iq},{pooled},{scores}},{blocks,tokens,offset},32*blocks,tokens);
        if(options_.sparse_selection=="gpu") gpu_.sparse_select(scores,mask,sparse_status_,tokens,offset,length);
        else {
            auto start=monotonic_ns();gpu_.finish();check_sparse_status();
            sparse_selection_wait_ns_+=monotonic_ns()-start;start=monotonic_ns();
            const auto selected=sparse_mask(scores->floats(),tokens,offset,length);
            std::memcpy(mask->data,selected.data(),selected.size());
            sparse_selection_cpu_ns_+=monotonic_ns()-start;
        }
        if(!options_.sparse_capture.empty()) capture_sparse(q,state.keys,state.values,qg,scores,tokens,offset,layer);
    }
    auto out=gpu_.allocate(uint64_t(tokens)*6144*4);
    auto scores=gpu_.allocate(uint64_t(tokens)*24*length*4);
    gpu_.attention_scores(q,state.keys,mask,scores,tokens,offset,length,sparse);
    gpu_.dispatch("attention_softmax",{{scores}},{length},tokens*24*32);
    gpu_.dispatch("attention_values",{{scores},{state.values},{qg},{out}},
        {tokens,length},32*32,(tokens+7)/8,24);
    if(!options_.trace_dir.empty()) {
        gpu_.finish();trace(b+".qg",qg);trace(b+".q",q);trace(b+".k",k);
        trace(b+".v",v);trace(b+".gated",out);trace(b+".probabilities",scores);
    }
    return gpu_.linear(resident_->linear(b+".o_proj"),out,tokens);
}
Buf Model::moe(const Buf& x,int layer,uint32_t tokens,const std::atomic<bool>* cancel) {
    gpu_.label("router",layer,tokens,trace_offset_);
    const auto b="model.layers."+std::to_string(layer)+".mlp";
    auto router=gpu_.linear(resident_->linear(b+".gate"),x,tokens,true);
    auto ids=gpu_.allocate(uint64_t(tokens)*TopK*4),weights=gpu_.allocate(uint64_t(tokens)*TopK*4);
    gpu_.route(router,ids,weights,tokens);
    gpu_.finish();finish_expert_tail(); check_sparse_status(); cancelled(cancel);
    trace("route_"+std::to_string(layer),ids);
    trace("router_"+std::to_string(layer),router);
    std::array<std::vector<int>,Experts> positions;
    auto raw=std::span<const int>(reinterpret_cast<const int*>(ids->data),tokens*TopK);
    if(route_trace_) route_trace_->routes(layer,trace_offset_,tokens,raw);
    if(options_.audit_routes) {
        auto& history=route_history_[layer];
        if(trace_offset_+tokens>8192) throw std::runtime_error("route audit exceeds context");
        history.resize(uint64_t(trace_offset_+tokens)*TopK);
        std::copy(raw.begin(),raw.end(),history.begin()+uint64_t(trace_offset_)*TopK);
    }
    for(uint32_t p=0;p<tokens*TopK;++p) {
        if(raw[p]<0 || raw[p]>=Experts) throw std::runtime_error("invalid router output");
        positions[raw[p]].push_back(int(p));
    }
    auto expert_out=gpu_.allocate(uint64_t(tokens)*TopK*Hidden*4);
    std::vector<int> selected;
    for(int e=0;e<Experts;++e) if(!positions[e].empty()) selected.push_back(e);
    if(tokens<=uint32_t(options_.short_append)) {
        // Consume live hits before miss admission can evict another selected
        // hit. Large prefill keeps expert/file order for predictable reads.
        std::stable_partition(selected.begin(),selected.end(),[&](int e){return cache_->ready({uint32_t(layer),uint32_t(e)});});
    }
    // Encode shared work now. The first miss batch is submitted before this
    // command group, allowing resident computation to cover some read latency.
    gpu_.label("shared_expert",layer,tokens,trace_offset_);
    auto shared_activation=gpu_.gated_linear(resident_->linear(b+".shared_expert.gate_proj"),resident_->linear(b+".shared_expert.up_proj"),x,tokens);
    auto shared=gpu_.linear(resident_->linear(b+".shared_expert.down_proj"),shared_activation,tokens);
    auto gate=gpu_.linear(resident_->linear(b+".shared_expert_gate"),x,tokens);
    if(options_.completion_pipeline) {
        std::vector<ExpertKey> keys;for(auto e:selected) keys.push_back({uint32_t(layer),uint32_t(e)});
        EncodeReadyGroup grouped;
        if(tokens==1 && options_.decode_path=="grouped") grouped=[&](std::span<const ReadyExpert> ready,size_t slot) {
            std::vector<Buf> records;std::vector<uint32_t> destinations,identities;
            for(const auto& item:ready) { records.push_back(item.record);destinations.push_back(uint32_t(positions[item.key.expert].at(0)));identities.push_back(item.key.expert); }
            gpu_.label("routed_expert",layer,1,trace_offset_,identities);
            gpu_.grouped_experts(records,destinations,x,expert_out,expert_scratch_.at(slot));
        };
        auto timing=execute_experts(keys,*cache_,reads_,gpu_,options_.ready_group,[&](ExpertKey key,const Buf& record){
            if(options_.expert_observer) {
                gpu_.finish();options_.expert_observer(key,record,x,tokens,phase_,trace_offset_);
            }
            encode_expert_rows(gpu_,record,x,expert_out,positions[key.expert],tokens,options_.chunk,
                options_.decode_path!="reference",layer,trace_offset_,cancel);
        },cancel,(!options_.dependency_trace.empty() && (!options_.kernels.profile_decode_only || phase_=="decode")) || (options_.kernels.profile && detailed_passes_[phase_]<48 && detailed_reads_[phase_]<8192),grouped,
            tokens==1 && options_.expert_tail=="overlap"?expert_tail_.get():nullptr,
            tokens==1 && options_.decode_submission=="coalesced");
        if(timing.is_null()) {
            ++tail_deferrals_;tail_layer_=layer;tail_offset_=trace_offset_;tail_phase_=phase_;
            if(!options_.dependency_trace.empty()) tail_routes_.assign(raw.begin(),raw.end());
        } else record_expert_timing(std::move(timing),layer,tokens,trace_offset_,phase_,raw);

    } else {
    const size_t batch_limit=std::min<size_t>(32,cache_->capacity());
    for(size_t start=0;start<selected.size();start+=batch_limit) {
        cancelled(cancel);
        const size_t end=std::min(start+batch_limit,selected.size());
        std::vector<ExpertCache::Lease> leases;
        for(size_t i=start;i<end;++i) {
            cancelled(cancel);leases.push_back(cache_->acquire({uint32_t(layer),uint32_t(selected[i])}));
        }
        auto compute=[&](size_t i) {
            cancelled(cancel);
            const auto& record=leases[i-start].wait();
            cancelled(cancel);
            encode_expert_rows(gpu_,record,x,expert_out,positions[selected[i]],tokens,options_.chunk,
                options_.decode_path!="reference",layer,trace_offset_,cancel);
        };
        std::vector<bool> done(end-start,false);
        // Available experts execute while the persistent reader fills misses.
        for(size_t i=start;i<end;++i) if(leases[i-start].ready()) { compute(i); done[i-start]=true; }
        gpu_.submit();
        try {
            for(size_t i=start;i<end;++i) if(!done[i-start]) { compute(i); cancelled(cancel); }
            gpu_.finish();
        } catch(...) { gpu_.finish(); reads_.drain(); throw; }
        // Leases release only after every encoded use, including the hit pass.
    }
    }
    if(!options_.trace_dir.empty()) {
        gpu_.finish();trace(b+".expert_out",expert_out);trace(b+".weights",weights);
        trace(b+".shared",shared);trace(b+".gate",gate);
    }
    auto out=gpu_.allocate(uint64_t(tokens)*Hidden*4);
    gpu_.label("expert_reduce",layer,tokens,trace_offset_);
    gpu_.dispatch("moe_sum",{{expert_out},{weights},{shared},{gate},{out}},{tokens},Hidden,tokens);
    return out;
}
void Model::finish_expert_tail() {
    if(!expert_tail_->pending()) return;
    auto timing=expert_tail_->finish();
    record_expert_timing(std::move(timing),tail_layer_,1,tail_offset_,tail_phase_,tail_routes_);
    tail_routes_.clear();
}
void Model::record_expert_timing(Json timing,int layer,uint32_t tokens,uint32_t offset,
                               const std::string& phase,std::span<const int> routes) {
    expert_wait_ns_[layer]+=timing["coordinator_wait_ns"].get<uint64_t>();
    expert_gpu_ns_[layer]+=timing["gpu_execution_sum_ns"].get<uint64_t>();++expert_passes_[layer];
    auto& aggregate=phase_dependencies_[phase];if(aggregate.is_null()) aggregate=Json::object();
    for(const auto* metric:{"ready_hits","loading_joins","new_misses","read_queue_sum_ns","read_service_sum_ns",
        "ready_to_encode_sum_ns","ready_to_gpu_sum_ns","coordinator_wait_ns","completion_waits"})
        aggregate[metric]=aggregate.value(metric,uint64_t(0))+timing.at(metric).get<uint64_t>();
    if(options_.kernels.profile && detailed_passes_[phase]<48) {
        auto event=timing;event["layer"]=layer;event["offset"]=offset;event["tokens"]=tokens;event["request_phase"]=phase;
        auto& records=event["records"];const auto remaining=8192-detailed_reads_[phase];
        if(records.size()>remaining) records.erase(records.begin()+remaining,records.end());
        detailed_reads_[phase]+=records.size();++detailed_passes_[phase];
        dependency_events_.push_back(std::move(event));
    }
    if(!options_.dependency_trace.empty() && (!options_.kernels.profile_decode_only || phase=="decode")) {
        timing["layer"]=layer;timing["offset"]=offset;timing["tokens"]=tokens;
        timing["routes"]=std::vector<int>(routes.begin(),routes.end());
        timing["build"]=gpu_.statistics()["build_fingerprint"];timing["artifact_revision"]=checkpoint_.revision();
        std::ofstream f(options_.dependency_trace,std::ios::app);f<<timing.dump()<<'\n';
        if(!f) throw std::runtime_error("cannot write dependency trace");
    }
}
Buf Model::ple(const Buf& x,const Buf& embedding,LayerState& state,uint32_t tokens) {
    const std::string b="model.layers.1.ple";
    auto projected=gpu_.linear(resident_->linear(b+".key_proj"),embedding,tokens);
    auto key=norm(projected,b+".norm_key.weight",Hyper,Hidden,tokens,true);
    auto query=norm(x,b+".norm_query.weight",Hyper,Hidden,tokens,true);
    auto value=gpu_.linear(resident_->linear(b+".value_proj"),embedding,tokens);
    auto gated=gpu_.allocate(uint64_t(tokens)*Hyper*4);
    gpu_.dispatch("ple_gate",{{key},{query},{value},{gated}},{tokens},640*4,tokens,1,640);
    auto normalized=norm(gated,b+".norm_conv.weight",Hyper,Hidden,tokens,true);
    if(mtp_recovery::capturing) mtp_recovery::capturing->capture_ple(gpu_,normalized);
    auto convolved=conv(normalized,state.ple_conv,b+".conv1d.weight",Hyper,tokens,3);
    if(!options_.trace_dir.empty()) {
        gpu_.finish();trace("ple.key_proj",projected);trace("ple.norm_key",key);trace("ple.norm_query",query);
        trace("ple.value_proj",value);trace("ple.norm_conv.input",gated);trace("ple.norm_conv",normalized);trace("ple.conv",convolved);
    }
    return binary(gated,convolved,0);
}
std::vector<float> Model::forward_impl(std::span<const int> ids,State& state,bool logits,const std::atomic<bool>* cancel) {
    if(!state.valid) throw std::invalid_argument("session state is invalid; rebuild before inference");
    if(state.artifact!=options_.artifact) throw std::invalid_argument("state artifact differs from model; rebuild before inference");
    if(logits && options_.probe_layers!=Layers) throw std::invalid_argument("truncated probes cannot produce model logits");
    if(ids.empty() || ids.size()>input_limit() || state.tokens+ids.size()>uint64_t(options_.context))
        throw std::invalid_argument("forward pass exceeds chunk/context limit");
    for(auto id:ids) if(id<0 || id>=Vocab) throw std::out_of_range("token outside vocabulary");
    for(int l=0;l<options_.probe_layers;++l) if(state.layers[l].position!=state.tokens)
        throw std::invalid_argument("layer position differs from committed history; rebuild state");
    cancelled(cancel);
    pressure_boundary(-1);
    if(available_memory()<GiB && cache_->capacity()>32) {
        ++pressure_resizes_;
        gpu_.finish(); cache_->resize(std::max<size_t>(32,cache_->capacity()/2));
        plan_.slots=cache_->capacity();plan_.experts=plan_.slots*ExpertStride;
    }
    // Each preceding forward has drained its users, including on failure.
    *reinterpret_cast<uint32_t*>(sparse_status_->data)=0;
    if(ids.size()>uint64_t(options_.chunk)) return forward_panel(ids,state,logits,cancel);
    StateUpdate update(state,ids);
    trace_offset_=state.tokens;
    const uint32_t T=uint32_t(ids.size());
    if(T==1) ++decode_passes_; else if(T<=uint32_t(options_.short_append)) ++append_passes_; else ++prefill_passes_;
    Buf h,ng;std::future<void> ng_future;
    try {
        const auto emb=resident_->linear("model.embed_tokens");
        if(!emb.quantized || emb.output!=Vocab || emb.input!=Hidden) throw std::runtime_error("unsupported embedding layout");
        gpu_.label("embedding",-1,T,state.tokens);
        h=gpu_.embedding(emb,ids,4);
        ng=gpu_.allocate(uint64_t(T)*Hidden*4);
        // Only known-token ngram addresses are prefetched. Every outstanding
        // user is drained even when the initial encoding or allocation fails.
        ng_future=std::async(std::launch::async,[this,ids,history=state.history,ng]{ngrams_->embedding(ids,history,ng->floats());});
        for(int l=0;l<options_.probe_layers;++l) {
            cancelled(cancel);
            pressure_boundary(l);
            if(T>1) observe_memory("layer_begin",l,T,state.tokens);
            resident_->activate_layer(l);
            const auto b="model.layers."+std::to_string(l);
            auto& st=state.layers[l];
            if(!options_.trace_dir.empty()) {gpu_.finish();trace("input_"+std::to_string(l),h);}
            if(l==1) { ng_future.get(); h=binary(h,ple(h,ng,st,T),0);trace("ngram",ng); }
            gpu_.label("attention_input",l,T,state.tokens);
            auto [x,inj]=hyper(h,b+".attn_hyper_connection",T);
            auto a=(l+1)%4 ? gdn(x,st,l,T) : attention(x,st,l,T,state.tokens);
            gpu_.label("attention_residual",l,T,state.tokens);
            auto after=gpu_.allocate(h->bytes);
            gpu_.dispatch("hc_add",{{h},{a},{inj},{after}},{T},Hyper,T);
            gpu_.label("mlp_input",l,T,state.tokens);
            auto [mx,mi]=hyper(after,b+".mlp_hyper_connection",T);
            if(T>1) observe_memory("attention_encoded",l,T,state.tokens);
            auto m=moe(mx,l,T,cancel);
            gpu_.label("mlp_residual",l,T,state.tokens);
            h=gpu_.allocate(after->bytes);
            gpu_.dispatch("hc_add",{{after},{m},{mi},{h}},{T},Hyper,T);
            st.position+=T;
            if(T>1) observe_memory("layer_encoded",l,T,state.tokens);
            // The next layer's router is the next CPU dependency. Keep the
            // residual and next attention work in the same command group.
            if(!options_.trace_dir.empty()) {
                gpu_.finish();trace("layer_"+std::to_string(l),h);
                trace("x1_"+std::to_string(l),x);trace("attn_"+std::to_string(l),a);
                trace("x2_"+std::to_string(l),mx);trace("moe_"+std::to_string(l),m);
            }
        }
        capture_mtp_hidden(gpu_,h,T,state.tokens);
        if(!logits) {gpu_.finish();finish_expert_tail();update.commit();return {};}
        auto result=compute_logits(h,T);update.commit();return result;
    } catch(...) {
        const auto error=std::current_exception();
        if(ng_future.valid()) ng_future.wait();
        try { gpu_.finish(); } catch(...) {}
        reads_.drain();try {finish_expert_tail();} catch(...) {} std::rethrow_exception(error);
    }
}
std::vector<float> Model::compute_logits(const Buf& h,uint32_t T) {
    // Developer perfect-draft diagnostic; production last-row output is unchanged.
    if(phase_=="decode" && T!=1 && T!=2 && T!=4 && T!=8)
        throw std::invalid_argument("perfect-draft decode block must contain 1, 2, 4, or 8 tokens");
    if(phase_=="decode" && T>1) {
        gpu_.label("logits",-1,T,trace_offset_);
        auto [mixed,unused]=hyper(h,"model.hyper_connection_mixer",T,false); (void)unused;
        auto out=gpu_.linear(resident_->linear("lm_head"),mixed,T);
        gpu_.finish();finish_expert_tail();
        const auto values=out->floats();
        if(values.size()!=uint64_t(T)*Vocab) throw std::runtime_error("invalid perfect-draft logits width");
        for(float value:values) if(!std::isfinite(value))
            throw std::runtime_error("non-finite model logits");
        return {values.begin(),values.end()};
    }
    gpu_.label("logits",-1,T,trace_offset_);
    auto [mixed,unused]=hyper(h,"model.hyper_connection_mixer",T,false); (void)unused;
    auto last=gpu_.allocate(Hidden*4);
    const std::array<int,1> row={int(T-1)};auto rowbuf=ints(gpu_,row);
    gpu_.dispatch("gather_rows",{{mixed},{rowbuf},{last}},{Hidden,1},Hidden);
    auto out=gpu_.linear(resident_->linear("lm_head"),last,1);gpu_.finish();finish_expert_tail();
    auto values=out->floats();
    for(float v:values) if(!std::isfinite(v)) throw std::runtime_error("non-finite model logits");
    return {values.begin(),values.end()};
}
Json Model::decode_counters() const {
    const auto start=monotonic_ns();
    const auto& cache=cache_->stats();
    auto dependencies=phase_dependencies_.value(phase_,Json::object());
    for(const auto* key:{"ready_hits","loading_joins","new_misses","read_queue_sum_ns","read_service_sum_ns",
                         "ready_to_encode_sum_ns","ready_to_gpu_sum_ns","coordinator_wait_ns","completion_waits"})
        if(!dependencies.contains(key)) dependencies[key]=0; // No expert work yet in this phase.
    Json result={{"process",process_memory()},{"metal",gpu_.timing_counters()},
        {"expert_cache",{{"hits",cache.hits},{"misses",cache.misses},{"application_read_bytes",cache.bytes}}},
        {"expert_dependencies",std::move(dependencies)}};
    result["sample_ns"]=monotonic_ns()-start;
    return result;
}
Json Model::stats() const {
    return {{"memory_pressure",{{"policy",options_.memory_pressure_policy},{"source","dispatch-memorypressure"},
                {"counts",pressure_monitor_->inbox().counts()},{"event_count",pressure_event_count_},{"events",pressure_events_}}},
        {"sparse_selection_cpu_ns",sparse_selection_cpu_ns_},{"sparse_selection_wait_ns",sparse_selection_wait_ns_},
        {"artifact_revision",checkpoint_.revision()},{"model_id",checkpoint_.model_id()},
        {"execution",{{"cache_policy",options_.cache_policy},{"sparse_selection",options_.sparse_selection},{"residency",options_.residency},{"decode_path",options_.decode_path},{"expert_tail",options_.expert_tail},{"decode_scratch",options_.decode_scratch},{"prefill_pipeline",options_.prefill_pipeline},{"phase_memory",options_.phase_memory},{"cached_token_replay",options_.cached_token_replay}}},
        {"phase_memory",{{"policy",options_.phase_memory},{"phase",prompt_memory_?"prompt":"generation"},
            {"prompt_plan",prompt_plan_.json()},{"generation_plan",generation_plan_.json()},{"pressure_resizes",pressure_resizes_},
            {"transition_count",transition_count_},{"transitions",memory_transitions_}}},
        {"memory_plan",plan_.json()},{"metal",gpu_.statistics()},{"process",process_memory()},{"storage",disk_counters()},
        {"diagnostic_stream_trunk",options_.diagnostic_stream_trunk},
        {"expert_tail_deferrals",tail_deferrals_},{"expert_tail_pending",expert_tail_->pending()},
        {"decode_submission",options_.decode_submission},
        {"decode_scratch_passes",decode_scratch_passes_},
        {"completion_pipeline",options_.completion_pipeline},{"ready_group",options_.ready_group},
        {"chunk_tokens",options_.chunk},{"io_workers",options_.io_workers},{"short_append_tokens",options_.short_append},
        {"phase_dependencies",phase_dependencies_},
        {"gpu_timing_scope","expert pipeline command groups can include shared work"},{"layer_expert_wait_ns",expert_wait_ns_},{"layer_expert_gpu_ns",expert_gpu_ns_},{"layer_expert_passes",expert_passes_},
        {"expert_cache_occupancy",cache_->occupancy()},{"expert_cache",cache_->json()},{"ngram_hits",ngrams_->hits},{"ngram_misses",ngrams_->misses},
        {"prepared",prepared_?prepared_->inspect():Json(nullptr)},
        {"checkpoint_application_read_bytes",checkpoint_.bytes_read()},
        {"passes",{{"decode",decode_passes_},{"short_append",append_passes_},{"prefill",prefill_passes_},{"panel",panel_passes_}}}};
}
void Model::reset_expert_cache() { gpu_.finish(); cache_->clear();if(route_trace_) route_trace_->cache_reset(); }
Json Model::take_profile() {
    auto result=gpu_.take_profile();result["expert_dependencies"]=std::move(dependency_events_);
    result["dependency_capture_limits"]={{"passes_per_phase",48},{"read_records_per_phase",8192}};
    dependency_events_=Json::array();detailed_reads_.clear();detailed_passes_.clear();return result;
}
Json Model::route_identity() const {
    Json result=Json::array();
    if(!options_.audit_routes) return result;
    for(int l=0;l<options_.probe_layers;++l) {
        const auto& ids=route_history_[l];unsigned char digest[CC_SHA256_DIGEST_LENGTH];
        CC_SHA256(ids.data(),CC_LONG(ids.size()*sizeof(int)),digest);
        std::string hash;constexpr char digits[]="0123456789abcdef";
        for(auto c:digest){hash+=digits[c>>4];hash+=digits[c&15];}
        result.push_back({{"layer",l},{"tokens",ids.size()/TopK},{"sha256",hash}});
    }
    return result;
}
} // namespace freellm::qwen
