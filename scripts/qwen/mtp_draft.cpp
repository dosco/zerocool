#include "mtp_draft.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace freellm::qwen {
namespace {
void need(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
void cancelled(const std::atomic<bool>* flag) {if(flag && flag->load()) throw std::runtime_error("MTP cancelled");}
std::string digest(const Buf& b) {
    unsigned char d[CC_SHA256_DIGEST_LENGTH];CC_SHA256(b->data,CC_LONG(b->bytes),d);
    std::string out;for(auto c:d){out+="0123456789abcdef"[c>>4];out+="0123456789abcdef"[c&15];}return out;
}
constexpr auto state_fields=std::array{&DraftState::keys,&DraftState::values,&DraftState::index};
}
thread_local Buf mtp_target_hidden;
thread_local uint32_t mtp_target_rows=0,mtp_target_offset=0;
void capture_mtp_hidden(Metal& gpu,const Buf& hidden,uint32_t rows,uint32_t offset) {
    if(!mtp_target_hidden) return;
    need(rows>0 && rows<=MtpDraft::Panel && hidden->bytes==uint64_t(rows)*Hyper*4 &&
         mtp_target_hidden->bytes>=hidden->bytes,"target hidden capture exceeds bounded panel");
    gpu.copy(hidden,0,mtp_target_hidden,0,hidden->bytes);
    mtp_target_rows=rows;mtp_target_offset=offset;
}
DraftCheckpoint::DraftCheckpoint(Metal& gpu) {
    for(size_t i=0;i<3;++i) tail_[i]=gpu.zeros(4*(i==2?128:512),AllocationClass::Snapshot);
}
void DraftCheckpoint::save(Metal& gpu,const DraftState& state,uint32_t width) {
    need(state.valid && width>0 && width<=4,"invalid MTP checkpoint");
    for(size_t i=0;i<3;++i) {
        const auto& b=state.*state_fields[i];const uint64_t stride=(i==2?128:512)*4;
        need(b && (uint64_t(state.position)+width)*stride<=b->bytes,"MTP checkpoint outside state");
    }
    gpu.finish();saved_=false;
    for(size_t i=0;i<3;++i) {
        const auto& b=state.*state_fields[i];const uint64_t stride=(i==2?128:512)*4;
        geometry_[i]=b->bytes;std::memcpy(tail_[i]->data,b->data+state.position*stride,width*stride);
    }
    position_=state.position;width_=width;saved_=true;
}
void DraftCheckpoint::restore(Metal& gpu,DraftState& state) const {
    need(saved_,"unsaved MTP checkpoint");
    for(size_t i=0;i<3;++i) {
        const auto& b=state.*state_fields[i];need(b && b->bytes==geometry_[i],"MTP restore geometry changed");
    }
    gpu.finish();state.valid=false;
    for(size_t i=0;i<3;++i) {
        const uint64_t stride=(i==2?128:512)*4;
        std::memcpy((state.*state_fields[i])->data+position_*stride,tail_[i]->data,width_*stride);
    }
    state.position=position_;state.valid=true;
}
uint64_t MtpDraft::budget_bytes(size_t slots,uint32_t context) {
    need(slots>=10 && slots<=128 && context>0 && context<=8192,"invalid bounded MTP capacity");
    // Dense payload, aligned shifted norms, cache, attention/index state,
    // bounded hidden capture, scratch/logits/checkpoints and control metadata.
    return 98041856+512*1024+slots*ExpertStride+uint64_t(context)*1152*4+
           Panel*Hyper*4+64*MiB+16*MiB+4ull*Vocab*4+4*Hyper*4+3*16384;
}
MtpDraft::MtpDraft(Metal& gpu,ReadPool& reads,const std::filesystem::path& root,
                  Linear embedding,Linear head,size_t slots,uint32_t context)
    :gpu_(gpu),reads_(reads),embedding_(embedding),head_(head),context_(context),manifest_(read_json(root/"manifest.json")) {
    (void)budget_bytes(slots,context);
    need(manifest_.at("kind")=="qwen_mtp_prepared_v1" && manifest_.at("complete")==true &&
         manifest_.at("recipe")=="mtp-affine-q4-experts-q8-dense-64-v1" &&
         manifest_.at("source_revision")=="de4b8e4d43b917e7706784d8bb445c9af86a3540","wrong MTP artifact");
    need(embedding.input==Hidden && embedding.output==Vocab && head.input==Hidden && head.output==Vocab,
         "MTP shared embedding/head geometry differs");
    File file(root/"dense.bin");need(file.size()==98041856 && manifest_["files"]["dense.bin"]["bytes"]==file.size(),"MTP dense size differs");
    dense_=gpu_.allocate(file.size(),AllocationClass::Resident);file.read(0,{dense_->data,size_t(dense_->bytes)});
    need(digest(dense_)==manifest_["files"]["dense.bin"]["sha256"].get<std::string>(),"corrupt MTP dense payload");
    for(const auto& [name,t]:manifest_.at("tensors").items()) {
        for(const auto& [part,p]:t.at("parts").items()) {
            (void)part;const uint64_t at=p.at("offset"),bytes=p.at("bytes");
            need(at%16384==0 && at<=dense_->bytes && bytes<=dense_->bytes-at,"MTP tensor outside dense payload");
        }
        if(t.value("norm_convention",Json(nullptr)).is_null()) continue;
        need(t["norm_convention"]=="zero-centered-original; apply 1+w once" && t["format"]=="BF16" && t["shape"].size()==1,"MTP norm convention differs");
        const uint64_t count=t["shape"][0],offset=t["parts"]["weight"]["offset"];
        need(t["parts"]["weight"]["bytes"]==count*2,"MTP norm size differs");
        auto shifted=gpu_.allocate(count*4,AllocationClass::Resident);
        for(uint64_t i=0;i<count;++i) {uint16_t raw;std::memcpy(&raw,dense_->data+offset+i*2,2);shifted->floats()[i]=round_bf16(1+bf16(raw));}
        norms_.emplace(name,std::move(shifted));
    }
    experts_=std::make_unique<File>(root/"experts.bin");
    need(experts_->size()==512*ExpertStride && manifest_["experts"]["record_sha256"].size()==512,"MTP expert geometry differs");
    cache_=std::make_unique<ExpertCache>(slots,[&](uint64_t n){return gpu_.allocate(n,AllocationClass::Expert);},reads_,
        [this](ExpertKey key,const Buf& into) {
            need(key.layer==0 && key.expert<512 && into->bytes==ExpertStride,"invalid MTP expert read");
            experts_->read(uint64_t(key.expert)*ExpertStride,{into->data,size_t(into->bytes)});
            need(digest(into)==std::as_const(manifest_).at("experts").at("record_sha256").at(key.expert).get<std::string>(),"corrupt MTP expert record");
        });
}
MtpDraft::~MtpDraft() {try {gpu_.finish();} catch(...) {} reads_.drain();}
DraftState MtpDraft::make_state() {
    return {gpu_.zeros(uint64_t(context_)*512,AllocationClass::State),
            gpu_.zeros(uint64_t(context_)*512,AllocationClass::State),
            gpu_.zeros(uint64_t(context_)*128,AllocationClass::State),0,true};
}
Linear MtpDraft::linear(const std::string& name) const {
    const auto& t=manifest_.at("tensors").at(name+".weight");need(t["shape"].size()==2,"MTP linear rank differs");
    Linear l;l.input=t["shape"][1];l.output=t["shape"][0];l.weight={dense_,t["parts"]["weight"]["offset"]};
    l.quantized=t["format"]=="affine-Q8";
    if(l.quantized) {
        need(t["bits"]==8 && t["group_size"]==64 && l.input%64==0,"MTP affine format differs");
        l.bits=8;l.scales={dense_,t["parts"]["scales"]["offset"]};l.biases={dense_,t["parts"]["biases"]["offset"]};
    } else need(t["format"]=="BF16","unsupported MTP matrix");
    return l;
}
void MtpDraft::observe(const std::string& label,const Buf& buffer) {if(observer && buffer){gpu_.finish();observer(label,buffer);}}
Buf MtpDraft::norm(const Buf& x,const std::string& name,uint32_t width,uint32_t group,uint32_t tokens) {
    auto out=gpu_.allocate(uint64_t(tokens)*width*4);const auto& w=norms_.at(name+".weight");
    need(x->bytes==out->bytes && w->bytes==width*4,"MTP norm geometry differs");
    if(group==Hyper) gpu_.dispatch("mtp_wide_norm",{{x},{w},{out}},{tokens},tokens*32);
    else {need(group && width%group==0 && group<=4096,"unsupported MTP norm width");
        gpu_.dispatch("norm",{{x},{w},{out}},{group,width,tokens,1,0},tokens*(width/group)*32);}
    return out;
}
Buf MtpDraft::unary(const Buf& x,uint32_t op) {
    auto y=gpu_.allocate(x->bytes);gpu_.dispatch("unary",{{x},{y}},{uint32_t(x->bytes/4),op},uint32_t(x->bytes/4));return y;
}
std::pair<Buf,Buf> MtpDraft::hyper(const Buf& x,const std::string& base,uint32_t tokens,bool inject) {
    auto n=norm(x,base+".hc_norm",Hyper,Hidden,tokens);
    auto low=unary(gpu_.linear(linear(base+".input_mix_weight_down"),n,tokens),2);
    auto mix=unary(gpu_.linear(linear(base+".input_mix_weight_up"),low,tokens),1);
    auto out=gpu_.allocate(uint64_t(tokens)*Hidden*4);
    gpu_.dispatch("hc_mix",{{n},{mix},{out}},{tokens},Hidden,tokens);
    auto injection=inject?unary(gpu_.linear(linear(base+".block_inject_weight"),n,tokens),3):Buf{};
    return {out,injection};
}
Buf MtpDraft::attention(const Buf& x,DraftState& state,uint32_t tokens) {
    const std::string b="mtp.layers.0.self_attn";const uint32_t offset=state.position,length=offset+tokens,blocks=length/4;
    auto iqk=gpu_.linear(linear(b+".indexer.index_qk_proj"),x,tokens);
    gpu_.dispatch("index_store",{{iqk},{state.index}},{tokens,offset},128,tokens);
    auto qg=gpu_.linear(linear(b+".q_proj"),x,tokens),rk=gpu_.linear(linear(b+".k_proj"),x,tokens),v=gpu_.linear(linear(b+".v_proj"),x,tokens);
    auto q=gpu_.allocate(uint64_t(tokens)*6144*4),k=gpu_.allocate(uint64_t(tokens)*512*4);
    gpu_.dispatch("norm_rope",{{qg},{norms_.at(b+".q_norm.weight")},{q}},{256,24,12288,512,offset,tokens,1},32*24,tokens);
    gpu_.dispatch("norm_rope",{{rk},{norms_.at(b+".k_norm.weight")},{k}},{256,2,512,256,offset,tokens,1},32*2,tokens);
    gpu_.dispatch("kv_store",{{k},{state.keys}},{512,tokens,offset},512,tokens);
    gpu_.dispatch("kv_store",{{v},{state.values}},{512,tokens,offset},512,tokens);
    auto mask=gpu_.allocate(std::max<uint64_t>(4,uint64_t(tokens)*length));const bool sparse=length>2048;
    if(sparse) {
        auto iq=gpu_.allocate(uint64_t(tokens)*512*4),pooled=gpu_.allocate(uint64_t(blocks)*128*4);
        gpu_.dispatch("norm_rope",{{iqk},{norms_.at(b+".indexer.q_layernorm.weight")},{iq}},{128,4,640,128,offset,tokens,1},32*4,tokens);
        gpu_.dispatch("index_pool",{{state.index},{norms_.at(b+".indexer.k_layernorm.weight")},{pooled}},{blocks,1},32*blocks);
        auto scores=gpu_.allocate(uint64_t(tokens)*blocks*4);
        gpu_.dispatch("index_scores",{{iq},{pooled},{scores}},{blocks,tokens,offset},32*blocks,tokens);
        gpu_.finish();const auto selected=sparse_mask(scores->floats(),tokens,offset,length);
        std::memcpy(mask->data,selected.data(),selected.size());
    }
    auto scores=gpu_.allocate(uint64_t(tokens)*24*length*4),out=gpu_.allocate(uint64_t(tokens)*6144*4);
    gpu_.attention_scores(q,state.keys,mask,scores,tokens,offset,length,sparse);
    gpu_.dispatch("attention_softmax",{{scores}},{length},tokens*24*32);
    gpu_.dispatch("attention_values",{{scores},{state.values},{qg},{out}},{tokens,length},32*32,(tokens+7)/8,24);
    observe("q",q);observe("k",k);observe("v",v);observe("attention_gated",out);
    return gpu_.linear(linear(b+".o_proj"),out,tokens);
}
Buf MtpDraft::moe(const Buf& x,uint32_t tokens,const std::atomic<bool>* cancel) {
    const std::string b="mtp.layers.0.mlp";
    auto router=gpu_.linear(linear(b+".gate"),x,tokens,true);
    auto ids=gpu_.allocate(uint64_t(tokens)*TopK*4),weights=gpu_.allocate(uint64_t(tokens)*TopK*4);
    gpu_.route(router,ids,weights,tokens);gpu_.finish();cancelled(cancel);
    observe("router",router);observe("routes",ids);observe("route_weights",weights);
    std::array<std::vector<int>,Experts> positions;
    auto raw=std::span(reinterpret_cast<const int*>(ids->data),tokens*TopK);
    for(uint32_t p=0;p<raw.size();++p) {need(raw[p]>=0 && raw[p]<Experts,"invalid MTP route");positions[raw[p]].push_back(p);}
    std::vector<ExpertKey> selected;for(uint32_t e=0;e<Experts;++e) if(!positions[e].empty()) selected.push_back({0,e});
    std::stable_partition(selected.begin(),selected.end(),[&](auto key){return cache_->ready(key);});
    auto expert_out=gpu_.allocate(uint64_t(tokens)*TopK*Hidden*4);
    auto activation=gpu_.gated_linear(linear(b+".shared_expert.gate_proj"),linear(b+".shared_expert.up_proj"),x,tokens);
    auto shared=gpu_.linear(linear(b+".shared_expert.down_proj"),activation,tokens),gate=gpu_.linear(linear(b+".shared_expert_gate"),x,tokens);
    execute_experts(selected,*cache_,reads_,gpu_,4,[&](ExpertKey key,const Buf& record){
        encode_expert_rows(gpu_,record,x,expert_out,positions[key.expert],tokens,Panel,false,0,0,cancel);
    },cancel);
    auto out=gpu_.allocate(uint64_t(tokens)*Hidden*4);
    gpu_.dispatch("moe_sum",{{expert_out},{weights},{shared},{gate},{out}},{tokens},Hidden,tokens);
    observe("expert_out",expert_out);observe("shared",shared);observe("shared_gate",gate);return out;
}
DraftOutput MtpDraft::forward(std::span<const int> ids,const Buf& hidden,DraftState& state,bool logits,const std::atomic<bool>* cancel) {
    return execute(ids,hidden,state,logits,cancel,false);
}
void MtpDraft::catch_up(std::span<const int> ids,const Buf& hidden,DraftState& state,const std::atomic<bool>* cancel) {
    (void)execute(ids,hidden,state,false,cancel,true);
}
DraftOutput MtpDraft::execute(std::span<const int> ids,const Buf& hidden,DraftState& state,bool logits,
                             const std::atomic<bool>* cancel,bool state_only) {
    need(!ids.empty() && ids.size()<=16,"MTP microchunk exceeds bound");
    const uint32_t T=uint32_t(ids.size());
    need(state.valid && state.position<=context_ && T<=context_-state.position,"invalid MTP forward/context");
    need(hidden && hidden->bytes==uint64_t(T)*Hyper*4,"invalid MTP wide hidden input");
    for(auto id:ids) need(id>=0 && id<Vocab,"invalid MTP token");cancelled(cancel);
    for(size_t i=0;i<3;++i) need((state.*state_fields[i]) && (state.*state_fields[i])->bytes==uint64_t(context_)*(i==2?128:512)*4,"MTP state geometry differs");
    KernelConfig config;config.policy="candidate";config.q8_decode_rows=2;config.token_tile=T==4?4:T==2?2:1;config.route_selection="simd";
    gpu_.configure(config);gpu_.request_phase("mtp");gpu_.label("mtp",0,T,state.position);state.valid=false;
    try {
        auto embedding=gpu_.embedding(embedding_,ids);observe("embedding",embedding);
        auto en=norm(embedding,"mtp.pre_fc_norm_embedding",Hidden,Hidden,T);
        auto hn=norm(hidden,"mtp.pre_fc_norm_hidden",Hyper,Hyper,T);observe("embedding_norm",en);observe("hidden_norm",hn);
        auto e=gpu_.linear(linear("mtp.fc_embedding"),en,T),h=gpu_.linear(linear("mtp.fc_hidden"),hn,T*4);
        auto injection=gpu_.upload(std::vector<float>(T*4,1));auto fused=gpu_.allocate(uint64_t(T)*Hyper*4);
        gpu_.dispatch("hc_add",{{h},{e},{injection},{fused}},{T},Hyper,T);observe("fused",fused);
        auto [x,inj]=hyper(fused,"mtp.layers.0.attn_hyper_connection",T,!state_only);observe("attention_input",x);
        if(state_only) {
            // This one-layer draft has no recurrent/PLE state. Given corrected
            // target hidden inputs, only these projections update persistent
            // state. Attention queries, MoE, final mixing and logits are unused.
            const std::string b="mtp.layers.0.self_attn";const auto offset=state.position;
            auto iqk=gpu_.linear(linear(b+".indexer.index_qk_proj"),x,T);
            auto rk=gpu_.linear(linear(b+".k_proj"),x,T),v=gpu_.linear(linear(b+".v_proj"),x,T);
            auto k=gpu_.allocate(uint64_t(T)*512*4);
            gpu_.dispatch("norm_rope",{{rk},{norms_.at(b+".k_norm.weight")},{k}},{256,2,512,256,offset,T,1},32*2,T);
            gpu_.dispatch("index_store",{{iqk},{state.index}},{T,offset},128,T);
            gpu_.dispatch("kv_store",{{k},{state.keys}},{512,T,offset},512,T);
            gpu_.dispatch("kv_store",{{v},{state.values}},{512,T,offset},512,T);
            gpu_.finish();cancelled(cancel);state.position+=T;state.valid=true;return {};
        }
        auto a=attention(x,state,T);observe("attention",a);auto after=gpu_.allocate(fused->bytes);
        gpu_.dispatch("hc_add",{{fused},{a},{inj},{after}},{T},Hyper,T);observe("after_attention",after);
        auto [mx,mi]=hyper(after,"mtp.layers.0.mlp_hyper_connection",T);observe("moe_input",mx);
        auto m=moe(mx,T,cancel);observe("moe",m);auto wide=gpu_.allocate(after->bytes);
        gpu_.dispatch("hc_add",{{after},{m},{mi},{wide}},{T},Hyper,T);
        auto [mixed,unused]=hyper(wide,"mtp.hyper_connection_mixer",T,false);(void)unused;
        Buf result;if(logits) result=gpu_.linear(head_,mixed,T);
        gpu_.finish();cancelled(cancel);observe("wide",wide);observe("mixed",mixed);observe("logits",result);
        state.position+=T;state.valid=true;return {wide,mixed,result};
    } catch(...) {const auto error=std::current_exception();try{gpu_.finish();}catch(...){}reads_.drain();std::rethrow_exception(error);}
}
Json MtpDraft::stats() const {return {{"recipe",manifest_["recipe"]},{"expert_cache",cache_->json()},
    {"expert_read_bytes",experts_->read_bytes.load()},{"budget_bytes",budget_bytes(cache_->capacity(),context_)},
    {"context",context_},{"norm_convention","BF16(1+w), once at load"}};}
} // namespace freellm::qwen
