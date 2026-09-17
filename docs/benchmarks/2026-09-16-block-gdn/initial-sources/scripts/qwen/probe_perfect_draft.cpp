// Standalone optimistic verifier screen. Never linked into production inference.
#include "qwen/session.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <malloc/malloc.h>
#include <csignal>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>

using namespace freellm::qwen;
namespace {
std::atomic<bool> stopped=false;
void interrupt(int) { stopped=true; }
void check(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
std::string hash(std::span<const std::byte> bytes) {
    unsigned char d[CC_SHA256_DIGEST_LENGTH]; CC_SHA256(bytes.data(),CC_LONG(bytes.size()),d);
    std::string out;for(auto c:d) {out+="0123456789abcdef"[c>>4];out+="0123456789abcdef"[c&15];}return out;
}
std::string hash_file(const std::filesystem::path& path) {auto text=read_text(path);return hash(std::as_bytes(std::span(text)));}
std::string row_hash(std::span<const float> row) {return hash(std::as_bytes(row));}
constexpr auto fields=std::array{&LayerState::conv,&LayerState::recurrence,&LayerState::keys,&LayerState::values,&LayerState::index,&LayerState::ple_conv};
// Own only detached host bytes. Keeping a shallow State would retain replaced
// convolution buffers and hide live GPU ownership from the checkpoint budget.
class CheckpointCopy {
    struct Region {int layer,field;uint64_t full_bytes,offset=0,bytes=0;Buf host;};
    std::vector<Region> regions_;
    Artifact artifact_;std::array<uint32_t,Layers> positions_{};
    std::array<int,2> history_{};uint32_t tokens_=0;uint64_t trace_=0;bool valid_=false;
public:
    uint64_t allocated=0;
    explicit CheckpointCopy(const State& state) {
        regions_.reserve(Layers*6);
        for(int l=0;l<Layers;++l) for(int f=0;f<6;++f) if(auto& b=state.layers[l].*fields[f]) {
            const uint64_t capacity=f>=2 && f<=4?4ull*(f==4?128:512)*sizeof(float):b->bytes;
            auto host=Buffer::host(capacity);const auto host_bytes=malloc_size(host->data);allocated+=host_bytes;
            // Every arm reserves and commits the same four-token capacity before
            // priming; candidate save timing must not include first page faults.
            std::memset(host->data,0,host_bytes);
            regions_.push_back({l,f,b->bytes,0,0,std::move(host)});
        }
        allocated+=malloc_size(regions_.data());
        check(allocated<=128*MiB,"rollback snapshot exceeds fixed 128MiB admission");
    }
    void save(const State& state,uint32_t width) {
        check(state.valid && width && width<=4,"invalid checkpoint state or width");
        for(auto& r:regions_) {
            const auto& b=state.layers[r.layer].*fields[r.field];
            check(b && b->bytes==r.full_bytes,"checkpoint geometry changed");
            const bool rows=r.field>=2 && r.field<=4;
            const uint64_t stride=(r.field==4?128:512)*sizeof(float);
            r.offset=rows?state.tokens*stride:0;r.bytes=rows?width*stride:b->bytes;
            check(r.offset+r.bytes<=b->bytes && r.bytes<=r.host->bytes,"checkpoint exceeds state bounds");
        }
        for(auto& r:regions_) std::memcpy(r.host->data,(state.layers[r.layer].*fields[r.field])->data+r.offset,r.bytes);
        artifact_=state.artifact;history_=state.history;tokens_=state.tokens;trace_=state.trace_session_id;valid_=state.valid;
        for(int l=0;l<Layers;++l) positions_[l]=state.layers[l].position;
    }
    void restore(State& state) const {
        check(valid_ && state.artifact==artifact_,"checkpoint artifact mismatch");
        // Validate every destination before modifying any of them. Forward has
        // already drained its I/O/GPU users; restore into current buffer owners.
        for(auto& r:regions_) {const auto& b=state.layers[r.layer].*fields[r.field];
            check(b && b->bytes==r.full_bytes && r.offset+r.bytes<=b->bytes,"restore geometry changed");}
        state.valid=false;
        for(auto& r:regions_) std::memcpy((state.layers[r.layer].*fields[r.field])->data+r.offset,r.host->data,r.bytes);
        for(int l=0;l<Layers;++l) state.layers[l].position=positions_[l];
        state.history=history_;state.tokens=tokens_;state.trace_session_id=trace_;state.valid=valid_;
    }
};
Json checkpoint_self_test() {
    State state;state.artifact=Artifact::Mixed;state.valid=true;state.tokens=2;
    state.history={17,23};state.trace_session_id=71;
    for(auto& layer:state.layers) layer.position=2;
    state.layers[0].conv=Buffer::host(256);state.layers[0].recurrence=Buffer::host(512);
    state.layers[1].ple_conv=Buffer::host(128);
    state.layers[3].keys=Buffer::host(8*512*sizeof(float));
    state.layers[3].values=Buffer::host(8*512*sizeof(float));
    state.layers[3].index=Buffer::host(8*128*sizeof(float));
    for(int l=0;l<Layers;++l) for(int f=0;f<6;++f) if(auto& b=state.layers[l].*fields[f])
        for(uint64_t at=0;at<b->bytes;++at) b->data[at]=std::byte((at+17*l+f)%251);
    const auto initial=state_digest(state);CheckpointCopy checkpoint(state);checkpoint.save(state,4);
    check(checkpoint.allocated>0 && checkpoint.allocated<=128*MiB,"self-test snapshot memory bound");
    std::weak_ptr<Buffer> old_conv=state.layers[0].conv;
    state.layers[0].conv=Buffer::host(256);auto* current_conv=state.layers[0].conv->data;
    check(old_conv.expired(),"checkpoint retained replaced convolution owner");
    auto dirty=[&] {
        for(int l=0;l<Layers;++l) for(int f=0;f<6;++f) if(auto& b=state.layers[l].*fields[f]) {
            const bool rows=f>=2 && f<=4;const uint64_t stride=(f==4?128:512)*sizeof(float);
            std::memset(b->data+(rows?2*stride:0),0xee,rows?4*stride:b->bytes);
        }
        for(auto& layer:state.layers) layer.position=6;
        state.tokens=6;state.history={99,100};state.trace_session_id=999;state.valid=false;
    };
    dirty();checkpoint.restore(state);
    check(state_digest(state)==initial && state.trace_session_id==71 && state.layers[0].conv->data==current_conv,
          "checkpoint failed to restore replacement owner and metadata");
    // Simulate acceptance of every proper prefix after restoring all four
    // speculative rows. Unaccepted rows must retain their pre-verification bytes.
    for(uint32_t accepted=1;accepted<4;++accepted) {
        dirty();checkpoint.restore(state);check(state_digest(state)==initial,"full speculative tail was not restored");
        for(int f=2;f<=4;++f) {
            auto& b=state.layers[3].*fields[f];const uint64_t stride=(f==4?128:512)*sizeof(float);
            std::memset(b->data+2*stride,0xdd,accepted*stride);
            for(uint64_t at=(2+accepted)*stride;at<6*stride;++at)
                check(b->data[at]==std::byte((at+17*3+f)%251),"accepted prefix overwrote rejected tail");
        }
    }
    checkpoint.restore(state);
    // A late geometry error must not restore earlier buffers or change metadata.
    std::memset(state.layers[0].conv->data,0xaa,state.layers[0].conv->bytes);
    state.layers[3].index=Buffer::host(7*128*sizeof(float));
    std::memset(state.layers[3].index->data,0xbb,state.layers[3].index->bytes);
    state.trace_session_id=73;const auto before_error=state_digest(state);bool rejected=false;
    try {checkpoint.restore(state);} catch(const std::runtime_error&) {rejected=true;}
    check(rejected && state_digest(state)==before_error && state.trace_session_id==73,
          "geometry error modified checkpoint destinations");
    return {{"kind","perfect_draft_checkpoint_self_test_v1"},{"passed",true},{"gpu_used",false},
        {"replacement_owner_restored",true},{"full_speculative_tail_restored",true},
        {"geometry_rejected_before_writes",true},{"snapshot_allocated_bytes",checkpoint.allocated}};
}
void save_report(const std::filesystem::path& path,const Json& report) {std::ofstream out(path);out<<report.dump(2)<<'\n';check(bool(out),"cannot save probe report");}
void phase(const std::filesystem::path& path,const char* name,uint32_t consumed=0) {
    std::cout<<name<<" "<<consumed<<std::endl;
    auto progress=path;progress.replace_extension(".progress.jsonl");std::ofstream out(progress,std::ios::app);
    out<<Json{{"phase",name},{"consumed_tokens",consumed},{"monotonic_ns",monotonic_ns()}}.dump()<<'\n';
}
}
int main(int argc,char** argv) {
    Json report={{"kind","perfect_draft_probe_v1"},{"complete",false},{"optimistic_upper_bound_only",true},
        {"normal_request_latency_qualified",false},{"production_promoted",false},{"route_audit_enabled",true}};
    bool may_write=false;
    try {
        if(argc==2 && std::string_view(argv[1])=="--checkpoint-self-test") {
            std::cout<<checkpoint_self_test().dump()<<'\n';return 0;
        }
        check(argc==7 || argc==8,"usage: probe MODEL PREPARED INPUT_JSON REPORT WIDTH timing|validate [1072|1460]");
        const std::filesystem::path output=argv[4];check(!std::filesystem::exists(output),"report already exists");may_write=true;
        const int width=std::stoi(argv[5]);check(width==1 || width==2 || width==4,"invalid verification width");
        const std::string_view slots_arg=argc==8?argv[7]:"1072";
        check(slots_arg=="1072" || slots_arg=="1460","invalid fixed expert capacity");
        const uint32_t slots=slots_arg=="1460"?1460:1072;
        const std::string mode=argv[6];check(mode=="timing" || mode=="validate","invalid probe mode");const bool validation=mode=="validate";
        check(validation?std::getenv("MTL_DEBUG_LAYER") && std::getenv("MTL_SHADER_VALIDATION"):
            !std::getenv("MTL_DEBUG_LAYER") && !std::getenv("MTL_SHADER_VALIDATION"),"wrong Metal validation mode");
        std::signal(SIGINT,interrupt);std::signal(SIGTERM,interrupt);
        const auto input=read_json(argv[3]);const auto prompt=input.at("prompt_ids").get<std::vector<int>>();
        const auto continuation=input.at("continuation_ids").get<std::vector<int>>(),expected=input.at("expected_next_ids").get<std::vector<int>>();
        check(prompt.size()==72 && continuation.size()==16 && expected.size()==16,"invalid fixed workload length");
        for(const auto* ids:{&prompt,&continuation,&expected}) for(int id:*ids) check(id>=0 && id<Vocab,"invalid token ID");
        check(input.at("expected_prompt_id")==continuation.front(),"wrong first continuation token");
        for(size_t i=1;i<continuation.size();++i) check(continuation[i]==expected[i-1],"inconsistent greedy continuation");
        const uint32_t count=validation?4:16;
        Options options;options.artifact=Artifact::Mixed;options.model=argv[1];options.prepared=argv[2];
        options.memory=12*GiB;options.context=8192;options.expert_slots=slots;options.panel=512;options.chunk=128;
        options.residency="core-cache";options.decode_scratch="reuse";options.kernels.policy="candidate";options.kernels.q8_decode_rows=2;
        options.kernels.route_selection="simd";options.audit_routes=true;
        report.update(Json{{"validation",validation},{"width",width},{"verified_tokens",count},
            {"prime_expert_schedule","fixed-lease-batches-32"},
            {"input_sha256",hash_file(argv[3])},{"artifact_revision",artifact_revision(Artifact::Mixed)},
            {"prepared_manifest_sha256",hash_file(std::filesystem::path(argv[2])/"manifest.json")},
            {"runtime_kernel_policy","decode token_tile=width; q8r2 atT1"},
            {"configuration",{{"memory_bytes",12*GiB},{"context",8192},{"expert_slots",slots},{"artifact","mixed-4_8bit"},
                {"q4_decode","reference"},{"q8_decode_rows",2},{"route_selection","simd"},{"residency","core-cache"},
                {"decode_scratch","reuse"},{"token_tile",width},{"cache_policy","clock"},{"ready_group",4},
                {"io_workers",8},{"panel",512},{"chunk",128},{"decode_submission","immediate"}}},
            {"host_before",host_conditions()},{"blocks",Json::array()},{"rollback_checks",Json::array()}});
        phase(output,"load_model");const auto setup_start=monotonic_ns();
        {
            Model model(options);const auto plan=model.memory_plan().json();report["memory_plan"]=plan;
            check(plan.at("limit_bytes")==12*GiB && plan.at("expert_slots")==slots && plan.at("panel_tokens")==512,"fixed memory admission failed");
            // Reserve admission before allocating detached checkpoint bytes. The
            // fixed logits allowance covers four simultaneous max-width vectors,
            // including validation temporaries, proposals and vector metadata.
            constexpr uint64_t logits_bound=4*4ull*Vocab*sizeof(float)+MiB;
            check(plan.at("planned_bytes").get<uint64_t>()+128*MiB+logits_bound<=12*GiB,"checkpoint workspace not admitted");
            auto state=model.make_state();CheckpointCopy checkpoint(state);
            report["snapshot_allocated_bytes"]=checkpoint.allocated;report["host_logits_bound_bytes"]=logits_bound;
            model.prepare_pipelines();std::mt19937_64 rng(0);
            phase(output,"prime");model.phase("prefill");model.prepare_ingest(prompt.size());
            auto prime=model.forward(prompt,state,true,&stopped);model.finish_ingest();model.diagnostic_drain();
            check(model.options().completion_pipeline,"priming failed to restore completion-driven decode");
            check(prime.size()==Vocab && sample(prime,0,20,.95f,rng)==continuation.front(),"prompt greedy result changed");
            const auto prime_state=state_digest(state);report["before"]=model.stats();
            report["prime"]={{"logits_sha256",row_hash(prime)},{"state",prime_state},{"routes",model.route_identity()},
                {"cache_state",report["before"]["expert_cache"].at("diagnostic_cache_state")}};
            prime.clear();prime.shrink_to_fit();report["setup_ns"]=monotonic_ns()-setup_start;
            model.phase("decode");uint64_t total_wall=0;
            for(uint32_t at=0;at<count;at+=width) {
                check(!stopped,"probe cancelled");phase(output,"verify_block",at);
                const auto ids=std::span(continuation).subspan(at,width);const auto memory_before=process_memory();
                const auto begin=monotonic_ns();uint64_t checkpoint_ns=0;
                if(width>1) {checkpoint.save(state,width);checkpoint_ns=monotonic_ns()-begin;}
                const auto forward_start=monotonic_ns();auto logits=model.forward(ids,state,true,&stopped);
                const auto forward_end=monotonic_ns();check(logits.size()==uint64_t(width)*Vocab,"missing per-token logits");
                std::vector<int> next;next.reserve(width);
                for(int row=0;row<width;++row) {const auto id=sample(std::span(logits).subspan(row*Vocab,Vocab),0,20,.95f,rng);
                    next.push_back(id);check(id==expected[at+row],"verified greedy token changed");}
                const auto end=monotonic_ns();const auto memory_after=process_memory();total_wall+=end-begin;
                Json hashes=Json::array();for(int row=0;row<width;++row) hashes.push_back(row_hash(std::span(logits).subspan(row*Vocab,Vocab)));
                // Hash the full persistent state at the same token boundaries
                // in every timing arm, avoiding width-dependent memory scans.
                const bool evidence_boundary=validation || (at+width)%4==0;
                report["blocks"].push_back({{"offset",72+at},{"input_tokens",std::vector<int>(ids.begin(),ids.end())},
                    {"next_ids",next},{"logits_sha256",hashes},
                    {"state",evidence_boundary?state_digest(state):Json(nullptr)},
                    {"routes",evidence_boundary?model.route_identity():Json(nullptr)},
                    {"checkpoint_ns",checkpoint_ns},{"forward_ns",forward_end-forward_start},{"accept_ns",end-forward_end},
                    {"wall_ns",end-begin},{"memory_before",memory_before},{"memory_after",memory_after}});
                if(validation && width>1 && at==0) {
                    phase(output,"rollback_validation");model.diagnostic_drain();checkpoint.restore(state);
                    std::vector<int> changed(ids.begin(),ids.end());changed.back()=(changed.back()+1)%Vocab;
                    auto changed_logits=model.forward(changed,state,true,&stopped);Json causal=Json::array();bool equal=true;
                    for(int row=0;row<width-1;++row) {auto h=row_hash(std::span(changed_logits).subspan(row*Vocab,Vocab));causal.push_back(h);equal&=h==hashes[row].get<std::string>();}
                    report["rollback_checks"].push_back({{"name","causal_prefix_unchanged"},{"passed",equal},{"logits_sha256",causal}});
                    check(equal,"future proposal changed earlier logits");model.diagnostic_drain();checkpoint.restore(state);
                    auto recovery=state_digest(state);report["rollback_checks"].push_back({{"name","zero_accept_restores_state"},{"passed",recovery==prime_state},{"recovery_state",recovery}});
                    check(recovery==prime_state,"zero-accept rollback changed state");
                    for(int accepted=1;accepted<width;++accepted) {
                        checkpoint.restore(state);std::vector<float> last;
                        for(int k=0;k<accepted;++k) last=model.forward(ids.subspan(k,1),state,true,&stopped);
                        report["rollback_checks"].push_back({{"name","accepted_prefix_replay_"+std::to_string(accepted)},
                            {"passed",sample(last,0,20,.95f,rng)==expected[accepted-1]},
                            {"recovery_state",state_digest(state)},{"logits_sha256",row_hash(last)}});
                    }
                    model.diagnostic_drain();checkpoint.restore(state);
                    auto repeated=model.forward(ids,state,true,&stopped);
                    check(row_hash(repeated)==row_hash(logits) && state_digest(state)==report["blocks"].back()["state"],"rollback/replay changed correct block");
                }
                if(evidence_boundary) save_report(output,report);
            }
            model.diagnostic_drain();report["after"]=model.stats();report["final_state"]=state_digest(state);
            report["decode_wall_ns"]=total_wall;report["verified_tokens_per_second"]=double(count)*1e9/total_wall;
            check(model.memory_plan().json()==plan,"memory admission changed during probe");
            phase(output,"destroy_model");
        }
        report["process_after_destroy"]=process_memory();report["host_after"]=host_conditions();report["complete"]=true;
        save_report(output,report);phase(output,"complete",count);return 0;
    } catch(const std::exception& error) {
        report["error"]=error.what();report["process_on_error"]=process_memory();
        if(may_write) save_report(argv[4],report);std::cerr<<error.what()<<'\n';return 2;
    }
}
