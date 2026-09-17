#include "mtp_target_recovery.hpp"
#include <cerrno>
#include <fcntl.h>
#include <system_error>
#include <unistd.h>
#include "mtp_direct_output.hpp"
#include "mtp_ngram_init.hpp"
#include "mtp_expert_scratch.hpp"
#include "mtp_draft.hpp"
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
            const uint64_t capacity=f>=2 && f<=4?8ull*(f==4?128:512)*sizeof(float):b->bytes;
            auto host=Buffer::host(capacity);const auto host_bytes=malloc_size(host->data);allocated+=host_bytes;
            // Every arm reserves and commits the same eight-token capacity before
            // timing; candidate save timing must not include first page faults.
            std::memset(host->data,0,host_bytes);
            regions_.push_back({l,f,b->bytes,0,0,std::move(host)});
        }
        allocated+=malloc_size(regions_.data());
        check(allocated<=128*MiB,"rollback snapshot exceeds fixed 128MiB admission");
    }
    void save(const State& state,uint32_t width) {
        check(state.valid && width && width<=8,"invalid checkpoint state or width");
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

    void check_prefix(const State& state,uint32_t keep,uint32_t width=4) const {
        check(valid_ && state.artifact==artifact_ && (width==2 || width==4) && keep>=1 && keep<=width && state.tokens==tokens_+width,"invalid prefix checkpoint");
        for(const auto& r:regions_) {
            const auto& b=state.layers[r.layer].*fields[r.field];
            const bool rows=r.field>=2 && r.field<=4;const uint64_t stride=(r.field==4?128:512)*4;
            check(b && b->bytes==r.full_bytes && r.offset+r.bytes<=b->bytes && r.host && r.bytes<=r.host->bytes &&
                (!rows || (r.bytes==width*stride && r.offset==tokens_*stride)),"invalid prefix restore geometry");
        }
    }
    void restore_prefix(State& state,uint32_t keep,uint32_t width=4) const {
        check_prefix(state,keep,width);state.valid=false;
        for(const auto& r:regions_) {
            const bool rows=r.field>=2 && r.field<=4;const uint64_t skip=rows?keep*(r.field==4?128ull:512ull)*4:0;
            std::memcpy((state.layers[r.layer].*fields[r.field])->data+r.offset+skip,r.host->data+skip,r.bytes-skip);
        }
        for(int l=0;l<Layers;++l) state.layers[l].position=positions_[l];
        state.history=history_;state.tokens=tokens_;state.trace_session_id=trace_;
    }
    void commit_prefix(State& state,std::span<const int> ids) const {
        check(!state.valid && state.tokens==tokens_ && ids.size()>=1 && ids.size()<=4,"invalid prefix commit");
        for(auto id:ids) check(id>=0 && id<Vocab,"invalid prefix history");
        for(int l=0;l<Layers;++l) state.layers[l].position=positions_[l]+uint32_t(ids.size());
        for(auto id:ids) state.history={state.history[1],id};
        state.tokens=tokens_+uint32_t(ids.size());state.valid=true;
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
// Appended to the existing verifier's checkpoint/report helpers by the builder.
namespace {
Linear load_shared(const Checkpoint& cp,Metal& gpu,const std::string& name) {
    Linear l;l.quantized=true;l.group=64;
    const auto& q=cp.config.at("quantization");l.bits=q.value(name,q).at("bits");
    const auto& w=cp.at(name+".weight");l.input=uint32_t(w.shape[1])*32/l.bits;l.output=w.shape[0];
    auto load=[&](const std::string& suffix){return Binding{cp.load(name+suffix,[&](uint64_t n){return gpu.allocate(n,AllocationClass::Resident);})};};
    l.weight=load(".weight");l.scales=load(".scales");l.biases=load(".biases");return l;
}
int argmax(std::span<const float> values) {
    check(values.size()==Vocab,"missing MTP/target logits");
    for(float x:values) check(std::isfinite(x),"nonfinite logits");
    return int(std::max_element(values.begin(),values.end())-values.begin());
}
Json draft_digest(const DraftState& state) {
    return {{"position",state.position},{"valid",state.valid},
        {"keys",hash({state.keys->data,size_t(state.keys->bytes)})},
        {"values",hash({state.values->data,size_t(state.values->bytes)})},
        {"index",hash({state.index->data,size_t(state.index->bytes)})}};
}
void save_buffer(const std::filesystem::path& path,const Buf& b) {
    std::ofstream file(path,std::ios::binary);file.write(reinterpret_cast<const char*>(b->data),b->bytes);check(bool(file),"cannot save MTP fixture");
}
Buf detached(Metal& gpu,const Buf& buffer,uint32_t first,uint32_t count) {
    auto out=gpu.allocate(uint64_t(count)*Hyper*4,AllocationClass::Workspace);
    gpu.copy(buffer,uint64_t(first)*Hyper*4,out,0,out->bytes);gpu.finish();return out;
}
void fixture(const std::filesystem::path& model,const std::filesystem::path& prepared,const Json& input,
             const std::filesystem::path& output,Json& report) {
    Metal gpu;gpu.budget(2*GiB);ReadPool reads(8);Checkpoint cp(model,true,Artifact::Mixed);
    auto embedding=load_shared(cp,gpu,"model.embed_tokens"),head=load_shared(cp,gpu,"lm_head");
    MtpDraft draft(gpu,reads,prepared,embedding,head,10,32);auto state=draft.make_state();
    const auto ids=input.at("ids").get<std::vector<int>>();check(ids.size()==4,"fixture expects four tokens");
    File hidden_file(input.at("hidden_file").get<std::string>());check(hidden_file.size()==4*Hyper*4,"fixture hidden size differs");
    auto hidden=gpu.allocate(hidden_file.size());hidden_file.read(0,{hidden->data,size_t(hidden->bytes)});
    auto trace=output.parent_path()/"trace";std::filesystem::create_directories(trace);Json captured=Json::object();
    draft.observer=[&](const std::string& name,const Buf& b) {
        save_buffer(trace/(name+".bin"),b);captured[name]={{"bytes",b->bytes},{"sha256",hash({b->data,size_t(b->bytes)})}};
    };
    DraftCheckpoint rollback(gpu);rollback.save(gpu,state,4);const auto initial=draft_digest(state);
    phase(output,"fixture_forward");auto full=draft.forward(ids,hidden,state,true);const auto full_state=draft_digest(state);
    const auto full_logit_hash=hash({full.logits->data,size_t(full.logits->bytes)});draft.observer={};
    full={};rollback.restore(gpu,state);check(draft_digest(state)==initial,"MTP rollback changed untouched prefix");
    const auto cache_before_catchup=draft.stats();draft.catch_up(ids,hidden,state);
    check(draft_digest(state)==full_state && draft.stats()==cache_before_catchup,"state-only MTP catch-up changed state or expert cache");
    rollback.restore(gpu,state);
    phase(output,"fixture_serial_replay");Json replay=Json::array();
    for(size_t at=0;at<ids.size();++at) {
        auto h=detached(gpu,hidden,at,1);auto one=draft.forward(std::span(ids).subspan(at,1),h,state,true);
        File expected(trace/"logits.bin");std::vector<float> row(Vocab);expected.read(at*Vocab*4,std::as_writable_bytes(std::span(row)));
        check(std::equal(row.begin(),row.end(),one.logits->floats().begin()),"MTP serial/block logits differ");
        replay.push_back(argmax(one.logits->floats()));
    }
    check(draft_digest(state)==full_state,"MTP serial/block state differs");
    rollback.restore(gpu,state);auto changed=ids;changed.back()=(changed.back()+1)%Vocab;
    auto other=draft.forward(changed,hidden,state,true);File baseline(trace/"logits.bin");std::vector<float> prefix(3*Vocab);
    baseline.read(0,std::as_writable_bytes(std::span(prefix)));
    check(std::equal(prefix.begin(),prefix.end(),other.logits->floats().begin()),"MTP future token changed earlier outputs");other={};
    rollback.restore(gpu,state);std::atomic<bool> stop=true;bool rejected=false;
    try{draft.forward(ids,hidden,state,true,&stop);}catch(const std::runtime_error&){rejected=true;}
    check(rejected && draft_digest(state)==initial,"cancelled MTP changed state");
    state.position=UINT32_MAX;rejected=false;
    try{draft.forward(ids,hidden,state,true);}catch(const std::runtime_error&){rejected=true;}
    check(rejected && state.valid && state.position==UINT32_MAX,"MTP overflow changed state");rollback.restore(gpu,state);
    auto wrong=gpu.zeros(31*128,AllocationClass::State);auto original=state.index;state.index=wrong;
    const auto before_bad_restore=draft_digest(state);rejected=false;
    try{rollback.restore(gpu,state);}catch(const std::runtime_error&){rejected=true;}
    check(rejected && draft_digest(state)==before_bad_restore,"invalid MTP restore modified state");state.index=original;
    // Every proper prefix is compared with a fresh zero-state replay. The cache
    // has just ten slots, forcing eviction throughout these checks.
    for(uint32_t count=1;count<4;++count) {
        rollback.restore(gpu,state);auto h=detached(gpu,hidden,0,count);auto partial=draft.forward(std::span(ids).first(count),h,state,false);
        auto recovered=draft_digest(state);auto fresh=draft.make_state();auto reference=draft.forward(std::span(ids).first(count),h,fresh,false);
        check(draft_digest(fresh)==recovered,"MTP proper-prefix recovery differs from fresh replay");
        auto minimal=draft.make_state();draft.catch_up(std::span(ids).first(count),h,minimal);
        check(draft_digest(minimal)==recovered,"partial state-only catch-up differs");
    }
    gpu.finish();report.update(Json{{"fixture_tensors",captured},{"greedy_ids",replay},{"full_logits_sha256",full_logit_hash},
        {"serial_block_exact",true},{"causal_prefix_exact",true},{"rollback_exact",true},{"proper_prefix_replay_exact",true},
        {"cancellation_before_update",true},{"overflow_rejected_before_write",true},{"bad_restore_rejected_before_write",true},
        {"state_only_catchup_exact",true},{"state_only_expert_cache_untouched",true},
        {"draft",draft.stats()},{"metal",gpu.statistics()},{"after",process_memory()}});
}
// Replaces only the developer short-screen harness. MTP/target arithmetic is shared.
// Included in the isolated continuation harness, using unchanged real Q4 bytes.
Json expert_scratch_test(const std::filesystem::path& prepared) {
    check(std::getenv("MTL_DEBUG_LAYER") && std::getenv("MTL_SHADER_VALIDATION"),"scratch fixture requires Metal validation");
    Metal gpu;gpu.budget(256*MiB);ReadPool reads(4);gpu.prepare_pipelines();
    const auto before=process_memory();const auto manifest=read_json(prepared/"manifest.json");
    check(manifest.at("format")=="freellm-affine-records-v1","wrong expert fixture artifact");
    const auto& layout=manifest.at("experts").at(0);
    check(layout.at("length")==ExpertBytes && layout.at("stride")==ExpertStride,"changed expert layout");
    File file(prepared/layout.at("file").get<std::string>());
    std::array<Buf,8> records;Json hashes=Json::array();
    for(uint32_t i=0;i<8;++i) {
        records[i]=gpu.allocate(ExpertBytes,AllocationClass::Expert);
        file.read(uint64_t(i)*ExpertStride,{records[i]->data,size_t(ExpertBytes)});
        hashes.push_back(hash({records[i]->data,size_t(ExpertBytes)}));
    }
    auto input=gpu.zeros(4*Hidden,AllocationClass::Workspace);
    for(size_t i=0;i<input->floats().size();++i) input->floats()[i]=round_bf16(float(int((i*3+i/Hidden)%19)-9)/64);
    auto output=gpu.zeros(4*TopK*Hidden,AllocationClass::Workspace);
    auto expected=gpu.zeros(4*TopK*Hidden,AllocationClass::Workspace);
    std::vector<ExpertKey> keys;for(uint32_t i=0;i<8;++i) keys.push_back({0,i});
    std::atomic<bool> reverse=false,release_first=false,fail_read=false;
    auto loader=[&](ExpertKey k,const Buf& b) {
        if(reverse && k.expert==0) {
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
            while(!release_first && std::chrono::steady_clock::now()<deadline) std::this_thread::yield();
            check(release_first,"ready expert work did not precede delayed read");
        }
        if(fail_read && k.expert==7) throw std::runtime_error("injected expert read failure");
        file.read(uint64_t(k.expert)*ExpertStride,{b->data,size_t(ExpertBytes)});
    };
    auto allocation=[&](uint64_t n){return gpu.allocate(n,AllocationClass::Expert);};
    Json cases=Json::array();
    {
        ExpertCache cache(3,allocation,reads,loader);
        for(uint32_t n:{1u,2u,3u,4u}) {
            std::array<std::vector<int>,8> positions;
            for(uint32_t e=0;e<8;++e) for(uint32_t row=0;row<(e%2?1:n);++row)
                positions[e].push_back(int(row*TopK+e));
            std::fill(expected->floats().begin(),expected->floats().end(),-19);
            for(uint32_t e=0;e<8;++e) encode_expert_rows(gpu,records[e],input,expected,positions[e],n,128,false,0,0);
            gpu.finish();reverse=true;release_first=false;std::vector<uint32_t> order;
            std::fill(output->floats().begin(),output->floats().end(),-19);
            Json result;
            {
                mtp_scratch::ForwardScope scope(gpu,true);
                result=execute_experts(keys,cache,reads,gpu,2,[&](ExpertKey k,const Buf& b) {
                    order.push_back(k.expert);if(k.expert!=0) release_first=true;
                    encode_expert_rows(gpu,b,input,output,positions[k.expert],n,128,false,0,0);
                },nullptr,true);
            }
            check(order.size()==8 && order.front()!=0,"reversed read was not exercised");
            check(std::memcmp(output->data,expected->data,expected->bytes)==0,"pooled expert rows changed output");
            check(result.at("peak_leases").get<size_t>()<=3 && result.at("peak_gpu_groups").get<size_t>()<=2,"unbounded expert resources");
            check(gpu.statistics().at("live_command_groups")==0 && gpu.statistics().at("active_scratch_slot")==-1,"expert users survived drain");
            cases.push_back({{"rows",n},{"exact",true},{"dependency",result},{"metal",gpu.statistics()}});
            reverse=false;cache.clear();
        }
        for(int failure=0;failure<3;++failure) {
            std::atomic<bool> cancel=false;uint32_t encoded=0;bool failed=false;fail_read=failure==2;
            try {
                mtp_scratch::ForwardScope scope(gpu,true);
                execute_experts(keys,cache,reads,gpu,2,[&](ExpertKey k,const Buf& b) {
                    const std::array<int,1> pos={int(k.expert)};
                    encode_expert_rows(gpu,b,input,output,pos,4,128,false,0,0);++encoded;
                    if(failure==0) cancel=true;
                    if(failure==1) throw std::runtime_error("injected expert encode failure");
                },&cancel);
            } catch(const std::exception&) {failed=true;}
            check(failed && encoded>0,"failure did not exercise outstanding GPU users");
            check(gpu.statistics().at("live_command_groups")==0 && gpu.statistics().at("active_scratch_slot")==-1,"failed work was not drained");
            cache.clear();fail_read=false;
        }
    }
    // Cached replays exercise reuse with no I/O and with both pool shapes warm.
    {
        ExpertCache cache(8,allocation,reads,loader);
        for(auto key:keys) {auto lease=cache.acquire(key);lease.wait();}
        const auto before_reuse=gpu.statistics().at("scratch_reuses").get<uint64_t>();
        for(int repeat=0;repeat<3;++repeat) {
            mtp_scratch::ForwardScope scope(gpu,true);
            const auto result=execute_experts(keys,cache,reads,gpu,4,[&](ExpertKey k,const Buf& b) {
                const std::array<int,4> pos={int(k.expert),int(k.expert+TopK),int(k.expert+2*TopK),int(k.expert+3*TopK)};
                encode_expert_rows(gpu,b,input,output,pos,4,128,false,0,0);
            });
            check(result.at("ready_hits")==8 && result.at("new_misses")==0,"all-hit case missed");
        }
        check(gpu.statistics().at("scratch_reuses").get<uint64_t>()>before_reuse,"no expert scratch reuse");
        cache.clear();
    }
    // Retain the first callback while a second slot is used. Re-entering slot 0
    // must wait before a host overwrite can race its old GPU copy.
    gpu.release_scratch();auto first=gpu.zeros(32,AllocationClass::Workspace),second=gpu.zeros(32,AllocationClass::Workspace);
    mtp_scratch::held_callbacks=0;mtp_scratch::hold_completions=true;
    std::thread releaser([] {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
        while(!mtp_scratch::held_callbacks && std::chrono::steady_clock::now()<deadline) std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds(30));mtp_scratch::hold_completions=false;
    });
    struct Join {std::thread& t;~Join(){mtp_scratch::hold_completions=false;if(t.joinable()) t.join();}} join{releaser};
    gpu.begin_scratch(0,mtp_scratch::PoolBytes);auto a=gpu.zeros(32);a->floats()[0]=17;
    gpu.copy(a,0,first,0,128);a.reset();gpu.end_scratch();
    gpu.begin_scratch(1,mtp_scratch::PoolBytes);auto b=gpu.zeros(32);b->floats()[0]=23;
    gpu.copy(b,0,second,0,128);b.reset();gpu.end_scratch();
    gpu.begin_scratch(0,mtp_scratch::PoolBytes);auto overwritten=gpu.zeros(32);overwritten->floats()[0]=-99;
    gpu.end_scratch();gpu.finish();releaser.join();
    check(mtp_scratch::held_callbacks>0 && first->floats()[0]==17 && second->floats()[0]==23,"delayed completion reused live memory");
    const auto retained=gpu.statistics();overwritten.reset();gpu.release_scratch();
    for(const auto& p:gpu.statistics().at("scratch_pools")) check(p.at("allocated_bytes")==0,"pool survived release");
    return {{"kind","mtp_expert_scratch_fixture_v1"},{"complete",true},{"cases",cases},{"expert_hashes",hashes},
        {"rows_exact",true},{"reversed_reads",true},{"forced_eviction",true},{"all_hits",true},
        {"cancellation_drained",true},{"encode_failure_drained",true},{"read_failure_drained",true},
        {"delayed_completion_safe",true},{"held_callbacks",mtp_scratch::held_callbacks.load()},
        {"pools_released",true},{"retained",retained},{"before",before},{"after",process_memory()},
        {"production_promoted",false},{"performance_measurement",false}};
}

Json ngram_init_test(const std::filesystem::path& model,const std::filesystem::path& prepared) {
    Checkpoint cp(model,true,Artifact::Mixed);ReadPool reads(4);
    auto artifact=std::make_shared<PreparedArtifact>(prepared,cp);Json cases=Json::array();
    // A small ring forces many wraparounds; full capacity checks that unused
    // payload stays unconstructed. Both stores read the same real packed tables.
    for(uint64_t budget:{64*1024ull,64*MiB}) {
        check(!setenv("FREELLM_MTP_NGRAM_INIT","eager",1),"cannot select eager fixture");
        NgramStore eager(cp,reads,budget,artifact);
        check(!setenv("FREELLM_MTP_NGRAM_INIT","lazy",1),"cannot select lazy fixture");
        NgramStore lazy(cp,reads,budget,artifact);
        const auto initial=NgramAudit::summary(lazy);check(initial.at("constructed_rows")==0,"lazy constructor touched rows");
        check(initial.at("capacity_rows")==NgramAudit::summary(eager).at("capacity_rows"),"cache capacities differ");
        std::array<int,2> history={248044,248044};uint64_t tokens=0;Json boundaries=Json::array();
        for(uint32_t step=0;step<18;++step) {
            const uint32_t n=std::array<uint32_t,6>{1,2,3,7,11,17}[step%6];std::vector<int> ids(n);
            for(uint32_t i=0;i<n;++i) ids[i]=(step%3==0)?77091:int(100+step*97+i*29);
            if(step%4==0) ids[n/2]=248044;
            std::vector<float> a(uint64_t(n)*Hidden),b(a.size());
            check(eager.row_ids(ids,history)==lazy.row_ids(ids,history),"ngram addresses changed");
            eager.embedding(ids,history,a);lazy.embedding(ids,history,b);
            check(std::memcmp(a.data(),b.data(),a.size()*4)==0 && NgramAudit::same(eager,lazy),"ngram bytes, replacement or hit counts changed");
            // An exact repeat also checks live hits and duplicate-row fan-out.
            eager.embedding(ids,history,a);lazy.embedding(ids,history,b);
            check(std::memcmp(a.data(),b.data(),a.size()*4)==0 && NgramAudit::same(eager,lazy),"ngram repeat changed cache behavior");
            for(int id:ids) history={history[1],id};tokens+=n;
            boundaries.push_back({{"step",step},{"lazy",NgramAudit::summary(lazy)},{"output_sha256",row_hash(a)}});
        }
        const auto final=NgramAudit::summary(lazy);
        if(budget==64*1024ull) check(final.at("constructed_rows")==final.at("capacity_rows") && lazy.misses>initial.at("capacity_rows").get<uint64_t>()*2,"missing ring wraparound");
        else check(final.at("constructed_rows")==final.at("cached_rows") && final.at("initialized_row_bytes").get<uint64_t>()<MiB,"unused full cache rows initialized");
        cases.push_back({{"budget_bytes",budget},{"tokens",tokens},{"initial",initial},{"final",final},{"boundaries",boundaries}});
    }
    return {{"kind","mtp_ngram_init_fixture_v1"},{"complete",true},{"exact_outputs_and_cache",true},
        {"duplicate_rows_and_hits",true},{"eos_history",true},{"ring_wraparound",true},{"cases",cases},
        {"model_loaded",false},{"gpu_used",false},{"production_promoted",false}};
}

// Included only in the isolated harness. Every destination is checked, including
// untouched sentinel rows. Existing Q4 kernels and final reduction are unchanged.
Json direct_output_test(const std::filesystem::path& prepared) {
    check(std::getenv("MTL_DEBUG_LAYER") && std::getenv("MTL_SHADER_VALIDATION"),"direct fixture requires Metal validation");
    mtp_scratch::enabled=false;mtp_direct::enabled=true;
    Metal gpu;gpu.budget(256*MiB);
    KernelConfig config;config.policy="candidate";config.token_tile=4;gpu.configure(config);gpu.request_phase("decode");
    ReadPool reads(4);gpu.prepare_pipelines();const auto before=process_memory();
    const auto manifest=read_json(prepared/"manifest.json");const auto& layout=manifest.at("experts").at(0);
    check(manifest.at("format")=="freellm-affine-records-v1" && layout.at("length")==ExpertBytes &&
        layout.at("stride")==ExpertStride,"changed real expert layout");
    File file(prepared/layout.at("file").get<std::string>());
    std::array<Buf,8> records;Json hashes=Json::array();std::vector<ExpertKey> keys;
    for(uint32_t i=0;i<8;++i) {
        records[i]=gpu.allocate(ExpertBytes,AllocationClass::Expert);
        file.read(uint64_t(i)*ExpertStride,{records[i]->data,size_t(ExpertBytes)});
        hashes.push_back(hash({records[i]->data,size_t(ExpertBytes)}));keys.push_back({0,i});
    }
    auto input=gpu.zeros(4*Hidden,AllocationClass::Workspace);
    for(size_t i=0;i<input->floats().size();++i) input->floats()[i]=round_bf16(float(int((i*3+i/Hidden)%19)-9)/64);
    auto output=gpu.zeros(4*TopK*Hidden,AllocationClass::Workspace),expected=gpu.zeros(4*TopK*Hidden,AllocationClass::Workspace);
    auto clear=[&](const Buf& b){std::fill(b->floats().begin(),b->floats().end(),-19);};
    auto exact=[&]{gpu.finish();check(std::memcmp(output->data,expected->data,expected->bytes)==0,"direct output or untouched sentinel changed");};
    std::array<std::vector<int>,8> positions;
    auto geometry=[&](uint32_t n) {
        for(uint32_t e=0;e<8;++e) {
            positions[e].clear();for(uint32_t row=0;row<(e%2?1:n);++row)
                positions[e].push_back(int(((e+row+n)%4)*TopK+e));
        }
    };
    auto reference=[&] {
        clear(expected);
        for(uint32_t e=0;e<8;++e) encode_expert_rows(gpu,records[e],input,expected,positions[e],4,128,false,0,0);
        gpu.finish();
    };
    std::atomic<bool> reverse=false,release_first=false,fail_read=false;
    auto loader=[&](ExpertKey k,const Buf& b) {
        if(reverse && k.expert==0) {
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
            while(!release_first && std::chrono::steady_clock::now()<deadline) std::this_thread::yield();
            check(release_first,"ready work did not pass delayed first read");
        }
        if(fail_read && k.expert==7) throw std::runtime_error("injected direct-output read failure");
        file.read(uint64_t(k.expert)*ExpertStride,{b->data,size_t(ExpertBytes)});
    };
    auto allocation=[&](uint64_t bytes){return gpu.allocate(bytes,AllocationClass::Expert);};
    Json cases=Json::array();
    {
        ExpertCache cache(3,allocation,reads,loader);
        for(uint32_t n:{1u,2u,3u,4u}) {
            geometry(n);reference();clear(output);reverse=true;release_first=false;std::vector<uint32_t> order;
            const auto old=mtp_direct::direct_writes;Json timing;
            {
                mtp_direct::Scope scope(true);
                timing=execute_experts(keys,cache,reads,gpu,2,[&](ExpertKey k,const Buf& b) {
                    order.push_back(k.expert);if(k.expert!=0) release_first=true;
                    encode_expert_rows(gpu,b,input,output,positions[k.expert],4,128,false,0,0);
                },nullptr,true);
            }
            exact();check(order.size()==8 && order.front()!=0,"read order was not reversed");
            check(mtp_direct::direct_writes-old==(n==1?8:4),"wrong eligible direct row count");
            check(timing.at("peak_leases").get<uint64_t>()<=3 && timing.at("peak_gpu_groups").get<uint64_t>()<=2,
                "direct output exceeded resource bounds");
            check(gpu.statistics().at("live_command_groups")==0,"direct users survived drain");
            cases.push_back({{"rows",n},{"exact",true},{"direct_writes",mtp_direct::direct_writes-old},{"dependency",timing}});
            reverse=false;cache.clear();
        }
        for(int failure=0;failure<3;++failure) {
            std::atomic<bool> cancel=false;uint32_t encoded=0;bool failed=false;fail_read=failure==2;
            try {
                mtp_direct::Scope scope(true);
                execute_experts(keys,cache,reads,gpu,2,[&](ExpertKey k,const Buf& b) {
                    const std::array<int,1> pos={int(((k.expert+1)%4)*TopK+k.expert)};
                    encode_expert_rows(gpu,b,input,output,pos,4,128,false,0,0);++encoded;
                    if(failure==0) cancel=true;
                    if(failure==1) throw std::runtime_error("injected direct-output encode failure");
                },&cancel);
            } catch(const std::exception&) {failed=true;}
            check(failed && encoded>0 && !mtp_direct::in_verifier,"failure did not unwind active work");
            check(gpu.statistics().at("live_command_groups")==0,"failed direct users were not drained");
            cache.clear();fail_read=false;
        }
    }
    // The coordinator must keep output and expert records alive even while its
    // GPU completion callbacks are deliberately delayed.
    {
        ExpertCache cache(8,allocation,reads,loader);
        for(auto k:keys) {auto lease=cache.acquire(k);lease.wait();}
        geometry(3);reference();clear(output);
        mtp_scratch::held_callbacks=0;mtp_scratch::hold_completions=true;
        std::thread releaser([] {
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
            while(!mtp_scratch::held_callbacks && std::chrono::steady_clock::now()<deadline) std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::milliseconds(30));mtp_scratch::hold_completions=false;
        });
        struct Join {std::thread& t;~Join(){mtp_scratch::hold_completions=false;if(t.joinable()) t.join();}} join{releaser};
        Json timing;
        {
            mtp_direct::Scope scope(true);
            timing=execute_experts(keys,cache,reads,gpu,4,[&](ExpertKey k,const Buf& b) {
                encode_expert_rows(gpu,b,input,output,positions[k.expert],4,128,false,0,0);
            });
        }
        releaser.join();exact();check(mtp_scratch::held_callbacks>0,"GPU delay not exercised");
        check(timing.at("ready_hits")==8 && timing.at("new_misses")==0,"all-hit fixture missed");cache.clear();
    }
    // Invalid destinations must be rejected before encoding even a valid prefix.
    for(const auto& pos:std::array<std::array<int,2>,2>{{{{3,-1}},{{3,40}}}}) {
        clear(output);const auto count=gpu.statistics().at("dispatches");bool failed=false;
        try {mtp_direct::Scope scope(true);encode_expert_rows(gpu,records[0],input,output,pos,4,128,false,0,0);}
        catch(const std::invalid_argument&) {failed=true;}
        check(failed && gpu.statistics().at("dispatches")==count,"invalid destination encoded work");
        for(float x:output->floats()) check(x==-19,"invalid destination wrote output");
    }
    const auto writes=mtp_direct::direct_writes;
    // A one-token replay and a priming-like call must retain the original path.
    for(uint32_t tokens:{1u,4u}) {
        clear(output);const std::array<int,1> pos={3};
        mtp_direct::Scope scope(false);encode_expert_rows(gpu,records[0],input,output,pos,tokens,128,false,0,0);gpu.finish();
    }
    check(mtp_direct::direct_writes==writes,"direct output leaked outside verifier scope");
    gpu.finish();const auto stats=gpu.statistics();
    for(auto& r:records) r.reset();input.reset();output.reset();expected.reset();gpu.reap();
    check(gpu.allocated()==0 && gpu.statistics().at("live_command_groups")==0,"fixture retained Metal buffers");
    return {{"kind","mtp_direct_output_fixture_v1"},{"complete",true},{"cases",cases},{"expert_hashes",hashes},
        {"destinations_and_sentinels_exact",true},{"mixed_rows_exact",true},{"reversed_reads",true},{"forced_eviction",true},
        {"all_hits",true},{"invalid_destinations_rejected",true},{"scope_exclusion",true},{"cancellation_drained",true},
        {"read_failure_drained",true},{"encode_failure_drained",true},{"delayed_completion_safe",true},{"all_buffers_released",true},
        {"direct_output",mtp_direct::counters()},{"metal",stats},{"before",before},{"after",process_memory()},
        {"performance_measurement",false},{"production_promoted",false}};
}

// Developer-only saved-state fixture. Payloads never enter the production binary.
constexpr uint64_t FixtureLimit=2*GiB;
// A fixture is larger than a gigabyte. Do not leave its writes in the filesystem
// cache alongside the live model; the reader already uses uncached I/O.
class FixtureWriter {
    int fd_=-1;
public:
    explicit FixtureWriter(const std::filesystem::path& path) {
        fd_=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
        if(fd_<0) throw std::system_error(errno,std::generic_category(),"create fixture payload");
        if(::fcntl(fd_,F_NOCACHE,1)!=0) {
            const int error=errno;::close(fd_);fd_=-1;
            throw std::system_error(error,std::generic_category(),"uncached fixture writes");
        }
    }
    FixtureWriter(const FixtureWriter&)=delete;
    ~FixtureWriter(){if(fd_>=0)::close(fd_);}
    void write(std::span<const std::byte> bytes) {
        check(fd_>=0,"fixture writer is closed");
        while(!bytes.empty()) {
            const auto n=::write(fd_,bytes.data(),std::min<size_t>(bytes.size(),32*MiB));
            if(n<0 && errno==EINTR)continue;
            if(n<0)throw std::system_error(errno,std::generic_category(),"write fixture payload");
            check(n>0,"short fixture write");bytes=bytes.subspan(size_t(n));
        }
    }
    void close() {
        check(fd_>=0,"fixture writer is closed");const int fd=fd_;fd_=-1;
        if(::close(fd)!=0)throw std::system_error(errno,std::generic_category(),"close fixture payload");
    }
};
void capture_memory(Json& report,const char* event,uint32_t prefix=0) {
    auto& samples=report["capture_memory"];
    if(samples.is_null())samples=Json::array();
    check(samples.size()<12,"too many capture memory boundaries");
    samples.push_back({{"event",event},{"prefix",prefix},{"monotonic_ns",monotonic_ns()},{"process",process_memory()}});
}
std::string stream_hash(const std::filesystem::path& path) {
    File file(path);check(file.size()<=FixtureLimit,"fixture payload exceeds 2GiB");
    CC_SHA256_CTX context;CC_SHA256_Init(&context);std::array<std::byte,65536> bytes;
    for(uint64_t at=0;at<file.size();) {
        auto n=std::min<uint64_t>(bytes.size(),file.size()-at);file.read(at,{bytes.data(),size_t(n)});
        CC_SHA256_Update(&context,bytes.data(),CC_LONG(n));at+=n;
    }
    unsigned char out[32];CC_SHA256_Final(out,&context);std::string result;
    for(auto c:out){result+="0123456789abcdef"[c>>4];result+="0123456789abcdef"[c&15];}return result;
}
uint64_t fixture_geometry(int layer,int field) {
    if(field==5) return layer==1?9ull*Hyper*4:0;
    if((layer+1)%4) return field==0?3*10240*4:field==1?48*128*128*4:0;
    return field>=2 && field<=4?8192ull*(field==4?128:512)*4:0;
}
struct RecoveryBundle {
    std::filesystem::path path;std::unique_ptr<FixtureWriter> payload;uint64_t bytes=0;
    Json manifest={{"kind","target_recovery_fixture_v1"},{"complete",false},{"limit_bytes",FixtureLimit},
        {"reference_origin","full-target-forward"},
        {"artifact_revision",artifact_revision(Artifact::Mixed)},{"layers",Layers},{"context",8192},
        {"arithmetic","original-gdn_scan-conv_update-v1"},{"expected",Json::array()}};
    explicit RecoveryBundle(const std::filesystem::path& p):path(p) {
        check(!std::filesystem::exists(p),"fixture directory exists");std::filesystem::create_directories(p);
        payload=std::make_unique<FixtureWriter>(p/"payload.bin");
    }
    Json buffer(const Buf& b) {
        if(!b)return nullptr;check(bytes+b->bytes+MiB<=FixtureLimit,"fixture bundle exceeds 2GiB");
        Json record={{"offset",bytes},{"bytes",b->bytes},{"sha256",hash({b->data,size_t(b->bytes)})}};
        payload->write({b->data,size_t(b->bytes)});
        bytes+=b->bytes;return record;
    }
    void state(const char* name,const State& state) {
        Json rows=Json::array();
        for(int l=0;l<Layers;++l) {
            Json buffers=Json::array();
            for(int f=0;f<6;++f){const auto& b=state.layers[l].*fields[f];
                check((b?b->bytes:0)==fixture_geometry(l,f),"fixture state geometry differs");buffers.push_back(buffer(b));}
            rows.push_back({{"position",state.layers[l].position},{"buffers",buffers}});
        }
        manifest[name]={{"layers",rows},{"tokens",state.tokens},{"history",state.history},
            {"trace_session_id",state.trace_session_id},{"valid",state.valid},{"digest",state_digest(state)}};
    }
    void journal(const mtp_recovery::Journal& journal) {
        Json entries=Json::array();
        for(int l=0;l<Layers;++l) if((l+1)%4) {
            const auto& e=journal.entries[l];entries.push_back({{"layer",l},{"ad",e.ad},{"dd",e.dd},
                {"convolution",buffer(e.convolution)},{"normalized",buffer(e.normalized)},
                {"a",buffer(e.a)},{"b",buffer(e.b)},{"alog",buffer(e.alog)},{"dt",buffer(e.dt)}});
        }
        manifest["journal"]={{"entries",entries},{"ple",buffer(journal.ple)},
            {"offset",journal.offset},{"ids",journal.ids},{"accounting",journal.stats()}};
    }
    void finish() {
        payload->close();manifest["payload_bytes"]=bytes;manifest["payload_sha256"]=stream_hash(path/"payload.bin");
        manifest["complete"]=true;check(manifest.dump().size()<MiB,"fixture manifest exceeds bound");
        save_report(path/"manifest.json",manifest);
    }
};
[[maybe_unused]] void capture_recovery_fixture(RecoveryBundle& bundle,Model& model,State& state,CheckpointCopy& checkpoint,
        mtp_recovery::Journal& journal,std::span<const int> ids,Json& report,const std::filesystem::path& output) {
    model.diagnostic_drain();journal.validate(state,4);capture_memory(report,"verified_before_write");
    bundle.state("verified",state);capture_memory(report,"verified_written");
    bundle.journal(journal);capture_memory(report,"journal_written");
    bundle.manifest["source"]={{"request_id",report.at("request_id")},{"draft_manifest_sha256",report.at("draft_manifest_sha256")},
        {"input_sha256",report.at("input_sha256")},
        {"producer_binary_sha256",report.at("producer_binary_sha256")},
        {"native_build_fingerprint",report.at("before").at("metal").at("build_fingerprint")},
        {"kernel_policy",report.at("before").at("metal").at("kernels")},
        {"target_prepared_sha256",report.at("before").at("prepared").at("manifest_sha256")}};
    // Golden states use complete target forwards, without calling Journal::apply.
    for(uint32_t keep=1;keep<=4;++keep) {
        phase(output,"capture_reference_prefix",keep);
        check(!stopped,"capture cancelled");checkpoint.restore(state);Json logits=Json::array();
        for(uint32_t i=0;i<keep;++i) {
            model.phase("decode");auto row=model.forward(ids.subspan(i,1),state,true,&stopped);logits.push_back(row_hash(row));
        }
        model.diagnostic_drain();bundle.manifest["expected"].push_back({{"keep",keep},{"state",state_digest(state)},
            {"trace_session_id",state.trace_session_id},{"row_logits_sha256",logits}});
        capture_memory(report,"reference_prefix_complete",keep);
    }
    bundle.finish();capture_memory(report,"payload_closed_and_hashed");
    report["kind"]="target_recovery_capture_v1";report["performance_measurement"]=false;
    report["capture"]={{"manifest",(bundle.path/"manifest.json").string()},{"sha256",hash_file(bundle.path/"manifest.json")},
        {"payload_bytes",bundle.bytes},{"independent_full_target_prefixes",{1,2,3,4}},{"fixture_limit_bytes",FixtureLimit}};
    report["after"]=model.stats();
}
struct FixtureReader {
    Json manifest;File file;uint64_t cursor=0;
    static uint32_t integer(const Json& value,uint32_t maximum) {
        check(value.is_number_unsigned() && value.get<uint64_t>()<=maximum,"invalid fixture integer");
        return value.get<uint32_t>();
    }
    explicit FixtureReader(const std::filesystem::path& path,const char* origin="full-target-forward"):file(path/"payload.bin") {
        check(std::filesystem::file_size(path/"manifest.json")<MiB,"fixture manifest too large");manifest=read_json(path/"manifest.json");
        check(manifest.at("kind")=="target_recovery_fixture_v1" && manifest.at("complete")==true &&
            manifest.at("reference_origin")==origin &&
            manifest.at("limit_bytes")==FixtureLimit && manifest.at("artifact_revision")==artifact_revision(Artifact::Mixed) &&
            manifest.at("layers")==Layers && manifest.at("context")==8192 &&
            manifest.at("arithmetic")=="original-gdn_scan-conv_update-v1" &&
            manifest.at("payload_bytes")==file.size() && file.size()+MiB<=FixtureLimit &&
            manifest.at("payload_sha256")==stream_hash(path/"payload.bin"),"corrupt or incompatible recovery fixture");
        // Validate every tensor range and shape before allocating or writing state.
        for(auto name:{"before","verified"}) {
            const auto& s=manifest.at(name);check(s.at("layers").size()==Layers && s.at("valid")==true,"invalid fixture state");
            const uint32_t tokens=integer(s.at("tokens"),8192);
            check(s.at("history").size()==2 && s.at("trace_session_id").is_number_unsigned(),"invalid fixture history");
            for(const auto& id:s.at("history"))integer(id,Vocab-1);
            for(int l=0;l<Layers;++l) {
                const auto& row=s.at("layers").at(l);check(row.at("position")==tokens && row.at("buffers").size()==6,"fixture positions differ");
                for(int f=0;f<6;++f) validate(row.at("buffers").at(f),fixture_geometry(l,f));
            }
        }
        const auto& j=manifest.at("journal");const uint32_t offset=integer(j.at("offset"),8188);
        check(j.at("ids").size()==4 && offset<=8188 && manifest["before"]["tokens"]==offset &&
            manifest["verified"]["tokens"]==offset+4 && j.at("entries").size()==36,"fixture journal coverage differs");
        for(const auto& id:j.at("ids"))integer(id,Vocab-1);
        int row=0;
        for(int l=0;l<Layers;++l) if((l+1)%4) {
            const auto& e=j.at("entries").at(row++);const uint32_t ad=integer(e.at("ad"),2),dd=integer(e.at("dd"),2);
            check(e.at("layer")==l && ad<=2 && dd<=2,"invalid journal layer or dtype");
            for(auto k:{"convolution","normalized"})validate(e.at(k),4*10240*4);
            for(auto k:{"a","b"})validate(e.at(k),4*48*4);
            validate(e.at("alog"),48*(ad==1?4:2));validate(e.at("dt"),48*(dd==1?4:2));
        }
        validate(j.at("ple"),4*Hyper*4);check(cursor==file.size(),"fixture has unindexed payload");
        check(manifest.at("expected").size()==4,"missing reference prefixes");
        for(int i=0;i<4;++i)check(manifest["expected"][i]["keep"]==i+1 &&
            manifest["expected"][i]["state"]["tokens"]==offset+i+1,"invalid reference prefix");
    }
    void validate(const Json& r,uint64_t bytes) {
        if(!bytes){check(r.is_null(),"unexpected fixture tensor");return;}
        check(r.at("offset").is_number_unsigned() && r.at("bytes").is_number_unsigned() &&
            r.at("offset")==cursor && r.at("bytes")==bytes && cursor+bytes<=file.size() &&
            r.at("sha256").is_string() && r.at("sha256").get<std::string>().size()==64,"invalid fixture tensor range");cursor+=bytes;
    }
    void load(const Json& record,const Buf& b) {
        check(b && record.at("bytes")==b->bytes,"fixture destination differs");
        file.read(record.at("offset"),{b->data,size_t(b->bytes)});
        check(record.at("sha256")==hash({b->data,size_t(b->bytes)}),"fixture tensor hash differs");
    }
    void state(Metal& gpu,const char* name,State& state) {
        const auto& source=manifest.at(name);state.valid=false;state.artifact=Artifact::Mixed;
        for(int l=0;l<Layers;++l) {
            for(int f=0;f<6;++f) if(auto n=fixture_geometry(l,f)) {
                auto& b=state.layers[l].*fields[f];if(!b)b=gpu.allocate(n,AllocationClass::State);
                load(source.at("layers").at(l).at("buffers").at(f),b);
            }
            state.layers[l].position=source.at("layers").at(l).at("position");
        }
        state.tokens=source.at("tokens");state.history=source.at("history").get<std::array<int,2>>();
        state.trace_session_id=source.at("trace_session_id");state.valid=true;
        check(state_digest(state)==source.at("digest"),"fixture state digest differs");
    }
    void journal(Metal& gpu,mtp_recovery::Journal& journal) {
        const auto& source=manifest.at("journal");const auto ids=source.at("ids").get<std::vector<int>>();
        journal.begin(ids,source.at("offset"));
        for(const auto& row:source.at("entries")) {
            auto& e=journal.entries.at(row.at("layer").get<size_t>());e.ad=row.at("ad");e.dd=row.at("dd");
            load(row.at("convolution"),e.convolution);load(row.at("normalized"),e.normalized);load(row.at("a"),e.a);load(row.at("b"),e.b);
            e.alog=gpu.allocate(row.at("alog").at("bytes"),AllocationClass::Resident);load(row.at("alog"),e.alog);
            e.dt=gpu.allocate(row.at("dt").at("bytes"),AllocationClass::Resident);load(row.at("dt"),e.dt);e.seen=true;
        }
        load(source.at("ple"),journal.ple);journal.ple_seen=true;journal.finish();
    }
};
Json replay_recovery_fixture(const std::filesystem::path& path,const std::string& producer_sha256,
        const char* origin="full-target-forward") {
    check(std::getenv("MTL_DEBUG_LAYER") && std::getenv("MTL_SHADER_VALIDATION"),"recovery fixture requires Metal validation");
    const auto before=process_memory(),host_before=host_conditions();FixtureReader reader(path,origin);
    check(reader.manifest.at("source").at("producer_binary_sha256")==producer_sha256,"fixture producer differs");
    Metal gpu;gpu.budget(FixtureLimit);
    KernelConfig config;gpu.configure(config);gpu.prepare_pipelines();Json cases=Json::array(),accounting;
    uint64_t peak=0;
    {
        State state;reader.state(gpu,"before",state);CheckpointCopy checkpoint(state);checkpoint.save(state,4);
        mtp_recovery::Journal journal(gpu);reader.journal(gpu,journal);
        auto corrupt=reader.manifest["journal"]["entries"][0]["a"];corrupt["sha256"]=std::string(64,'0');bool corrupted=false;
        try{reader.load(corrupt,journal.entries[0].a);}catch(const std::exception&){corrupted=true;}
        check(corrupted,"corrupt tensor hash accepted");
        auto reset=[&]{gpu.finish();reader.state(gpu,"verified",state);};
        auto run=[&](uint32_t keep,const std::atomic<bool>* cancel=nullptr){
            journal.validate(state,keep);checkpoint.check_prefix(state,keep);mtp_recovery::cancelled(cancel);
            if(keep<4){checkpoint.restore_prefix(state,keep);journal.apply(gpu,state,keep,cancel);checkpoint.commit_prefix(state,std::span(journal.ids).first(keep));}
        };
        for(int repetition=0;repetition<2;++repetition) for(uint32_t keep=1;keep<=4;++keep) {
            reset();run(keep);const auto& expected=reader.manifest.at("expected").at(keep-1);
            check(state_digest(state)==expected.at("state") && state.trace_session_id==expected.at("trace_session_id"),"recovery differs from full-target prefix state");
            cases.push_back({{"keep",keep},{"repetition",repetition},{"every_buffer_exact",true},{"state",state_digest(state)}});
            peak=std::max(peak,process_memory().at("physical_footprint_peak_bytes").get<uint64_t>());check(peak<=FixtureLimit,"replay process exceeds 2GiB");
        }
        // Bad late geometry/coverage must leave earlier state and all metadata intact.
        for(int test=0;test<3;++test) {
            reset();auto old=state.layers[47].index;const bool seen=journal.entries[46].seen;
            if(test==0)state.layers[47].index=Buffer::host(16);if(test==1)journal.entries[46].seen=false;
            const auto initial=state_digest(state);bool failed=false;
            try{run(test==2?0:2);}catch(const std::exception&){failed=true;}
            check(failed && state_digest(state)==initial,"invalid recovery wrote state");state.layers[47].index=old;journal.entries[46].seen=seen;
        }
        reset();const auto initial=state_digest(state);std::atomic<bool> cancel=true;bool failed=false;
        try{run(2,&cancel);}catch(const std::exception&){failed=true;}
        check(failed && state_digest(state)==initial,"pre-cancelled recovery wrote state");
        reset();journal.test_fail_after_layer=5;failed=false;
        try{run(2);}catch(const std::exception&){failed=true;}
        check(failed && !state.valid && gpu.statistics().at("live_command_groups")==0,"failed recovery did not invalidate and drain");journal.test_fail_after_layer=-1;
        // Delay actual completion callbacks, then exercise cancellation with work in flight.
        for(bool interrupt:{false,true}) {
            reset();cancel=false;mtp_scratch::held_callbacks=0;mtp_scratch::hold_completions=true;
            std::thread releaser([&]{
                const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(2);
                while(!mtp_scratch::held_callbacks && std::chrono::steady_clock::now()<end)std::this_thread::yield();
                std::this_thread::sleep_for(std::chrono::milliseconds(30));cancel=interrupt;mtp_scratch::hold_completions=false;
            });
            struct Join {std::thread& t;~Join(){mtp_scratch::hold_completions=false;if(t.joinable())t.join();}} join{releaser};
            failed=false;try{run(3,&cancel);}catch(const std::exception&){failed=true;}releaser.join();
            check(mtp_scratch::held_callbacks>0 && failed==interrupt && gpu.statistics().at("live_command_groups")==0,"delayed completion handling differs");
            if(interrupt)check(!state.valid,"cancelled partial state remained valid");
            else check(state_digest(state)==reader.manifest["expected"][2]["state"],"delayed recovery state differs");
        }
        accounting=journal.stats();check(journal.peak_groups<=2,"too many recovery groups");
    }
    gpu.finish();check(gpu.allocated()==0,"recovery buffers retained after cleanup");const auto after=process_memory();
    check(after.at("physical_footprint_peak_bytes").get<uint64_t>()<=FixtureLimit,"replay process peak exceeds 2GiB");
    return {{"kind","target_recovery_replay_v1"},{"complete",true},{"performance_measurement",false},
        {"reference_origin",origin},{"host_before",host_before},{"host_after",host_conditions()},
        {"source_manifest_sha256",hash_file(path/"manifest.json")},{"cases",cases},{"journal",accounting},
        {"invalid_geometry_atomic",true},{"missing_coverage_atomic",true},{"invalid_prefix_atomic",true},
        {"corrupt_tensor_rejected",true},
        {"cancellation_drained",true},{"failure_drained",true},{"delayed_completion_safe",true},{"all_buffers_released",true},
        {"full_model_loaded",false},{"before",before},{"after",after},{"process_limit_bytes",FixtureLimit}};
}

// A low-memory operator/transaction check, explicitly synthetic. This is useful
// before real capture is admitted, and cannot satisfy the real-model gate.
Json recovery_self_test(const std::filesystem::path& fixture={},const std::string& producer_sha256={},uint32_t width=4) {
    check(std::getenv("MTL_DEBUG_LAYER") && std::getenv("MTL_SHADER_VALIDATION"),"recovery self-test needs Metal validation");
    for(const auto& value:Json::array({uint64_t(1)<<32,-1,1.5,true,"1"})) {
        bool rejected=false;try{FixtureReader::integer(value,8192);}catch(const std::exception&){rejected=true;}
        check(rejected,"invalid fixture integer accepted or narrowed");
    }
    check((width==2 || width==4) && (fixture.empty() || width==4),"invalid recovery fixture width");
    const auto before=process_memory(),host_before=host_conditions();
    Metal gpu;gpu.budget(FixtureLimit);gpu.prepare_pipelines();Json cases=Json::array();uint64_t peak=0;
    {
        State state;state.artifact=Artifact::Mixed;
        for(int l=0;l<Layers;++l)for(int f=0;f<6;++f)if(auto n=fixture_geometry(l,f))state.layers[l].*fields[f]=gpu.zeros(n/4,AllocationClass::State);
        mtp_recovery::Journal journal(gpu);const std::array<int,4> ids={17,248046,23,248044};
        for(uint32_t offset:{3u,8192-width}) {
            state.tokens=offset;state.history={248044,248046};state.trace_session_id=91;state.valid=true;
            for(auto& layer:state.layers)layer.position=offset;
            std::unique_ptr<RecoveryBundle> bundle;
            if(offset==3 && !fixture.empty()) {
                bundle=std::make_unique<RecoveryBundle>(fixture);bundle->manifest["reference_origin"]="synthetic-operators";
                bundle->manifest["source"]={{"producer_binary_sha256",producer_sha256}};bundle->state("before",state);
            }
            CheckpointCopy checkpoint(state);checkpoint.save(state,width);journal.begin(std::span(ids).first(width),offset);
            auto fill=[](const Buf& b,int seed){for(size_t i=0;i<b->floats().size();++i)b->floats()[i]=float(int((i+seed)%19)-9)/128;};
            for(int l=0;l<Layers;++l)if((l+1)%4) {
                auto& e=journal.entries[l];auto c=gpu.zeros(width*10240),n=gpu.zeros(width*10240);
                auto a=gpu.zeros(width*48),b=gpu.zeros(width*48);fill(c,l);fill(n,l+1);fill(a,l+2);fill(b,l+3);
                auto alog=gpu.zeros(48,AllocationClass::Resident),dt=gpu.zeros(48,AllocationClass::Resident);
                journal.capture(gpu,l,c,n,a,b,alog,dt,1,1);gpu.finish();
                check(e.seen,"width journal input was not captured");
            }
            auto ple_input=gpu.zeros(width*Hyper);fill(ple_input,7);journal.capture_ple(gpu,ple_input);
            gpu.finish();ple_input.reset();journal.finish();Json expected=Json::array();
            // Reference processes one row at a time through the existing public
            // operators. It never calls either candidate restore_prefix/apply.
            for(uint32_t keep=1;keep<=width;++keep) {
                checkpoint.restore(state);
                for(uint32_t row=0;row<keep;++row) {
                    auto p=gpu.slice(journal.ple,uint64_t(row)*Hyper*4,Hyper*4);
                    auto ple=gpu.allocate(9*Hyper*4,AllocationClass::State);
                    gpu.dispatch("conv_update",{{p},{state.layers[1].ple_conv},{ple}},{Hyper,1,9},9*Hyper);state.layers[1].ple_conv=ple;
                    for(int l=0;l<Layers;++l)if((l+1)%4) {
                        auto& s=state.layers[l];const auto& e=journal.entries[l];
                        auto c=gpu.slice(e.convolution,uint64_t(row)*10240*4,10240*4);
                        auto n=gpu.slice(e.normalized,uint64_t(row)*10240*4,10240*4);
                        auto a=gpu.slice(e.a,row*48*4,48*4),b=gpu.slice(e.b,row*48*4,48*4);
                        auto conv=gpu.allocate(3*10240*4,AllocationClass::State);
                        gpu.dispatch("conv_update",{{c},{s.conv},{conv}},{10240,1,3},3*10240);s.conv=conv;
                        auto y=gpu.gdn_scan(n,a,b,e.alog,e.dt,s.recurrence,1,e.ad,e.dd);gpu.finish();
                    } else for(int f=2;f<=4;++f) {
                        auto& b=state.layers[l].*fields[f];const uint64_t stride=(f==4?128:512)*4;
                        std::memset(b->data+(offset+row)*stride,int(31+row+l),stride);
                    }
                    state.history={state.history[1],ids[row]};
                }
                gpu.finish();for(auto& layer:state.layers)layer.position=offset+keep;state.tokens=offset+keep;
                expected.push_back(state_digest(state));
                if(bundle)bundle->manifest["expected"].push_back({{"keep",keep},{"state",state_digest(state)},
                    {"trace_session_id",state.trace_session_id},{"row_logits_sha256",Json::array()}});
            }
            if(bundle){bundle->state("verified",state);bundle->journal(journal);bundle->finish();}
            auto post=snapshot_state(gpu,state);
            for(uint32_t keep=1;keep<=width;++keep) {
                restore_state(gpu,post,state);journal.validate(state,keep);checkpoint.check_prefix(state,keep,width);
                if(keep<width){checkpoint.restore_prefix(state,keep,width);journal.apply(gpu,state,keep);checkpoint.commit_prefix(state,std::span(ids).first(keep));}
                check(state_digest(state)==expected.at(keep-1),"synthetic state recovery differs from row-at-a-time reference");
                cases.push_back({{"width",width},{"offset",offset},{"keep",keep},{"all_persistent_buffers_exact",true}});
                peak=std::max(peak,process_memory().at("physical_footprint_peak_bytes").get<uint64_t>());check(peak<=FixtureLimit,"self-test exceeds 2GiB");
            }
        }
    }
    gpu.finish();check(gpu.allocated()==0,"synthetic recovery users leaked");
    Json serialized=nullptr;
    if(!fixture.empty()) {
        bool rejected=false;try{FixtureReader wrong_origin(fixture);}catch(const std::exception&){rejected=true;}
        check(rejected,"synthetic fixture accepted as real target evidence");
        serialized=replay_recovery_fixture(fixture,producer_sha256,"synthetic-operators");
    }
    return {{"kind","target_recovery_synthetic_v1"},{"complete",true},{"real_model_evidence",false},
        {"before",before},{"after",process_memory()},{"host_before",host_before},{"host_after",host_conditions()},
        {"invalid_integers_rejected",true},
        {"serialized_replay",serialized},{"performance_measurement",false},{"cases",cases},
        {"peak_physical_bytes",process_memory().at("physical_footprint_peak_bytes")},{"all_buffers_released",true}};
}

struct ContinuationInput {
    std::vector<int> prompt,expected,eos;
    uint32_t count=0;
    bool validation=false,serial=false,fast=false;
    ContinuationInput(const Json& input,const std::string& mode) {
        check(mode=="serial" || mode=="timing" || mode=="fast-timing" || mode=="validate" || mode=="fast-validate","invalid continuation mode");
        validation=mode=="validate" || mode=="fast-validate";serial=mode=="serial";fast=mode.starts_with("fast-");
        prompt=input.at("prompt_ids").get<std::vector<int>>();
        expected=input.value("expected_next_ids",std::vector<int>{});eos=input.at("eos_ids").get<std::vector<int>>();
        check(input.at("max_tokens").is_number_unsigned() || input.at("max_tokens").is_number_integer(),"invalid output bound");
        const int64_t requested=input.at("max_tokens");check(requested>=1 && requested<=256,"continuation exceeds output bound");
        count=validation?std::min(8u,uint32_t(requested)):uint32_t(requested);
        check(prompt.size()>=2 && prompt.size()<=512 && prompt.size()+count<=8192,"continuation exceeds prompt/context bound");
        check(!eos.empty() && eos.size()<=4,"invalid EOS set");
        for(const auto* ids:{&prompt,&expected,&eos}) for(int id:*ids) check(id>=0 && id<Vocab,"invalid continuation token");
        check(expected.empty() || expected.size()>=count,"incomplete expected sequence");
        check(!validation || expected.size()>=count,"validation requires independent expected sequence");
        check(input.value("draft_slots",32u)==32,"continuation draft capacity must remain 32");
    }
    bool is_eos(int id) const {return std::find(eos.begin(),eos.end(),id)!=eos.end();}
};
Json continuation_self_test() {
    Json input={{"prompt_ids",{10,11}},{"eos_ids",{248046,248044}},{"max_tokens",128},{"draft_slots",32}};
    const ContinuationInput good(input,"fast-timing");check(good.count==128 && good.fast && good.is_eos(248046) && !good.is_eos(10),"continuation parsing failed");
    for(auto key:{"max_tokens","prompt_ids","eos_ids","draft_slots"}) {
        auto bad=input;
        if(std::string_view(key)=="max_tokens") bad[key]=-1;
        else if(std::string_view(key)=="draft_slots") bad[key]=128;
        else bad[key]=Json::array();
        bool rejected=false;try{ContinuationInput ignored(bad,"fast-timing");}catch(const std::exception&){rejected=true;}
        check(rejected,"invalid continuation input accepted");
    }
    bool rejected=false;try{ContinuationInput ignored(input,"fast-validate");}catch(const std::exception&){rejected=true;}
    check(rejected,"unreferenced rejection validation accepted");
    return {{"passed",true},{"gpu_used",false},{"bounds_checked",true},{"unreferenced_validation_rejected",true}};
}
// Offline known-continuation ceiling. No proposal quality or generation claim.
Json horizon_kernel_test() {
    check(std::getenv("MTL_DEBUG_LAYER") && std::getenv("MTL_SHADER_VALIDATION"),"horizon kernel test requires Metal validation");
    Json report={{"kind","verifier_horizon_kernel_test_v1"},{"performance_measurement",false},
        {"before",process_memory()},{"host_before",host_conditions()},{"cases",Json::array()}};
    {
        Metal gpu;gpu.budget(128*MiB);KernelConfig config;gpu.configure(config);
        for(auto shape:{std::pair{2560u,6144u},std::pair{6144u,2560u}}) {
            const auto [K,N]=shape;const uint64_t bytes=uint64_t(8)*N*4;
            auto w=gpu.allocate(uint64_t(K)*N),s=gpu.allocate(uint64_t(K/64)*N*2),b=gpu.allocate(s->bytes);
            auto x=gpu.allocate(uint64_t(8)*K*4),expected=gpu.allocate(bytes+128),actual=gpu.allocate(bytes+128);
            for(uint64_t i=0;i<w->bytes;++i) w->data[i]=std::byte((i*17+i/19)%256);
            for(uint64_t i=0;i<s->bytes/2;++i) {
                reinterpret_cast<uint16_t*>(s->data)[i]=uint16_t(0x3c00+i%63);
                reinterpret_cast<uint16_t*>(b->data)[i]=uint16_t(0xbf00+i%127);
            }
            for(uint64_t i=0;i<x->bytes/4;++i) x->floats()[i]=round_bf16(float(int((i*23+i/K)%257)-128)/64);
            const auto input_hash=hash(std::span<const std::byte>(x->data,x->bytes));
            Linear l;l.weight={w};l.scales={s};l.biases={b};l.input=K;l.output=N;l.bits=8;l.group=64;l.quantized=true;
            for(uint32_t fp32:{0u,1u}) {
                std::memset(expected->data,0xa5,expected->bytes);std::memset(actual->data,0xa5,actual->bytes);
                gpu.linear_into(l,x,8,{expected,64},fp32);
                gpu.dispatch("q8_horizon_t8_w8",{l.weight,l.scales,l.biases,{x},{actual,64}},{K,N,8,64,fp32},N*32);
                gpu.finish();check(std::memcmp(expected->data,actual->data,actual->bytes)==0,"eight-row Q8 changed arithmetic");
                for(uint64_t i=0;i<64;++i) check(actual->data[i]==std::byte(0xa5) && actual->data[64+bytes+i]==std::byte(0xa5),"eight-row Q8 overwrote guards");
                check(hash(std::span<const std::byte>(x->data,x->bytes))==input_hash,"eight-row Q8 modified input");
                report["cases"].push_back({{"K",K},{"N",N},{"rows",8},{"float_output",fp32==1},
                    {"exact_reference",true},{"guards_intact",true},{"output_sha256",hash(std::span<const std::byte>(actual->data+64,bytes))}});
            }
        }
        report["peak_gpu_bytes"]=gpu.peak();
    }
    report["after"]=process_memory();report["host_after"]=host_conditions();report["complete"]=true;return report;
}

void joint(const std::filesystem::path& model_path,const std::filesystem::path& prepared,const Json& input,
           const std::filesystem::path& output,const std::string& mode,Json& report) {
    check(mode=="fast-validate" || mode=="fast-timing","invalid horizon mode");
    const bool validation=mode=="fast-validate";
    const char* setting=std::getenv("FREELLM_VERIFIER_HORIZON");
    check(setting && (std::string_view(setting)=="1" || std::string_view(setting)=="4" ||
                      std::string_view(setting)=="8"),"explicit horizon must be 1, 4 or 8");
    const uint32_t requested_width=uint32_t(setting[0]-'0');
    const auto prompt=input.at("prompt_ids").get<std::vector<int>>();
    const auto known=input.at("continuation_ids").get<std::vector<int>>();
    const auto expected=input.at("expected_next_ids").get<std::vector<int>>();
    const auto hashes=input.at("row_logits_sha256").get<std::vector<std::string>>();
    const uint32_t count=uint32_t(known.size());
    check(prompt.size()>=2 && prompt.size()<=512 && count>=1 && count<=256 &&
          prompt.size()+count<=8192 && expected.size()==count && hashes.size()==count,"invalid horizon bounds");
    for(const auto* v:{&prompt,&known,&expected}) for(int id:*v) check(id>=0 && id<Vocab,"invalid horizon token");
    for(uint32_t i=1;i<count;++i) check(known[i]==expected[i-1],"discontinuous known continuation");
    for(const auto& h:hashes) check(h.size()==64 && h.find_first_not_of("0123456789abcdef")==std::string::npos,"invalid reference hash");
    mtp_scratch::enabled=false;mtp_direct::enabled=true;
    Options options;options.artifact=Artifact::Mixed;options.model=model_path;options.prepared=input.at("target_prepared").get<std::string>();
    options.memory=12*GiB;options.context=8192;options.expert_slots=1460;options.panel=512;options.chunk=128;
    options.residency="core-cache";options.decode_scratch="reuse";options.kernels.policy="candidate";
    options.kernels.q8_decode_rows=2;options.kernels.route_selection="simd";options.audit_routes=validation;
    phase(output,"load_target");Model model(options);auto& gpu=DraftAccess::gpu(model);auto& resident=DraftAccess::resident(model);
    const auto plan=model.memory_plan().json();const uint64_t host_bound=128*MiB+4*8ull*Vocab*4+MiB;
    const auto combined=plan.at("planned_bytes").get<uint64_t>()+host_bound+MtpDraft::budget_bytes(32,8192)+mtp_scratch::ReserveBytes+mtp_recovery::ReserveBytes;
    check(plan["expert_slots"]==1460 && plan["limit_bytes"]==12*GiB && combined<=12*GiB,"horizon exceeds joint admission");
    report["admission"]={{"target",plan},{"draft_slots",32},{"draft_bytes",MtpDraft::budget_bytes(32,8192)},
        {"host_checkpoint_logits_bytes",host_bound},{"combined_bytes",combined},{"checkpoint_rows",8},
        {"expert_scratch_reserve_bytes",mtp_scratch::ReserveBytes},{"target_recovery_reserve_bytes",mtp_recovery::ReserveBytes}};
    auto state=model.make_state();
    MtpDraft draft(gpu,DraftAccess::reads(model),prepared,resident.linear("model.embed_tokens"),resident.linear("lm_head"),32,8192);
    auto draft_state=draft.make_state();DraftCheckpoint draft_checkpoint(gpu);
    mtp_target_hidden=gpu.allocate(MtpDraft::Panel*Hyper*4,AllocationClass::Workspace);
    struct ClearCapture {~ClearCapture(){mtp_target_hidden.reset();}} clear_capture;
    model.prepare_pipelines();int next=-1;const auto prime_start=monotonic_ns();
    // Same target/draft priming as the real-MTP control. Retain draft allocations
    // throughout measurement, but exclude its execution to measure a ceiling.
    for(uint32_t at=0;at<prompt.size();) {
        const uint32_t n=std::min<uint32_t>(128,prompt.size()-at);
        model.phase("prefill");phase(output,"prime_target",at);
        auto prime=model.forward(std::span(prompt).subspan(at,n),state,true,&stopped);model.diagnostic_drain();
        check(mtp_target_rows==n && mtp_target_offset==at,"horizon hidden alignment differs");
        if(at+n==prompt.size()) {
            next=argmax(prime);report["prime_logits_sha256"]=row_hash(prime);
            check(next==known.front() && report["prime_logits_sha256"]==input.at("prime_logits_sha256"),"horizon prime changed");
        }
        phase(output,"prime_draft",at);
        const uint32_t rows=std::min<uint32_t>(n,prompt.size()-1-at);
        for(uint32_t row=0;row<rows;) {
            const uint32_t t=std::min(16u,rows-row);auto h=detached(gpu,mtp_target_hidden,row,t);
            draft.forward(std::span(prompt).subspan(at+row+1,t),h,draft_state,false,&stopped);row+=t;
        }
        at+=n;
    }
    check(draft_state.position+1==state.tokens && state.tokens==prompt.size(),"horizon priming state differs");
    report["priming_wall_ns"]=monotonic_ns()-prime_start;
    mtp_recovery::Journal journal(gpu);CheckpointCopy checkpoint(state);
    report["before"]=model.stats();report["draft_before"]=draft.stats();report["initial_draft_state"]=draft_digest(draft_state);
    report["direct_output_before"]=mtp_direct::counters();report["target_recovery_journal"]=journal.stats();
    report["requested_width"]=requested_width;report["prompt_tokens"]=prompt.size();report["requested_tokens"]=count;
    report["perfect_proposals_only"]=true;report["normal_request_latency_qualified"]=false;
    report["excluded_costs"]={"proposal_generation","rejection_recovery","draft_state_catchup"};
    Json boundaries=Json::array(),row_hashes=Json::array();uint64_t total=0,verify_total=0;
    uint32_t consumed=0;report["cycles"]=Json::array();const auto request_start=monotonic_ns();
    while(consumed<count) {
        phase(output,"verify_known_tokens",consumed);check(!stopped,"horizon cancelled");
        const uint32_t width=count-consumed>=requested_width?requested_width:1;
        auto ids=std::span(known).subspan(consumed,width);check(ids.front()==next,"horizon next input differs");
        const auto memory_before=process_memory();const auto start=monotonic_ns();
        uint64_t checkpoint_ns=0;
        if(width>1) {checkpoint.save(state,width);checkpoint_ns=monotonic_ns()-start;}
        model.phase("decode");const auto vstart=monotonic_ns();auto logits=model.forward(ids,state,true,&stopped);
        model.diagnostic_drain();const auto end=monotonic_ns();const auto verify_ns=end-vstart;
        check(logits.size()==uint64_t(width)*Vocab,"horizon logits width differs");
        for(uint32_t i=0;i<width;++i) {
            auto row=std::span(logits).subspan(i*Vocab,Vocab);next=argmax(row);
            const auto h=row_hash(row);check(next==expected[consumed+i] && h==hashes[consumed+i],"horizon changed full logits");
            row_hashes.push_back(h);
        }
        total+=end-start;verify_total+=verify_ns;
        report["cycles"].push_back({{"offset",state.tokens-width},{"width",width},{"committed_tokens",width},
            {"checkpoint_save_ns",checkpoint_ns},{"verify_ns",verify_ns},{"wall_ns",end-start},
            {"memory_before",memory_before},{"memory_after",process_memory()}});
        consumed+=width;
        if(validation) boundaries.push_back({{"tokens",consumed},{"target_state",state_digest(state)}});
        if(consumed%16<width || consumed==count) save_report(output,report);
    }
    report.update(Json{{"generated_tokens",consumed},{"committed_token_ids",known},{"row_logits_sha256",row_hashes},
        {"next_id",next},{"decode_wall_ns",total},{"verify_wall_ns",verify_total},
        {"verified_tokens_per_second",double(consumed)*1e9/total},{"decode_including_reporting_ns",monotonic_ns()-request_start},
        {"boundaries",boundaries},{"after",model.stats()},{"draft_after",draft.stats()},
        {"final_target_state",state_digest(state)},{"final_draft_state",draft_digest(draft_state)},
        {"host_checkpoint_allocated_bytes",checkpoint.allocated},{"direct_output_after",mtp_direct::counters()}});
    check(report["initial_draft_state"]==report["final_draft_state"],"ceiling executed draft work");
    if(input.contains("final_target_state")) check(report["final_target_state"]==input["final_target_state"],"horizon final reference state differs");
    check(model.memory_plan().json()==plan,"horizon memory admission changed");
}


}
int main(int argc,char** argv) {
    Json report={{"kind","native_verifier_horizon_v1"},{"complete",false},{"production_promoted",false}};bool write=false;
    try {
        if(argc==2 && std::string_view(argv[1])=="--checkpoint-self-test") {std::cout<<checkpoint_self_test().dump()<<'\n';return 0;}
        if(argc==2 && std::string_view(argv[1])=="--continuation-self-test") {std::cout<<continuation_self_test().dump()<<'\n';return 0;}
        if(argc==4 && std::string_view(argv[1])=="--expert-scratch-test") {
            check(!std::filesystem::exists(argv[3]),"scratch test output exists");
            save_report(argv[3],expert_scratch_test(argv[2]));return 0;
        }
        if(argc==5 && std::string_view(argv[1])=="--ngram-init-test") {
            check(!std::filesystem::exists(argv[4]),"ngram fixture output exists");
            save_report(argv[4],ngram_init_test(argv[2],argv[3]));return 0;
        }
        if(argc==4 && std::string_view(argv[1])=="--direct-output-test") {
            check(!std::filesystem::exists(argv[3]),"direct-output fixture exists");
            save_report(argv[3],direct_output_test(argv[2]));return 0;
        }
        if(argc==3 && std::string_view(argv[1])=="--width-recovery-self-test") {
            check(!std::filesystem::exists(argv[2]),"width self-test output exists");
            Json tests=Json::array();
            for(uint32_t width:{2u,4u}) tests.push_back(recovery_self_test({},hash_file(argv[0]),width));
            save_report(argv[2],{{"kind","mtp_width_recovery_synthetic_v1"},{"complete",true},
                {"real_model_evidence",false},{"performance_measurement",false},{"tests",tests}});return 0;
        }
        if((argc==3 || argc==4) && std::string_view(argv[1])=="--recovery-self-test") {
            check(!std::filesystem::exists(argv[2]),"self-test output exists");
            save_report(argv[2],recovery_self_test(argc==4?argv[3]:"",hash_file(argv[0])));return 0;
        }
        if(argc==4 && std::string_view(argv[1])=="--replay-recovery") {
            check(!std::filesystem::exists(argv[3]),"replay output exists");
            save_report(argv[3],replay_recovery_fixture(argv[2],hash_file(argv[0])));return 0;
        }
        if(argc==3 && std::string_view(argv[1])=="--horizon-kernel-test") {
            check(!std::filesystem::exists(argv[2]),"horizon kernel report exists");
            save_report(argv[2],horizon_kernel_test());return 0;
        }
        check(argc==6,"usage: probe MODEL MTP_PREPARED INPUT_JSON REPORT fixture|serial|validate|timing");
        const std::filesystem::path output=argv[4];check(!std::filesystem::exists(output),"MTP report exists");write=true;
        const std::string mode=argv[5];check(mode=="fixture" || mode=="serial" || mode=="validate" || mode=="timing" ||
            mode=="fast-validate" || mode=="fast-timing","invalid MTP mode");
        const bool validation=mode=="fixture" || mode=="validate" || mode=="fast-validate";
        check(validation?std::getenv("MTL_DEBUG_LAYER") && std::getenv("MTL_SHADER_VALIDATION"):
            !std::getenv("MTL_DEBUG_LAYER") && !std::getenv("MTL_SHADER_VALIDATION"),"wrong MTP validation environment");
        std::signal(SIGINT,interrupt);std::signal(SIGTERM,interrupt);
        report["producer_binary_sha256"]=hash_file(argv[0]);
        auto input=read_json(argv[3]);report.update(Json{{"mode",mode},{"validation",validation},{"input_sha256",hash_file(argv[3])},
            {"draft_manifest_sha256",hash_file(std::filesystem::path(argv[2])/"manifest.json")},
            {"before_load",process_memory()},{"host_before",host_conditions()}});
        if(mode=="fixture") fixture(argv[1],argv[2],input,output,report);else joint(argv[1],argv[2],input,output,mode,report);
        report["after_destroy"]=process_memory();report["host_after"]=host_conditions();report["complete"]=true;
        save_report(output,report);phase(output,"complete");return 0;
    } catch(const std::exception& error) {
        report["error"]=error.what();report["after_error"]=process_memory();
        if(write) save_report(argv[4],report);std::cerr<<error.what()<<'\n';return 2;
    }
}
