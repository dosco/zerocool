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
    }
    gpu.finish();report.update(Json{{"fixture_tensors",captured},{"greedy_ids",replay},{"full_logits_sha256",full_logit_hash},
        {"serial_block_exact",true},{"causal_prefix_exact",true},{"rollback_exact",true},{"proper_prefix_replay_exact",true},
        {"cancellation_before_update",true},{"overflow_rejected_before_write",true},{"bad_restore_rejected_before_write",true},
        {"draft",draft.stats()},{"metal",gpu.statistics()},{"after",process_memory()}});
}
void joint(const std::filesystem::path& model_path,const std::filesystem::path& prepared,const Json& input,
           const std::filesystem::path& output,const std::string& mode,Json& report) {
    const auto prompt=input.at("prompt_ids").get<std::vector<int>>(),expected=input.at("expected_next_ids").get<std::vector<int>>();
    check(prompt.size()==72 && expected.size()==16,"changed short screen workload");
    const auto draft_slots=input.value("draft_slots",128u);check(draft_slots==32 || draft_slots==128,"invalid fixed draft capacity");
    const bool serial=mode=="serial",validation=mode=="validate";
    Options options;options.artifact=Artifact::Mixed;options.model=model_path;options.prepared=input.at("target_prepared").get<std::string>();
    options.memory=12*GiB;options.context=8192;options.expert_slots=1460;options.panel=512;options.chunk=128;
    options.residency="core-cache";options.decode_scratch="reuse";options.kernels.policy="candidate";
    options.kernels.q8_decode_rows=2;options.kernels.route_selection="simd";options.audit_routes=validation;
    phase(output,"load_target");Model model(options);auto& gpu=DraftAccess::gpu(model);auto& resident=DraftAccess::resident(model);
    const auto plan=model.memory_plan().json();const uint64_t host_bound=128*MiB+4*4ull*Vocab*4+MiB;
    const auto combined=plan.at("planned_bytes").get<uint64_t>()+host_bound+MtpDraft::budget_bytes(draft_slots,8192);
    check(plan["expert_slots"]==1460 && plan["limit_bytes"]==12*GiB && combined<=12*GiB,"joint MTP admission exceeds 12GiB");
    report["admission"]={{"target",plan},{"draft_slots",draft_slots},{"draft_bytes",MtpDraft::budget_bytes(draft_slots,8192)},
        {"host_checkpoint_logits_bytes",host_bound},{"combined_bytes",combined}};
    auto state=model.make_state();
    // Both timing arms reserve the same draft resources. Control has no draft
    // computation; target cache capacity and total admitted memory stay fixed.
    MtpDraft draft(gpu,DraftAccess::reads(model),prepared,resident.linear("model.embed_tokens"),resident.linear("lm_head"),draft_slots,8192);
    auto draft_state=draft.make_state();DraftCheckpoint draft_checkpoint(gpu);
    mtp_target_hidden=gpu.allocate(MtpDraft::Panel*Hyper*4,AllocationClass::Workspace);
    struct ClearCapture {~ClearCapture(){mtp_target_hidden.reset();}} clear_capture;
    model.prepare_pipelines();model.phase("prefill");phase(output,"prime_target");
    auto prime=model.forward(prompt,state,true,&stopped);model.diagnostic_drain();
    int next=argmax(prime);check(next==input.at("expected_prompt_id"),"target prime changed");
    report["prime_logits_sha256"]=row_hash(prime);prime.clear();prime.shrink_to_fit();
    check(mtp_target_rows==72 && mtp_target_offset==0,"target hidden alignment differs");
    if(validation) {
        auto real=detached(gpu,mtp_target_hidden,0,4);save_buffer(output.parent_path()/"real-hidden.f32",real);
        save_report(output.parent_path()/"real-input.json",{{"ids",std::vector<int>(prompt.begin()+1,prompt.begin()+5)},
            {"hidden_file",(output.parent_path()/"real-hidden.f32").string()}});
    }
    phase(output,"prime_draft");
    for(uint32_t at=0;at<71;) {
        const uint32_t n=std::min(16u,71-at);auto h=detached(gpu,mtp_target_hidden,at,n);
        draft.forward(std::span(prompt).subspan(at+1,n),h,draft_state,false,&stopped);at+=n;
    }
    // Admission reserved these bytes before loading. Commit the host checkpoint
    // when generation needs it, rather than leaving cold pages during prefill.
    CheckpointCopy checkpoint(state);
    auto seed=detached(gpu,mtp_target_hidden,71,1);report["before"]=model.stats();report["draft_before"]=draft.stats();
    Json boundaries=Json::array();const auto count=validation?4u:16u;uint64_t total=0;uint32_t consumed=0,proposed=0,accepted=0;
    report["cycles"]=Json::array();
    while(consumed<count) {
        phase(output,serial?"serial_token":"draft_verify",consumed);check(!stopped,"MTP probe cancelled");
        const auto memory_before=process_memory();const auto start=monotonic_ns();
        const uint32_t width=!serial && count-consumed>=4?4:1;
        std::vector<int> ids{next};uint64_t draft_ns=0,verify_ns=0,recovery_ns=0;
        if(width==4) {
            check(draft_state.position+1==state.tokens,"draft/target position mismatch");
            checkpoint.save(state,4);draft_checkpoint.save(gpu,draft_state,4);
            auto hidden=seed;const auto begin=monotonic_ns();
            for(uint32_t i=0;i<3;++i) {
                auto out=draft.forward(std::span(ids).last(1),hidden,draft_state,true,&stopped);
                ids.push_back(argmax(out.logits->floats()));hidden=out.wide;
            }
            draft_ns=monotonic_ns()-begin;proposed+=3;
        }
        // Force a real rejection in validation, after generating the proposals.
        // This is excluded from acceptance-rate and normal timing evidence.
        const bool forced=validation && consumed==0 && width==4;
        if(forced) ids[1]=(expected[0]+1)%Vocab;
        model.phase("decode");const auto vstart=monotonic_ns();auto logits=model.forward(ids,state,true,&stopped);verify_ns=monotonic_ns()-vstart;
        uint32_t good=0;
        while(good+1<width && ids[good+1]==argmax(std::span(logits).subspan(good*Vocab,Vocab))) ++good;
        const uint32_t keep=good+1;next=argmax(std::span(logits).subspan(good*Vocab,Vocab));accepted+=good;
        if(forced) check(good==0,"forced target rejection failed");
        for(uint32_t i=0;i<keep;++i) check(argmax(std::span(logits).subspan(i*Vocab,Vocab))==expected[consumed+i],"MTP verification changed target greedy sequence");
        // Preserve target rows before a rejected prefix replay overwrites capture.
        auto verified_hidden=detached(gpu,mtp_target_hidden,0,keep);
        const auto recovery_start=monotonic_ns();
        if(width==4) {
            if(keep<width) {
                model.diagnostic_drain();checkpoint.restore(state);
                for(uint32_t i=0;i<keep;++i) {
                    model.phase("decode");auto row=model.forward(std::span(ids).subspan(i,1),state,true,&stopped);
                    check(row_hash(row)==row_hash(std::span(logits).subspan(i*Vocab,Vocab)),"target rejection replay changed full logits");
                }
            }
            draft_checkpoint.restore(gpu,draft_state);
            // Replace tentative hidden inputs with actual target hidden states.
            // All of this catch-up work counts toward the request latency.
            for(uint32_t i=0;i<keep;++i) {
                auto hidden=i==0?seed:detached(gpu,verified_hidden,i-1,1);
                draft.forward(std::span(ids).subspan(i,1),hidden,draft_state,false,&stopped);
            }
        } else if(!serial) draft.forward(ids,seed,draft_state,false,&stopped);
        recovery_ns=monotonic_ns()-recovery_start;seed=detached(gpu,verified_hidden,keep-1,1);
        model.diagnostic_drain();const auto end=monotonic_ns();total+=end-start;consumed+=keep;
        report["cycles"].push_back({{"width",width},{"proposals",ids},{"accepted_proposals",good},{"committed_tokens",keep},
            {"forced_rejection",forced},{"draft_ns",draft_ns},{"verify_ns",verify_ns},{"recovery_ns",recovery_ns},{"wall_ns",end-start},
            {"memory_before",memory_before},{"memory_after",process_memory()},{"next_id",next}});
        if(validation) boundaries.push_back({{"tokens",consumed},{"target_state",state_digest(state)},
            {"target_logits_sha256",row_hash(std::span(logits).subspan(good*Vocab,Vocab))},{"draft_state",draft_digest(draft_state)}});
        save_report(output,report);
    }
    report.update(Json{{"generated_tokens",consumed},{"proposed_tokens",proposed},{"accepted_proposals",accepted},
        {"decode_wall_ns",total},{"tokens_per_second",double(consumed)*1e9/total},{"boundaries",boundaries},
        {"after",model.stats()},{"draft_after",draft.stats()},{"final_target_state",state_digest(state)},
        {"host_checkpoint_allocated_bytes",checkpoint.allocated},{"normal_request_latency_qualified",false}});
    check(model.memory_plan().json()==plan,"target memory admission changed");
}
}
int main(int argc,char** argv) {
    Json report={{"kind","native_mtp_forward_v1"},{"complete",false},{"production_promoted",false}};bool write=false;
    try {
        if(argc==2 && std::string_view(argv[1])=="--checkpoint-self-test") {std::cout<<checkpoint_self_test().dump()<<'\n';return 0;}
        check(argc==6,"usage: probe MODEL MTP_PREPARED INPUT_JSON REPORT fixture|serial|validate|timing");
        const std::filesystem::path output=argv[4];check(!std::filesystem::exists(output),"MTP report exists");write=true;
        const std::string mode=argv[5];check(mode=="fixture" || mode=="serial" || mode=="validate" || mode=="timing","invalid MTP mode");
        const bool validation=mode=="fixture" || mode=="validate";
        check(validation?std::getenv("MTL_DEBUG_LAYER") && std::getenv("MTL_SHADER_VALIDATION"):
            !std::getenv("MTL_DEBUG_LAYER") && !std::getenv("MTL_SHADER_VALIDATION"),"wrong MTP validation environment");
        std::signal(SIGINT,interrupt);std::signal(SIGTERM,interrupt);
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
