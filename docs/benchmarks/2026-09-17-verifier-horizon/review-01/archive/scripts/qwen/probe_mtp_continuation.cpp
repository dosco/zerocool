// Replaces only the developer short-screen harness. MTP/target arithmetic is shared.
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
        count=validation?8u:uint32_t(requested);
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
void joint(const std::filesystem::path& model_path,const std::filesystem::path& prepared,const Json& input,
           const std::filesystem::path& output,const std::string& mode,Json& report) {
    const ContinuationInput work(input,mode);const auto& prompt=work.prompt;
    Options options;options.artifact=Artifact::Mixed;options.model=model_path;options.prepared=input.at("target_prepared").get<std::string>();
    options.memory=12*GiB;options.context=8192;options.expert_slots=1460;options.panel=512;options.chunk=128;
    options.residency="core-cache";options.decode_scratch="reuse";options.kernels.policy="candidate";
    options.kernels.q8_decode_rows=2;options.kernels.route_selection="simd";options.audit_routes=work.validation;
    phase(output,"load_target");Model model(options);auto& gpu=DraftAccess::gpu(model);auto& resident=DraftAccess::resident(model);
    const auto plan=model.memory_plan().json();const uint64_t host_bound=128*MiB+4*4ull*Vocab*4+MiB;
    const auto combined=plan.at("planned_bytes").get<uint64_t>()+host_bound+MtpDraft::budget_bytes(32,8192);
    check(plan["expert_slots"]==1460 && plan["limit_bytes"]==12*GiB && combined<=12*GiB,"joint continuation admission exceeds 12GiB");
    report["admission"]={{"target",plan},{"draft_slots",32},{"draft_bytes",MtpDraft::budget_bytes(32,8192)},
        {"host_checkpoint_logits_bytes",host_bound},{"combined_bytes",combined}};
    auto state=model.make_state();
    MtpDraft draft(gpu,DraftAccess::reads(model),prepared,resident.linear("model.embed_tokens"),resident.linear("lm_head"),32,8192);
    auto draft_state=draft.make_state();DraftCheckpoint draft_checkpoint(gpu);
    mtp_target_hidden=gpu.allocate(MtpDraft::Panel*Hyper*4,AllocationClass::Workspace);
    struct ClearCapture {~ClearCapture(){mtp_target_hidden.reset();}} clear_capture;
    model.prepare_pipelines();int next=-1;Buf seed;const auto prime_start=monotonic_ns();
    // Feed each captured target panel through the draft before reusing it.
    // A boundary row uses the next prompt token, already known to the caller.
    for(uint32_t at=0;at<prompt.size();) {
        const uint32_t n=std::min<uint32_t>(128,prompt.size()-at);
        model.phase("prefill");phase(output,"prime_target",at);
        auto prime=model.forward(std::span(prompt).subspan(at,n),state,true,&stopped);model.diagnostic_drain();
        check(mtp_target_rows==n && mtp_target_offset==at,"target continuation hidden alignment differs");
        if(at+n==prompt.size()) {
            next=argmax(prime);report["prime_logits_sha256"]=row_hash(prime);
            if(input.contains("expected_prompt_id")) check(next==input.at("expected_prompt_id"),"target prime changed");
            seed=detached(gpu,mtp_target_hidden,n-1,1);
        }
        phase(output,"prime_draft",at);
        const uint32_t draft_rows=std::min<uint32_t>(n,prompt.size()-1-at);
        for(uint32_t row=0;row<draft_rows;) {
            const uint32_t count=std::min(16u,draft_rows-row);auto h=detached(gpu,mtp_target_hidden,row,count);
            draft.forward(std::span(prompt).subspan(at+row+1,count),h,draft_state,false,&stopped);row+=count;
        }
        at+=n;
    }
    check(seed && draft_state.position+1==state.tokens && state.tokens==prompt.size(),"continuation priming state differs");
    report["priming_wall_ns"]=monotonic_ns()-prime_start;
    CheckpointCopy checkpoint(state);
    report["before"]=model.stats();report["draft_before"]=draft.stats();
    report["prompt_tokens"]=prompt.size();report["requested_tokens"]=work.count;report["eos_ids"]=work.eos;
    Json boundaries=Json::array(),row_hashes=Json::array();std::vector<int> committed;
    uint64_t total=0;uint32_t consumed=0,proposed=0,accepted=0;bool eos=false;
    report["cycles"]=Json::array();const auto request_start=monotonic_ns();
    while(consumed<work.count && !eos) {
        phase(output,work.serial?"serial_token":"draft_verify",consumed);check(!stopped,"MTP continuation cancelled");
        const auto memory_before=process_memory();const auto start=monotonic_ns();
        const uint32_t width=!work.serial && !work.is_eos(next) && work.count-consumed>=4?4:1;
        std::vector<int> ids{next};uint64_t draft_ns=0,verify_ns=0,recovery_ns=0;
        if(width==4) {
            check(draft_state.position+1==state.tokens,"draft/target continuation position mismatch");
            checkpoint.save(state,4);draft_checkpoint.save(gpu,draft_state,4);
            auto hidden=seed;const auto begin=monotonic_ns();
            for(uint32_t i=0;i<3;++i) {
                auto out=draft.forward(std::span(ids).last(1),hidden,draft_state,true,&stopped);
                ids.push_back(argmax(out.logits->floats()));hidden=out.wide;
            }
            draft_ns=monotonic_ns()-begin;proposed+=3;
        }
        const bool forced=work.validation && consumed==0 && width==4;
        if(forced) ids[1]=(work.expected[0]+1)%Vocab;
        model.phase("decode");const auto vstart=monotonic_ns();auto logits=model.forward(ids,state,true,&stopped);verify_ns=monotonic_ns()-vstart;
        uint32_t good=0;
        while(good+1<width && ids[good+1]==argmax(std::span(logits).subspan(good*Vocab,Vocab))) ++good;
        uint32_t keep=good+1;
        for(uint32_t i=0;i<keep;++i) if(work.is_eos(ids[i])) {keep=i+1;eos=true;break;}
        good=keep-1;next=argmax(std::span(logits).subspan(good*Vocab,Vocab));accepted+=good;
        if(forced) check(good==0,"forced continuation rejection failed");
        for(uint32_t i=0;i<keep && !work.expected.empty();++i)
            check(argmax(std::span(logits).subspan(i*Vocab,Vocab))==work.expected[consumed+i],"continuation greedy sequence differs from independent reference");
        auto verified_hidden=detached(gpu,mtp_target_hidden,0,keep);const auto recovery_start=monotonic_ns();
        if(width==4) {
            if(keep<width) {
                model.diagnostic_drain();checkpoint.restore(state);
                for(uint32_t i=0;i<keep;++i) {
                    model.phase("decode");auto row=model.forward(std::span(ids).subspan(i,1),state,true,&stopped);
                    check(row_hash(row)==row_hash(std::span(logits).subspan(i*Vocab,Vocab)),"continuation rejection replay changed full logits");
                }
            }
            draft_checkpoint.restore(gpu,draft_state);
            if(work.fast) {
                auto aligned=gpu.allocate(uint64_t(keep)*Hyper*4);gpu.copy(seed,0,aligned,0,Hyper*4);
                if(keep>1) gpu.copy(verified_hidden,0,aligned,Hyper*4,uint64_t(keep-1)*Hyper*4);
                draft.catch_up(std::span(ids).first(keep),aligned,draft_state,&stopped);
            } else for(uint32_t i=0;i<keep;++i) {
                auto hidden=i==0?seed:detached(gpu,verified_hidden,i-1,1);
                draft.forward(std::span(ids).subspan(i,1),hidden,draft_state,false,&stopped);
            }
        } else if(!work.serial) {
            if(work.fast) draft.catch_up(ids,seed,draft_state,&stopped);
            else draft.forward(ids,seed,draft_state,false,&stopped);
        }
        recovery_ns=monotonic_ns()-recovery_start;seed=detached(gpu,verified_hidden,keep-1,1);
        model.diagnostic_drain();const auto end=monotonic_ns();total+=end-start;
        const uint32_t offset=state.tokens-keep;
        committed.insert(committed.end(),ids.begin(),ids.begin()+keep);
        for(uint32_t i=0;i<keep;++i) row_hashes.push_back(row_hash(std::span(logits).subspan(i*Vocab,Vocab)));
        consumed+=keep;
        report["cycles"].push_back({{"offset",offset},{"width",width},{"proposals",ids},{"accepted_proposals",good},
            {"committed_tokens",keep},{"forced_rejection",forced},{"draft_ns",draft_ns},{"verify_ns",verify_ns},
            {"recovery_ns",recovery_ns},{"wall_ns",end-start},{"memory_before",memory_before},
            {"memory_after",process_memory()},{"next_id",next}});
        if(work.validation) boundaries.push_back({{"tokens",consumed},{"target_state",state_digest(state)},
            {"target_logits_sha256",row_hash(std::span(logits).subspan(good*Vocab,Vocab))},{"draft_state",draft_digest(draft_state)}});
        if(consumed%16<keep || consumed==work.count || eos) save_report(output,report);
    }
    const auto request_end=monotonic_ns();
    report.update(Json{{"generated_tokens",consumed},{"proposed_tokens",proposed},{"accepted_proposals",accepted},
        {"committed_token_ids",committed},{"row_logits_sha256",row_hashes},{"next_id",next},{"stop_reason",eos?"eos":"length"},
        {"decode_wall_ns",total},{"tokens_per_second",double(consumed)*1e9/total},
        {"decode_including_reporting_ns",request_end-request_start},{"boundaries",boundaries},
        {"after",model.stats()},{"draft_after",draft.stats()},{"final_target_state",state_digest(state)},
        {"final_draft_state",work.serial?Json(nullptr):draft_digest(draft_state)},
        {"host_checkpoint_allocated_bytes",checkpoint.allocated},{"normal_request_latency_qualified",false}});
    check(model.memory_plan().json()==plan,"continuation memory admission changed");
}
