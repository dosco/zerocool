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
