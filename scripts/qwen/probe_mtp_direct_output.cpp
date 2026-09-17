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
