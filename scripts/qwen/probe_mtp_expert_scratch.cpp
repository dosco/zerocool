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
