// Controlled prepared-record arrivals through the production expert coordinator.
// Included only by the developer checker; no production library changes.
#pragma once
namespace {
int replay_q4_reads(int argc,char** argv) {
    require(argc==8,"usage: qwen_q4_check FIXTURES REPORT arrivals MODEL PREPARED HITS check|timing|trace");
    const std::string mode=argv[7];const auto hits=std::string(argv[6]);
    require(hits=="2" || hits=="8","arrivals requires exactly two or eight primed hits");
    const uint32_t hit_count=hits=="2"?2:8,cycles=mode=="timing"?4:1;
    require(mode=="check" || mode=="timing" || mode=="trace","invalid arrivals mode");
    const bool validation=mode=="check",detailed=mode=="trace";
    require(validation==(std::getenv("MTL_DEBUG_LAYER")!=nullptr) &&
            validation==(std::getenv("MTL_SHADER_VALIDATION")!=nullptr),"arrivals validation mode differs");
    require(!std::filesystem::exists(argv[2]),"arrivals output already exists");
    replay_stopped=0;std::signal(SIGINT,stop_replay);std::signal(SIGTERM,stop_replay);
    const auto started=monotonic_ns();const std::filesystem::path root=argv[1];
    const auto manifest=read_json(root/"manifest.json");
    require(manifest.at("complete")==true && manifest.at("origin")=="existing-Q4-experts" &&
            manifest.at("layers").size()==4,"incomplete original Q4 fixtures");
    const auto prepared_text=read_text(std::filesystem::path(argv[5])/"manifest.json");
    const auto prepared_sha=hash({reinterpret_cast<const std::byte*>(prepared_text.data()),prepared_text.size()});
    Metal gpu;gpu.budget(12*GiB);gpu.residency("core-cache");gpu.prepare_pipelines();gpu.request_phase("decode");
    Checkpoint checkpoint(argv[4],true,Artifact::Mixed);
    const auto expected_resident=checkpoint.resident_bytes();
    require(expected_resident+256*MiB<12*GiB,"arrivals exceed fixed memory budget");
    if(available_memory()<expected_resident+256*MiB+GiB+GiB/2)
        throw std::runtime_error("insufficient currently available memory for arrivals replay");
    const auto resident_before=gpu.allocated();Resident resident(checkpoint,gpu);
    const auto resident_bytes=gpu.allocated()-resident_before;
    require(resident_bytes==expected_resident,"arrivals resident allocation differs");
    PreparedArtifact prepared(argv[5],checkpoint);
    struct Fixture { ExpertKey key;Buf original;std::array<Buf,8> inputs; };
    std::vector<Fixture> fixtures;std::vector<ExpertKey> keys;
    for(int layer:{0,16,32,47}) {
        const auto& item=manifest.at("layers").at(std::to_string(layer));
        require(item.at("experts").size()==2 && item.at("offsets").size()==8,"arrivals fixture coverage differs");
        auto all=read(gpu,root,item.at("inputs"));require(all->bytes==8*Hidden*4,"arrivals input size differs");
        std::array<Buf,8> inputs;
        for(uint32_t row=0;row<8;++row) {
            inputs[row]=gpu.allocate(Hidden*4);std::memcpy(inputs[row]->data,all->data+row*Hidden*4,Hidden*4);
        }
        for(const auto& e:item.at("experts")) {
            auto original=read(gpu,root,e.at("record"),AllocationClass::Expert);
            require(original->bytes==ExpertBytes,"arrivals record size differs");
            ExpertKey key{uint32_t(layer),e.at("expert").get<uint32_t>()};
            // Native detailed records identify experts within one layer. This fixture
            // spans layers; enforce unique expert IDs before joining those records.
            require(std::none_of(keys.begin(),keys.end(),[&](auto k){return k.expert==key.expert;}),
                    "arrivals trace requires unique fixture expert IDs");
            keys.push_back(key);fixtures.push_back({key,original,inputs});
        }
    }
    auto fixture_index=[&](ExpertKey key) {
        const auto it=std::find(keys.begin(),keys.end(),key);require(it!=keys.end(),"unknown arrivals expert");
        return size_t(it-keys.begin());
    };
    constexpr std::array<int,8> positions={9,1,7,3,5,0,8,4};
    auto output=gpu.allocate(TopK*Hidden*4);std::fill(output->floats().begin(),output->floats().end(),-19);
    std::vector<std::byte> expected(64*Hidden*4);
    auto check_guards=[&] {
        for(uint32_t slot:{2u,6u}) for(uint32_t i=0;i<Hidden;++i)
            require(output->floats()[slot*Hidden+i]==-19,"arrivals overwrote unused output");
    };
    auto check_batch=[&](uint32_t batch) {
        for(uint32_t i=0;i<8;++i) require(std::memcmp(expected.data()+(batch*8+i)*Hidden*4,
            output->data+positions[i]*Hidden*4,Hidden*4)==0,"arrivals changed expert output");
        check_guards();
    };
    KernelConfig reference;reference.policy="candidate";reference.q8_decode_rows=2;reference.route_selection="simd";
    auto packed=reference;packed.q4_decode="packed-r2";gpu.configure(reference);
    auto encode=[&](size_t i,uint32_t batch,const Buf& record) {
        replay_check();const auto& f=fixtures[i];const int position=positions[i];
        encode_expert_rows(gpu,record,f.inputs[batch],output,std::span(&position,1),1,128,false,f.key.layer,72+batch);
    };
    // Independent of all prepared arrivals: establish expected bytes from the
    // original saved records, using the existing reference arithmetic.
    for(uint32_t batch=0;batch<8;++batch) {
        gpu.begin_scratch(0,128*MiB);
        for(uint32_t at=0;at<8;at+=4) {for(uint32_t i=at;i<at+4;++i) encode(i,batch,fixtures[i].original);gpu.wait(gpu.submit());}
        gpu.end_scratch();gpu.finish();check_guards();
        for(uint32_t i=0;i<8;++i) std::memcpy(expected.data()+(batch*8+i)*Hidden*4,
            output->data+positions[i]*Hidden*4,Hidden*4);
    }
    std::array<Buf,8> slots;
    for(auto& slot:slots) slot=gpu.allocate(ExpertStride,AllocationClass::Expert);
    for(uint32_t i=0;i<8;++i) {
        prepared.expert(keys[i],slots[i]);
        require(std::memcmp(slots[i]->data,fixtures[i].original->data,ExpertBytes)==0,"prepared record differs from fixture");
        fixtures[i].original.reset();
    }
    gpu.finish();
    ReadPool reads(8);size_t pool_cursor=0;std::atomic<uint64_t> load_calls=0;
    ExpertCache cache(8,[&](uint64_t bytes) {
        require(bytes==ExpertStride && pool_cursor<slots.size(),"arrivals slot pool exhausted");
        return slots[pool_cursor++];
    },reads,[&](ExpertKey key,const Buf& into){prepared.expert(key,into);++load_calls;});
    auto prepare_batch=[&](uint32_t batch) {
        // Clear only at a drained boundary. The fixed pool retains every physical
        // buffer; no allocator can recycle one twice within an admitted batch.
        replay_check();gpu.finish();reads.drain();cache.clear();pool_cursor=0;
        for(uint32_t j=0;j<hit_count;++j) {auto lease=cache.acquire(keys[(batch+j)%8]);lease.wait();}
        reads.drain();
        for(uint32_t i=0;i<8;++i) require(cache.ready(keys[i])==((i+8-batch)%8<hit_count),"arrivals primed wrong hit mask");
    };
    auto execute=[&](bool candidate,uint32_t group,uint32_t repetitions,bool capture,bool check_outputs) {
        gpu.configure(candidate?packed:reference);const auto before=gpu.statistics();
        const auto arm_start=monotonic_ns();uint64_t wall=0,prep_ns=0,prep_bytes=0,prep_calls=0;
        uint64_t wait=0,ready=0,misses=0,joins=0,bytes=0,calls=0;Json batches=Json::array();
        for(uint32_t cycle=0;cycle<repetitions;++cycle) for(uint32_t batch=0;batch<8;++batch) {
            auto pbytes=prepared.bytes_read(),pcalls=load_calls.load(),pstart=monotonic_ns();
            prepare_batch(batch);prep_ns+=monotonic_ns()-pstart;
            prep_bytes+=prepared.bytes_read()-pbytes;prep_calls+=load_calls.load()-pcalls;
            const auto read_before=prepared.bytes_read(),calls_before=load_calls.load();
            const auto begin=monotonic_ns();gpu.begin_scratch(0,128*MiB);
            auto result=execute_experts(keys,cache,reads,gpu,group,
                [&](ExpertKey key,const Buf& record){encode(fixture_index(key),batch,record);},nullptr,capture);
            gpu.end_scratch();gpu.finish();reads.drain();wall+=monotonic_ns()-begin;
            bytes+=prepared.bytes_read()-read_before;calls+=load_calls.load()-calls_before;
            wait+=result.at("coordinator_wait_ns").get<uint64_t>();ready+=result.at("ready_hits").get<uint64_t>();
            misses+=result.at("new_misses").get<uint64_t>();joins+=result.at("loading_joins").get<uint64_t>();
            require(cache.capacity()==8 && cache.occupancy()==8 && pool_cursor==8,"arrivals cache geometry changed");
            if(capture) {
                for(auto& record:result["records"]) {
                    const auto expert=record.at("expert").get<uint32_t>();
                    const auto it=std::find_if(keys.begin(),keys.end(),[&](auto key){return key.expert==expert;});
                    require(it!=keys.end(),"unmapped arrivals trace expert");const auto i=size_t(it-keys.begin());
                    record["layer"]=it->layer;record["position"]=positions[i];
                }
                Json primed=Json::array();for(uint32_t j=0;j<hit_count;++j) primed.push_back((batch+j)%8);
                batches.push_back({{"cycle",cycle},{"batch",batch},{"hit_indices",primed},{"timing",std::move(result)}});
            }
            if(check_outputs) check_batch(batch);
        }
        const auto after=gpu.statistics();const uint64_t count=uint64_t(repetitions)*8;
        require(ready==count*hit_count && misses==count*(8-hit_count) && !joins &&
                bytes==misses*ExpertBytes && calls==misses && prep_bytes==ready*ExpertBytes && prep_calls==ready,
                "arrivals did not execute the declared hit/read pattern");
        Json sample={{"expert_executions",count*8},{"wall_ns",wall},{"arm_elapsed_ns",monotonic_ns()-arm_start},
            {"preparation_ns",prep_ns},{"preparation_read_bytes",prep_bytes},{"preparation_load_calls",prep_calls},
            {"coordinator_wait_ns",wait},{"ready_hits",ready},{"new_misses",misses},{"loading_joins",joins},
            {"read_bytes",bytes},{"expert_load_calls",calls},{"batches",std::move(batches)}};
        for(auto key:{"cpu_encode_ns","cpu_gpu_wait_ns","gpu_command_ns","submissions","allocation_count","scratch_reuses"})
            sample[key]=after[key].get<uint64_t>()-before[key].get<uint64_t>();
        Json counts=Json::object();
        for(const auto& [key,value]:after["kernel_dispatches"].items()) {
            const auto delta=value.get<uint64_t>()-before["kernel_dispatches"].value(key,uint64_t(0));if(delta) counts[key]=delta;
        }
        sample["kernel_dispatches"]=counts;return sample;
    };
    const auto initial=gpu.statistics();Json pairs=Json::array();
    auto persist=[&](bool complete) {
        const Json report={{"kind","native_q4_arrivals_v1"},{"mode",mode},{"complete",complete},
            {"hits_per_batch",hit_count},{"fixture_manifest",manifest},{"prepared_manifest_sha256",prepared_sha},
            {"resident_bytes",resident_bytes},{"expected_resident_bytes",expected_resident},{"budget_bytes",12*GiB},
            {"cache_slots",8},{"io_workers",8},{"max_live_groups",2},{"scratch_scope","eight-expert-batch"},
            {"disk_counter_scope","entire-arm-including-preparation-and-other-processes"},
            {"wall_scope","sum-of-coordinator-windows-excludes-preparation"},
            {"cycles",cycles},{"initial_metal",initial},{"final_metal",gpu.statistics()},{"pairs",pairs},
            {"validation",validation},{"exact",true},{"output_sha256",hash(expected)},
            {"timed_final_outputs_checked",true},{"untouched_destinations_checked",true},
            {"elapsed_ns",monotonic_ns()-started},{"normal_request_latency_qualified",false},{"production_promoted",false}};
        std::ofstream out(argv[2]);out<<report.dump(2)<<'\n';require(bool(out),"cannot write arrivals report");
    };
    try {
        persist(false);
        for(int pair=0;pair<(mode=="timing"?5:1);++pair) for(uint32_t group:{1u,4u}) {
            Json arms=Json::array();std::println("{} pair {} group {} hits {}",mode,pair,group,hit_count);std::fflush(stdout);
            for(bool candidate:std::array<bool,2>{bool(pair%2),!bool(pair%2)}) {
                execute(candidate,group,1,false,true); // identical warmup, exact every batch
                const auto memory_before=process_memory(),host_before=host_conditions(),disk_before=disk_counters(),metal_before=gpu.statistics();
                auto sample=execute(candidate,group,cycles,detailed,mode!="timing");
                const auto metal_after=gpu.statistics(),disk_after=disk_counters(),host_after=host_conditions(),memory_after=process_memory();
                check_batch(7);
                arms.push_back({{"variant",candidate?"packed-r2":"reference"},{"sample",std::move(sample)},
                    {"memory_before",memory_before},{"memory_after",memory_after},{"host_before",host_before},{"host_after",host_after},
                    {"disk_before",disk_before},{"disk_after",disk_after},{"metal_before",metal_before},{"metal_after",metal_after}});
                if(mode=="timing") execute(candidate,group,1,false,true); // exact post-timing comparison outside measured windows
            }
            pairs.push_back({{"pair",pair},{"group",group},{"arms",std::move(arms)}});persist(false);
        }
        gpu.release_scratch();reads.drain();cache.clear();persist(true);
    } catch(...) {
        const auto failure=std::current_exception();try {gpu.release_scratch();} catch(...) {} reads.drain();
        std::rethrow_exception(failure);
    }
    return 0;
}
}
