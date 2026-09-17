// Developer-only replay included by check_q4.cpp; the native library is unchanged.
#pragma once
namespace {
volatile std::sig_atomic_t replay_stopped=0;
void stop_replay(int) { replay_stopped=1; }
void replay_check() { if(replay_stopped) throw std::runtime_error("native Q4 replay cancelled"); }

int replay_q4(int argc,char** argv) {
    const std::string condition=argv[4];
    require(condition=="direct" || condition=="scatter" || condition=="resident","unknown replay condition");
    require(argc==(condition=="resident"?6:5),"replay FIXTURES REPORT replay direct|scatter|resident [MODEL]");
    require(!std::getenv("MTL_DEBUG_LAYER") && !std::getenv("MTL_SHADER_VALIDATION"),"timing cannot enable validation");
    require(!std::filesystem::exists(argv[2]),"replay output exists");
    std::signal(SIGINT,stop_replay);std::signal(SIGTERM,stop_replay);
    const auto started=monotonic_ns();
    const std::filesystem::path root=argv[1];const auto manifest=read_json(root/"manifest.json");
    require(manifest.at("complete")==true && manifest.at("origin")=="existing-Q4-experts" &&
            manifest.at("layers").size()==4,"incomplete original Q4 fixtures");
    Metal gpu;gpu.budget(12*GiB);gpu.residency("core-cache");gpu.prepare_pipelines();gpu.request_phase("decode");
    std::unique_ptr<Checkpoint> checkpoint;std::unique_ptr<Resident> resident;
    uint64_t resident_bytes=0,expected_resident_bytes=0;
    if(condition=="resident") {
        checkpoint=std::make_unique<Checkpoint>(argv[5],true,Artifact::Mixed);
        expected_resident_bytes=checkpoint->resident_bytes();
        require(expected_resident_bytes+128*MiB<12*GiB,"resident replay exceeds fixed budget");
        if(available_memory()<expected_resident_bytes+128*MiB+GiB+GiB/2)
            throw std::runtime_error("insufficient currently available memory for resident replay");
        const auto before=gpu.allocated();resident=std::make_unique<Resident>(*checkpoint,gpu);
        resident_bytes=gpu.allocated()-before;
        require(resident_bytes==expected_resident_bytes,"resident accounting differs");replay_check();
    }
    struct Fixture { Buf record;std::array<Buf,8> inputs;int layer,expert; };
    std::vector<Fixture> fixtures;
    for(int layer:{0,16,32,47}) {
        const auto& item=manifest.at("layers").at(std::to_string(layer));
        require(item.at("experts").size()==2 && item.at("offsets").size()==8,"fixture coverage differs");
        auto all=read(gpu,root,item.at("inputs"));require(all->bytes==8*Hidden*4,"input size differs");
        std::array<Buf,8> inputs;
        for(uint32_t row=0;row<8;++row) {inputs[row]=gpu.allocate(Hidden*4);std::memcpy(inputs[row]->data,all->data+row*Hidden*4,Hidden*4);}
        for(const auto& e:item.at("experts")) {
            auto record=read(gpu,root,e.at("record"),AllocationClass::Expert);
            require(record->bytes==ExpertBytes,"expert size differs");
            fixtures.push_back({record,inputs,layer,e.at("expert").get<int>()});
        }
    }
    constexpr std::array<int,8> positions={9,1,7,3,5,0,8,4};
    auto output=gpu.allocate(TopK*Hidden*4);std::vector<std::byte> expected(64*Hidden*4);
    std::fill(output->floats().begin(),output->floats().end(),-19);
    auto check_destinations=[&] {
        for(uint32_t slot:{2u,6u}) for(uint32_t i=0;i<Hidden;++i)
            require(output->floats()[slot*Hidden+i]==-19,"replay overwrote an unused destination");
    };
    auto check_batch=[&](uint32_t batch) {
        for(uint32_t i=0;i<8;++i)
            require(std::memcmp(expected.data()+(batch*8+i)*Hidden*4,
                output->data+positions[i]*Hidden*4,Hidden*4)==0,"native replay output changed");
        check_destinations();
    };
    KernelConfig reference;reference.policy="candidate";reference.q8_decode_rows=2;reference.route_selection="simd";
    auto packed=reference;packed.q4_decode="packed-r2";
    const auto initial=gpu.statistics();Json rows=Json::array();
    auto persist=[&](bool complete) {
        const Json report={{"kind","native_q4_replay_v1"},{"complete",complete},{"condition",condition},
            {"fixture_manifest",manifest},{"cycles",4},{"cases_per_cycle",64},{"max_live_groups",2},
            {"positions",positions},{"scratch_scope","eight-expert-batch"},{"validation",false},{"exact",true},
            {"timed_final_outputs_checked",true},{"untouched_destinations_checked",true},
            {"initial_metal",initial},{"final_metal",gpu.statistics()},{"resident_bytes",resident_bytes},
            {"expected_resident_bytes",expected_resident_bytes},{"budget_bytes",12*GiB},{"pairs",rows},
            {"elapsed_ns",monotonic_ns()-started},{"normal_request_latency_qualified",false},
            {"output_sha256",hash(expected)},{"production_promoted",false}};
        std::ofstream out(argv[2]);out<<report.dump(2)<<'\n';require(bool(out),"cannot write replay report");
    };
    auto execute=[&](bool candidate,uint32_t group,uint32_t cycles,int validation) {
        replay_check();gpu.configure(candidate?packed:reference);
        // No pipeline compilation, copies, per-pass profiling, or JSON writes in the timed loop.
        const auto before=gpu.statistics();const auto begin=monotonic_ns();
        for(uint32_t cycle=0;cycle<cycles;++cycle) for(uint32_t batch=0;batch<8;++batch) {
            replay_check();gpu.begin_scratch(0,128*MiB);
            std::deque<std::shared_ptr<Metal::Completion>> pending;
            for(uint32_t at=0;at<8;at+=group) {
                replay_check();
                for(uint32_t i=at;i<at+group;++i) {
                    const auto& f=fixtures[i];const int position=positions[i];
                    encode_expert_rows(gpu,f.record,f.inputs[batch],output,std::span(&position,1),1,128,
                                       condition=="direct",f.layer,batch);
                }
                pending.push_back(gpu.submit());
                if(pending.size()>=2) {gpu.wait(pending.front());pending.pop_front();}
            }
            for(const auto& completion:pending) gpu.wait(completion);
            gpu.end_scratch();gpu.finish();
            if(validation==1) {
                for(uint32_t i=0;i<8;++i) std::memcpy(expected.data()+(batch*8+i)*Hidden*4,
                    output->data+positions[i]*Hidden*4,Hidden*4);
                check_destinations();
            }
            else if(validation==2) check_batch(batch);
        }
        const auto wall=monotonic_ns()-begin;const auto after=gpu.statistics();
        Json sample={{"expert_executions",uint64_t(cycles)*64},{"wall_ns",wall}};
        for(auto key:{"cpu_encode_ns","cpu_gpu_wait_ns","gpu_command_ns","submissions","allocation_count","scratch_reuses"})
            sample[key]=after[key].get<uint64_t>()-before[key].get<uint64_t>();
        Json counts=Json::object();
        for(const auto& [key,value]:after["kernel_dispatches"].items()) {
            const auto n=value.get<uint64_t>()-before["kernel_dispatches"].value(key,uint64_t(0));if(n) counts[key]=n;
        }
        sample["kernel_dispatches"]=counts;return sample;
    };
    try {
        // Explicit reference-output checks in both geometries, entirely outside timing samples.
        for(uint32_t group:{1u,4u}) {execute(false,group,1,1);execute(true,group,1,2);}
        persist(false);
        for(int pair=0;pair<5;++pair) for(uint32_t group:{1u,4u}) {
            Json arms=Json::array();std::println("pair {} group {}",pair,group);std::fflush(stdout);
            for(bool candidate:std::array<bool,2>{bool(pair%2),!bool(pair%2)}) {
                execute(candidate,group,1,0); // identical warmup count before each arm
                const auto memory_before=process_memory(),host_before=host_conditions(),metal_before=gpu.statistics();
                const auto sample=execute(candidate,group,4,0);
                const auto metal_after=gpu.statistics(),host_after=host_conditions(),memory_after=process_memory();
                check_batch(7); // Inspect the actual timed output after all measured work has drained.
                arms.push_back({{"variant",candidate?"packed-r2":"reference"},{"sample",sample},
                    {"memory_before",memory_before},{"memory_after",memory_after},{"host_before",host_before},
                    {"host_after",host_after},{"metal_before",metal_before},{"metal_after",metal_after}});
                execute(candidate,group,1,2); // verify post-timing outputs outside the measurement
            }
            rows.push_back({{"pair",pair},{"group",group},{"arms",arms}});persist(false);
        }
        gpu.release_scratch();persist(true);
    } catch(...) {gpu.release_scratch();throw;}
    return 0;
}
}
