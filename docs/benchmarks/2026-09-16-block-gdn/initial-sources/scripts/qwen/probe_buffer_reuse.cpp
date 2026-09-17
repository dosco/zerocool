// Isolated lifetime/timing experiment using the existing scratch pool.
// This is synthetic allocation pressure, not a model or recorded-route replay.
#include "qwen/metal.hpp"
#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace freellm::qwen;
namespace {
constexpr uint32_t Groups=48, Buffers=64, Repeats=8;
constexpr std::array<uint32_t,4> Widths{2560,6144,10240,768};
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}

Json iteration(Metal& gpu,const Buf& input,bool pool) {
    const auto before=gpu.statistics(),memory_before=process_memory();
    const auto start=monotonic_ns();
    if(pool) gpu.begin_scratch(0,512*MiB);
    uint64_t elements=0;
    for(uint32_t group=0;group<Groups;++group) {
        std::array<Buf,Buffers> outputs;
        for(uint32_t i=0;i<Buffers;++i) {
            const auto n=Widths[(i+group)%Widths.size()];
            outputs[i]=gpu.allocate(uint64_t(n)*4);
            gpu.dispatch("binary",{{input},{input},{outputs[i]}},{n,0},n);
        }
        gpu.finish(); // All owners remain alive through their last GPU use.
        for(uint32_t i=0;i<Buffers;++i) {
            const auto n=Widths[(i+group)%Widths.size()];
            const auto* values=reinterpret_cast<const float*>(outputs[i]->data);
            for(uint32_t j=0;j<n;++j) require(values[j]==float(j%64)*2,"GPU output mismatch");
            elements+=n;
        }
        // Control releases after each group; pool retains the whole iteration.
    }
    if(pool) gpu.end_scratch();
    const auto elapsed=monotonic_ns()-start;
    const auto after=gpu.statistics(),memory_after=process_memory();
    require(after.at("live_command_groups")==0,"Outstanding GPU users");
    require(gpu.allocated()<=512*MiB,"Probe memory limit exceeded");
    return {{"elapsed_ns",elapsed},{"validated_elements",elements},
        {"allocation_count",after.at("allocation_count").get<uint64_t>()-before.at("allocation_count").get<uint64_t>()},
        {"scratch_reuses",after.at("scratch_reuses").get<uint64_t>()-before.at("scratch_reuses").get<uint64_t>()},
        {"live_bytes",gpu.allocated()},{"process_before",memory_before},{"process_after",memory_after}};
}
}
int main(int argc,char** argv) {
    try {
        require(argc==2,"usage: probe_buffer_reuse output.json");
        require(!std::getenv("MTL_DEBUG_LAYER") && !std::getenv("MTL_SHADER_VALIDATION"),"Disable Metal validation for timing");
        Metal gpu;gpu.budget(512*MiB);
        auto input=gpu.allocate(Widths[2]*4,AllocationClass::Resident);
        for(uint32_t j=0;j<Widths[2];++j) reinterpret_cast<float*>(input->data)[j]=float(j%64);
        Json report={{"kind","buffer_reuse_probe_v1"},{"complete",false},{"synthetic",true},
            {"normal_request_latency_qualified",false},{"production_promoted",false},
            {"build",build_fingerprint()},{"device",gpu.device_name()},{"physical_bytes",gpu.physical()},
            {"groups",Groups},{"buffers_per_group",Buffers},{"widths",Widths},{"iterations_per_arm",Repeats},
            {"budget_bytes",512*MiB},{"timers_enabled",false},{"pairs",Json::array()},
            {"limitations",{"No model, SSD, routing, recurrent state or real allocation trace; not a tokens/s result.",
                "Iteration wall time includes GPU completion, CPU output validation, and final-owner release.",
                "Warm scratch timing excludes its first allocation; cold time and retained bytes are reported separately."}}};
        for(int pair=0;pair<5;++pair) {
            const Json order=pair%2?Json{"pool","control"}:Json{"control","pool"};
            Json row={{"pair",pair},{"order",order}};
            for(const auto& name:order) {
                std::cout<<"pair "<<pair<<' '<<name.get<std::string>()<<std::endl;
                const bool pool=name=="pool";
                gpu.release_scratch();
                Json arm={{"cold",iteration(gpu,input,pool)},{"warm",Json::array()}};
                // A second untimed-in-decision warmup exercises actual reuse.
                arm["warmup"]=iteration(gpu,input,pool);
                for(uint32_t i=0;i<Repeats;++i) {
                    auto sample=iteration(gpu,input,pool);
                    require(sample["allocation_count"]==(pool?0:Groups*Buffers),"Unexpected physical allocations");
                    require(sample["scratch_reuses"]==(pool?Groups*Buffers:0),"Unexpected scratch reuse");
                    arm["warm"].push_back(std::move(sample));
                }
                const auto start=monotonic_ns();gpu.release_scratch();
                arm["release_ns"]=monotonic_ns()-start;arm["after_release"]=process_memory();
                require(gpu.allocated()==49152,"Scratch ownership retained after release");
                row[name.get<std::string>()]=std::move(arm);
            }
            report["pairs"].push_back(std::move(row));
        }
        input.reset();gpu.finish();require(gpu.allocated()==0,"Final buffer leak");
        report["complete"]=true;report["final_live_bytes"]=gpu.allocated();
        std::ofstream output(argv[1]);output<<report.dump(2)<<'\n';require(bool(output),"Cannot write report");
        std::cout<<"completed five isolated pairs\n";return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 2;}
}
