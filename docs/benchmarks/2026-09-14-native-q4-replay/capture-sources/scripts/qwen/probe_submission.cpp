// Bounded real-Q4-weight ownership experiment. No SSD/request-speed claim.
#include "qwen/metal.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iostream>

using namespace freellm::qwen;
namespace {
std::atomic<bool> stopped=false;
void interrupt(int) {stopped=true;}
void check(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
std::string hash(std::span<const std::byte> bytes) {
    unsigned char d[CC_SHA256_DIGEST_LENGTH];CC_SHA256(bytes.data(),CC_LONG(bytes.size()),d);
    std::string out;for(auto c:d) {out+="0123456789abcdef"[c>>4];out+="0123456789abcdef"[c&15];}return out;
}
Buf load(Metal& gpu,const std::filesystem::path& directory,const Json& entry) {
    const std::filesystem::path name=entry.at("file").get<std::string>();
    check(name==name.filename() && !name.empty(),"invalid fixture path");
    File file(directory/name);const uint64_t size=entry.at("bytes");
    check(size && size<=4*MiB && file.size()==size,"invalid fixture length");
    auto b=gpu.allocate(size);file.read(0,{b->data,size_t(size)});
    check(hash({b->data,size_t(size)})==entry.at("sha256").get<std::string>(),"fixture hash mismatch");return b;
}
struct Fixture {Buf record,input,hidden,output;std::string expected;};
void encode(Metal& gpu,const Fixture& f) {
    auto gate=expert_linear(f.record,0),up=expert_linear(f.record,1);
    gpu.dispatch("q4_gate_up",{gate.weight,gate.scales,gate.biases,up.weight,up.scales,up.biases,{f.input},{f.input},{f.hidden}},
        {Hidden,Intermediate,1,64,0},Intermediate*32);
    gpu.linear_into(expert_linear(f.record,2),f.hidden,1,{f.output});
}
Json sample(Metal& gpu,std::span<const Fixture> fixtures,uint32_t group_size,uint32_t repeats) {
    Json timing=Json::array();const auto before=process_memory();
    for(uint32_t i=0;i<repeats;++i) {
        check(!stopped,"probe cancelled");const auto start=monotonic_ns();
        for(uint32_t j=0;j<group_size;++j) encode(gpu,fixtures[(i*group_size+j)%fixtures.size()]);
        auto c=gpu.submit();gpu.wait(c);const auto end=monotonic_ns();
        timing.push_back({{"wall_ns",end-start},{"submitted_ns",c->submitted_ns},{"commit_returned_ns",c->commit_returned_ns},
            {"driver_start_seconds",c->driver_start},{"driver_end_seconds",c->driver_end},
            {"gpu_start_seconds",c->gpu_start},{"gpu_end_seconds",c->gpu_end},{"completed_ns",c->completed_ns}});
        // Check outside the measured group. Equal inputs/destinations in both arms.
        for(uint32_t j=0;j<group_size;++j) {
            const auto& f=fixtures[(i*group_size+j)%fixtures.size()];
            check(hash({f.output->data,size_t(f.output->bytes)})==f.expected,"ownership changed expert output");
        }
    }
    check(gpu.statistics()["live_command_groups"]==0,"GPU users remain after probe");
    return {{"samples",timing},{"memory_before",before},{"memory_after",process_memory()}};
}
}
int main(int argc,char** argv) {
    Json report={{"kind","command_ownership_probe_v1"},{"complete",false},{"normal_request_latency_qualified",false},{"production_promoted",false}};
    try {
        check(argc==4,"usage: qwen_submission_probe FIXTURES REPORT timing|validate");
        const std::string mode=argv[3];check(mode=="timing" || mode=="validate","invalid probe mode");
        const bool validation=mode=="validate";
        if(!validation) check(!std::getenv("MTL_DEBUG_LAYER") && !std::getenv("MTL_SHADER_VALIDATION"),"timing must disable Metal validation");
        else check(std::getenv("MTL_DEBUG_LAYER") && std::getenv("MTL_SHADER_VALIDATION"),"validation mode requires Metal checks");
        check(!std::filesystem::exists(argv[2]),"report already exists");
        std::signal(SIGINT,interrupt);std::signal(SIGTERM,interrupt);
        const std::filesystem::path directory=argv[1];auto manifest=read_json(directory/"manifest.json");
        check(manifest.at("complete")==true && manifest.at("kind")=="q3_probe_capture_v2" &&
            manifest.at("artifact_revision")==artifact_revision(Artifact::Mixed) && manifest.at("origin")=="existing-Q4-experts", "wrong Q4 fixture identity");
        Metal gpu;gpu.budget(64*MiB);gpu.prepare_pipelines();std::vector<Fixture> fixtures;
        for(auto& [layer,item]:manifest.at("layers").items()) {
            auto input=load(gpu,directory,item.at("inputs"));check(input->bytes==8*Hidden*4,"wrong input shape");
            check(item.at("experts").size()==2,"expected two experts per layer");
            for(auto& expert:item.at("experts")) {
                auto record=load(gpu,directory,expert.at("record"));check(record->bytes==ExpertBytes,"wrong expert shape");
                Fixture f{record,input,gpu.allocate(Intermediate*4),gpu.allocate(Hidden*4),{}};
                encode(gpu,f);gpu.finish();
                for(uint32_t i=0;i<Hidden;++i) check(std::isfinite(f.output->floats()[i]),"nonfinite reference");
                f.expected=hash({f.output->data,size_t(f.output->bytes)});fixtures.push_back(std::move(f));
            }
        }
        check(fixtures.size()==8,"expected eight real experts");
        report.update(Json{{"build",build_fingerprint()},{"device",gpu.device_name()},{"physical_bytes",gpu.physical()},
            {"fixture_manifest",manifest},{"validation",validation},{"budget_bytes",64*MiB},{"pairs",Json::array()},
            {"limitations",{"Resident fixture weights; no SSD waits, full-model state, or request-speed claim.",
                "One group at a time; preallocated outputs, unchanged gate/up/down arithmetic; hashes checked outside timing.",
                "The eight expert inputs are captured model inputs, not a replay of full-model routing."}}});
        for(uint32_t pair=0;pair<(validation?1:5);++pair) {
            for(uint32_t size:{1u,2u,4u}) {
                Json row={{"pair",pair},{"experts_per_group",size},{"arms",Json::array()}};
                for(uint32_t arm=0;arm<2;++arm) {
                    const bool retained=(pair+arm)%2==0;gpu.command_references(retained);
                    std::cout<<"pair "<<pair<<" group "<<size<<' '<<(retained?"retained":"external")<<std::endl;
                    (void)sample(gpu,fixtures,size,8);
                    auto result=sample(gpu,fixtures,size,validation?8:64);result["retained_references"]=retained;
                    row["arms"].push_back(std::move(result));
                }
                report["pairs"].push_back(std::move(row));
            }
        }
        report["tracked_peak_bytes"]=gpu.peak();fixtures.clear();gpu.finish();
        check(gpu.allocated()==0,"fixture ownership leaked");report["complete"]=true;report["final_live_bytes"]=gpu.allocated();
        std::ofstream out(argv[2]);out<<report.dump(2)<<'\n';check(bool(out),"cannot save probe");return 0;
    } catch(const std::exception& error) {
        report["error"]=error.what();if(argc==4 && !std::filesystem::exists(argv[2])) {std::ofstream out(argv[2]);out<<report.dump(2)<<'\n';}
        std::cerr<<error.what()<<'\n';return 2;
    }
}
