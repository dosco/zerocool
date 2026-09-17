#include "qwen/model.hpp"
#include "qwen/pipeline.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <cstring>
#include <fstream>
#include <print>
#include <csignal>
#include <deque>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
using namespace freellm::qwen;
namespace {
void require(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
std::string hash(std::span<const std::byte> bytes) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256(bytes.data(),CC_LONG(bytes.size()),digest);
    std::string out;for(auto b:digest) {out+="0123456789abcdef"[b>>4];out+="0123456789abcdef"[b&15];}return out;
}
Buf read(Metal& gpu,const std::filesystem::path& root,const Json& entry,AllocationClass kind=AllocationClass::Temporary) {
    const auto name=entry.at("file").get<std::string>();
    require(std::filesystem::path(name).filename()==name,"fixture path must be local");
    File file(root/name);require(file.size()==entry.at("bytes") && file.size()<=32*MiB,"fixture size mismatch");
    auto out=gpu.allocate(file.size(),kind);file.read(0,{out->data,size_t(out->bytes)});
    require(hash({out->data,size_t(out->bytes)})==entry.at("sha256").get<std::string>(),"fixture hash mismatch");return out;
}
}
#include "replay_q4.hpp"
#include "replay_q4_reads.hpp"
int main(int argc,char** argv) {
    try {
        if(argc>=5 && std::string(argv[3])=="replay") return replay_q4(argc,argv);
        if(argc>=4 && std::string(argv[3])=="arrivals") return replay_q4_reads(argc,argv);
        if(argc>=4 && std::string(argv[3])=="cache_probe") return q4_cache_probe(argc,argv);
        require(argc==3,"usage: qwen_q4_check FIXTURE_DIR REPORT");
        const std::filesystem::path root=argv[1];const auto manifest=read_json(root/"manifest.json");
        require(manifest.at("complete")==true && manifest.at("origin")=="existing-Q4-experts" &&
                manifest.at("layers").size()==4,"incomplete original Q4 fixtures");
        Metal gpu;gpu.budget(128*MiB);gpu.prepare_pipelines();gpu.request_phase("decode");
        Json cases=Json::array();KernelConfig packed;packed.policy="candidate";packed.q4_decode="packed-r2";
        for(int layer:{0,16,32,47}) {
            const auto& item=manifest.at("layers").at(std::to_string(layer));
            require(item.at("experts").size()==2 && item.at("offsets").size()==8,"fixture coverage differs");
            auto inputs=read(gpu,root,item.at("inputs"));require(inputs->bytes==8*Hidden*4,"input size differs");
            for(const auto& expert:item.at("experts")) {
                auto record=read(gpu,root,expert.at("record"));require(record->bytes==ExpertBytes,"expert size differs");
                const auto gate=expert_linear(record,0),up=expert_linear(record,1),down=expert_linear(record,2);
                for(uint32_t row=0;row<8;++row) {
                    auto input=gpu.slice(inputs,row*Hidden*4,Hidden*4);
                    gpu.configure({});auto expected_gate=gpu.gated_linear(gate,up,input,1);
                    auto expected=gpu.linear(down,expected_gate,1),fp32=gpu.linear(down,expected_gate,1,true);gpu.finish();
                    gpu.configure(packed);auto actual_gate=gpu.gated_linear(gate,up,input,1);
                    auto actual=gpu.zeros(3*Hidden),actual_float=gpu.zeros(3*Hidden);
                    std::fill(actual->floats().begin(),actual->floats().end(),-19);
                    std::fill(actual_float->floats().begin(),actual_float->floats().end(),-19);
                    gpu.linear_into(down,actual_gate,1,{actual,Hidden*4});
                    gpu.linear_into(down,actual_gate,1,{actual_float,Hidden*4},true);gpu.finish();
                    require(std::memcmp(actual_gate->data,expected_gate->data,Intermediate*4)==0,"gate/up changed");
                    require(std::memcmp(actual->data+Hidden*4,expected->data,Hidden*4)==0,"BF16 down changed");
                    require(std::memcmp(actual_float->data+Hidden*4,fp32->data,Hidden*4)==0,"FP32 down changed");
                    for(auto out:{actual,actual_float}) for(uint32_t i=0;i<Hidden;++i)
                        require(out->floats()[i]==-19 && out->floats()[2*Hidden+i]==-19,"output destination overwritten");
                    cases.push_back({{"layer",layer},{"expert",expert.at("expert")},{"row",row},
                        {"exact_gate",true},{"exact_bf16_down",true},{"exact_fp32_down",true},{"destination_preserved",true}});
                }
            }
        }
        const auto stats=gpu.statistics();const auto counts=stats.at("kernel_dispatches");
        require(counts.at("q4_gate_up_packed_r2")==64 && counts.at("q4_down_packed_r2")==128,"native candidate not exercised");
        Json report={{"kind","native_q4_exact_check_v1"},{"complete",true},{"cases",cases},
            {"fixture",manifest},{"metal",stats},{"validation",std::getenv("MTL_DEBUG_LAYER")!=nullptr},
            {"normal_request_latency_qualified",false}};
        std::ofstream out(argv[2]);out<<report.dump(2)<<'\n';require(bool(out),"cannot write report");
        std::println("64 native Q4 expert/input cases passed");return 0;
    } catch(const std::exception& e) {std::println(stderr,"Q4 check: {}",e.what());return 1;}
}
