// Developer-only six-case replay. Weight bytes come directly from pinned ranges.
#include "qwen/metal.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
using namespace freellm::qwen;
namespace {
void check(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
std::string hash(const void* data,uint64_t bytes) {
    check(bytes<=UINT32_MAX,"hash bound exceeded");unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data,CC_LONG(bytes),digest);std::string out;
    for(auto c:digest){out+="0123456789abcdef"[c>>4];out+="0123456789abcdef"[c&15];}return out;
}
}
int main(int argc,char** argv) {
    Json report={{"kind","q8_expanded_operator_v1"},{"complete",false},{"production_promoted",false},
        {"normal_request_latency_qualified",false}};bool may_write=false;
    try {
        check(argc==5,"usage: probe MANIFEST MODEL REPORT validate|timing");
        const std::filesystem::path input=argv[1],output=argv[3];check(!std::filesystem::exists(output),"report exists");may_write=true;
        const bool validation=std::string_view(argv[4])=="validate";
        check(validation || std::string_view(argv[4])=="timing","invalid mode");
        check(validation?(getenv("MTL_DEBUG_LAYER") && getenv("MTL_SHADER_VALIDATION")):
            (!getenv("MTL_DEBUG_LAYER") && !getenv("MTL_SHADER_VALIDATION")),"wrong validation mode");
        check(!getenv("FREELLM_Q8_EXPANDED"),"operator probe requires explicit arm selection");
        const auto manifest=read_json(input),cases=Json::parse(R"CASES(@CASES@)CASES");
        check(manifest.at("kind")=="q8_expanded_fixture_v1" && manifest.at("build_fingerprint")==build_fingerprint() &&
            manifest.at("artifact_revision")==artifact_revision(Artifact::Mixed) && manifest.at("cases").size()==cases.size(),"fixture identity");
        report.update(Json{{"validation",validation},{"build",build_fingerprint()},{"artifact_revision",artifact_revision(Artifact::Mixed)},
            {"source",manifest},{"host_before",host_conditions()},{"memory_before",process_memory()},{"cases",Json::array()}});
        {
            Checkpoint checkpoint(argv[2],true,Artifact::Mixed);Metal gpu;gpu.budget(GiB);gpu.prepare_pipelines();
            size_t index=0;
            for(const auto& item:manifest.at("cases")) {
                const auto& expected_case=cases.at(index++);const auto& m=item.at("matrix");
                const uint32_t K=expected_case.at("K"),N=expected_case.at("N");
                check(item.at("name")==expected_case.at("name") && item.at("frequency")==expected_case.at("frequency") &&
                    m==Json{{"K",K},{"N",N},{"rows",4},{"group",64},{"bits",8},{"fused",false},{"gathered",false}} &&
                    item.at("phase")=="decode" && item.at("context").at("stage")==expected_case.at("stage") &&
                    item.at("context").at("layer")==expected_case.at("layer") && item.at("context").at("offset")==72,
                    "changed fixture geometry or scope");
                auto load_weight=[&](const char* key,const char* suffix,uint64_t bytes) {
                    const auto name=expected_case.at("tensor").get<std::string>()+suffix;
                    const auto& entry=item.at("tensors").at(key);const auto& tensor=checkpoint.at(name);
                    check(entry.at("tensor")==name && entry.at("bytes")==bytes && tensor.bytes==bytes && bytes<=768*MiB &&
                        entry.at("file")==std::filesystem::path(tensor.file->name()).filename().string() &&
                        entry.at("offset")==tensor.offset,"changed checkpoint range");
                    auto data=checkpoint.load(name,[&](uint64_t size){return gpu.allocate(size);});
                    check(hash(data->data,data->bytes)==entry.at("sha256").get<std::string>(),"changed checkpoint bytes");return data;
                };
                const uint64_t metadata=uint64_t(K/64)*N*2;
                Linear l{{load_weight("w",".weight",uint64_t(K)*N)},
                    {load_weight("s",".scales",metadata)},{load_weight("b",".biases",metadata)},K,N,64,0,true,8};
                const auto& entry=item.at("tensors").at("x");const auto name=entry.at("file").get<std::string>();
                check(!name.empty() && std::filesystem::path(name).filename()==name && entry.at("bytes")==4ull*K*4,"invalid input record");
                File file(input.parent_path()/name);check(file.size()==4ull*K*4,"wrong input bytes");auto x=gpu.allocate(file.size());
                file.read(0,{x->data,size_t(x->bytes)});check(hash(x->data,x->bytes)==entry.at("sha256").get<std::string>(),"changed input hash");
                auto out=gpu.allocate(4ull*N*4);gpu.configure({});auto reference=gpu.linear(l,x,4);gpu.finish();
                const auto golden=hash(reference->data,reference->bytes);
                Json row={{"name",item.at("name")},{"frequency",item.at("frequency")},{"K",K},{"N",N},
                    {"tokens",4},{"reference_sha256",golden},{"pairs",Json::array()}};
                auto sample=[&](bool candidate,int repeats) {
                    KernelConfig cfg;cfg.policy="candidate";cfg.token_tile=4;cfg.affine_rows=1;gpu.configure(cfg);
                    const auto before=process_memory(),counts=gpu.statistics().at("kernel_dispatches");const auto begin=monotonic_ns();
                    for(int r=0;r<repeats;++r) {
                        if(candidate) gpu.dispatch("q8_expanded_t4_w8",{l.weight,l.scales,l.biases,{x},{out}},{K,N,4,64,0},N*32);
                        else gpu.linear_into(l,x,4,{out});
                    }
                    auto command=gpu.submit();gpu.finish();const auto end=monotonic_ns();
                    check(hash(out->data,out->bytes)==golden,"expanded packed Q8 changed output");
                    const auto after=gpu.statistics().at("kernel_dispatches");Json dispatches=Json::object();
                    for(const auto& [key,value]:after.items()) {
                        const auto n=value.get<uint64_t>()-counts.value(key,uint64_t(0));if(n) dispatches[key]=n;
                    }
                    check(dispatches==Json{{candidate?"q8_expanded_t4_w8":"q8_mm_t4",repeats}},"wrong kernel selection");
                    return Json{{"candidate",candidate},{"repeats",repeats},{"exact",true},{"output_sha256",golden},
                        {"kernel_dispatches",dispatches},{"gpu_ns",uint64_t((command->gpu_end-command->gpu_start)*1e9)},
                        {"wall_ns",end-begin},{"memory_before",before},{"memory_after",process_memory()}};
                };
                row["warmup"]=Json::array({sample(false,1),sample(true,1)});
                for(int pair=0;pair<(validation?1:5);++pair) {
                    const bool first=pair%2;row["pairs"].push_back({{"pair",pair},
                        {"arms",Json::array({sample(first,validation?1:32),sample(!first,validation?1:32)})}});
                }
                report["cases"].push_back(std::move(row));std::cout<<item.at("name")<<" complete"<<std::endl;
            }
            report["peak_gpu_bytes"]=gpu.peak();report["device"]=gpu.device_name();
        }
        report["host_after"]=host_conditions();report["memory_after_destroy"]=process_memory();report["complete"]=true;
        std::ofstream out(output);out<<report.dump(2)<<'\n';out.close();check(bool(out),"cannot save operator report");return 0;
    } catch(const std::exception& e) {
        report["error"]=e.what();if(may_write){std::ofstream out(argv[3]);out<<report.dump(2)<<'\n';}
        std::cerr<<e.what()<<'\n';return 2;
    }
}
