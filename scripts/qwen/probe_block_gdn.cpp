// Developer-only four-token Q8 row-pair screen with actual captured inputs.
#include "engine/metal.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
using namespace zerocool::engine;
namespace {
void check(bool ok,const char* text) { if(!ok) throw std::runtime_error(text); }
std::string hash(const void* data,size_t bytes) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];check(bytes<=UINT32_MAX,"oversized hash");
    CC_SHA256(data,CC_LONG(bytes),digest);std::string out;
    for(auto c:digest) {out+="0123456789abcdef"[c>>4];out+="0123456789abcdef"[c&15];}return out;
}
}
int main(int argc,char** argv) {
    Json report={{"kind","block_gdn_operator_v1"},{"complete",false},{"production_promoted",false},
        {"normal_request_latency_qualified",false}};bool may_write=false;
    try {
        check(argc==4,"usage: probe FIXTURES REPORT validate|timing");
        const std::filesystem::path manifest_path=argv[1],output=argv[2];
        check(!std::filesystem::exists(output),"output exists");may_write=true;
        const bool validation=std::string_view(argv[3])=="validate";
        check(validation || std::string_view(argv[3])=="timing","invalid mode");
        check(validation?(getenv("MTL_DEBUG_LAYER") && getenv("MTL_SHADER_VALIDATION")):
            (!getenv("MTL_DEBUG_LAYER") && !getenv("MTL_SHADER_VALIDATION")),"wrong validation mode");
        const auto manifest=read_json(manifest_path);
        check(manifest.at("kind")=="captured_affine_operators" && manifest.at("cases").size()==3 &&
            manifest.at("artifact_revision")==artifact_revision(Artifact::Mixed) &&
            manifest.at("build_fingerprint")==build_fingerprint(),"fixture identity mismatch");
        report.update(Json{{"validation",validation},{"build",build_fingerprint()},
            {"artifact_revision",artifact_revision(Artifact::Mixed)},{"host_before",host_conditions()},
            {"memory_before",process_memory()},{"source",manifest},{"cases",Json::array()}});
        {
            Metal gpu;gpu.budget(256*MiB);gpu.prepare_pipelines();
            const std::array<std::pair<uint32_t,uint32_t>,3> shapes={{{2560,10240},{2560,6144},{6144,2560}}};
            size_t index=0;
            for(const auto& item:manifest.at("cases")) {
                const auto& m=item.at("matrix");const auto [K,N]=shapes.at(index++);
                check(m.at("K")==K && m.at("N")==N && m.at("rows")==4 && m.at("bits")==8 &&
                    m.at("group")==64 && m.at("fused")==false && m.at("gathered")==false &&
                    item.at("phase")=="decode" && item.at("context").at("layer")==0 &&
                    item.at("context").at("stage")=="gdn" && item.at("context").at("offset")==72,
                    "unexpected captured matrix");
                auto load=[&](const char* name,uint64_t bytes) {
                    const auto& entry=item.at("tensors").at(name);const auto file=entry.at("file").get<std::string>();
                    check(!file.empty() && std::filesystem::path(file).filename()==file && entry.at("bytes")==bytes &&
                        bytes<=128*MiB,"invalid fixture file");
                    File src(manifest_path.parent_path()/file);check(src.size()==bytes,"fixture size mismatch");
                    auto b=gpu.allocate(bytes);src.read(0,{b->data,size_t(bytes)});
                    check(hash(b->data,bytes)==entry.at("sha256").get<std::string>(),"fixture hash mismatch");return b;
                };
                const uint64_t metadata=uint64_t(K/64)*N*2;
                Linear l{{load("w",uint64_t(K)*N)},{load("s",metadata)},{load("b",metadata)},K,N,64,0,true,8};
                auto x=load("x",4ull*K*4),out=gpu.allocate(4ull*N*4);
                gpu.configure({});auto reference=gpu.linear(l,x,4);gpu.finish();
                const auto expected=hash(reference->data,reference->bytes);
                Json row={{"K",K},{"N",N},{"tokens",4},{"reference_sha256",expected},{"pairs",Json::array()}};
                auto sample=[&](bool candidate,int repeats) {
                    KernelConfig cfg;cfg.policy="candidate";cfg.token_tile=4;cfg.affine_rows=candidate?2:1;
                    gpu.configure(cfg);const auto before=process_memory();const auto start=monotonic_ns();
                    for(int i=0;i<repeats;++i) gpu.linear_into(l,x,4,{out});
                    auto command=gpu.submit();gpu.finish();const auto end=monotonic_ns();
                    check(hash(out->data,out->bytes)==expected,"Q8 row pair changed output bits");
                    return Json{{"candidate",candidate},{"repeats",repeats},{"exact",true},
                        {"output_sha256",expected},{"gpu_ns",uint64_t((command->gpu_end-command->gpu_start)*1e9)},
                        {"wall_ns",end-start},{"memory_before",before},{"memory_after",process_memory()}};
                };
                // One explicit warmup per arm, retained in evidence but excluded
                // from paired timing. Both arms reuse the same allocated buffers.
                row["warmup"]=Json::array({sample(false,1),sample(true,1)});
                for(int pair=0;pair<(validation?1:5);++pair) {
                    Json arms=Json::array();const bool first=pair%2;
                    arms.push_back(sample(first,validation?1:32));arms.push_back(sample(!first,validation?1:32));
                    row["pairs"].push_back({{"pair",pair},{"arms",arms}});
                }
                report["cases"].push_back(std::move(row));
            }
            report["peak_gpu_bytes"]=gpu.peak();report["device"]=gpu.device_name();
        }
        report["memory_after_destroy"]=process_memory();report["host_after"]=host_conditions();report["complete"]=true;
        std::ofstream out(argv[2]);out<<report.dump(2)<<'\n';out.close();check(bool(out),"cannot write report");return 0;
    } catch(const std::exception& e) {
        report["error"]=e.what();if(may_write) {std::ofstream out(argv[2]);out<<report.dump(2)<<'\n';}
        std::cerr<<e.what()<<'\n';return 2;
    }
}
