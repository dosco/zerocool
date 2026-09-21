#include "engine/metal.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <print>

using namespace zerocool::engine;
namespace {
std::string hash(const Buf& b) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256(b->data,CC_LONG(b->bytes),digest);
    std::string out;constexpr char hex[]="0123456789abcdef";
    for(auto c:digest) {out+=hex[c>>4];out+=hex[c&15];}return out;
}
bool equal(const Buf& a,const Buf& b) {return a->bytes==b->bytes && std::memcmp(a->data,b->data,a->bytes)==0;}
}
int main(int argc,char** argv) {
    try {
        if(argc!=3) throw std::invalid_argument("usage: qwen_route_check FIXTURE_MANIFEST REPORT");
        const std::filesystem::path path(argv[1]);const auto manifest=read_json(path);
        Metal gpu;gpu.budget(32*MiB);gpu.prepare_pipelines();
        if(manifest.at("kind")!="captured_router_logits_v1" || manifest.at("build")!=build_fingerprint() ||
           manifest.at("artifact_revision")!=artifact_revision(Artifact::Mixed) ||
           manifest.at("cases").empty() || manifest.at("cases").size()>64)
            throw std::invalid_argument("wrong or unbounded route fixture");
        Json measurements=Json::array(),checks=Json::array();uint64_t total=0;bool exact=true;
        for(const auto& c:manifest.at("cases")) {
            const std::string file=c.at("file");const uint32_t tokens=c.at("tokens");
            if(!tokens || tokens>128 || std::filesystem::path(file).filename()!=file || c.at("bytes")!=uint64_t(tokens)*512*4)
                throw std::invalid_argument("invalid route fixture dimensions or file");
            File f(path.parent_path()/file);total+=f.size();
            if(f.size()!=uint64_t(tokens)*512*4 || total>16*MiB) throw std::invalid_argument("route fixture exceeds bounds");
            auto x=gpu.allocate(f.size());f.read(0,{x->data,size_t(x->bytes)});
            if(hash(x)!=c.at("sha256").get<std::string>()) throw std::invalid_argument("changed router logits");
            auto ids=gpu.allocate(tokens*40),weights=gpu.allocate(tokens*40);
            auto reference_ids=gpu.allocate(tokens*40),reference_weights=gpu.allocate(tokens*40);
            gpu.configure({});gpu.route(x,reference_ids,reference_weights,tokens);gpu.finish();
            for(uint32_t t=0;t<tokens;++t) {
                std::array<int,512> cpu;std::iota(cpu.begin(),cpu.end(),0);
                for(uint32_t e=0;e<512;++e) if(!std::isfinite(x->floats()[t*512+e])) throw std::runtime_error("non-finite captured route score");
                std::stable_sort(cpu.begin(),cpu.end(),[&](int a,int b){return x->floats()[t*512+a]>x->floats()[t*512+b];});
                for(uint32_t j=0;j<10;++j) if(cpu[j]!=reinterpret_cast<int*>(reference_ids->data)[t*10+j])
                    throw std::runtime_error("reference differs from independent stable CPU ranking");
            }
            // Warm both pipelines; allocation and capture are outside all timings.
            for(const std::string variant:{"serial","simd"}) {
                KernelConfig config;config.policy="candidate";config.route_selection=variant;gpu.configure(config);
                gpu.route(x,ids,weights,tokens);gpu.finish();
                if(!equal(ids,reference_ids) || !equal(weights,reference_weights)) throw std::runtime_error("routing mismatch before timing");
            }
            for(int rep=0;rep<10;++rep) for(int arm=0;arm<2;++arm) {
                const std::string variant=(arm^(rep%2))?"simd":"serial";
                KernelConfig config;config.policy="candidate";config.route_selection=variant;gpu.configure(config);
                const auto start=monotonic_ns();
                for(int i=0;i<8;++i) gpu.route(x,ids,weights,tokens);
                const auto completion=gpu.submit();gpu.finish();
                const auto wall=monotonic_ns()-start;
                const auto duration=uint64_t(std::max(0.0,completion->gpu_end-completion->gpu_start)*1e9);
                const bool matches=equal(ids,reference_ids) && equal(weights,reference_weights);exact&=matches;
                measurements.push_back({{"case",c.at("sha256")},{"tokens",tokens},{"repetition",rep},{"variant",variant},
                    {"dispatches",8},{"wall_ns",wall},{"gpu_ns",duration},{"exact",matches}});
            }
            checks.push_back({{"case",c.at("sha256")},{"cpu_ranking_exact",true},
                {"ids_sha256",hash(reference_ids)},{"weights_sha256",hash(reference_weights)}});
        }
        Json report={{"kind","captured_router_operator_check_v1"},{"complete",true},{"exact",exact},
            {"source",manifest},{"machine",gpu.statistics()},{"measurements",measurements},{"checks",checks},
            {"normal_request_latency_qualified",false},{"production_promoted",false}};
        std::ofstream out(argv[2]);out<<report.dump(2)<<'\n';if(!out) throw std::runtime_error("cannot write route report");
        std::println("{} captured routing cases: {}",checks.size(),exact?"exact":"mismatch");return exact?0:1;
    } catch(const std::exception& error) {std::println(stderr,"route check: {}",error.what());return 1;}
}
