#include "qwen/model.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <cstring>
#include <fstream>
#include <print>
using namespace freellm::qwen;
namespace {
std::string hash(const Buf& b) {
    unsigned char d[32];CC_SHA256(b->data,CC_LONG(b->bytes),d);std::string s;
    constexpr char hex[]="0123456789abcdef";for(auto c:d){s+=hex[c>>4];s+=hex[c&15];}return s;
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=3) throw std::invalid_argument("usage: qwen_sparse_replay MANIFEST REPORT");
        const std::filesystem::path path=argv[1];auto manifest=read_json(path);Metal gpu;gpu.budget(GiB);
        const auto build=gpu.statistics()["build_fingerprint"];
        if(manifest.at("kind")!="sparse_attention_fixture_v1" || manifest.at("build_fingerprint")!=build ||
           (manifest.at("artifact_revision")!=artifact_revision(Artifact::Q4) && manifest.at("artifact_revision")!=artifact_revision(Artifact::Mixed)) ||
           manifest.at("cases").empty() || manifest.at("cases").size()>16 || manifest.at("bytes").get<uint64_t>()>256*MiB)
            throw std::invalid_argument("incompatible sparse fixture identity or bounds");
        gpu.prepare_pipelines();Json results=Json::array();uint64_t bytes=0;
        for(const auto& c:manifest.at("cases")) {
            // The previous case ends with an instrumented pass. Validation work
            // for this case must not leak into the next arm's measurements.
            gpu.configure({});gpu.take_profile();
            const auto T=c.at("tokens").get<uint32_t>(),offset=c.at("offset").get<uint32_t>(),L=c.at("length").get<uint32_t>();
            if(!T || T>256 || L<=2048 || L>8192 || uint64_t(offset)+T!=L) throw std::invalid_argument("invalid sparse fixture geometry");
            auto load=[&](const char* name,uint64_t expected) {
                const auto& meta=c.at("tensors").at(name);const auto filename=meta.at("file").get<std::string>();
                if(std::filesystem::path(filename).filename()!=filename || meta.at("bytes")!=expected || expected>256*MiB-bytes)
                    throw std::invalid_argument("invalid sparse fixture tensor");
                File file(path.parent_path()/filename);if(file.size()!=expected) throw std::runtime_error("sparse fixture size mismatch");
                auto b=gpu.allocate(expected);file.read(0,{b->data,size_t(expected)});bytes+=expected;
                if(hash(b)!=meta.at("sha256").get<std::string>()) throw std::runtime_error("sparse fixture hash mismatch");return b;
            };
            auto q=load("q",uint64_t(T)*6144*4),k=load("keys",uint64_t(L)*512*4),v=load("values",uint64_t(L)*512*4);
            auto qg=load("qg",uint64_t(T)*12288*4),index=load("index_scores",uint64_t(T)*(L/4)*4);
            auto status=gpu.zeros(1,AllocationClass::State),mask=gpu.allocate(uint64_t(T)*L),cpu=gpu.allocate(uint64_t(T)*L);
            auto selected=sparse_mask(index->floats(),T,offset,L);std::memcpy(cpu->data,selected.data(),selected.size());
            gpu.sparse_select(index,mask,status,T,offset,L);gpu.finish();
            if(*reinterpret_cast<uint32_t*>(status->data) || std::memcmp(mask->data,cpu->data,cpu->bytes)) throw std::runtime_error("sparse mask mismatch");
            uint64_t total=0,skipped=0;
            for(uint32_t t=0;t<T;t+=8) for(uint32_t b=0;b<L;b+=8) {
                bool visible=false;++total;
                for(uint32_t i=t;i<std::min(T,t+8);++i) for(uint32_t j=b;j<std::min(L,b+8);++j)
                    visible|=j<=offset+i && bool(cpu->data[uint64_t(i)*L+j]);
                skipped+=!visible;
            }
            auto scores=gpu.zeros(uint64_t(T)*24*L),out=gpu.zeros(uint64_t(T)*6144);
            std::array<std::string,3> expected;Json observations=Json::array();
            for(int arm=0;arm<3;++arm) {
                KernelConfig config;config.attention_score_tiles=arm==2?"skip-masked":"full";gpu.configure(config);
                if(arm) gpu.sparse_select(index,mask,status,T,offset,L);
                gpu.attention_scores(q,k,arm?mask:cpu,scores,T,offset,L,true);gpu.finish();const auto sh=hash(scores);
                gpu.dispatch("attention_softmax",{{scores}},{L},T*24*32);gpu.finish();const auto ph=hash(scores);
                gpu.dispatch("attention_values",{{scores},{v},{qg},{out}},{T,L},32*32,(T+7)/8,24);gpu.finish();
                const std::array<std::string,3> hashes={sh,ph,hash(out)};
                if(!arm) expected=hashes;else if(hashes!=expected) throw std::runtime_error("sparse attention arithmetic changed");
                // Separate instrumented pass, not a normal latency measurement.
                config.profile=true;config.counter_profile=true;gpu.configure(config);gpu.label("attention",c.at("layer"),T,offset);
                if(arm) gpu.sparse_select(index,mask,status,T,offset,L);
                gpu.attention_scores(q,k,arm?mask:cpu,scores,T,offset,L,true);
                gpu.dispatch("attention_softmax",{{scores}},{L},T*24*32);
                gpu.dispatch("attention_values",{{scores},{v},{qg},{out}},{T,L},32*32,(T+7)/8,24);gpu.finish();
                if(hash(out)!=expected[2]) throw std::runtime_error("profile pass arithmetic changed");
                observations.push_back({{"arm",arm},{"hashes",hashes},{"profile",gpu.take_profile()}});
            }
            results.push_back({{"layer",c["layer"]},{"phase",c["phase"]},{"tokens",T},{"length",L},
                {"score_tiles",total},{"wholly_masked_tiles",skipped},{"arms",observations},{"exact",true}});
        }
        if(bytes!=manifest.at("bytes")) throw std::runtime_error("sparse fixture total bytes mismatch");
        Json report={{"kind","sparse_attention_replay_v1"},{"build_fingerprint",build},{"artifact_revision",manifest["artifact_revision"]},
            {"passed",true},{"normal_request_latency_qualified",false},{"cases",results}};
        std::ofstream f(argv[2]);f<<report.dump(2)<<'\n';if(!f) throw std::runtime_error("cannot write sparse replay report");
        std::println("Sparse attention: {} real cases exact",results.size());
    } catch(const std::exception& e) {std::println(stderr,"sparse replay: {}",e.what());return 1;}
}
