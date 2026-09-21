#include "engine/pipeline.hpp"
#include <algorithm>
#include <CommonCrypto/CommonDigest.h>
#include <cstring>
#include <fstream>
#include <print>

using namespace zerocool::engine;
// Explicit real-asset diagnostic: missing files fail. This checks every routed
// projection and ngram decoding, but is not a full-model or latency release gate.
int main(int argc,char** argv) {
    try {
        if(argc!=5 && argc!=6) throw std::invalid_argument("usage: qwen_storage_check MODEL PREPARED FIVE_TOKEN_TRACE REPORT [q4-control|mixed-4_8bit]");
        const std::string artifact=argc==6?argv[5]:"q4-control";
        if(artifact!="q4-control" && artifact!="mixed-4_8bit") throw std::invalid_argument("unknown artifact");
        Checkpoint cp(argv[1],true,artifact=="q4-control"?Artifact::Q4:Artifact::Mixed);
        auto prepared=std::make_shared<PreparedArtifact>(argv[2],cp);
        Metal gpu;gpu.budget(192*MiB);ReadPool reads(8);
        ExpertStore original(cp),store(cp,prepared);
        ExpertCache cache(32,[&](auto n){return gpu.allocate(n);},reads,[&](auto k,const auto& b){store.read(k,b);});
        std::filesystem::path trace(argv[3]);Json fixture_hashes=Json::object();
        auto load=[&](const std::string& name) {File f(trace/(name+".bin"));auto b=gpu.allocate(f.size());f.read(0,{b->data,size_t(b->bytes)});
            unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256(b->data,CC_LONG(b->bytes),digest);
            std::string hex;constexpr char digits[]="0123456789abcdef";for(auto c:digest) {hex+=digits[c>>4];hex+=digits[c&15];}
            fixture_hashes[name]=hex;return b;};
        auto upload=[&](std::span<const int> values) {auto b=gpu.allocate(values.size_bytes());std::memcpy(b->data,values.data(),values.size_bytes());return b;};
        auto control=Buffer::host(ExpertBytes);uint64_t records=0;Json layers=Json::array();
        for(uint32_t l=0;l<Layers;++l) {
            auto x=load("x2_"+std::to_string(l)),route=load("route_"+std::to_string(l));
            if(x->bytes!=5*Hidden*4 || route->bytes!=5*TopK*4) throw std::runtime_error("expected five-token fixture");
            const auto raw=reinterpret_cast<const int*>(route->data);
            std::array<std::vector<int>,Experts> positions;
            for(int i=0;i<5*TopK;++i) {
                if(raw[i]<0 || raw[i]>=Experts) throw std::runtime_error("invalid fixture expert");
                positions[raw[i]].push_back(i);
            }
            std::vector<ExpertKey> keys;for(uint32_t e=0;e<Experts;++e) if(!positions[e].empty()) keys.push_back({l,e});
            auto output=gpu.allocate(5*TopK*Hidden*4);
            auto timing=execute_experts(keys,cache,reads,gpu,4,[&](ExpertKey k,const Buf& record) {
                original.read(k,control);
                if(std::memcmp(control->data,record->data,ExpertBytes)) throw std::runtime_error("prepared expert differs from source bytes");
                ++records;
                const auto& pos=positions[k.expert];std::vector<int> rows;for(auto p:pos) rows.push_back(p/TopK);
                auto rowbuf=upload(rows),posbuf=upload(pos);
                auto activated=gpu.gated_linear(expert_linear(record,0),expert_linear(record,1),x,uint32_t(pos.size()),rowbuf);
                auto down=gpu.linear(expert_linear(record,2),activated,uint32_t(pos.size()));
                gpu.dispatch("scatter_experts",{{down},{posbuf},{output}},{uint32_t(pos.size())},Hidden,uint32_t(pos.size()));
            });
            auto expected=load("model.layers."+std::to_string(l)+".mlp.expert_out");
            if(expected->bytes!=output->bytes || std::memcmp(expected->data,output->data,output->bytes))
                throw std::runtime_error("expert contribution differs at layer "+std::to_string(l));
            layers.push_back({{"layer",l},{"records",keys.size()},{"contributions_bit_identical",true},
                              {"peak_leases",timing["peak_leases"]},{"peak_gpu_groups",timing["peak_gpu_groups"]}});
        }
        NgramStore a(cp,reads,4*MiB),b(cp,reads,4*MiB,prepared),evicting(cp,reads,1024,prepared);
        std::vector<int> tokens={760,6511,314,9338,369,248044,369,248044,248044,760};
        for(int i=0;i<120;++i) tokens.push_back((i*7919)%248000);
        std::vector<float> expected(tokens.size()*Hidden),actual(expected.size());
        a.embedding(tokens,{248044,248044},expected);
        for(auto* cache_ptr:{&b,&b,&evicting,&evicting}) {
            cache_ptr->embedding(tokens,{248044,248044},actual);
            if(std::memcmp(expected.data(),actual.data(),expected.size()*4)) throw std::runtime_error("prepared/BF16 ngram cache changed values");
        }
        Json report={{"kind","real_prepared_storage_and_moe_check"},{"artifact_revision",cp.revision()},
            {"prepared",prepared->inspect()},{"machine",gpu.statistics()},{"source_records_compared",records},
            {"fixture_sha256",fixture_hashes},{"layers",layers},{"ngram_token_ids",tokens},{"ngram_tokens",tokens.size()},{"ngram_passes",4},{"ngram_bit_identical",true},
            {"full_model_verified",false},{"performance_qualified",false}};
        std::ofstream out(argv[4]);out<<report.dump(2)<<'\n';if(!out) throw std::runtime_error("cannot write report");
        std::println("All 48 MoE layers and ngram cache replays match the control.");
    } catch(const std::exception& e) {std::println(stderr,"storage check: {}",e.what());return 1;}
}
