#include "engine/bench.hpp"
#include <algorithm>
#include <cstring>
#include <CommonCrypto/CommonDigest.h>

namespace zerocool::engine {
namespace {
Linear load_linear(const Checkpoint& cp,Metal& gpu,const std::string& name) {
    const auto& q=cp.config.at("quantization");
    const auto format=q.value(name,Json::object());
    const auto bits=format.value("bits",q.at("bits").get<uint32_t>());
    const auto group=format.value("group_size",q.at("group_size").get<uint32_t>());
    const auto& w=cp.at(name+".weight");
    auto load=[&](const std::string& suffix){return cp.load(name+suffix,[&](uint64_t n){return gpu.allocate(n);});};
    return {{load(".weight")},{load(".scales")},{load(".biases")},uint32_t(w.shape[1])*32/bits,uint32_t(w.shape[0]),group,0,true,bits};
}
Json timed(Metal& gpu,const std::function<Buf()>& encode,Buf& output) {
    const auto start=monotonic_ns();output=encode();auto c=gpu.submit();gpu.finish();
    return {{"wall_ns",monotonic_ns()-start},{"gpu_ns",uint64_t(std::max(0.0,c->gpu_end-c->gpu_start)*1e9)}};
}
bool equal(const Buf& a,const Buf& b) {return a->bytes==b->bytes && std::memcmp(a->data,b->data,a->bytes)==0;}
}
Json kernel_bench(const Options& o,int repetitions) {
    Checkpoint cp(o.model,true,o.artifact);Metal gpu;
    gpu.budget(512*MiB);gpu.prepare_pipelines();
    std::shared_ptr<PreparedArtifact> prepared=o.prepared.empty()?nullptr:std::make_shared<PreparedArtifact>(o.prepared,cp);
    ExpertStore store(cp,prepared);auto expert=gpu.allocate(ExpertBytes);store.read({0,0},expert);
    Json measurements=Json::array();bool exact=true;
    for(const auto& name:{"model.layers.0.linear_attn.in_proj_qkv","model.layers.0.linear_attn.in_proj_z","expert_gate","expert_down"}) {
        const std::string key=name;
        auto l=key=="expert_gate"?expert_linear(expert,0):key=="expert_down"?expert_linear(expert,2):load_linear(cp,gpu,key);
        auto up=key=="expert_gate"?expert_linear(expert,1):l;
        for(uint32_t T:{1u,2u,4u,8u,16u,32u,64u,128u}) {
            auto x=gpu.zeros(uint64_t(T)*l.input),rows=gpu.allocate(T*4);
            for(size_t i=0;i<x->floats().size();++i) x->floats()[i]=round_bf16(float(int(i%47)-23)/256);
            for(uint32_t t=0;t<T;++t) reinterpret_cast<int*>(rows->data)[t]=int((T-1-t)/2);
            for(bool fused:{false,true}) {
                if(fused && key!="expert_gate") continue;
                auto encode=[&]{return fused?gpu.gated_linear(l,up,x,T,rows):gpu.linear(l,x,T);};
                gpu.configure({});auto reference=encode();gpu.finish();
                for(int rep=0;rep<repetitions;++rep) {
                    std::array<uint32_t,4> tiles={1,2,4,8};if(rep%2) std::reverse(tiles.begin(),tiles.end());
                    for(auto tile:tiles) {
                        KernelConfig c;if(tile!=1){c.policy="candidate";c.token_tile=tile;}gpu.configure(c);
                        Buf out;auto result=timed(gpu,encode,out);const auto matches=equal(out,reference);exact&=matches;
                        result.update({{"kind","linear"},{"name",name},{"tokens",T},{"K",l.input},{"N",l.output},
                            {"bits",l.bits},{"group",l.group},{"fused_gathered",fused},{"tile",tile},{"repetition",rep},{"exact",matches}});
                        measurements.push_back(std::move(result));
                    }
                }
            }
        }
    }
    auto alog=cp.load("model.layers.0.linear_attn.A_log",[&](uint64_t n){return gpu.allocate(n);});
    auto dt=cp.load("model.layers.0.linear_attn.dt_bias",[&](uint64_t n){return gpu.allocate(n);});
    auto dtype=[&](const std::string& key){const auto& d=cp.at(key).dtype;return d=="F32"?1u:d=="F16"?2u:0u;};
    for(uint32_t T:{1u,8u,32u,128u}) {
        auto q=gpu.zeros(T*10240),a=gpu.zeros(T*48),b=gpu.zeros(T*48),initial=gpu.zeros(48*128*128);
        for(size_t i=0;i<q->floats().size();++i) q->floats()[i]=round_bf16(float(int(i%29)-14)/128);
        for(size_t i=0;i<a->floats().size();++i) {a->floats()[i]=round_bf16(float(int(i%11)-5)/4);b->floats()[i]=round_bf16(float(int(i%13)-6)/4);}
        for(size_t i=0;i<initial->floats().size();++i) initial->floats()[i]=float(int(i%31)-15)/4096;
        auto state=gpu.upload(initial->floats());
        auto encode=[&]{return gpu.gdn_scan(q,a,b,alog,dt,state,T,dtype("model.layers.0.linear_attn.A_log"),dtype("model.layers.0.linear_attn.dt_bias"));};
        gpu.configure({});auto ref=encode();gpu.finish();auto expected=gpu.upload(state->floats());
        std::vector<KernelConfig> configs(1);
        KernelConfig pre;pre.policy="candidate";pre.gdn="precompute";configs.push_back(pre);
        for(uint32_t rows:{4u,8u}) for(uint32_t block:{4u,8u,16u}) {auto c=pre;c.gdn="staged";c.gdn_rows=rows;c.gdn_block=block;configs.push_back(c);}
        for(int rep=0;rep<repetitions;++rep) {
            if(rep) std::reverse(configs.begin(),configs.end());
            for(const auto& c:configs) {
                std::memcpy(state->data,initial->data,state->bytes);gpu.configure(c);
                Buf out;auto result=timed(gpu,encode,out);const auto matches=equal(out,ref)&&equal(state,expected);exact&=matches;
                result.update({{"kind","gdn"},{"tokens",T},{"config",c.json()},{"repetition",rep},{"exact",matches}});
                measurements.push_back(std::move(result));
            }
        }
    }
    if(!exact) throw std::runtime_error("real-weight kernel screening failed bitwise equality");
    return {{"kind","isolated_real_weight_kernel_screen"},{"artifact_revision",cp.revision()},{"machine",gpu.statistics()},
        {"prepared",prepared?prepared->inspect():Json(nullptr)},{"repetitions",repetitions},{"exact",exact},
        {"normal_request_latency_qualified",false},{"inputs","deterministic BF16 activations; real pinned weights and GDN parameters"},
        {"measurements",measurements}};
}
Json fixture_kernel_bench(const Options& o,int repetitions) {
    const auto manifest=read_json(o.operator_fixtures);
    Metal gpu;gpu.budget(512*MiB);gpu.prepare_pipelines();
    if(manifest.at("kind")!="captured_affine_operators" || manifest.at("artifact_revision")!=artifact_revision(o.artifact) ||
       manifest.at("build_fingerprint")!=gpu.statistics().at("build_fingerprint") ||
       manifest.at("cases").empty() || manifest.at("cases").size()>32 || repetitions<1 || repetitions>20)
        throw std::invalid_argument("operator fixture identity or coverage mismatch");
    Json measurements=Json::array();uint64_t total=0;
    for(const auto& item:manifest.at("cases")) {
        const auto& m=item.at("matrix");const auto K=m.at("K").get<uint32_t>(),N=m.at("N").get<uint32_t>(),T=m.at("rows").get<uint32_t>();
        if(!K || !N || !T || T>8192) throw std::invalid_argument("invalid fixture geometry");
        if(o.kernels.q8_decode_rows && (T!=1 || m.at("bits")!=8 || m.at("fused").get<bool>())) continue;
        auto load=[&](const char* key) {
            const auto& entry=item.at("tensors").at(key);const auto name=entry.at("file").get<std::string>();
            const auto bytes=entry.at("bytes").get<uint64_t>();
            if(name.empty() || std::filesystem::path(name).filename()!=name || !bytes || bytes>256*MiB || total>256*MiB-bytes)
                throw std::invalid_argument("operator fixture exceeds bounds");
            total+=bytes;File file(o.operator_fixtures.parent_path()/name);
            if(file.size()!=bytes) throw std::invalid_argument("operator fixture file size differs");
            auto b=gpu.allocate(bytes);file.read(0,{b->data,size_t(bytes)});
            unsigned char hash[CC_SHA256_DIGEST_LENGTH];CC_SHA256(b->data,CC_LONG(bytes),hash);
            std::string hex;constexpr char digits[]="0123456789abcdef";for(auto c:hash){hex+=digits[c>>4];hex+=digits[c&15];}
            if(hex!=entry.at("sha256").get<std::string>()) throw std::invalid_argument("operator fixture hash differs");return b;
        };
        Linear l{{load("w")},{load("s")},{load("b")},K,N,m.at("group").get<uint32_t>(),0,true,m.at("bits").get<uint32_t>()};
        auto up=l;const bool fused=m.at("fused").get<bool>(),gathered=m.at("gathered").get<bool>();
        if(fused) {up.weight={load("uw")};up.scales={load("us")};up.biases={load("ub")};}
        auto x=load("x"),rows=gathered?load("rows"):Buf{};
        auto encode=[&]{return fused?gpu.gated_linear(l,up,x,T,rows):gpu.linear(l,x,T);};
        gpu.configure({});auto reference=encode();gpu.finish();
        std::vector<KernelConfig> variants;
        for(uint32_t tile:{1u,2u,4u,8u}) {KernelConfig c;c.policy="candidate";c.token_tile=tile;variants.push_back(c);}
        if(!fused && l.bits==8 && T>1) for(uint32_t tile:{4u,8u}) for(uint32_t rows:{2u,4u}) {
            KernelConfig c;c.policy="candidate";c.token_tile=tile;c.affine_rows=rows;variants.push_back(c);
        }
        if(T==1) {KernelConfig c;c.policy="candidate";if(fused) c.gate_pair=true;else c.affine_rows=2;variants.push_back(c);}
        if(o.kernels.q8_decode_rows) {
            KernelConfig baseline;baseline.policy="candidate";baseline.affine_rows=2;
            auto candidate=baseline;candidate.q8_decode_rows=o.kernels.q8_decode_rows;
            variants={baseline,candidate};
        }
        for(int rep=0;rep<repetitions;++rep) {
            if(rep) std::reverse(variants.begin(),variants.end());
            for(const auto& c:variants) {
                gpu.configure(c);Buf out;auto result=timed(gpu,encode,out);
                if(!equal(reference,out)) throw std::runtime_error("captured operator changed output bits");
                result.update({{"matrix",m},{"tile",c.token_tile},{"output_rows",c.affine_rows},{"gate_pair",c.gate_pair},
                    {"q8_decode_rows",c.q8_decode_rows},
                    {"repetition",rep},{"exact",true},{"case",item.at("tensors").at("x").at("sha256")}});
                measurements.push_back(result);
            }
        }
    }
    if(measurements.empty()) throw std::runtime_error("no matching operator fixtures");
    return {{"kind",o.kernels.q8_decode_rows?"captured_q8_decode_screen":"captured_operator_screen"},{"artifact_revision",artifact_revision(o.artifact)},
        {"build_fingerprint",gpu.statistics()["build_fingerprint"]},{"source",manifest},{"exact",true},
        {"normal_request_latency_qualified",false},{"measurements",measurements}};
}

}
