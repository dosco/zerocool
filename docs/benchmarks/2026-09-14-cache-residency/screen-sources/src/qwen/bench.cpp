#include "qwen/bench.hpp"
#include "qwen/pipeline.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>

namespace freellm::qwen {
std::vector<Json> read_route_trace(const std::filesystem::path& path,Artifact artifact) {
    std::istringstream f(read_text(path,32*MiB));
    std::vector<Json> trace;std::string line;
    auto integer=[](const Json& value,uint64_t limit) {
        if(!value.is_number_integer() || (!value.is_number_unsigned() && value.get<int64_t>()<0) ||
           value.get<uint64_t>()>limit) throw std::runtime_error("invalid route trace integer");
        return value.get<uint64_t>();
    };
    while(std::getline(f,line)) {
        if(line.size()>4*MiB || trace.size()>=4096) throw std::runtime_error("route replay exceeds metadata bound");
        auto row=Json::parse(line);
        const auto tokens=integer(row.at("tokens"),8192);
        integer(row.at("offset"),8192-tokens);integer(row.at("layer"),Layers-1);
        if(tokens<1 || !row.at("routes").is_array() ||
           row.at("artifact_revision")!=artifact_revision(artifact) || row.at("routes").size()!=uint64_t(tokens)*TopK)
            throw std::runtime_error("route trace artifact or geometry mismatch");
        for(size_t t=0;t<size_t(tokens);++t) {
            std::set<uint64_t> unique;
            for(size_t k=0;k<TopK;++k) if(!unique.insert(integer(row.at("routes").at(t*TopK+k),Experts-1)).second)
                throw std::runtime_error("invalid recorded top-10 selection");
        }
        trace.push_back(std::move(row));
    }
    if(trace.size()<Layers || trace.size()%Layers) throw std::runtime_error("replay requires complete 48-layer passes");
    for(size_t i=0;i<trace.size();++i) {
        const auto& first=trace[i/Layers*Layers];
        if(trace[i].at("layer")!=i%Layers || trace[i].at("tokens")!=first.at("tokens") ||
           trace[i].at("offset")!=first.at("offset") || trace[i].value("build","")!=trace.front().value("build",""))
            throw std::runtime_error("route trace is not a consistent 48-layer pass");
    }
    return trace;
}
Json dependency_bench(const Options& o,const std::filesystem::path& path,int repetitions,int hit_count) {
    if(hit_count<-1 || hit_count>TopK) throw std::invalid_argument("replay hits must be 0..10, or -1 for recorded ready hits");
    const auto trace=read_route_trace(path,o.artifact);
    Checkpoint cp(o.model,true,o.artifact);Metal gpu;ReadPool reads(o.io_workers);
    auto prepared=o.prepared.empty()?nullptr:std::make_shared<PreparedArtifact>(o.prepared,cp);
    ExpertStore store(cp,prepared);
    // Replay does not load the resident trunk or session state and cannot
    // qualify normal request latency. Still honor available memory/headroom.
    const auto available=available_memory();
    if(available<=GiB+GiB/2+128*MiB) throw std::runtime_error("insufficient memory for dependency replay");
    gpu.budget(std::min({o.memory,gpu.recommended(),available-GiB-GiB/2}));
    ExpertCache cache(32,[&](auto n){return gpu.allocate(n);},reads,[&](auto k,const auto& b){store.read(k,b);});
    auto x=gpu.zeros(Hidden);for(int i=0;i<Hidden;++i) x->floats()[i]=round_bf16(float(i%17-8)/32);
    Json runs=Json::array();
    for(int rep=0;rep<repetitions;++rep) for(size_t pass=0;pass<trace.size()/Layers;++pass) {
        Json layers=Json::array();uint64_t total=0;
        for(int l=0;l<Layers;++l) {
            cache.clear();const auto& row=trace[pass*Layers+l];
            const auto raw=row.at("routes").get<std::vector<int>>();
            std::vector<ExpertKey> selected;std::set<uint32_t> unique;
            for(size_t i=raw.size()-TopK;i<raw.size();++i) {
                if(raw[i]<0 || raw[i]>=Experts || !unique.insert(uint32_t(raw[i])).second)
                    throw std::runtime_error("invalid recorded top-10 selection");
                selected.push_back({uint32_t(l),uint32_t(raw[i])});
            }
            std::set<uint32_t> warm;
            if(hit_count>=0) for(int i=0;i<hit_count;++i) warm.insert(selected[i].expert);
            else for(const auto& r:row.at("records")) if(r.at("acquisition")=="ready_hit") warm.insert(r.at("expert").get<uint32_t>());
            for(auto k:selected) if(warm.contains(k.expert)) {auto lease=cache.acquire(k);lease.wait();}
            auto before=disk_counters();const auto bytes_before=cp.bytes_read()+(prepared?prepared->bytes_read():0);
            std::array<Buf,TopK> outputs;
            auto encode=[&](ExpertKey k,const Buf& record){
                auto activated=gpu.gated_linear(expert_linear(record,0),expert_linear(record,1),x,1);
                const auto it=std::find(selected.begin(),selected.end(),k);
                outputs[size_t(it-selected.begin())]=gpu.linear(expert_linear(record,2),activated,1);
            };
            auto result=o.completion_pipeline?execute_experts(selected,cache,reads,gpu,o.ready_group,encode,nullptr,true):
                execute_experts_batched(selected,cache,reads,gpu,encode);
            CC_SHA256_CTX hash;CC_SHA256_Init(&hash);
            double checksum=0;for(const auto& output:outputs) {
                for(auto v:output->floats()) checksum+=v;
                CC_SHA256_Update(&hash,output->data,CC_LONG(output->bytes));
            }
            if(!std::isfinite(checksum)) throw std::runtime_error("non-finite expert replay");
            unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256_Final(digest,&hash);
            std::string hex;constexpr char digits[]="0123456789abcdef";
            for(auto c:digest) {hex+=digits[c>>4];hex+=digits[c&15];}result["output_sha256"]=hex;
            result["output_checksum"]=checksum;result["layer"]=l;result["routes"]=std::vector<int>(raw.end()-TopK,raw.end());
            result["application_read_bytes"]=cp.bytes_read()+(prepared?prepared->bytes_read():0)-bytes_before;
            result["storage_before"]=before;result["storage_after"]=disk_counters();
            total+=result["duration_ns"].get<uint64_t>();layers.push_back(std::move(result));
        }
        runs.push_back({{"repetition",rep},{"recorded_pass",pass},{"expert_dependency_ns",total},
            {"token_budget_5tps_ns",200000000},{"token_budget_8tps_ns",125000000},{"layers",std::move(layers)}});
    }
    return {{"kind","expert_dependency_replay"},{"normal_generation_benchmark",false},
        {"artifact_revision",cp.revision()},{"machine",gpu.statistics()},
        {"prepared",prepared?prepared->inspect():Json(nullptr)},{"route_source",path.string()},
        {"route_source_build",trace.front().value("build","")},{"ready_group",o.ready_group},
        {"completion_pipeline",o.completion_pipeline},{"io_workers",o.io_workers},{"hit_count_override",hit_count},{"runs",std::move(runs)},
        {"note","Last token of each recorded pass; actual Q4 expert GPU work with fixed BF16 input. Ready hits are seeded before each layer outside timing. Attention, router, shared experts, ngrams, and full request latency are excluded. Device counters include other processes."}};
}
}
