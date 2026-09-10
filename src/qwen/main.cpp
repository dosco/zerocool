#include "qwen/session.hpp"
#include "qwen/bench.hpp"
#include "qwen/cached_progress.hpp"
#include "qwen/route_trace.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <cmath>
#include <print>
#include <stdexcept>

using namespace freellm::qwen;
namespace {
std::atomic<bool> cancelled{false};
void interrupt(int) {cancelled.store(true);}
double milliseconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
void print_json(const Json& j) {std::println("{}",j.dump(2,' ',false,Json::error_handler_t::replace));}
Json state_identity(const State& state) {
    // The diagnostic caller has completed forward(): all buffers are CPU-visible.
    Json layers=Json::array();
    for(const auto& layer:state.layers) {
        Json buffers=Json::object();
        const std::array<std::pair<const char*,Buf>,6> fields={{{"conv",layer.conv},{"recurrence",layer.recurrence},
            {"keys",layer.keys},{"values",layer.values},{"index",layer.index},{"ple_conv",layer.ple_conv}}};
        for(const auto& [name,buffer]:fields) if(buffer) {
            unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256(buffer->data,CC_LONG(buffer->bytes),digest);
            std::string hex;constexpr char digits[]="0123456789abcdef";
            for(auto c:digest) {hex+=digits[c>>4];hex+=digits[c&15];}
            buffers[name]={{"bytes",buffer->bytes},{"sha256",hex}};
        }
        buffers["position"]=layer.position;layers.push_back(std::move(buffers));
    }
    return {{"tokens",state.tokens},{"history",state.history},{"valid",state.valid},
        {"artifact_revision",artifact_revision(state.artifact)},{"layers",layers}};
}
Json storage_bench(const std::filesystem::path& path,int repeats) {
    auto file=std::make_shared<File>(path);
    if(file->size()<ExpertStride) throw std::runtime_error("I/O fixture must contain at least one expert-sized record");
    Json cases=Json::array();
    for(const auto size:std::array<uint64_t,2>{ExpertBytes,100}) for(const int workers:{1,4,8,16,32}) {
        for(int rep=0;rep<repeats;++rep) {
            ReadPool pool(workers); std::vector<Buf> buffers;
            for(int i=0;i<workers;++i) buffers.push_back(Buffer::host(size));
            const int count=size==100?4096:256;
            const auto start=std::chrono::steady_clock::now();
            std::vector<std::shared_future<void>> pending;
            for(int lane=0;lane<workers;++lane) pending.push_back(pool.submit([&,lane] {
                std::mt19937_64 rng(1234+lane);
                for(int i=lane;i<count;i+=workers) {
                    const auto off=(rng()%((file->size()-size)/16384+1))*16384;
                    file->read(off,{buffers[lane]->data,size_t(size)});
                }
            }));
            for(auto& f:pending) f.get();
            const auto elapsed=milliseconds(start),bps=double(size)*count*1000/elapsed;
            Json row={{"pattern",size==100?"row_sized_random":"record_sized_random"},{"read_bytes",size},
                {"workers",workers},{"repetition",rep},{"requests",count},{"elapsed_ms",elapsed},
                {"bytes_per_second",bps},{"os_cache_policy","F_NOCACHE"},
                {"physical_ssd_bytes",nullptr}};
            if(size==ExpertBytes) for(int rate:{5,8}) row["optimistic_hit_rate_for_"+std::to_string(rate)+"tps"]=
                std::max(0.0,1-bps/(double(ExpertBytes)*Layers*TopK*rate));
            cases.push_back(std::move(row));
        }
    }
    return {{"kind","storage_probe"},{"file",path.string()},{"cases",cases},
        {"note","Application read throughput; whole-record reads are an optimistic bound for nine-piece experts. Physical SSD traffic is not inferred from pread bytes."}};
}
Json checkpoint_storage_bench(const Options& o,int repeats) {
    Checkpoint cp(o.model,true,o.artifact);
    auto prepared=o.prepared.empty()?nullptr:std::make_shared<PreparedArtifact>(o.prepared,cp);
    auto bytes_read=[&]{return cp.bytes_read()+(prepared?prepared->bytes_read():0);};
    ExpertStore store(cp,prepared);Json cases=Json::array();
    for(int workers:{1,4,8,16,32}) for(int rep=0;rep<repeats;++rep) {
        ReadPool pool(workers);std::vector<Buf> buffers;
        for(int i=0;i<workers;++i) buffers.push_back(Buffer::host(ExpertStride));
        auto before=disk_counters();const auto read_before=bytes_read();
        auto start=std::chrono::steady_clock::now();std::vector<std::shared_future<void>> pending;
        for(int lane=0;lane<workers;++lane) pending.push_back(pool.submit([&,lane]{
            std::mt19937_64 rng(1000+rep);
            // The request sequence is independent of worker count.
            for(int i=0;i<256;++i) {
                ExpertKey k{uint32_t(rng()%Layers),uint32_t(rng()%Experts)};
                if(i%workers==lane) store.read(k,buffers[lane]);
            }
        }));
        for(auto& f:pending) f.get();
        const double elapsed=milliseconds(start),bps=double(bytes_read()-read_before)*1000/elapsed;
        Json row={{"pattern",prepared?"contiguous_real_experts":"nine_piece_real_experts"},{"workers",workers},{"repetition",rep},{"records",256},
            {"application_read_bytes",bytes_read()-read_before},{"elapsed_ms",elapsed},{"bytes_per_second",bps},
            {"storage_before",before},{"storage_after",disk_counters()}};
        for(int rate:{5,8}) row["optimistic_hit_rate_for_"+std::to_string(rate)+"tps"]=std::max(0.0,1-bps/(double(ExpertBytes)*Layers*TopK*rate));
        cases.push_back(row);
        NgramStore ngrams(cp,pool,1024,prepared);std::mt19937_64 rng(1000+rep);
        std::vector<int> tokens(256);for(auto& id:tokens) id=int(rng()%248000);
        std::vector<float> embedding(tokens.size()*Hidden);
        before=disk_counters();const auto ng_before=bytes_read();start=std::chrono::steady_clock::now();
        ngrams.embedding(tokens,{248044,248044},embedding);
        cases.push_back({{"pattern",prepared?"contiguous_real_ngram_rows":"three_piece_real_ngram_rows"},{"workers",workers},{"repetition",rep},
            {"rows",tokens.size()*16},{"application_read_bytes",bytes_read()-ng_before},{"elapsed_ms",milliseconds(start)},
            {"storage_before",before},{"storage_after",disk_counters()}});
    }
    return {{"kind","checkpoint_storage_probe"},{"revision",cp.revision()},{"os_cache_policy","F_NOCACHE"},{"cases",cases},
        {"note","Hit-rate bounds leave zero time for computation and are necessary, not sufficient. Device counters include other processes."}};
}
}
int main(int argc,char** argv) {
    try {
        if(argc<2 || std::string(argv[1])=="--help") {
            std::println("FreeLLM — Qwen3.8-Flash-Next on Apple Silicon\n"
                "  freellm inspect --model DIR\n"
                "  freellm run --model DIR [--prompt TEXT] [--raw]\n"
                "  freellm bench --model DIR --prompt-file FILE [--repetitions 3]\n"
                "  freellm bench --storage [--repetitions 3]\n"
                "  freellm bench --workload-file FILE [--repetitions 3]\n"
                "  freellm bench --io-file FILE [--repetitions 3]\n"
                "  freellm serve --model DIR [--port 8080]\n"
                "Limits: --memory-gb 22 --context 8192 --chunk 128 --io-workers 8\n"
                "Prefill panels: --panel 0|256|512|1024 (0 keeps the existing schedule)\n"
                "Sampling: --max-tokens 256 --temperature 0 --top-p 0.95 --top-k 20 --seed 0\n"
                "Diagnostics: --json FILE --tokens-file FILE --logits-file FILE --tokenize TEXT\n"
                "Layer diagnostics: --probe-layers N --trace-dir DIR [--stream-trunk]\n"
                "Cache/schedule: --expert-slots N --short-append 32 --ready-group 4 [--legacy-schedule]\n"
                "Prepared storage: --prepared DIR; profiling: --dependency-trace FILE\n"
                "  --route-trace FILE records committed routes for normal bench workloads\n"
                "Artifact: --artifact q4-control|mixed-4_8bit with its explicit --model DIR\n"
                "Dependency replay: bench --replay-routes FILE [--replay-hits 0..10]\n"
                "Execution experiments: --residency off|core|core-cache --decode-path reference|direct|grouped\n"
                "  --decode-diagnostics (first 32 decode steps per normal request; instrumented)\n"
                "  --cache-policy clock|slru (experimental; bench/inspect only)\n"
                "  --phase-memory fixed|reclaim --prefill-pipeline serial|double --cached-token-replay (last token is continuation)\n"
                "  --cached-progress FILE writes flushed phase progress for cached replay outside forward timing\n"
                "Kernel experiments (bench only): --kernel-policy reference|auto|candidate --token-tile 1|2|4|8\n"
                "  --q8-decode-rows 0|2|4|8; diagnostic per-pass timing: --dispatch-profile FILE\n"
                "  --kernel-bench measures bounded real-weight operators, not request latency\n"
                "  --soak-seconds 1200 repeats a workload conversation for a memory soak\n"
                "  --gdn-path original|precompute|staged --gdn-rows 4|8 --gdn-block 4|8|16 --phase-profile FILE\n"
                "Model files: scripts/qwen/download.sh; verify: scripts/qwen/verify_checkpoint.py");
            return 0;
        }
        const std::string command=argv[1];
        if(command!="inspect" && command!="run" && command!="bench" && command!="serve") throw std::invalid_argument("unknown command: "+command);
        Options o; o.model=".cache/models/qwen38-flash-next";
        std::string prompt,json_path,io_path,tokens_path,logits_path,tokenize,render_path,workloads_path;
        std::string replay_routes,phase_profile,cached_progress;int replay_hits=-1,soak_seconds=0;
        bool probe=false,storage=false,kernel_probe=false;
        bool raw=command=="bench",thinking=false; int repetitions=3,port=8080;
        for(int i=2;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--raw") {raw=true;continue;} if(arg=="--chat") {raw=false;continue;}
            if(arg=="--thinking") {thinking=true;continue;}
            if(arg=="--legacy-schedule") {o.completion_pipeline=false;continue;}
            if(arg=="--decode-diagnostics") {o.decode_diagnostics=true;continue;}
            if(arg=="--storage") {storage=true;continue;}
            if(arg=="--cached-token-replay") {o.cached_token_replay=true;continue;}
            if(arg=="--cached-compare") {o.cached_compare=true;continue;}
            if(arg=="--kernel-bench") {kernel_probe=true;continue;}
            if(arg=="--stream-trunk") {o.diagnostic_stream_trunk=true;continue;}
            if(i+1==argc) throw std::invalid_argument("missing value for "+arg);
            const std::string value=argv[++i];
            if(arg=="--model") o.model=value;
            else if(arg=="--artifact") {
                if(value=="q4-control") o.artifact=Artifact::Q4;
                else if(value=="mixed-4_8bit") o.artifact=Artifact::Mixed;
                else throw std::invalid_argument("artifact must be q4-control or mixed-4_8bit");
            }
            else if(arg=="--prepared") o.prepared=value;
            else if(arg=="--memory-gb") {auto n=std::stod(value);if(!std::isfinite(n)||n<=0||n>22)throw std::invalid_argument("memory must be >0 and <=22GiB");o.memory=uint64_t(n*GiB);}
            else if(arg=="--context") o.context=std::stoi(value);
            else if(arg=="--chunk") o.chunk=std::stoi(value);
            else if(arg=="--kernel-policy") o.kernels.policy=value;
            else if(arg=="--affine-rows") o.kernels.affine_rows=std::stoul(value);
            else if(arg=="--gate-pair") {if(value!="on" && value!="off") throw std::invalid_argument("gate pair must be on or off");o.kernels.gate_pair=value=="on";}
            else if(arg=="--token-tile") o.kernels.token_tile=std::stoul(value);
            else if(arg=="--gdn-path") o.kernels.gdn=value;
            else if(arg=="--gdn-rows") o.kernels.gdn_rows=std::stoul(value);
            else if(arg=="--gdn-block") o.kernels.gdn_block=std::stoul(value);
            else if(arg=="--shape-policy") o.kernels.shape_table=read_json(value);
            else if(arg=="--operator-capture") {o.kernels.operator_capture=value;o.kernels.profile=true;}
            else if(arg=="--capture-phase") o.kernels.capture_phase=value;
            else if(arg=="--capture-operator") o.kernels.capture_operator=value;
            else if(arg=="--capture-layer") o.kernels.capture_layer=std::stoi(value);
            else if(arg=="--operator-fixtures") o.operator_fixtures=value;
            else if(arg=="--residency") o.residency=value;
            else if(arg=="--decode-path") o.decode_path=value;
            else if(arg=="--prefill-pipeline") o.prefill_pipeline=value;
            else if(arg=="--cache-policy") o.cache_policy=value;
            else if(arg=="--phase-memory") o.phase_memory=value;
            else if(arg=="--phase-profile") {phase_profile=value;o.kernels.profile=true;}
            else if(arg=="--dispatch-profile") {phase_profile=value;o.kernels.profile=true;o.kernels.counter_profile=true;}
            else if(arg=="--sparse-selection") o.sparse_selection=value;
            else if(arg=="--attention-score-tiles") o.kernels.attention_score_tiles=value;
            else if(arg=="--cached-compare-axis") o.cached_compare_axis=value;
            else if(arg=="--cached-progress") cached_progress=value;
            else if(arg=="--sparse-capture") {o.sparse_capture=value;o.kernels.profile=true;}
            else if(arg=="--q8-decode-rows") o.kernels.q8_decode_rows=std::stoul(value);
            else if(arg=="--soak-seconds") soak_seconds=std::stoi(value);
            else if(arg=="--panel") o.panel=std::stoi(value);
            else if(arg=="--short-append") o.short_append=std::stoi(value);
            else if(arg=="--replay-routes") replay_routes=value;
            else if(arg=="--replay-hits") replay_hits=std::stoi(value);
            else if(arg=="--ready-group") o.ready_group=std::stoi(value);
            else if(arg=="--dependency-trace") o.dependency_trace=value;
            else if(arg=="--route-trace") o.route_trace=value;
            else if(arg=="--io-workers") o.io_workers=std::stoi(value);
            else if(arg=="--max-tokens") o.max_tokens=std::stoi(value);
            else if(arg=="--temperature") o.temperature=std::stof(value);
            else if(arg=="--top-p") o.top_p=std::stof(value);
            else if(arg=="--top-k") o.top_k=std::stoi(value);
            else if(arg=="--seed") o.seed=std::stoull(value);
            else if(arg=="--prompt") prompt=value;
            else if(arg=="--prompt-file") prompt=read_text(value);
            else if(arg=="--tokens-file") tokens_path=value;
            else if(arg=="--logits-file") logits_path=value;
            else if(arg=="--tokenize") tokenize=value;
            else if(arg=="--render-chat") render_path=value;
            else if(arg=="--workload-file") workloads_path=value;
            else if(arg=="--trace-dir") o.trace_dir=value;
            else if(arg=="--probe-layers") {o.probe_layers=std::stoi(value);probe=true;}
            else if(arg=="--expert-slots") o.expert_slots=std::stoull(value);
            else if(arg=="--json") json_path=value;
            else if(arg=="--io-file") io_path=value;
            else if(arg=="--repetitions") repetitions=std::stoi(value);
            else if(arg=="--port") port=std::stoi(value);
            else throw std::invalid_argument("unknown option: "+arg);
        }
        if(repetitions<1 || repetitions>20 || port<1 || port>65535) throw std::invalid_argument("invalid repetitions or port");
        o.kernels.validate();
        if(o.sparse_selection!="cpu" && o.sparse_selection!="gpu") throw std::invalid_argument("sparse selection must be cpu or gpu");
        if(o.cached_compare_axis!="q8_decode_rows" && o.cached_compare_axis!="sparse_selection" && o.cached_compare_axis!="attention_score_tiles")
            throw std::invalid_argument("invalid cached comparison axis");
        if(o.decode_diagnostics && (command!="bench" || o.cached_token_replay || o.diagnostic_stream_trunk || o.probe_layers!=Layers || workloads_path.empty()))
            throw std::invalid_argument("decode diagnostics require normal bench --workload-file");
        if(o.cached_compare && (!o.cached_token_replay || o.kernels.profile || repetitions<5 ||
           (o.cached_compare_axis=="q8_decode_rows" && !o.kernels.q8_decode_rows) ||
           (o.cached_compare_axis=="sparse_selection" && o.sparse_selection!="gpu") ||
           (o.cached_compare_axis=="attention_score_tiles" && o.kernels.attention_score_tiles!="skip-masked")))
            throw std::invalid_argument("cached comparison requires an unprofiled candidate and at least five pairs");
        if((o.cache_policy!="clock" || o.residency!="off" || o.decode_path!="reference" || o.prefill_pipeline!="serial" || o.phase_memory!="fixed" || o.cached_token_replay ||
            o.sparse_selection!="cpu" || o.kernels.attention_score_tiles!="full" || !o.sparse_capture.empty()) && command!="bench" && command!="inspect")
            throw std::invalid_argument("execution experiments require bench or inspect");
        if(o.cached_token_replay && ((command!="bench" && command!="inspect") || (command=="bench" && tokens_path.empty()) || kernel_probe || storage || !io_path.empty() || !o.operator_fixtures.empty() || !logits_path.empty() || probe || !replay_routes.empty() || !workloads_path.empty() || !o.trace_dir.empty() || !o.kernels.operator_capture.empty()))
            throw std::invalid_argument("cached replay requires bench --tokens-file and no other diagnostic");
        if(soak_seconds<0 || soak_seconds>86400 || (soak_seconds && (command!="bench" || workloads_path.empty())))
            throw std::invalid_argument("soak requires a benchmark workload and 1..86400 seconds");
        if(!o.route_trace.empty() && (command!="bench" || workloads_path.empty() || repetitions!=1 || soak_seconds ||
            o.cached_token_replay || probe || storage || kernel_probe || !replay_routes.empty() || !logits_path.empty() || !io_path.empty() || !o.operator_fixtures.empty()))
            throw std::invalid_argument("route trace requires one normal bench workload repetition");
        if(!o.route_trace.empty() && !json_path.empty() && std::filesystem::absolute(o.route_trace)==std::filesystem::absolute(json_path))
            throw std::invalid_argument("route trace and benchmark report must use different files");
        if((o.kernels.policy=="candidate" || o.kernels.profile) && command!="bench" && command!="inspect")
            throw std::invalid_argument("experimental kernels and profiling require bench or inspect");
        std::signal(SIGINT,interrupt); std::signal(SIGTERM,interrupt);
        auto emit=[&](const Json& j) {
            if(json_path.empty()) print_json(j);
            else {std::ofstream f(json_path);if(!f)throw std::runtime_error("cannot open report file");f<<j.dump(2,' ',false,Json::error_handler_t::replace)<<'\n';if(!f)throw std::runtime_error("cannot write report file");}
        };
        if(o.residency!="off" && o.residency!="core" && o.residency!="core-cache") throw std::invalid_argument("invalid residency mode");
        if(o.decode_path!="reference" && o.decode_path!="direct" && o.decode_path!="grouped") throw std::invalid_argument("invalid decode path");
        if(o.prefill_pipeline!="serial" && o.prefill_pipeline!="double") throw std::invalid_argument("invalid prefill pipeline");
        (void)parse_cache_policy(o.cache_policy);
        if(o.phase_memory!="fixed" && o.phase_memory!="reclaim") throw std::invalid_argument("invalid phase memory policy");
        if(o.phase_memory=="reclaim" && o.prefill_pipeline!="double") throw std::invalid_argument("phase reclamation requires double prefill");
        if(!cached_progress.empty() && (command!="bench" || !o.cached_token_replay))
            throw std::invalid_argument("cached progress requires bench --cached-token-replay");
        if(!cached_progress.empty() && ((!json_path.empty() && std::filesystem::weakly_canonical(cached_progress)==std::filesystem::weakly_canonical(json_path)) ||
            (!phase_profile.empty() && std::filesystem::weakly_canonical(cached_progress)==std::filesystem::weakly_canonical(phase_profile))))
            throw std::invalid_argument("cached progress must have a separate output path");
        if(o.cached_token_replay && !o.expert_slots) o.expert_slots=480;
        if(o.cached_token_replay && command=="bench") {
            if(!o.expert_slots) o.expert_slots=480;
            auto input=read_json(tokens_path).get<std::vector<int>>();
            if(input.size()<2 || input.size()>size_t(o.context)) throw std::invalid_argument("cached replay requires prompt plus continuation within context");
            std::unique_ptr<CachedProgress> progress;
            if(!cached_progress.empty()) progress=std::make_unique<CachedProgress>(cached_progress,Json{
                {"build",build_fingerprint()},{"artifact_revision",artifact_revision(o.artifact)},
                {"prompt_tokens",input.size()-1},{"budget_bytes",o.memory},{"comparison_axis",o.cached_compare_axis},
                {"paired_comparison",o.cached_compare}});
            try {
                if(progress) progress->begin("model_load");
                Model model(o);
                if(progress) progress->end();
                auto report=model.cached_token_replay(input,repetitions,&cancelled,progress.get());
                if(progress) progress->begin("report_write");
                if(!phase_profile.empty()) {
                    std::ofstream stream(phase_profile);stream<<model.take_profile().dump()<<'\n';
                    if(!stream) throw std::runtime_error("cannot write cached-token profile");
                }
                emit(report);
                if(progress) {progress->end();progress->finish("complete");}
            } catch(const std::exception& error) {
                if(progress) try {progress->finish(cancelled.load()?"interrupted":"failed",error.what());}
                    catch(const std::exception& logging) {std::cerr<<"cached progress: "<<logging.what()<<'\n';}
                throw;
            }
            return 0;
        }
        if(!io_path.empty()) {if(command!="bench")throw std::invalid_argument("--io-file requires bench");emit(storage_bench(io_path,repetitions));return 0;}
        if(!o.operator_fixtures.empty()) {if(command!="bench") throw std::invalid_argument("operator fixtures require bench");emit(fixture_kernel_bench(o,repetitions));return 0;}
        if(kernel_probe) {if(command!="bench")throw std::invalid_argument("--kernel-bench requires bench");emit(kernel_bench(o,repetitions));return 0;}
        if(!replay_routes.empty()) {if(command!="bench")throw std::invalid_argument("--replay-routes requires bench");emit(dependency_bench(o,replay_routes,repetitions,replay_hits));return 0;}
        if(storage) {if(command!="bench")throw std::invalid_argument("--storage requires bench");emit(checkpoint_storage_bench(o,repetitions));return 0;}
        if(!tokenize.empty()) {Tokenizer t(o.model);auto ids=t.encode(tokenize);emit({{"tokens",ids},{"decoded",t.decode(ids)}});return 0;}
        if(!render_path.empty()) {
            Tokenizer t(o.model);auto request=read_json(render_path);
            auto text=t.render(request.at("messages"),request.value("tools",Json::array()),request.value("enable_thinking",false));
            emit({{"text",text},{"tokens",t.encode(text)}});return 0;
        }
        if(probe && (command!="bench" || o.trace_dir.empty() || !logits_path.empty()))
            throw std::invalid_argument("--probe-layers requires bench and --trace-dir, and cannot produce logits");
        if(o.diagnostic_stream_trunk && (command!="bench" || (logits_path.empty() && !probe)))
            throw std::invalid_argument("--stream-trunk is limited to layer/logit diagnostics; it cannot serve or benchmark generation");
        if(command=="inspect") {
            Checkpoint cp(o.model,true,o.artifact);Metal gpu;gpu.residency(o.residency);
            auto report=cp.inspect();
            if(!o.prepared.empty()) report["prepared"]=PreparedArtifact(o.prepared,cp).inspect();
            o.kernels.artifact_revision=cp.revision();gpu.configure(o.kernels);
            auto planned=MemoryPlan::make(o.memory,gpu.physical(),gpu.recommended(),cp.resident_bytes(),o.context,o.chunk,o.panel,Layers,o.kernels.scratch_bytes(o.chunk),o.prefill_pipeline=="double",o.decode_path=="grouped"?2*((uint64_t(o.ready_group)*Intermediate*4+16383)/16384)*16384:0,o.cached_token_replay);planned.cap_experts(o.expert_slots);report["memory_plan"]=planned.json();report["cache_policy"]=o.cache_policy;
            if(o.phase_memory=="reclaim") {report["phase_memory"]={{"policy",o.phase_memory},{"prompt_plan",planned.json()},{"generation_plan",planned.without_prompt_workspaces(o.expert_slots).json()}};report["memory_plan"]=report["phase_memory"]["generation_plan"];}
            report["sparse_selection"]=o.sparse_selection;report["machine"]=gpu.statistics();report["process"]=process_memory();
            const auto available=available_memory();
            try {
                if(available<=GiB+GiB/2) throw std::runtime_error("insufficient currently available memory");
                auto admitted=MemoryPlan::make(std::min(o.memory,available-GiB-GiB/2),gpu.physical(),gpu.recommended(),cp.resident_bytes(),o.context,o.chunk,o.panel,Layers,o.kernels.scratch_bytes(o.chunk),o.prefill_pipeline=="double",o.decode_path=="grouped"?2*((uint64_t(o.ready_group)*Intermediate*4+16383)/16384)*16384:0,o.cached_token_replay);admitted.cap_experts(o.expert_slots);report["current_admission"]=admitted.json();
                if(o.phase_memory=="reclaim") {report["current_phase_admission"]={{"prompt_plan",admitted.json()},{"generation_plan",admitted.without_prompt_workspaces(o.expert_slots).json()}};report["current_admission"]=report["current_phase_admission"]["generation_plan"];}
            } catch(const std::exception& e) {report["current_admission"]={{"error",e.what()},{"reclaimable_bytes",available}};}
            emit(report);return 0;
        }
        std::signal(SIGINT,interrupt); std::signal(SIGTERM,interrupt);
        std::println(stderr,"Loading pinned Qwen trunk; routed experts and ngram rows remain on SSD.");
        const auto initialization_start=monotonic_ns();
        Model model(o);Tokenizer tokenizer(o.model);Session session(model,tokenizer);
        if(command=="bench") model.prepare_pipelines();
        const auto initialization_ns=monotonic_ns()-initialization_start;
        auto flush_profile=[&] {
            if(phase_profile.empty()) return;
            std::ofstream stream(phase_profile);stream<<model.take_profile().dump()<<'\n';
            if(!stream) throw std::runtime_error("cannot write phase profile");
        };
        std::println(stderr,"Expert slots: {}. Total planned memory: {:.2f} GiB.",model.memory_plan().slots,double(model.memory_plan().json()["planned_bytes"].get<uint64_t>())/GiB);
        if(command=="serve") {serve(model,tokenizer,o,uint16_t(port),&cancelled);return 0;}
        if(probe) {
            auto ids=tokens_path.empty()?tokenizer.encode(prompt):read_json(tokens_path).get<std::vector<int>>();
            if(ids.empty() || ids.size()>size_t(o.context)) throw std::invalid_argument("probe requires 1..context prompt tokens");
            auto state=model.make_state();
            for(size_t at=0;at<ids.size();) {
                const auto n=std::min<size_t>(model.input_limit(),ids.size()-at);
                model.forward(std::span<const int>(ids).subspan(at,n),state,false,&cancelled);at+=n;
            }
            emit({{"kind","truncated_real_weight_probe"},{"layers",o.probe_layers},{"tokens",ids},{"full_model_verified",false},{"statistics",model.stats()}});return 0;
        }
        if(!logits_path.empty()) {
            auto ids=tokens_path.empty()?tokenizer.encode(prompt):read_json(tokens_path).get<std::vector<int>>();
            if(ids.empty()) throw std::invalid_argument("logit probe needs prompt tokens");
            auto state=model.make_state(); std::vector<float> logits;
            for(size_t at=0;at<ids.size();) {
                const size_t n=std::min<size_t>(model.input_limit(),ids.size()-at);
                logits=model.forward(std::span<const int>(ids).subspan(at,n),state,at+n==ids.size(),&cancelled);at+=n;
            }
            std::ofstream f(logits_path,std::ios::binary);f.write(reinterpret_cast<const char*>(logits.data()),std::streamsize(logits.size()*4));
            if(!f) throw std::runtime_error("cannot write logit probe");
            emit({{"tokens",ids},{"layers",Layers},{"logits_file",logits_path},
                {"state",state_identity(state)},{"statistics",model.stats()}});return 0;
        }
        if(command=="bench") {
            if(!workloads_path.empty()) {
                const auto cases=read_json(workloads_path);
                if(!cases.is_array() || cases.empty()) throw std::invalid_argument("workload file must be a nonempty array");
                Json runs=Json::array();
                const Json sampling={{"temperature",o.temperature},{"top_k",o.top_k},{"top_p",o.top_p},{"seed",o.seed}};
                const auto soak_start=monotonic_ns();
                for(int rep=0;rep<repetitions || monotonic_ns()-soak_start<uint64_t(soak_seconds)*1000000000ull;++rep) {
                    session.clear();std::vector<int> history;
                    for(const auto& task:cases) {
                        Options settings=o;settings.max_tokens=task.value("max_tokens",o.max_tokens);
                        auto ids=task.at("tokens").get<std::vector<int>>();
                        if(task.value("append",false)) {history.insert(history.end(),ids.begin(),ids.end());ids=history;}
                        else {session.clear();history=ids;}
                        if(auto* trace=model.route_trace()) trace->request_begin(task.at("name"),ids,settings.max_tokens,task.value("prime",false));
                        const auto before=model.stats();auto result=task.value("prime",false)?session.prime(ids,&cancelled):session.generate(ids,settings,{},&cancelled);
                        if(auto* trace=model.route_trace()) trace->request_end(result.tokens,result.finish_reason,result.reused_tokens);
                        auto row=result.json();row["name"]=task.at("name");row["repetition"]=rep;row["before"]=before;row["after"]=model.stats();
                        row["runtime_cache_state"]=runs.empty()?"empty_at_process_start":"retained";
                        row["initialization_ns"]=initialization_ns;row["profiling_enabled"]=o.decode_diagnostics || o.kernels.profile || !o.dependency_trace.empty() || !o.route_trace.empty() || !o.trace_dir.empty();
                        runs.push_back(row);history.insert(history.end(),result.tokens.begin(),result.tokens.end());
                        if(!json_path.empty()) emit({{"model_revision",model.checkpoint().revision()},{"sampling",sampling},{"workloads",cases},{"runs",runs},{"complete",false}});
                        std::println(stderr,"{}: {:.2f} tokens/s, first token {:.1f}s, {} tokens reused",task.at("name").get<std::string>(),row["tokens_per_second"].get<double>(),result.first_token_ms/1000,result.reused_tokens);
                    }
                }
                if(auto* trace=model.route_trace()) trace->finish();
                flush_profile();emit({{"model_revision",model.checkpoint().revision()},{"sampling",sampling},{"workloads",cases},{"runs",runs},{"complete",true},
                    {"soak_seconds_requested",soak_seconds},{"benchmark_elapsed_ns",monotonic_ns()-soak_start}});return 0;
            }
            if(prompt.empty() && tokens_path.empty()) throw std::invalid_argument("bench requires --prompt, --prompt-file, or --tokens-file");
            auto ids=tokens_path.empty()?tokenizer.encode(raw?prompt:tokenizer.render(Json::array({{{"role","user"},{"content",prompt}}}),Json::array(),thinking)):read_json(tokens_path).get<std::vector<int>>();
            Json runs=Json::array();
            for(int rep=0;rep<repetitions;++rep) {
                session.clear(); const auto before=model.stats();
                auto result=session.generate(ids,o,{},&cancelled);auto row=result.json();row["repetition"]=rep;
                row["runtime_cache_state"]=rep==0?"empty_at_process_start":"retained";
                row["initialization_ns"]=initialization_ns;row["profiling_enabled"]=o.decode_diagnostics || o.kernels.profile || !o.dependency_trace.empty() || !o.trace_dir.empty();
                row["before"]=before;row["after"]=model.stats();runs.push_back(row);
                std::println(stderr,"Run {}: {:.2f} tokens/s, first token {:.1f}s",rep+1,row["tokens_per_second"].get<double>(),result.first_token_ms/1000);
            }
            flush_profile();emit({{"model_revision",model.checkpoint().revision()},{"sampling",{{"temperature",o.temperature},{"top_k",o.top_k},{"top_p",o.top_p},{"seed",o.seed}}},
                {"prompt_token_ids",ids},{"runs",runs},{"physical_ssd_bytes",nullptr}});return 0;
        }
        Json messages=Json::array();
        for(;;) {
            if(prompt.empty()) {std::print("you> ");std::fflush(stdout);if(!std::getline(std::cin,prompt) || prompt=="/quit")break;}
            messages.push_back({{"role","user"},{"content",prompt}});
            const auto ids=tokenizer.encode(raw?prompt:tokenizer.render(messages,Json::array(),thinking));
            auto result=session.generate(ids,o,[&](int id){const std::array<int,1> one={id};std::print("{}",tokenizer.decode(one));std::fflush(stdout);},&cancelled);
            std::println("");messages.push_back(parse_output(result.text,Json::array(),thinking));
            if(argc>2 && std::find_if(argv+2,argv+argc,[](const char* a){return std::string(a)=="--prompt" || std::string(a)=="--prompt-file";})!=argv+argc) {
                if(!json_path.empty()) emit(result.json());break;
            }
            prompt.clear();
        }
        return 0;
    } catch(const std::exception& e) {std::println(stderr,"freellm: {}",e.what());return 1;}
}
