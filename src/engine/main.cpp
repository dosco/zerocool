#include "engine/cli.hpp"
#include "engine/server.hpp"
#ifdef ZEROCOOL_WITH_TUI
#include "engine/chat.hpp"
#endif
#include "engine/bench.hpp"
#include "engine/cached_progress.hpp"
#include "engine/fetch.hpp"
#include "engine/route_trace.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <cmath>
#include <print>
#include <stdexcept>

using namespace zerocool::engine;
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
// Acquire and verify a pinned checkpoint without an interpreter. Parsed apart
// from the engine options because neither command opens a model or a GPU.
int checkpoint_main(const std::string& command,int argc,char** argv) {
    std::filesystem::path model;
    auto artifact=Artifact::Q4;
    bool cached=false;
    for(int i=2;i<argc;i++) {
        const std::string arg=argv[i];
        auto value=[&]{ if(i+1>=argc) throw std::invalid_argument("missing value for "+arg); return std::string(argv[++i]); };
        if(arg=="--model") model=value();
        else if(arg=="--artifact") {
            const auto name=value();
            if(name=="q4-control") artifact=Artifact::Q4;
            else if(name=="mixed-4_8bit") artifact=Artifact::Mixed;
            else throw std::invalid_argument("artifact must be q4-control or mixed-4_8bit");
        }
        else if(arg=="--check-receipt" && command=="verify") cached=true;
        else throw std::invalid_argument("unknown option for "+command+": "+arg);
    }
    if(model.empty())
        model=artifact==Artifact::Mixed?".cache/qwen-mixed-reference":".cache/models/qwen38-flash-next";

    uint64_t last=0;
    const ProgressFn progress=[&](const FetchProgress& p) {
        const auto done=p.resumed+p.received;
        const auto now=uint64_t(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
        if(done<p.total && now==last) return;
        last=now;
        std::println("{} {:.1f}/{:.1f} GiB{}",p.name,double(done)/double(GiB),double(p.total)/double(GiB),
                     p.resumed&&p.received?" (resumed)":"");
        std::fflush(stdout);
    };
    const auto receipt=command=="download" ? download_checkpoint(model,artifact,cancelled,progress)
                                           : verify_checkpoint(model,artifact,cached,cancelled,progress);
    std::println("{} {} files at {}",command=="download"?"downloaded and verified":"verified",
                 receipt.at("files").size(),model.string());
    return 0;
}
}
int main(int argc,char** argv) {
    std::unique_ptr<CachedProgress> bench_progress;
    try {
        if(argc<2 || std::string(argv[1])=="--help") {
            std::println("ZeroCool — Qwen3.8-Flash-Next on Apple Silicon\n"
                "  zerocool download [--model DIR] [--artifact q4-control|mixed-4_8bit]\n"
                "  zerocool verify [--model DIR] [--artifact ...] [--check-receipt]\n"
                "  zerocool inspect --model DIR\n"
                "  zerocool run --model DIR [--prompt TEXT] [--raw]\n"
                "  zerocool bench --model DIR --prompt-file FILE [--repetitions 3]\n"
                "  zerocool bench --storage [--repetitions 3]\n"
                "  zerocool bench --workload-file FILE [--repetitions 3]\n"
                "  zerocool bench --io-file FILE [--repetitions 3]\n"
                "  zerocool serve --model DIR [--port 8080]\n"
                "  zerocool chat [--model DIR] [--prepared DIR] [--memory-gb 12]\n"
                "  zerocool chat --connect http://127.0.0.1:8080\n"
                "Chat defaults (relative to current directory):\n"
                "  model: .cache/models/qwen38-flash-next; prepared: .cache/prepared/q4-records-v1\n"
                "  --artifact mixed-4_8bit uses .cache/qwen-mixed-reference unless --model is set\n"
                "  12GiB memory, 8192 context tokens, 256 output tokens, greedy, thinking off\n"
                "Limits: --memory-gb 22 --context 8192 --chunk 128 --io-workers 8\n"
                "Prefill panels: --panel 0|256|512|1024 (0 keeps the existing schedule)\n"
                "Sampling: --max-tokens 256 --temperature 0 --top-p 0.95 --top-k 20 --seed 0\n"
                "Diagnostics: --json FILE --tokens-file FILE --logits-file FILE --tokenize TEXT\n"
                "Layer diagnostics: --probe-layers N --trace-dir DIR [--stream-trunk]\n"
                "Cache/schedule: --expert-slots N --short-append 32 --ready-group 4 [--legacy-schedule]\n"
                "Prepared storage: --prepared DIR; profiling: --dependency-trace FILE\n"
                "  --route-trace FILE records committed routes for normal bench workloads\n"
                "Artifact: --artifact q4-control|mixed-4_8bit (non-chat commands require matching --model DIR)\n"
                "Dependency replay: bench --replay-routes FILE [--replay-hits 0..10]\n"
                "Execution experiments: --residency off|core|core-cache --decode-path reference|direct|grouped\n"
                "  --decode-diagnostics (first 32 decode steps per normal request; instrumented)\n"
                "  --cache-policy clock|slru (experimental; bench/inspect only)\n"
                "  --phase-memory fixed|reclaim --prefill-pipeline serial|double --cached-token-replay (last token is continuation)\n"
                "  --expert-tail wait|overlap (single-token scheduling experiment)\n"
                "  --decode-scratch none|reuse (bounded single-token temporary experiment)\n"
                "  --memory-pressure-policy observe|shrink (experimental cache release)\n"
                "  --profile-decode-only 0|1 (command-group trace excludes ingestion)\n"
                "  --gpu-reference off|resident-q8-v1 --bench-progress FILE (normal workload diagnostics)\n"
                "  --cached-progress FILE writes flushed phase progress for cached replay outside forward timing\n"
                "Kernel experiments (bench only): --kernel-policy reference|auto|candidate --token-tile 1|2|4|8\n"
                "  --q8-decode-rows 0|2|4|8; --q4-decode reference|packed-r2 (candidate policy)\n"
                "  diagnostic per-pass timing: --dispatch-profile FILE\n"
                "  --route-selection serial|simd (simd requires experimental candidate policy)\n"
                "  --kernel-bench measures bounded real-weight operators, not request latency\n"
                "  --soak-seconds 1200 repeats a workload conversation for a memory soak\n"
                "  --gdn-path original|precompute|staged --gdn-rows 4|8 --gdn-block 4|8|16 --phase-profile FILE\n"
                "Model files: scripts/qwen/download.sh; verify: scripts/qwen/verify_checkpoint.py");
            return 0;
        }
        const std::string command=argv[1];
        if(command=="chat") {
#ifdef ZEROCOOL_WITH_TUI
            return chat_main(argc,argv);
#else
            throw std::invalid_argument("this build omits the TUI; use run or serve");
#endif
        }
        if(command=="download" || command=="verify") {
            std::signal(SIGINT,interrupt); std::signal(SIGTERM,interrupt);
            return checkpoint_main(command,argc,argv);
        }
        if(command!="inspect" && command!="run" && command!="bench" && command!="serve") throw std::invalid_argument("unknown command: "+command);
        auto cli=parse_cli(argc,argv);
        validate_cli(cli);
        auto& o=cli.options;
        auto& prompt=cli.prompt;auto& json_path=cli.json_path;auto& io_path=cli.io_path;
        auto& tokens_path=cli.tokens_path;auto& logits_path=cli.logits_path;auto& tokenize=cli.tokenize;
        auto& render_path=cli.render_path;auto& workloads_path=cli.workloads_path;
        auto& replay_routes=cli.replay_routes;auto& phase_profile=cli.phase_profile;
        auto& cached_progress=cli.cached_progress;auto& bench_progress_path=cli.bench_progress_path;
        auto& replay_hits=cli.replay_hits;auto& soak_seconds=cli.soak_seconds;
        auto& repetitions=cli.repetitions;auto& port=cli.port;auto& control_fd=cli.control_fd;
        auto& probe=cli.probe;auto& storage=cli.storage;auto& kernel_probe=cli.kernel_probe;
        auto& raw=cli.raw;auto& thinking=cli.thinking;

        std::signal(SIGINT,interrupt); std::signal(SIGTERM,interrupt);
        auto emit=[&](const Json& j) {
            if(json_path.empty()) print_json(j);
            else {std::ofstream f(json_path);if(!f)throw std::runtime_error("cannot open report file");f<<j.dump(2,' ',false,Json::error_handler_t::replace)<<'\n';if(!f)throw std::runtime_error("cannot write report file");}
        };

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
        if(command=="serve") {serve(o,uint16_t(port),&cancelled,control_fd);return 0;}
        std::println(stderr,"Loading pinned Qwen trunk; routed experts and ngram rows remain on SSD.");
        if(!bench_progress_path.empty()) {
            bench_progress=std::make_unique<CachedProgress>(bench_progress_path,Json{
                {"build",build_fingerprint()},{"artifact_revision",artifact_revision(o.artifact)},
                {"gpu_reference",o.gpu_reference}},"benchmark_progress_v1");
            bench_progress->begin("model_load");
        }
        const auto initialization_start=monotonic_ns();
        Model model(o);Tokenizer tokenizer(o.model);Session session(model,tokenizer);
        if(command=="bench") model.prepare_pipelines();
        const auto initialization_ns=monotonic_ns()-initialization_start;
        if(bench_progress) bench_progress->end();
        auto flush_profile=[&] {
            if(phase_profile.empty()) return;
            std::ofstream stream(phase_profile);stream<<model.take_profile().dump()<<'\n';
            if(!stream) throw std::runtime_error("cannot write phase profile");
        };
        std::println(stderr,"Expert slots: {}. Total planned memory: {:.2f} GiB.",model.memory_plan().slots,double(model.memory_plan().json()["planned_bytes"].get<uint64_t>())/GiB);
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
                Json runs=Json::array(),references=Json::array();
                const Json sampling={{"temperature",o.temperature},{"top_k",o.top_k},{"top_p",o.top_p},{"seed",o.seed}};
                auto write_report=[&](bool complete) {
                    emit({{"model_revision",model.checkpoint().revision()},{"sampling",sampling},{"workloads",cases},
                        {"runs",runs},{"complete",complete},{"gpu_reference_mode",o.gpu_reference},{"gpu_references",references}});
                };
                auto reference=[&](int rep,const char* boundary) {
                    if(o.gpu_reference=="off") return;
                    if(bench_progress) bench_progress->begin(std::string(boundary)+"_probe",{{"repetition",rep}});
                    auto sample=model.gpu_reference(&cancelled);sample["repetition"]=rep;sample["boundary"]=boundary;
                    references.push_back(std::move(sample));write_report(false);
                    if(bench_progress) bench_progress->end();
                };
                write_report(false);
                const auto soak_start=monotonic_ns();
                for(int rep=0;rep<repetitions || monotonic_ns()-soak_start<uint64_t(soak_seconds)*1000000000ull;++rep) {
                    session.clear();std::vector<int> history;reference(rep,"before");
                    for(const auto& task:cases) {
                        Options settings=o;settings.max_tokens=task.value("max_tokens",o.max_tokens);
                        auto ids=task.at("tokens").get<std::vector<int>>();
                        if(task.value("append",false)) {history.insert(history.end(),ids.begin(),ids.end());ids=history;}
                        else {session.clear();history=ids;}
                        if(bench_progress) bench_progress->begin(task.value("append",false)?"append_request":"initial_request",{{"repetition",rep},{"name",task.at("name")}});
                        if(auto* trace=model.route_trace()) trace->request_begin(task.at("name"),ids,settings.max_tokens,task.value("prime",false));
                        const auto before=model.stats();auto result=task.value("prime",false)?session.prime(ids,&cancelled):session.generate(ids,settings,{},&cancelled);
                        if(auto* trace=model.route_trace()) trace->request_end(result.tokens,result.finish_reason,result.reused_tokens);
                        auto row=result.json();row["name"]=task.at("name");row["repetition"]=rep;row["before"]=before;row["after"]=model.stats();
                        row["runtime_cache_state"]=runs.empty()?"empty_at_process_start":"retained";
                        row["initialization_ns"]=initialization_ns;row["profiling_enabled"]=o.decode_diagnostics || o.kernels.profile || !o.dependency_trace.empty() || !o.route_trace.empty() || !o.trace_dir.empty();
                        row["gpu_reference_mode"]=o.gpu_reference;
                        runs.push_back(row);history.insert(history.end(),result.tokens.begin(),result.tokens.end());
                        if(bench_progress) bench_progress->end();
                        if(cancelled.load()) {write_report(false);throw std::runtime_error("benchmark cancelled");}
                        if(!json_path.empty()) write_report(false);
                        std::println(stderr,"{}: {:.2f} tokens/s, first token {:.1f}s, {} tokens reused",task.at("name").get<std::string>(),row["tokens_per_second"].get<double>(),result.first_token_ms/1000,result.reused_tokens);
                    }
                    reference(rep,"after");
                }
                if(auto* trace=model.route_trace()) trace->finish();
                flush_profile();emit({{"model_revision",model.checkpoint().revision()},{"sampling",sampling},{"workloads",cases},{"runs",runs},{"complete",true},
                    {"gpu_reference_mode",o.gpu_reference},{"gpu_references",references},
                    {"soak_seconds_requested",soak_seconds},{"benchmark_elapsed_ns",monotonic_ns()-soak_start}});
                if(bench_progress) bench_progress->finish("complete");return 0;
            }
            if(prompt.empty() && tokens_path.empty()) throw std::invalid_argument("bench requires --prompt, --prompt-file, or --tokens-file");
            auto ids=!tokens_path.empty()?read_json(tokens_path).get<std::vector<int>>():
                raw?tokenizer.encode(prompt):tokenizer.encode_chat(Json::array({{{"role","user"},{"content",prompt}}}),Json::array(),thinking);
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
            const auto ids=raw?tokenizer.encode(prompt):tokenizer.encode_chat(messages,Json::array(),thinking);
            auto result=session.generate(ids,o,[&](int id){const std::array<int,1> one={id};std::print("{}",tokenizer.decode(one));std::fflush(stdout);},&cancelled);
            std::println("");messages.push_back(parse_output(result.text,Json::array(),thinking));
            if(argc>2 && std::find_if(argv+2,argv+argc,[](const char* a){return std::string(a)=="--prompt" || std::string(a)=="--prompt-file";})!=argv+argc) {
                if(!json_path.empty()) emit(result.json());break;
            }
            prompt.clear();
        }
        return 0;
    } catch(const std::exception& e) {
        if(bench_progress) try {bench_progress->finish(cancelled.load()?"interrupted":"failed",e.what());} catch(...) {}
        std::println(stderr,"zerocool: {}",e.what());return 1;
    }
}
