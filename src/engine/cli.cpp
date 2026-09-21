#include "engine/cli.hpp"
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace zerocool::engine {
Cli parse_cli(int argc,char** argv) {
    Cli cli;
    if(argc<2) throw std::invalid_argument("missing command");
    cli.command=argv[1];
    auto& o=cli.options;auto& command=cli.command;
    auto& prompt=cli.prompt;auto& json_path=cli.json_path;auto& io_path=cli.io_path;
    auto& tokens_path=cli.tokens_path;auto& logits_path=cli.logits_path;auto& tokenize=cli.tokenize;
    auto& render_path=cli.render_path;auto& workloads_path=cli.workloads_path;
    auto& replay_routes=cli.replay_routes;auto& phase_profile=cli.phase_profile;
    auto& cached_progress=cli.cached_progress;auto& bench_progress_path=cli.bench_progress_path;
    auto& replay_hits=cli.replay_hits;auto& soak_seconds=cli.soak_seconds;
    auto& repetitions=cli.repetitions;auto& port=cli.port;auto& control_fd=cli.control_fd;
    auto& probe=cli.probe;auto& storage=cli.storage;auto& kernel_probe=cli.kernel_probe;
    auto& raw=cli.raw;auto& thinking=cli.thinking;
    o.model=".cache/models/qwen38-flash-next";
    raw=command=="bench";
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
        else if(arg=="--gpu-reference") o.gpu_reference=value;
        else if(arg=="--bench-progress") bench_progress_path=value;
        else if(arg=="--residency") o.residency=value;
        else if(arg=="--expert-tail") o.expert_tail=value;
        else if(arg=="--decode-scratch") o.decode_scratch=value;
        else if(arg=="--memory-pressure-policy") {
            if(value!="observe" && value!="shrink") throw std::invalid_argument("memory-pressure-policy must be observe or shrink");
            o.memory_pressure_policy=value;
        }
        else if(arg=="--profile-decode-only") {
            if(value!="0" && value!="1") throw std::invalid_argument("profile-decode-only must be 0 or 1");
            o.kernels.profile_decode_only=value=="1";
        }
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
        else if(arg=="--q4-decode") o.kernels.q4_decode=value;
        else if(arg=="--route-selection") o.kernels.route_selection=value;
        else if(arg=="--soak-seconds") soak_seconds=std::stoi(value);
        else if(arg=="--panel") o.panel=std::stoi(value);
        else if(arg=="--short-append") o.short_append=std::stoi(value);
        else if(arg=="--replay-routes") replay_routes=value;
        else if(arg=="--replay-hits") replay_hits=std::stoi(value);
        else if(arg=="--ready-group") o.ready_group=std::stoi(value);
        else if(arg=="--decode-submission") o.decode_submission=value;
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
        else if(arg=="--control-fd" && command=="serve") control_fd=std::stoi(value);
        else throw std::invalid_argument("unknown option: "+arg);
    }
    return cli;
}
void validate_cli(Cli& cli) {
    auto& o=cli.options;auto& command=cli.command;
    auto& prompt=cli.prompt;auto& json_path=cli.json_path;auto& io_path=cli.io_path;
    auto& tokens_path=cli.tokens_path;auto& logits_path=cli.logits_path;auto& tokenize=cli.tokenize;
    auto& render_path=cli.render_path;auto& workloads_path=cli.workloads_path;
    auto& replay_routes=cli.replay_routes;auto& phase_profile=cli.phase_profile;
    auto& cached_progress=cli.cached_progress;auto& bench_progress_path=cli.bench_progress_path;
    auto& replay_hits=cli.replay_hits;auto& soak_seconds=cli.soak_seconds;
    auto& repetitions=cli.repetitions;auto& port=cli.port;auto& control_fd=cli.control_fd;
    auto& probe=cli.probe;auto& storage=cli.storage;auto& kernel_probe=cli.kernel_probe;
    auto& raw=cli.raw;auto& thinking=cli.thinking;
    (void)tokenize;(void)render_path;(void)control_fd;(void)raw;(void)thinking;(void)prompt;(void)replay_hits;
    if(repetitions<1 || repetitions>20 || port<0 || port>65535) throw std::invalid_argument("invalid repetitions or port");
    o.kernels.validate();
    if(o.sparse_selection!="cpu" && o.sparse_selection!="gpu") throw std::invalid_argument("sparse selection must be cpu or gpu");
    if(o.cached_compare_axis!="q8_decode_rows" && o.cached_compare_axis!="sparse_selection" && o.cached_compare_axis!="attention_score_tiles")
        throw std::invalid_argument("invalid cached comparison axis");
    if(o.gpu_reference!="off" && o.gpu_reference!="resident-q8-v1") throw std::invalid_argument("invalid GPU reference mode");
    if((o.gpu_reference!="off" || !bench_progress_path.empty()) && (command!="bench" || workloads_path.empty() ||
       o.cached_token_replay || o.diagnostic_stream_trunk || probe || kernel_probe || storage ||
       !io_path.empty() || !o.operator_fixtures.empty() || !logits_path.empty() || !replay_routes.empty()))
        throw std::invalid_argument("boundary diagnostics require normal bench --workload-file");
    if(o.gpu_reference!="off" && (o.artifact!=Artifact::Mixed || o.prefill_pipeline!="serial" ||
       o.phase_memory!="fixed" || o.kernels.profile || o.decode_diagnostics || !o.dependency_trace.empty() ||
       !o.trace_dir.empty() || !o.route_trace.empty()))
        throw std::invalid_argument("GPU reference requires mixed artifact, serial fixed memory and no other profiling");
    if(!bench_progress_path.empty() && !json_path.empty() &&
       std::filesystem::weakly_canonical(bench_progress_path)==std::filesystem::weakly_canonical(json_path))
        throw std::invalid_argument("benchmark progress requires a separate output path");
    if(o.decode_diagnostics && (command!="bench" || o.cached_token_replay || o.diagnostic_stream_trunk || o.probe_layers!=Layers || workloads_path.empty()))
        throw std::invalid_argument("decode diagnostics require normal bench --workload-file");
    if(o.cached_compare && (!o.cached_token_replay || o.kernels.profile || repetitions<5 ||
       (o.cached_compare_axis=="q8_decode_rows" && !o.kernels.q8_decode_rows) ||
       (o.cached_compare_axis=="sparse_selection" && o.sparse_selection!="gpu") ||
       (o.cached_compare_axis=="attention_score_tiles" && o.kernels.attention_score_tiles!="skip-masked")))
        throw std::invalid_argument("cached comparison requires an unprofiled candidate and at least five pairs");
    if((o.memory_pressure_policy!="observe" || o.decode_scratch!="none" || o.expert_tail!="wait" || o.cache_policy!="clock" || o.residency!="off" || o.decode_path!="reference" || o.prefill_pipeline!="serial" || o.phase_memory!="fixed" || o.cached_token_replay ||
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
    if(o.residency!="off" && o.residency!="core" && o.residency!="core-cache") throw std::invalid_argument("invalid residency mode");
    o.validate_decode_scratch();
    o.validate_decode_submission();
    if(o.decode_submission!="immediate" && command!="bench" && command!="inspect")
        throw std::invalid_argument("coalesced decode is a benchmark experiment");
    if(o.expert_tail!="wait" && o.expert_tail!="overlap") throw std::invalid_argument("expert tail must be wait or overlap");
    if(o.expert_tail=="overlap" && (!o.completion_pipeline || o.cached_token_replay))
        throw std::invalid_argument("expert tail overlap requires normal completion-pipeline execution");
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
}
} // namespace zerocool::engine
