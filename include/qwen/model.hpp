#pragma once
#include "qwen/metal.hpp"
#include "qwen/pressure.hpp"
#include <optional>

namespace freellm::qwen {
class CachedProgress;
class RouteTrace;
class ExpertTail;
struct Options {
    Artifact artifact = Artifact::Q4; // Explicit selection; precision never changes under pressure.
    std::filesystem::path model;
    std::filesystem::path prepared;
    uint64_t memory = 22*GiB;
    int context = MaxContext;
    int chunk = 128;
    int panel = 0; // Explicit experiment until request-level qualification; 256/512/1024.
    int short_append = 32;
    int io_workers = 8;
    int ready_group = 4;
    bool completion_pipeline = true;
    std::string expert_tail="wait"; // Explicit single-token encode-ahead experiment.
    std::string decode_submission="immediate"; // Benchmark-only coalesced read experiment.
    std::string decode_scratch="none"; // Explicit bounded single-token temporary reuse.
    std::string memory_pressure_policy="observe"; // Benchmark experiment; no automatic regrowth.
    void validate_decode_scratch() const;
    void validate_decode_submission() const;
    std::filesystem::path dependency_trace;
    std::filesystem::path route_trace;
    int max_tokens = 256;
    float temperature = 0;
    float top_p = 0.95f;
    int top_k = 20;
    uint64_t seed = 0;
    size_t expert_slots = 0; // A smaller fixed cache for reproducible eviction experiments.
    int probe_layers = Layers; // Truncated diagnostics only; never exposed by serve.
    std::filesystem::path trace_dir;
    bool diagnostic_stream_trunk = false;
    KernelConfig kernels;
    bool audit_routes=false; // Bounded correctness capture, disabled during timing.
    bool decode_diagnostics=false; // Bounded per-step observation; benchmark timings are instrumented.
    // Developer harness only. Runs on the coordinator; observers must not mutate
    // Metal or retain buffers. No production CLI enables this callback.
    std::function<void(const Json&,const Metal&)> memory_observer;
    // Developer fixture capture; invoked after data/GPU completion, never used by production CLI.
    std::function<void(ExpertKey,const Buf&,const Buf&,uint32_t,const std::string&,uint32_t)> expert_observer;
    std::string gpu_reference="off"; // Benchmark-only boundary instrumentation.
    std::string residency="off", decode_path="reference", prefill_pipeline="serial", phase_memory="fixed";
    bool cached_token_replay=false;
    bool cached_compare=false;
    std::string cached_compare_axis="q8_decode_rows";
    std::string sparse_selection="cpu";
    std::string cache_policy="clock"; // Experimental SLRU; no automatic policy changes.
    std::filesystem::path sparse_capture;
    std::filesystem::path operator_fixtures;
};

struct LayerState {
    Buf conv, recurrence, keys, values, index, ple_conv;
    uint32_t position = 0; // Absolute position; partial panels never advance global history.
};
std::vector<std::byte> sparse_mask(std::span<const float> scores,uint32_t tokens,uint32_t offset,uint32_t length);
struct State {
    Artifact artifact = Artifact::Q4;
    std::array<LayerState,Layers> layers;
    std::array<int,2> history{EndOfText,EndOfText};
    uint32_t tokens = 0;
    uint64_t trace_session_id = 0; // Diagnostic identity; never part of model arithmetic.
    bool valid = false; // Only make_state() or a committed update makes this reusable.
};
State snapshot_state(Metal& gpu,const State& state);
void restore_state(Metal& gpu,const State& snapshot,State& state);
Json state_digest(const State& state);

// Recurrent/attention writes cannot be rolled back. A failed update leaves
// state invalid and its committed history unchanged, including direct API use.
class StateUpdate {
public:
    StateUpdate(State& state, std::span<const int> tokens);
    StateUpdate(const StateUpdate&) = delete;
    void commit();
private:
    State& state_;
    uint32_t next_tokens_;
    std::array<int,2> next_history_;
    bool committed_ = false;
};

// The architecture is intentionally fixed. A CPU oracle belongs in tests,
// not behind a second production model/backend adapter.
class Model {
public:
    explicit Model(Options options);
    ~Model();
    State make_state();
    void prepare_ingest(size_t remaining_tokens);
    void finish_ingest();
    std::vector<float> forward(std::span<const int> ids, State& state, bool logits = true,
                              const std::atomic<bool>* cancel = nullptr);
    Json stats() const;
    Json decode_counters() const;
    Json memory_counters() const { return gpu_.memory_counters(); }
    void diagnostic_drain(); // Explicit lifecycle boundary, outside measured requests.
    Json gpu_reference(const std::atomic<bool>* cancel=nullptr);
    const MemoryPlan& memory_plan() const { return plan_; }
    const Options& options() const { return options_; }
    uint32_t input_limit() const { return plan_.panel_tokens?plan_.panel_tokens:uint32_t(options_.chunk); }
    const Checkpoint& checkpoint() const { return checkpoint_; }
    void reset_expert_cache();
    void prepare_pipelines() { gpu_.prepare_pipelines(); }
    Json take_profile();
    void phase(std::string name) { phase_=name;gpu_.request_phase(std::move(name)); }
    Json route_identity() const;
    RouteTrace* route_trace() const { return route_trace_.get(); }
    // Developer diagnostic: full forward, with real routes and deep state restore.
    Json cached_token_replay(std::span<const int> tokens, int repetitions,
                             const std::atomic<bool>* cancel=nullptr,CachedProgress* progress=nullptr);
private:
    void pressure_boundary(int layer);
    void observe_memory(const char* event,int layer,uint32_t tokens,uint32_t offset) const;
    std::vector<float> forward_impl(std::span<const int> ids, State& state, bool logits,
                                   const std::atomic<bool>* cancel);
    void transition_memory(bool prompt);
    void check_sparse_status() const;
    void finish_expert_tail();
    void record_expert_timing(Json timing,int layer,uint32_t tokens,uint32_t offset,
                              const std::string& phase,std::span<const int> routes);
    void capture_sparse(const Buf& q,const Buf& keys,const Buf& values,const Buf& qg,
                        const Buf& index_scores,uint32_t tokens,uint32_t offset,int layer);
    std::vector<float> forward_panel(std::span<const int> ids, State& state, bool logits,
                                    const std::atomic<bool>* cancel);
    std::vector<float> compute_logits(const Buf& h, uint32_t tokens);
    Buf norm(const Buf& x,const std::string& weight,uint32_t width,uint32_t group,uint32_t tokens,bool grouped=false);
    Buf unary(const Buf& x,uint32_t op);
    Buf binary(const Buf& x,const Buf& y,uint32_t op);
    std::pair<Buf,Buf> hyper(const Buf& x,const std::string& base,uint32_t tokens,bool inject=true);
    Buf conv(const Buf& x,Buf& state,const std::string& weight,uint32_t width,uint32_t tokens,uint32_t dilation);
    Buf gdn(const Buf& x,LayerState& state,int layer,uint32_t tokens);
    Buf attention(const Buf& x,LayerState& state,int layer,uint32_t tokens,uint32_t offset);
    Buf moe(const Buf& x,int layer,uint32_t tokens,const std::atomic<bool>* cancel);
    Buf ple(const Buf& x,const Buf& embedding,LayerState& state,uint32_t tokens);
    void trace(const std::string& name,const Buf& buffer);
    Options options_;
    Checkpoint checkpoint_;
    Metal gpu_;
    MemoryPlan plan_;
    MemoryPlan prompt_plan_,generation_plan_;
    bool ingest_active_=false, prompt_memory_=false;
    uint64_t pressure_resizes_=0,transition_count_=0;
    std::unique_ptr<PressureMonitor> pressure_monitor_;
    PressurePolicy pressure_policy_;
    uint64_t pressure_event_count_=0;
    Json pressure_events_=Json::array();
    Json memory_transitions_=Json::array();
    std::unique_ptr<Resident> resident_;
    ReadPool reads_;
    std::shared_ptr<PreparedArtifact> prepared_;
    ExpertStore store_;
    std::unique_ptr<ExpertCache> cache_;
    std::unique_ptr<NgramStore> ngrams_;
    uint64_t decode_passes_=0,append_passes_=0,prefill_passes_=0;
    uint64_t decode_scratch_passes_=0;
    uint64_t panel_passes_=0;
    uint32_t trace_offset_=0;
    std::array<uint64_t,Layers> expert_wait_ns_{},expert_gpu_ns_{},expert_passes_{};
    std::array<std::vector<int>,Layers> route_history_;
    std::string phase_="unspecified";
    std::array<Buf,2> expert_scratch_;
    Buf sparse_status_; // Persistent across scratch reuse; checked after completed GPU work.
    uint64_t sparse_capture_bytes_=0,sparse_selection_cpu_ns_=0,sparse_selection_wait_ns_=0;
    Json sparse_captures_=Json::array();
    Json phase_dependencies_=Json::object(),dependency_events_=Json::array();
    std::unordered_map<std::string,size_t> detailed_reads_,detailed_passes_;
    std::unique_ptr<ExpertTail> expert_tail_;
    int tail_layer_=0;uint32_t tail_offset_=0;
    uint64_t tail_deferrals_=0;
    std::string tail_phase_;
    std::vector<int> tail_routes_;
    std::unique_ptr<RouteTrace> route_trace_;
    uint64_t trace_session_sequence_=0;
};
} // namespace freellm::qwen
