#pragma once
#include "qwen/storage.hpp"
#include <initializer_list>

namespace freellm::qwen {
struct Binding { Buf buffer; uint64_t offset = 0; };
enum class AllocationClass { Temporary, Resident, State, Expert, Snapshot, Workspace };
struct Linear {
    Binding weight, scales, biases;
    uint32_t input = 0, output = 0, group = 64, dtype = 0;
    bool quantized = false;
    uint32_t bits = 4;
};

// Forced candidates are developer experiments. Automatic dispatch keeps the
// reference until a measured, exact-validated rule is explicitly promoted.
struct KernelConfig {
    std::string policy="reference";
    uint32_t token_tile=1, gdn_rows=4, gdn_block=8;
    uint32_t affine_rows=1;
    uint32_t q8_decode_rows=0; // 0 keeps the existing path; 2/4/8 use packed T1 loads.
    std::string route_selection="serial";
    bool gate_pair=false;
    std::string gdn="original";
    std::string attention_score_tiles="full";
    bool profile=false;
    bool profile_decode_only=false; // Exclude ingestion without adding GPU boundaries.
    bool counter_profile=false; // Diagnostic: one compute pass per dispatch.
    Json shape_table=nullptr;
    std::filesystem::path operator_capture;
    std::string capture_phase, capture_operator;
    int capture_layer=-2; // -2 accepts all layers, -1 selects non-layer operations.
    std::string artifact_revision;
    void validate() const;
    Json json() const;
    uint64_t scratch_bytes(uint32_t chunk) const { return gdn=="original"?0:2*((uint64_t(chunk)*384+16383)/16384)*16384; }
};

// One Metal executor, one queue. Dispatch only encodes. CPU visibility and
// cache-slot reuse require completion. The coordinator reaps finished groups;
// callbacks only publish timestamps and notifications, never mutate a cache.
class Metal {
public:
    struct Completion {
        std::atomic<bool> done{false};
        uint64_t submitted_ns=0, commit_returned_ns=0, completed_ns=0;
        double driver_start=0, driver_end=0;
        double gpu_start=0, gpu_end=0;
        std::string error;
    };
    Metal();
    ~Metal();
    Metal(const Metal&) = delete;
    Metal& operator=(const Metal&) = delete;
    Buf allocate(uint64_t bytes);
    Buf allocate(uint64_t bytes, AllocationClass kind);
    Buf zeros(uint64_t floats);
    Buf zeros(uint64_t floats, AllocationClass kind);
    Buf upload(std::span<const float> values);
    // Encode a bit-preserving copy on the same queue; no CPU visibility wait.
    void copy(const Buf& source, uint64_t source_offset, const Buf& destination,
              uint64_t destination_offset, uint64_t bytes);
    Buf slice(const Buf& source, uint64_t offset, uint64_t bytes);
    void budget(uint64_t bytes);
    // Opt-in scalar instrumentation, configured before any engine allocation.
    void buffer_diagnostics(bool enabled);
    void residency(const std::string& mode);
    // Two coordinator-owned temporary pools. The previous GPU user must finish
    // before a pool is reset; persistent allocations always bypass the pools.
    void begin_scratch(size_t slot, uint64_t capacity);
    void end_scratch();
    // Drain GPU users, drop pool ownership, and reap retired allocations.
    // External views retain their charge until their last owner disappears.
    void release_scratch();
    void wait(const std::shared_ptr<Completion>& completion);
    void configure(KernelConfig config);
    // Developer probe: engine-owned bindings and pipelines outlive GPU users.
    // Change only at a drained boundary; retained references remain the default.
    void command_references(bool retained);
    void prepare_pipelines();
    void label(std::string phase,int layer=-1,uint32_t tokens=0,uint32_t offset=0,std::span<const uint32_t> experts={});
    void request_phase(std::string phase);
    Json take_profile();
    void dispatch(const std::string& name, std::initializer_list<Binding> buffers,
                  std::initializer_list<uint32_t> params, uint32_t x,
                  uint32_t y = 1, uint32_t z = 1,
                  uint32_t tx = 32, uint32_t ty = 1, uint32_t tz = 1);
    std::shared_ptr<Completion> submit();
    void reap();
    void completion_events(std::shared_ptr<CompletionEvents> events);
    void finish();
    // Bounded diagnostic, using fixed reference arithmetic and temporary scratch.
    Json reference_probe(const Linear& layer, const std::atomic<bool>* cancel=nullptr);
    Buf linear(const Linear& layer, const Buf& x, uint32_t tokens, bool float_output = false);
    void linear_into(const Linear& layer, const Buf& x, uint32_t tokens,
                     Binding output, bool float_output = false);
    void grouped_experts(std::span<const Buf> records, std::span<const uint32_t> positions,
                         const Buf& x, const Buf& output, const Buf& scratch);
    Buf gated_linear(const Linear& gate, const Linear& up, const Buf& x, uint32_t tokens, const Buf& rows = {});
    Buf gdn_scan(const Buf& qkv,const Buf& a,const Buf& b,const Buf& alog,
                 const Buf& dt,const Buf& state,uint32_t tokens,uint32_t alog_dtype,uint32_t dt_dtype);
    void route(const Buf& logits,const Buf& ids,const Buf& weights,uint32_t tokens);
    void sparse_select(const Buf& scores,const Buf& mask,const Buf& status,
                       uint32_t tokens,uint32_t offset,uint32_t length);
    void attention_scores(const Buf& q,const Buf& keys,const Buf& mask,const Buf& scores,
                          uint32_t tokens,uint32_t offset,uint32_t length,bool sparse);
    Buf embedding(const Linear& layer, std::span<const int> ids, uint32_t copies = 1);
    uint64_t recommended() const;
    uint64_t physical() const;
    uint64_t allocated() const;
    uint64_t peak() const;
    std::string device_name() const;
    Json statistics() const;
    Json timing_counters() const; // Scalar snapshot; never submits, waits, reaps or allocates GPU buffers.
    Json memory_counters() const; // Same non-mutating contract, plus allocation/ownership gauges.
private:
    void capture_linear(const Linear& layer,const Linear* up,const Buf& input,uint32_t tokens,const Buf& rows={});
    struct LinearPolicy { uint32_t tile=1, rows=1; bool pair=false; };
    LinearPolicy linear_policy(const Linear& layer,uint32_t tokens,bool fused,bool gathered) const;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class Resident {
public:
    Resident(const Checkpoint& cp, Metal& gpu, int layers = Layers, bool diagnostic_streaming = false);
    void activate_layer(int layer);
    const Buf& at(const std::string& name) const;
    Linear linear(const std::string& name) const;
    uint32_t dtype(const std::string& name) const;
private:
    const Checkpoint& cp_;
    Metal& gpu_;
    bool streaming_;
    std::unordered_map<std::string,Buf> tensors_;
};
Linear expert_linear(const Buf& record, int projection);
Json process_memory();
Json system_memory(); // Overlapping VM categories, not additive process allocations.
Json host_conditions();
uint64_t available_memory();
Json disk_counters();
} // namespace freellm::qwen
