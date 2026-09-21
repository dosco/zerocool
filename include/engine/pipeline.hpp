#pragma once
#include "engine/metal.hpp"

namespace zerocool::engine {
// Internal production row encoder; queue submission and leases remain caller-owned.
void encode_expert_rows(Metal& gpu,const Buf& record,const Buf& input,const Buf& output,
                        std::span<const int> positions,uint32_t tokens,uint32_t chunk,
                        bool direct,int layer,uint32_t offset,const std::atomic<bool>* cancel=nullptr);
struct ReadyExpert { ExpertKey key; Buf record; };
using EncodeReadyGroup = std::function<void(std::span<const ReadyExpert>,size_t)>;
// Coordinator-owned tail: retains every lease until the submitted GPU users
// finish. Drain before another expert admission, scratch reset or cache resize.
class ExpertTail {
public:
    ExpertTail();
    ~ExpertTail();
    ExpertTail(const ExpertTail&) = delete;
    bool pending() const;
    Json finish();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend Json execute_experts(std::span<const ExpertKey>,ExpertCache&,ReadPool&,Metal&,size_t,
        const std::function<void(ExpertKey,const Buf&)>&,const std::atomic<bool>*,bool,const EncodeReadyGroup&,ExpertTail*,bool);
};
// Run all requested experts; only execution order may change. The caller
// scatters each contribution into its router-assigned position before reducing.
// This is also the dependency replay path, so its timings exercise real leases,
// reads, dispatch, completion, and resource release.
Json execute_experts(std::span<const ExpertKey> selected, ExpertCache& cache,
                     ReadPool& reads, Metal& gpu, size_t group_size,
                     const std::function<void(ExpertKey,const Buf&)>& encode,
                     const std::atomic<bool>* cancel = nullptr, bool detailed = false,
                     const EncodeReadyGroup& encode_group = {}, ExpertTail* tail = nullptr,
                     bool coalesce_reads = false);
Json execute_experts_batched(std::span<const ExpertKey> selected, ExpertCache& cache,
                     ReadPool& reads, Metal& gpu,
                     const std::function<void(ExpertKey,const Buf&)>& encode,
                     const std::atomic<bool>* cancel = nullptr);
}
