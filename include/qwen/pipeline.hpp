#pragma once
#include "qwen/metal.hpp"

namespace freellm::qwen {
struct ReadyExpert { ExpertKey key; Buf record; };
using EncodeReadyGroup = std::function<void(std::span<const ReadyExpert>,size_t)>;
// Run all requested experts; only execution order may change. The caller
// scatters each contribution into its router-assigned position before reducing.
// This is also the dependency replay path, so its timings exercise real leases,
// reads, dispatch, completion, and resource release.
Json execute_experts(std::span<const ExpertKey> selected, ExpertCache& cache,
                     ReadPool& reads, Metal& gpu, size_t group_size,
                     const std::function<void(ExpertKey,const Buf&)>& encode,
                     const std::atomic<bool>* cancel = nullptr, bool detailed = false,
                     const EncodeReadyGroup& encode_group = {});
Json execute_experts_batched(std::span<const ExpertKey> selected, ExpertCache& cache,
                     ReadPool& reads, Metal& gpu,
                     const std::function<void(ExpertKey,const Buf&)>& encode,
                     const std::atomic<bool>* cancel = nullptr);
}
