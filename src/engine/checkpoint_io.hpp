#pragma once
#include "engine/fetch.hpp"

// Internal I/O contracts. The public entry points supply the compiled locks and
// geometry; tests use tiny pinned files and a loopback HTTP server.
namespace zerocool::engine::checkpoint_io {
Json verify(const std::filesystem::path& directory, const Json& lock, bool cached, const std::atomic<bool>& cancel,
            const ProgressFn& progress = {});
Json download(const std::filesystem::path& directory, const Json& lock, const std::string& base_url,
              const std::atomic<bool>& cancel, const ProgressFn& progress, unsigned jobs);
struct Geometry {
    uint64_t layers = Layers, experts = Experts, ngram_shards = 128;
    uint64_t record_bytes = ExpertBytes, record_stride = ExpertStride;
};
Json prepare(const std::filesystem::path& model, const std::filesystem::path& output, const Json& source_lock,
             const Json& canonical_lock, const Geometry& geometry, bool verify_only, const std::atomic<bool>& cancel,
             const ProgressFn& progress = {});
} // namespace zerocool::engine::checkpoint_io
