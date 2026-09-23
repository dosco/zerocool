#pragma once
#include "engine/storage.hpp"
#include <atomic>
#include <filesystem>
#include <functional>

namespace zerocool::engine {

// Checkpoint acquisition without a Python interpreter. The pinned lock names
// the repository, the revision and every required file with its size and
// digest, so fetching and verifying need nothing the engine does not already
// link: libcurl for transport and CommonCrypto for the digests.

// Progress for the whole operation rather than one file, because transfers run
// concurrently and interleaved per-file counters read as noise.
struct FetchProgress {
    std::string name;       // the file this update concerns
    uint64_t done = 0;      // bytes settled across every file, including resumed
    uint64_t total = 0;     // bytes the lock pins across every file
    unsigned active = 0;    // transfers in flight; zero outside a download
    bool resumed = false;   // this file continued a partial transfer
    const char* phase = ""; // what happened to it: fetched, verified, prepared, reused
};
using ProgressFn = std::function<void(const FetchProgress&)>;

// Concurrent transfers. The payload is a score of multi-gigabyte shards, so
// running several at once is most of the available speedup; striping ranges
// within one file would need a record of which ranges landed to stay resumable.
inline constexpr unsigned DefaultFetchJobs = 4, MaxFetchJobs = 16;

// The stat identity a receipt records: size, device, inode and both timestamps.
// A file whose identity moved is reread rather than trusted.
Json file_fingerprint(const std::filesystem::path& path);
// SHA-256 of a whole file, read uncached so a 104GB scan does not evict the
// page cache the engine is about to want.
std::string file_digest(const std::filesystem::path& path, const std::atomic<bool>& cancel);
// Serialize with Python's `json.dumps(indent=2) + "\n"` spelling, then rename
// into place, so a partial write is never visible under the real name.
void write_json_atomic(const std::filesystem::path& path, const Json& value);

// Hash every required file and record its identity atomically. With `cached`,
// a file whose stat fingerprint and recorded digest are both unchanged is not
// reread, which keeps a repeat check off a 104GB scan. Release checks always
// pass cached=false: the receipt is a local integrity cache, not a trust
// boundary. Throws when a file is missing, the wrong size, or hashes wrong.
Json verify_checkpoint(const std::filesystem::path& directory, Artifact artifact, bool cached,
                       const std::atomic<bool>& cancel, const ProgressFn& progress = {});

// Fetch missing or corrupt required files, then verify the complete checkpoint.
// A partial transfer resumes from its own length with a ranged request. Completed
// files are verified before reuse; corrupt files are replaced only after the new
// bytes pass the pinned hash. Refuses to write into a directory holding a
// different pinned revision, and refuses to start
// without room for the remainder plus a reserve.
Json download_checkpoint(const std::filesystem::path& directory, Artifact artifact, const std::atomic<bool>& cancel,
                         const ProgressFn& progress = {}, unsigned jobs = DefaultFetchJobs);

// Repack the pinned checkpoint's routed experts and ngram tables into the
// contiguous records the engine reads. Nothing is requantized: the checkpoint
// routed experts and ngrams are already affine Q4, so this moves their bytes
// into records reached with one seek. Each file is published atomically, so an
// interrupted run reuses whatever finished. With `verify_only`, payloads are
// rehashed and no payload is rewritten; manifest and receipts are refreshed.
//
// Both artifacts produce the canonical record manifest pinned by models.lock.json.
// Mixed inputs require the existing pinned payload-equivalence proof. The receipt
// records the actual verified input separately from that canonical output identity.
Json prepare_storage(const std::filesystem::path& model, const std::filesystem::path& output, bool verify_only,
                     const std::atomic<bool>& cancel, const ProgressFn& progress = {},
                     Artifact artifact = Artifact::Q4);

} // namespace zerocool::engine
