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

// Progress for one file: bytes already present, bytes now transferred, and the
// total the lock pins. Reported often enough to drive a display and to let a
// caller observe a stalled transfer.
struct FetchProgress {
    std::string name;
    uint64_t resumed = 0, received = 0, total = 0;
};
using ProgressFn = std::function<void(const FetchProgress&)>;

// Hash every required file and record its identity atomically. With `cached`,
// a file whose stat fingerprint and recorded digest are both unchanged is not
// reread, which keeps a repeat check off a 104GB scan. Release checks always
// pass cached=false: the receipt is a local integrity cache, not a trust
// boundary. Throws when a file is missing, the wrong size, or hashes wrong.
Json verify_checkpoint(const std::filesystem::path& directory, Artifact artifact,
                       bool cached, const std::atomic<bool>& cancel, const ProgressFn& progress = {});

// Fetch every required file that is absent or partial, then verify the result.
// A partial transfer resumes from its own length with a ranged request; a file
// that is already complete and verified is never refetched. Refuses to write
// into a directory holding a different pinned revision, and refuses to start
// without room for the remainder plus a reserve.
Json download_checkpoint(const std::filesystem::path& directory, Artifact artifact,
                         const std::atomic<bool>& cancel, const ProgressFn& progress = {});

} // namespace zerocool::engine
