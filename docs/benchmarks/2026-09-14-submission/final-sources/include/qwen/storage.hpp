#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>

namespace freellm::qwen {
using Json = nlohmann::ordered_json;
inline constexpr uint64_t GiB = 1024ull * 1024 * 1024;
inline constexpr uint64_t MiB = 1024ull * 1024;
inline constexpr int Layers = 48, Hidden = 2560, Experts = 512, TopK = 10;
inline constexpr int Intermediate = 640, Vocab = 248320, Hyper = 10240;
inline constexpr uint64_t ExpertBytes = 2764800;
inline constexpr uint64_t ExpertStride = 2768896;
inline constexpr const char* ModelRevision = "aa7c790e804bbf9d491ddb109c3d61bc4a555f7c";
enum class Artifact { Q4, Mixed };
Json artifact_lock(Artifact artifact);
const char* artifact_revision(Artifact artifact);
const char* artifact_model_id(Artifact artifact);
const char* build_fingerprint();

uint64_t checked_add(uint64_t a, uint64_t b);
uint64_t checked_mul(uint64_t a, uint64_t b);
float bf16(uint16_t value);
float round_bf16(float value);
float fp16(uint16_t value);
Json read_json(const std::filesystem::path& path, uint64_t limit = 32 * MiB);
void verify_identity(const std::filesystem::path& directory, Artifact artifact = Artifact::Q4);
std::string read_text(const std::filesystem::path& path, uint64_t limit = 32 * MiB);

// An allocation is counted once even when it is visible to both CPU and GPU.
// gpu_owner owns the Metal object; consumers retain Buffer until completion.
struct Buffer {
    uint64_t bytes = 0;
    std::byte* data = nullptr;
    std::shared_ptr<void> owner;
    void* metal = nullptr;
    static std::shared_ptr<Buffer> host(uint64_t bytes);
    std::span<float> floats() const;
};
using Buf = std::shared_ptr<Buffer>;
using Allocator = std::function<Buf(uint64_t)>;

class File {
public:
    explicit File(const std::filesystem::path& path, bool uncached = true);
    ~File();
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    void read(uint64_t offset, std::span<std::byte> into) const;
    uint64_t size() const { return size_; }
    const std::string& name() const { return name_; }
    mutable std::atomic<uint64_t> read_bytes{0}, read_calls{0};
private:
    int fd_ = -1;
    uint64_t size_ = 0;
    std::string name_;
};

struct TensorRef {
    std::shared_ptr<File> file;
    uint64_t offset = 0, bytes = 0;
    std::vector<uint64_t> shape;
    std::string dtype;
    uint64_t elements() const;
    uint64_t row_bytes() const;
};

// Only headers and tensor ranges are retained; no whole-shard mmap wrappers.
class Checkpoint {
public:
    explicit Checkpoint(const std::filesystem::path& dir, bool validate_model = true, Artifact artifact = Artifact::Q4);
    Artifact artifact() const { return artifact_; }
    const char* revision() const { return artifact_revision(artifact_); }
    const char* model_id() const { return artifact_model_id(artifact_); }
    const TensorRef& at(const std::string& key) const;
    bool contains(const std::string& key) const;
    Buf load(const std::string& key, const Allocator& alloc) const;
    Json inspect() const;
    uint64_t resident_bytes(int layers = Layers) const;
    uint64_t diagnostic_resident_bytes(int layers = Layers) const;
    uint64_t bytes_read() const;
    void validate() const;
    std::filesystem::path directory;
    Json config;
    std::unordered_map<std::string, TensorRef> tensors;
private:
    Artifact artifact_ = Artifact::Q4;
    std::vector<std::shared_ptr<File>> files_;
};

bool is_resident(const std::string& key, int layers = Layers);

// Lossless sidecar; resident tensors and tokenizer still come from Checkpoint.
// A verified manifest binds every range to the source and current file identity.
struct ExpertKey;
class PreparedArtifact {
public:
    PreparedArtifact(const std::filesystem::path& directory, const Checkpoint& source);
    void expert(ExpertKey key, const Buf& into) const;
    void ngram(uint64_t shard, uint64_t row, std::span<std::byte> into) const;
    void ngram_range(uint64_t shard, uint64_t offset, std::span<std::byte> into) const;
    uint64_t bytes_read() const;
    Json inspect() const;
private:
    Json manifest_;
    std::string identity_;
    Artifact consumer_ = Artifact::Q4;
    std::array<std::shared_ptr<File>,Layers> experts_;
    std::array<std::shared_ptr<File>,128> ngrams_;
};

// Notifications carry no mutable cache state. Take a ticket before examining
// futures so a completion between the examination and wait cannot be lost.
class CompletionEvents {
public:
    uint64_t ticket() const;
    void publish();
    void wait(uint64_t ticket, std::chrono::milliseconds timeout = std::chrono::milliseconds(20));
private:
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    uint64_t sequence_ = 0;
};

enum class ReadPriority { Demand, Future };
struct ReadTiming { uint64_t queued_ns=0, started_ns=0, completed_ns=0; };
uint64_t monotonic_ns();

// Persistent bounded workers. Tasks own their destination until their future
// completes; cancellation never invalidates storage being written by pread.
class ReadPool {
public:
    explicit ReadPool(size_t workers = 8, size_t queue_limit = 256);
    ~ReadPool();
    std::shared_future<void> submit(std::function<void()> work,
        ReadPriority priority = ReadPriority::Demand, std::shared_ptr<ReadTiming> timing = {});
    void drain();
    std::shared_ptr<CompletionEvents> events() const { return events_; }
private:
    void worker();
    std::mutex mutex_;
    std::condition_variable ready_, space_, drained_;
    struct Task { std::packaged_task<void()> work; std::shared_ptr<ReadTiming> timing; };
    std::deque<Task> demand_, future_;
    std::vector<std::thread> workers_;
    size_t limit_, active_ = 0;
    bool stopping_ = false;
    std::shared_ptr<CompletionEvents> events_ = std::make_shared<CompletionEvents>();
};

struct ExpertKey {
    uint32_t layer, expert;
    bool operator==(const ExpertKey&) const = default;
    uint64_t value() const { return uint64_t(layer) * Experts + expert; }
};
class ExpertStore {
public:
    explicit ExpertStore(const Checkpoint& checkpoint, std::shared_ptr<PreparedArtifact> prepared = {});
    void read(ExpertKey key, const Buf& into) const;
    const std::array<uint64_t, 10>& offsets() const { return offsets_; }
private:
    std::array<std::array<TensorRef, 9>, Layers> refs_;
    std::array<uint64_t, 10> offsets_{};
    std::shared_ptr<PreparedArtifact> prepared_;
};

struct CacheStats {
    uint64_t hits = 0, misses = 0, evictions = 0, bytes = 0;
    uint64_t ready_hits = 0, loading_joins = 0;
    std::array<uint64_t, Layers> layer_hits{}, layer_misses{};
    Json json() const;
};

// Single inference coordinator; asynchronous I/O only changes future state.
// A lease pins a slot for all queued GPU uses, including hits overlapped with
// loading misses. CLOCK skips every leased slot and never pins prefill seeds.
enum class ExpertCachePolicy { Clock, SegmentedLRU };
ExpertCachePolicy parse_cache_policy(std::string_view name);
std::string_view cache_policy_name(ExpertCachePolicy policy);
class ExpertCache {
    struct Entry;
public:
    class Lease {
    public:
        Lease() = default;
        ~Lease();
        Lease(Lease&&) noexcept;
        Lease& operator=(Lease&&) noexcept;
        Lease(const Lease&) = delete;
        bool ready() const;
        const Buf& wait() const;
        ExpertKey key() const;
        const ReadTiming& timing() const;
        const char* acquisition() const;
    private:
        friend class ExpertCache;
        explicit Lease(std::shared_ptr<Entry> e, int acquisition);
        std::shared_ptr<Entry> entry_;
        int acquisition_ = 0;
    };
    ExpertCache(size_t slots, Allocator allocator, ReadPool& reads,
                std::function<void(ExpertKey, const Buf&)> loader,
                uint64_t stride = ExpertStride, ExpertCachePolicy policy = ExpertCachePolicy::Clock);
    ~ExpertCache();
    Lease acquire(ExpertKey key);
    bool ready(ExpertKey key) const;
    void clear();
    void resize(size_t slots);
    size_t capacity() const { return slots_.size(); }
    size_t occupancy() const { return lookup_.size(); }
    const CacheStats& stats() const { return stats_; }
    Json json() const;
private:
    void unlink(Entry* entry);
    void link(Entry* entry, unsigned queue);
    void touch(Entry* entry);
    void demote();
    Entry* slru_victim() const;
    std::vector<std::shared_ptr<Entry>> slots_;
    std::unordered_map<uint64_t, size_t> lookup_;
    size_t hand_ = 0;
    Allocator allocator_;
    ReadPool& reads_;
    std::function<void(ExpertKey, const Buf&)> loader_;
    uint64_t stride_;
    CacheStats stats_;
    ExpertCachePolicy policy_;
    // Intrusive links live in the existing entries; no separate queue allocations.
    // Metadata for both policies is covered by the common driver/control reserve.
    std::array<Entry*,2> oldest_{}, newest_{};
    size_t protected_ = 0;
};

struct MemoryPlan {
    uint64_t limit = 0, resident = 0, state = 0, scratch = 0;
    uint64_t ngram = 64 * MiB, reserve = GiB, experts = 0;
    uint64_t panel_scratch = 0, kernel_scratch = 0;
    uint64_t pipeline_scratch = 0, snapshot = 0;
    uint64_t runtime_control = 16384; // One aligned GPU status allocation, all configurations.
    uint32_t panel_tokens = 0;
    size_t slots = 0;
    void cap_experts(size_t count);
    MemoryPlan without_prompt_workspaces(size_t expert_cap = 0) const;
    static MemoryPlan make(uint64_t requested, uint64_t physical,
                           uint64_t metal_limit, uint64_t resident,
                           int context, int chunk, int panel = 0, int layers = Layers, uint64_t kernel_scratch = 0,
                           bool double_pipeline = false, uint64_t decode_scratch = 0, bool snapshot = false);
    Json json() const;
};

class NgramStore {
public:
    NgramStore(const Checkpoint& cp, ReadPool& reads, uint64_t cache_bytes = 64 * MiB,
               std::shared_ptr<PreparedArtifact> prepared = {});
    std::vector<std::array<int64_t, 16>> row_ids(std::span<const int> tokens,
                                               std::array<int, 2> history) const;
    void embedding(std::span<const int> tokens, std::array<int, 2> history, std::span<float> out);
    uint64_t hits = 0, misses = 0;
private:
    ReadPool& reads_;
    std::shared_ptr<PreparedArtifact> prepared_;
    std::array<int64_t, 3> multipliers_{};
    std::array<int64_t, 16> sizes_{}, offsets_{};
    std::array<std::array<TensorRef, 3>, 128> refs_;
    struct Row { int64_t key = -1; std::array<uint16_t, 160> values{}; };
    std::vector<Row> rows_;
    std::unordered_map<int64_t, size_t> lookup_;
    size_t next_ = 0;
    void read_row(int64_t id, std::span<float> into) const;
    static void decode_row(std::span<const std::byte> packed, std::span<float> into);
};
} // namespace freellm::qwen
