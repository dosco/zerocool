#pragma once

#include "tensor.hpp"
#include "model_loader.hpp"
#include "bounded_queue.hpp"
#include "quantiz/quantized_tensor.hpp"
#include "quantiz/types.hpp"
#include "safetensors.hh"
#include <string>
#include <thread>
#include <vector>
#include <memory>
#include <functional>
#include <optional>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <print>

namespace freellm {

/**
 * @brief Memory-mapped file handle with RAII cleanup
 *
 * Manages the lifetime of a memory-mapped file region.
 * Automatically unmaps on destruction.
 */
class MMapHandle {
public:
    MMapHandle() : data_(nullptr), size_(0), fd_(-1) {}

    ~MMapHandle() {
        close();
    }

    // Disable copy, enable move
    MMapHandle(const MMapHandle&) = delete;
    MMapHandle& operator=(const MMapHandle&) = delete;

    MMapHandle(MMapHandle&& other) noexcept
        : data_(other.data_)
        , size_(other.size_)
        , fd_(other.fd_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
        other.fd_ = -1;
    }

    MMapHandle& operator=(MMapHandle&& other) noexcept {
        if (this != &other) {
            close();
            data_ = other.data_;
            size_ = other.size_;
            fd_ = other.fd_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.fd_ = -1;
        }
        return *this;
    }

    /**
     * @brief Open and memory-map a file
     *
     * Uses mmap() to map the entire file into the process's virtual
     * address space. The OS will page in data on-demand.
     *
     * @param filepath Path to file
     * @return true on success, false on error
     */
    bool open(const std::string& filepath) {
        // Open file
        fd_ = ::open(filepath.c_str(), O_RDONLY);
        if (fd_ == -1) {
            return false;
        }

        // Get file size
        struct stat sb;
        if (fstat(fd_, &sb) == -1) {
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        size_ = static_cast<size_t>(sb.st_size);

        // Memory-map the file
        // PROT_READ: Pages may be read
        // MAP_PRIVATE: Changes are private (we won't modify)
        data_ = static_cast<uint8_t*>(
            mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));

        if (data_ == MAP_FAILED) {
            ::close(fd_);
            fd_ = -1;
            data_ = nullptr;
            return false;
        }

        // Hint to OS: we'll read sequentially
        // This helps the OS optimize its page-ahead strategy
        #ifdef __linux__
        madvise(data_, size_, MADV_SEQUENTIAL);
        #endif

        return true;
    }

    /**
     * @brief Close and unmap the file
     */
    void close() {
        if (data_ != nullptr && data_ != MAP_FAILED) {
            munmap(data_, size_);
            data_ = nullptr;
        }
        if (fd_ != -1) {
            ::close(fd_);
            fd_ = -1;
        }
        size_ = 0;
    }

    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }
    bool is_open() const { return data_ != nullptr && fd_ != -1; }

private:
    uint8_t* data_;
    size_t size_;
    int fd_;
};

/**
 * @brief Work item for tensor loading pipeline
 *
 * Contains all information needed to load and optionally quantize a tensor:
 * - Tensor metadata (name, shape, dtype)
 * - Pointer to raw data in mmap-ed region (zero-copy)
 * - Optional quantization type
 * - Shared pointer to mmap handle to keep it alive
 */
struct TensorWorkItem {
    std::string name;                    // Internal tensor name (after mapping)
    std::vector<size_t> shape;           // Tensor shape
    const uint8_t* raw_data_ptr;         // Pointer to data in mmap region (zero-copy)
    size_t raw_data_size;                // Size of data in bytes
    safetensors::dtype dtype;            // Original data type
    std::optional<quant::QuantType> quantize_to;  // Optional quantization target
    std::shared_ptr<MMapHandle> mmap_handle;  // Keep mmap alive until processed

    TensorWorkItem() = default;

    TensorWorkItem(std::string n, std::vector<size_t> s, const uint8_t* data_ptr, size_t data_size,
                   safetensors::dtype dt, std::shared_ptr<MMapHandle> mmap,
                   std::optional<quant::QuantType> qt = std::nullopt)
        : name(std::move(n))
        , shape(std::move(s))
        , raw_data_ptr(data_ptr)
        , raw_data_size(data_size)
        , dtype(dt)
        , quantize_to(qt)
        , mmap_handle(std::move(mmap))
    {}
};

/**
 * @brief Result of tensor processing
 *
 * Contains the loaded (and possibly quantized) tensor, ready to be
 * inserted into the model's weight map.
 */
struct TensorResult {
    std::string name;
    Tensor tensor;
    std::optional<QuantizedTensor> quantized;

    TensorResult() = default;

    TensorResult(std::string n, Tensor t, std::optional<QuantizedTensor> q = std::nullopt)
        : name(std::move(n))
        , tensor(std::move(t))
        , quantized(std::move(q))
    {}
};

/**
 * @brief Parallel tensor loader with producer-consumer pipeline
 *
 * Architecture:
 * ┌────────────────────────────────────────────────────────────┐
 * │ Producer Thread (I/O)                                       │
 * │ - Memory-maps safetensors file                             │
 * │ - Parses JSON header                                       │
 * │ - Creates work items with pointers to mmap region          │
 * │ - Pushes to bounded queue (backpressure if full)           │
 * └────────────────────────────────────────────────────────────┘
 *                           ↓
 * ┌────────────────────────────────────────────────────────────┐
 * │ Thread-Safe Bounded Queue                                   │
 * │ - Decouples I/O from CPU work                              │
 * │ - Size = 2x num_threads (prevents unbounded memory)        │
 * └────────────────────────────────────────────────────────────┘
 *                           ↓
 * ┌────────────────────────────────────────────────────────────┐
 * │ Consumer Threads (CPU pool)                                 │
 * │ - Pop work items from queue                                │
 * │ - Convert dtype (BF16→F32, FP16→F32, etc.)                │
 * │ - Optionally quantize (F32→Q4_K, Q8_0, etc.)              │
 * │ - Store results in thread-safe output map                  │
 * └────────────────────────────────────────────────────────────┘
 *
 * Performance:
 * - mmap: Zero-copy reads, OS handles paging
 * - Pipeline: Overlaps I/O with CPU work
 * - Bounded queue: Prevents memory explosion
 * - Thread pool: Utilizes all CPU cores for conversions
 *
 * Memory efficiency:
 * - mmap region: Size of .safetensors file (shared across threads)
 * - Queue: ~2x num_cores work items (small metadata structs)
 * - Output: Final F32 tensors (or quantized if requested)
 */
class ParallelTensorLoader {
public:
    /**
     * @brief Constructor
     *
     * @param num_threads Number of worker threads (default: hardware concurrency)
     * @param queue_capacity Queue size (default: 2x num_threads)
     */
    explicit ParallelTensorLoader(
        size_t num_threads = std::thread::hardware_concurrency(),
        size_t queue_capacity = 0  // 0 = auto (2x num_threads)
    )
        : num_threads_(num_threads)
        , queue_capacity_(queue_capacity == 0 ? num_threads * 2 : queue_capacity)
    {}

    /**
     * @brief Load tensors from safetensors file using parallel pipeline
     *
     * This is the main entry point. It:
     * 1. Opens and memory-maps the file
     * 2. Starts consumer threads
     * 3. Runs producer logic (parse metadata, enqueue work items)
     * 4. Waits for all workers to finish
     * 5. Returns the loaded weight map
     *
     * @param filepath Path to .safetensors file
     * @param name_mapper Function to map HF names to internal names
     * @param quant_resolver Function to determine quantization type (optional)
     * @return WeightMap with loaded tensors
     */
    WeightMap load(
        const std::string& filepath,
        std::function<std::string(const std::string&)> name_mapper,
        std::function<std::optional<quant::QuantType>(const std::string&)> quant_resolver = nullptr
    );

    /**
     * @brief Load tensors with parallel quantization
     *
     * Extended version that returns both F32 tensors and their quantized versions.
     * Quantization happens in parallel in the worker threads.
     *
     * @param filepath Path to .safetensors file
     * @param name_mapper Function to map HF names to internal names
     * @param quant_resolver Function to determine quantization type for each tensor
     * @return Pair of (F32 WeightMap, Quantized tensors map)
     */
    std::pair<WeightMap, std::unordered_map<std::string, QuantizedTensor>>
    load_with_quantization(
        const std::string& filepath,
        std::function<std::string(const std::string&)> name_mapper,
        std::function<std::optional<quant::QuantType>(const std::string&)> quant_resolver
    );

private:
    /**
     * @brief Producer thread function
     *
     * Memory-maps file, parses metadata, enqueues work items.
     * Runs in the calling thread (not spawned).
     */
    void producer_thread(
        const std::string& filepath,
        BoundedQueue<TensorWorkItem>& work_queue,
        std::function<std::string(const std::string&)> name_mapper,
        std::function<std::optional<quant::QuantType>(const std::string&)> quant_resolver
    );

    /**
     * @brief Consumer thread function
     *
     * Pops work items, converts/quantizes tensors, stores results.
     * Multiple instances run in parallel in worker threads.
     */
    void consumer_thread(
        BoundedQueue<TensorWorkItem>& work_queue,
        std::vector<TensorResult>& thread_results
    );

    /**
     * @brief Process a single tensor work item
     *
     * Converts from source dtype to F32 (or quantized format).
     * This is the CPU-intensive part that we parallelize.
     */
    TensorResult process_work_item(const TensorWorkItem& item);

    size_t num_threads_;
    size_t queue_capacity_;
    std::shared_ptr<MMapHandle> current_mmap_;  // Keep mmap alive during load operation
};

} // namespace freellm
