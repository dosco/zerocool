#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <variant>

namespace freellm {
namespace infra {

using KernelParam = std::variant<int, float, uint32_t>;

enum class DeviceType {
    CUDA,
    PJRT_TPU,
    PJRT_GPU,
    PJRT_CPU,
    METAL
};

enum class DType {
    FLOAT32,
    FLOAT16,
    INT8,
    INT32,
    INT64
};

struct Dim3 {
    unsigned int x = 1;
    unsigned int y = 1;
    unsigned int z = 1;
};

struct KernelConfig {
    Dim3 grid;
    Dim3 block;
    size_t shared_mem_bytes = 0;
};

// Forward declaration for DLPack
struct DLManagedTensor;

class DeviceBuffer {
public:
    virtual ~DeviceBuffer() = default;
    virtual void* raw_ptr() = 0;
    virtual const void* raw_ptr() const = 0;
    virtual size_t size_bytes() const = 0;
    virtual DeviceType device_type() const = 0;

    // DLPack interop for zero-copy with PyTorch/JAX
    virtual DLManagedTensor* to_dlpack() = 0;
    // static std::unique_ptr<DeviceBuffer> from_dlpack(DLManagedTensor* tensor); // To be implemented in concrete classes or factory
};

enum class ReduceOp {
    SUM,
    PROD,
    MIN,
    MAX
};

class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;

    // Factory: auto-detects best available backend
    static std::unique_ptr<ComputeBackend> Create(DeviceType type, int device_id = 0);

    // Memory management
    virtual std::unique_ptr<DeviceBuffer> allocate(size_t bytes, DType dtype) = 0;
    virtual void copy_to_device(DeviceBuffer* dst, const void* src, size_t bytes) = 0;
    virtual void copy_to_device(DeviceBuffer* dst, size_t dst_offset, const void* src, size_t bytes) {
        // Default implementation throws (or could be pure virtual)
        // Making it virtual with default implementation to avoid breaking other backends immediately if any
        (void)dst; (void)dst_offset; (void)src; (void)bytes;
        throw std::runtime_error("copy_to_device with offset not implemented");
    }
    virtual void copy_to_host(void* dst, const DeviceBuffer* src, size_t bytes) = 0;
    virtual void copy_device_to_device(DeviceBuffer* dst, size_t dst_offset, const DeviceBuffer* src, size_t src_offset, size_t bytes) {
         (void)dst; (void)dst_offset; (void)src; (void)src_offset; (void)bytes;
         throw std::runtime_error("copy_device_to_device not implemented");
    }

    // Kernel execution (dispatches to Triton PTX or PJRT StableHLO)
    virtual void execute_kernel(
        const std::string& kernel_name,
        const std::vector<DeviceBuffer*>& inputs,
        const std::vector<DeviceBuffer*>& outputs,
        const KernelConfig& config,
        const std::vector<KernelParam>& params = {}
    ) = 0;

    // Collective operations (NCCL for CUDA, PJRT collectives for TPU)
    virtual void all_reduce(DeviceBuffer* buffer, ReduceOp op = ReduceOp::SUM) = 0;
    virtual void all_gather(DeviceBuffer* send, DeviceBuffer* recv) = 0;

    // Synchronization
    virtual void synchronize() = 0;

    // Device info
    virtual DeviceType type() const = 0;
    virtual std::string device_name() const = 0;
    virtual size_t total_memory() const = 0;
};

} // namespace infra
} // namespace freellm
