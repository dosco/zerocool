#pragma once

#include "compute_backend.hpp"
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <unordered_map>
#include <string>

#ifdef __APPLE__
// No Metal includes here to keep it pure C++ compatible
#endif

namespace freellm {
namespace infra {

#ifdef __APPLE__
class MetalDeviceBuffer : public DeviceBuffer {
    void* buffer_; // id<MTLBuffer>
    size_t size_;
    // DType dtype_; // Unused for now

public:
    MetalDeviceBuffer(void* buffer, size_t size, DType /*dtype*/);
    ~MetalDeviceBuffer() override;

    void* raw_ptr() override; // Returns host pointer (shared memory)
    const void* raw_ptr() const override;
    size_t size_bytes() const override;
    DeviceType device_type() const override;

    DLManagedTensor* to_dlpack() override;
    
    void* metal_buffer() const { return buffer_; }
};

class MetalBackend : public ComputeBackend {
    void* device_;        // id<MTLDevice>
    void* command_queue_; // id<MTLCommandQueue>
    void* library_; // MTLLibrary
    std::map<std::string, void*> kernels_; // Name -> MTLFunction
    std::map<std::string, void*> pso_cache_; // Name -> MTLComputePipelineState
    std::map<std::string, void*> mps_matmul_cache_; // Key -> MPSMatrixMultiplication
    void* last_command_buffer_; // MTLCommandBuffer (retained)
    void* active_command_buffer_; // MTLCommandBuffer (retained, currently building)

public:
    MetalBackend(int device_id = 0);
    ~MetalBackend() override;

    std::unique_ptr<DeviceBuffer> allocate(size_t bytes, DType dtype) override;
    void copy_to_device(DeviceBuffer* dst, const void* src, size_t bytes) override;
    void copy_to_device(DeviceBuffer* dst, size_t dst_offset, const void* src, size_t bytes) override;
    void copy_to_host(void* dst, const DeviceBuffer* src, size_t bytes) override;
    void copy_device_to_device(DeviceBuffer* dst, size_t dst_offset, const DeviceBuffer* src, size_t src_offset, size_t bytes) override;

    void execute_kernel(
        const std::string& kernel_name,
        const std::vector<DeviceBuffer*>& inputs,
        const std::vector<DeviceBuffer*>& outputs,
        const KernelConfig& config,
        const std::vector<KernelParam>& params = {}
    ) override;

    void all_reduce(DeviceBuffer* buffer, ReduceOp op = ReduceOp::SUM) override;
    void all_gather(DeviceBuffer* send, DeviceBuffer* recv) override;

    void synchronize() override;

    DeviceType type() const override;
    std::string device_name() const override;
    size_t total_memory() const override;
    
    void* device() const { return device_; }
    void* command_queue() const { return command_queue_; }
};
#endif

} // namespace infra
} // namespace freellm
