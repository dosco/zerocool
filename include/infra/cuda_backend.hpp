#pragma once

#include "compute_backend.hpp"
#include <memory>
#include <string>
#include <vector>

#ifdef FREELM_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace freellm {
namespace infra {

class TritonKernelLoader;

#ifdef FREELM_ENABLE_CUDA

// Helper for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            throw std::runtime_error("CUDA error"); \
        } \
    } while (0)

class CUDADeviceBuffer : public DeviceBuffer {
    void* ptr_;
    size_t size_;
    DType dtype_;

public:
    CUDADeviceBuffer(size_t size, DType dtype);
    ~CUDADeviceBuffer() override;

    void* raw_ptr() override;
    const void* raw_ptr() const override;
    size_t size_bytes() const override;
    DeviceType device_type() const override;

    DLManagedTensor* to_dlpack() override;
};

class CUDABackend : public ComputeBackend {
    int device_id_;
    cudaStream_t stream_;
    std::unique_ptr<TritonKernelLoader> kernel_loader_;

public:
    CUDABackend(int device_id = 0);
    ~CUDABackend() override;

    std::unique_ptr<DeviceBuffer> allocate(size_t bytes, DType dtype) override;
    void copy_to_device(DeviceBuffer* dst, const void* src, size_t bytes) override;
    void copy_to_host(void* dst, const DeviceBuffer* src, size_t bytes) override;

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

    DeviceType type() const override { return DeviceType::CUDA; }
    std::string device_name() const override;
    size_t total_memory() const override;
    
    cudaStream_t stream() const { return stream_; }
};

#endif // FREELM_ENABLE_CUDA

} // namespace infra
} // namespace freellm
