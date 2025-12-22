#pragma once

#include "compute_backend.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

namespace freellm {
namespace infra {

// Forward declarations of PJRT C API structs
struct PJRT_Api;
struct PJRT_Client;
struct PJRT_Buffer;
struct PJRT_LoadedExecutable;

class PJRTDeviceBuffer : public DeviceBuffer {
    PJRT_Buffer* buffer_;
    size_t size_;

public:
    PJRTDeviceBuffer(PJRT_Buffer* buffer, size_t size, DType dtype);
    ~PJRTDeviceBuffer() override;

    void* raw_ptr() override;
    const void* raw_ptr() const override;
    size_t size_bytes() const override;
    DeviceType device_type() const override;

    DLManagedTensor* to_dlpack() override;

    PJRT_Buffer* pjrt_buffer();
};

class PJRTBackend : public ComputeBackend {
    void* lib_handle_;
    const PJRT_Api* api_;
    PJRT_Client* client_;
    std::unordered_map<std::string, PJRT_LoadedExecutable*> executables_;

public:
    PJRTBackend(const std::string& plugin_path);
    ~PJRTBackend() override;

    void load_stablehlo_kernels(const std::string& hlo_dir);

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

    DeviceType type() const override;
    std::string device_name() const override;
    size_t total_memory() const override;
};

} // namespace infra
} // namespace freellm
