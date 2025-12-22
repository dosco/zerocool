#pragma once

#include <string>
#ifdef FREELM_ENABLE_CUDA
#include <cuda_runtime.h>
#endif
#include <stdexcept>

namespace freellm {
namespace infra {

struct GPUCapabilities {
    int compute_capability;     // 80=A100, 86=3090, 89=4090, 90=H100
    bool has_fp8_tensor_cores;  // H100 only
    bool has_int8_tensor_cores; // A100, H100
    bool has_nvlink;
    size_t vram_bytes;
    size_t memory_bandwidth_gbps; // Estimated
    std::string device_name;

    static GPUCapabilities detect(int device_id = 0) {
#ifdef FREELM_ENABLE_CUDA
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to get device properties");
        }

        GPUCapabilities caps;
        caps.compute_capability = prop.major * 10 + prop.minor;
        caps.device_name = prop.name;
        caps.vram_bytes = prop.totalGlobalMem;
        
        // Heuristic for tensor cores
        caps.has_fp8_tensor_cores = (caps.compute_capability >= 90);
        caps.has_int8_tensor_cores = (caps.compute_capability >= 80);

        // NVLink detection (simplified, real detection might need NVML)
        // Checking if there are multiple devices and peer access is possible might be a proxy,
        // but prop.isMultiGpuBoard is deprecated/not enough.
        // For now, assume false unless we have a better way.
        caps.has_nvlink = false; 

        // Bandwidth estimation (simplified)
        // memoryClockRate is in kHz, memoryBusWidth in bits
        // Bandwidth (GB/s) = (Clock * BusWidth * 2 (DDR)) / 8 / 1e6
        // Note: memoryClockRate is often the boost clock.
        double clock_ghz = prop.memoryClockRate / 1e6;
        double bus_width_bytes = prop.memoryBusWidth / 8.0;
        // GDDR6/6X/HBM effective rate multiplier varies.
        // This is a rough estimate.
        caps.memory_bandwidth_gbps = static_cast<size_t>(clock_ghz * bus_width_bytes * 2); 

        return caps;
#else
        throw std::runtime_error("CUDA not enabled");
#endif
    }
};

} // namespace infra
} // namespace freellm
