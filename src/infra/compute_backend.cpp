#include "infra/compute_backend.hpp"
#ifdef FREELM_ENABLE_CUDA
#include "infra/cuda_backend.hpp"
#endif
#ifdef __APPLE__
#include "infra/metal_backend.hpp"
#endif
#include "infra/pjrt_backend.hpp"
#include <stdexcept>

namespace freellm {
namespace infra {

std::unique_ptr<ComputeBackend> ComputeBackend::Create(DeviceType type, int device_id) {
    (void)device_id; // Unused if CUDA is disabled
    switch (type) {
#ifdef FREELM_ENABLE_CUDA
        case DeviceType::CUDA:
            return std::make_unique<CUDABackend>(device_id);
#endif
#ifdef __APPLE__
        case DeviceType::METAL:
            return std::make_unique<MetalBackend>(device_id);
#endif
        case DeviceType::PJRT_TPU:
        case DeviceType::PJRT_GPU:
        case DeviceType::PJRT_CPU:
            // For PJRT, we need to know the plugin path.
            // For now, we'll hardcode or assume a default, or throw if not configured.
            // In a real app, this might come from a config or environment variable.
            // Let's assume a default libtpu.so for TPU.
            if (type == DeviceType::PJRT_TPU) {
                return std::make_unique<PJRTBackend>("libtpu.so");
            } else {
                throw std::runtime_error("PJRT backend type not yet fully supported in factory without path");
            }
        default:
#ifdef FREELM_ENABLE_CUDA
            if (type == DeviceType::CUDA) {
                // Should have been handled above, but just in case
                return std::make_unique<CUDABackend>(device_id);
            }
#endif
            throw std::runtime_error("Unknown device type or backend not enabled");
    }
}

} // namespace infra
} // namespace freellm
