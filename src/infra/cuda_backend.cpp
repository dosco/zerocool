#include "infra/cuda_backend.hpp"
#include "infra/kernel_loader.hpp"
#include <iostream>
#include <vector>

#ifdef FREELM_ENABLE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#endif

namespace freellm {
namespace infra {

#ifdef FREELM_ENABLE_CUDA

// CUDADeviceBuffer Implementation

CUDADeviceBuffer::CUDADeviceBuffer(size_t size, DType dtype) : size_(size), dtype_(dtype) {
    CUDA_CHECK(cudaMalloc(&ptr_, size));
}

CUDADeviceBuffer::~CUDADeviceBuffer() {
    if (ptr_) {
        cudaFree(ptr_);
    }
}

void* CUDADeviceBuffer::raw_ptr() {
    return ptr_;
}

const void* CUDADeviceBuffer::raw_ptr() const {
    return ptr_;
}

size_t CUDADeviceBuffer::size_bytes() const {
    return size_;
}

DeviceType CUDADeviceBuffer::device_type() const {
    return DeviceType::CUDA;
}

DLManagedTensor* CUDADeviceBuffer::to_dlpack() {
    // TODO: Implement DLPack export
    return nullptr;
}

// CUDABackend Implementation

CUDABackend::CUDABackend(int device_id) : device_id_(device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaStreamCreate(&stream_));
    kernel_loader_ = std::make_unique<TritonKernelLoader>();
}

CUDABackend::~CUDABackend() {
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

std::unique_ptr<DeviceBuffer> CUDABackend::allocate(size_t bytes, DType dtype) {
    return std::make_unique<CUDADeviceBuffer>(bytes, dtype);
}

void CUDABackend::copy_to_device(DeviceBuffer* dst, const void* src, size_t bytes) {
    if (dst->device_type() != DeviceType::CUDA) {
        throw std::runtime_error("Destination buffer is not a CUDA buffer");
    }
    CUDA_CHECK(cudaMemcpyAsync(dst->raw_ptr(), src, bytes, cudaMemcpyHostToDevice, stream_));
}

void CUDABackend::copy_to_host(void* dst, const DeviceBuffer* src, size_t bytes) {
    if (src->device_type() != DeviceType::CUDA) {
        throw std::runtime_error("Source buffer is not a CUDA buffer");
    }
    CUDA_CHECK(cudaMemcpyAsync(dst, src->raw_ptr(), bytes, cudaMemcpyDeviceToHost, stream_));
}

void CUDABackend::execute_kernel(
    const std::string& kernel_name,
    const std::vector<DeviceBuffer*>& inputs,
    const std::vector<DeviceBuffer*>& outputs,
    const KernelConfig& config,
    const std::vector<KernelParam>& /*params*/
) {
    // 1. Load kernel if not loaded
    // For this example, we assume kernels are pre-compiled and located in a known directory
    // or passed via some registry.
    // Let's assume a convention: kernels are in "kernels/ptx/<name>.ptx"
    
    if (!kernel_loader_->has_kernel(kernel_name)) {
        // TODO: Make this path configurable
        std::string ptx_path = "kernels/ptx/" + kernel_name + ".ptx";
        try {
            kernel_loader_->load_kernel(kernel_name, ptx_path);
        } catch (const std::exception& e) {
            std::cerr << "Failed to load kernel " << kernel_name << ": " << e.what() << std::endl;
            throw;
        }
    }

    // 2. Prepare arguments
    // Triton kernels expect arguments as pointers to values.
    // For pointers (DeviceBuffer), we pass the address of the pointer (void**).
    // But wait, cuLaunchKernel expects void** kernelParams, where kernelParams[i] is a pointer to the i-th argument.
    // If the i-th argument is a pointer (void*), kernelParams[i] should be void**.
    
    std::vector<void*> args;
    args.reserve(inputs.size() + outputs.size());

    // We need to store the actual pointer values somewhere so we can take their addresses
    // This is tricky because vector reallocations invalidates pointers.
    // We'll use a vector of void* to hold the device pointers, and then pass addresses of elements in that vector.
    
    // Actually, TritonKernelLoader::invoke takes vector<void*>& args.
    // And inside it calls cuLaunchKernel with args.data().
    // This implies args[i] MUST be the pointer to the argument value.
    
    // So if argument is a device pointer (void* ptr), args[i] must be &ptr.
    // We need a stable storage for these pointers.
    
    // Let's create a struct or vector to hold the parameter values.
    // Since we only have DeviceBuffers here, they are all pointers.
    // But real kernels might have scalar args (int, float).
    // The current ComputeBackend interface doesn't support scalar args well (everything is DeviceBuffer*).
    // We might need to extend it or assume scalars are passed as 1-element buffers (inefficient) or via config?
    // For now, let's assume all args are buffers.
    
    std::vector<void*> param_values;
    param_values.reserve(inputs.size() + outputs.size());
    
    for (auto* buf : inputs) {
        param_values.push_back(buf->raw_ptr());
    }
    for (auto* buf : outputs) {
        param_values.push_back(buf->raw_ptr());
    }
    
    // Now construct the args vector of pointers to values
    for (size_t i = 0; i < param_values.size(); ++i) {
        args.push_back(&param_values[i]);
    }

    // 3. Launch
    kernel_loader_->invoke(
        kernel_name,
        args,
        config.grid,
        config.block,
        config.shared_mem_bytes,
        (void*)stream_
    );
}

void CUDABackend::all_reduce(DeviceBuffer* buffer, ReduceOp op) {
    // TODO: Implement NCCL
}

void CUDABackend::all_gather(DeviceBuffer* send, DeviceBuffer* recv) {
    // TODO: Implement NCCL
}

void CUDABackend::synchronize() {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

std::string CUDABackend::device_name() const {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id_);
    return std::string(prop.name);
}

size_t CUDABackend::total_memory() const {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return total;
}

#endif // FREELM_ENABLE_CUDA

} // namespace infra
} // namespace freellm
