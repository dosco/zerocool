#pragma once

#include "compute_backend.hpp" // For Dim3
#include <string>
#include <vector>
#include <unordered_map>
#ifdef FREELM_ENABLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include <iostream>
#include <stdexcept>

namespace freellm {
namespace infra {

class TritonKernelLoader {
public:
    TritonKernelLoader() {}
    
    ~TritonKernelLoader() {
#ifdef FREELM_ENABLE_CUDA
        for (auto& pair : modules_) {
            cuModuleUnload(pair.second);
        }
#endif
    }

    // Load pre-compiled Triton kernel (.cubin or .ptx)
    void load_kernel(const std::string& name, const std::string& path) {
#ifdef FREELM_ENABLE_CUDA
        CUmodule module;
        CUfunction function;
        
        CUresult res = cuModuleLoad(&module, path.c_str());
        if (res != CUDA_SUCCESS) {
            const char* err;
            cuGetErrorString(res, &err);
            throw std::runtime_error("Failed to load module " + path + ": " + (err ? err : "unknown error"));
        }

        res = cuModuleGetFunction(&function, module, name.c_str());
        if (res != CUDA_SUCCESS) {
            const char* err;
            cuGetErrorString(res, &err);
            throw std::runtime_error("Failed to get function " + name + ": " + (err ? err : "unknown error"));
        }

        modules_[name] = module;
        kernels_[name] = function;
#else
        (void)name;
        (void)path;
        throw std::runtime_error("CUDA not enabled");
#endif
    }

    // Invoke with CUDA driver API
    void invoke(
        const std::string& name,
        const std::vector<void*>& args,
        Dim3 grid,
        Dim3 block,
        size_t shared_mem,
        void* stream // cudaStream_t cast to void*
    ) {
#ifdef FREELM_ENABLE_CUDA
        if (kernels_.find(name) == kernels_.end()) {
            throw std::runtime_error("Kernel " + name + " not found");
        }

        CUfunction func = kernels_[name];
        
        // cuLaunchKernel takes void** args
        // We need to be careful here. args is vector<void*>, which contains pointers to the arguments.
        // cuLaunchKernel expects an array of pointers to arguments.
        // So we can pass args.data() directly if args contains pointers to the actual data.
        // However, Triton/CUDA args usually need to be pointers to the values.
        // The caller of invoke needs to ensure args contains pointers to the values (e.g. &ptr, &int_val).
        
        CUresult res = cuLaunchKernel(
            func,
            grid.x, grid.y, grid.z,
            block.x, block.y, block.z,
            shared_mem,
            static_cast<cudaStream_t>(stream),
            const_cast<void**>(args.data()),
            nullptr // extra
        );

        if (res != CUDA_SUCCESS) {
            const char* err;
            cuGetErrorString(res, &err);
            throw std::runtime_error("Failed to launch kernel " + name + ": " + (err ? err : "unknown error"));
        }
#else
        (void)name;
        (void)args;
        (void)grid;
        (void)block;
        (void)shared_mem;
        (void)stream;
        throw std::runtime_error("CUDA not enabled");
#endif
    }

    // Check if kernel exists
    bool has_kernel(const std::string& name) const {
#ifdef FREELM_ENABLE_CUDA
        return kernels_.find(name) != kernels_.end();
#else
        (void)name;
        return false;
#endif
    }

private:
#ifdef FREELM_ENABLE_CUDA
    std::unordered_map<std::string, CUfunction> kernels_;
    std::unordered_map<std::string, CUmodule> modules_;
#endif
};

} // namespace infra
} // namespace freellm
