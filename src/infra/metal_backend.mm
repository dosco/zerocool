#include "infra/metal_backend.hpp"
#include "utils/cache_utils.hpp"
#include <iostream>
#include <stdexcept>
#include <CoreFoundation/CoreFoundation.h>
#include <fstream>
#include <sstream>
#include <vector>

#ifdef __APPLE__
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace freellm {
namespace infra {

// MetalDeviceBuffer Implementation

MetalDeviceBuffer::MetalDeviceBuffer(void* buffer, size_t size, DType /*dtype*/)
    : buffer_(buffer), size_(size) {}

MetalDeviceBuffer::~MetalDeviceBuffer() {
    if (buffer_) {
        CFRelease(buffer_);
    }
}

void* MetalDeviceBuffer::raw_ptr() {
    return [(__bridge id<MTLBuffer>)buffer_ contents];
}

const void* MetalDeviceBuffer::raw_ptr() const {
    return [(__bridge id<MTLBuffer>)buffer_ contents];
}

size_t MetalDeviceBuffer::size_bytes() const {
    return size_;
}

DeviceType MetalDeviceBuffer::device_type() const {
    return DeviceType::METAL;
}

DLManagedTensor* MetalDeviceBuffer::to_dlpack() {
    // TODO: Implement DLPack export
    return nullptr;
}

// MetalBackend Implementation

MetalBackend::MetalBackend(int device_id) {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (device_id >= (int)[devices count]) {
        throw std::runtime_error("Invalid Metal device ID");
    }
    id<MTLDevice> dev = devices[device_id];
    device_ = (__bridge_retained void*)dev;
    command_queue_ = (__bridge_retained void*)[dev newCommandQueue];
    last_command_buffer_ = nullptr;
    active_command_buffer_ = nullptr;

    // Load custom kernels with caching
    NSError* error = nil;
    std::string metal_lib_path = "/repo/kernels/metal/kernels.metal";
    
    // Check if source file exists
    if (!std::filesystem::exists(metal_lib_path)) {
        std::cerr << "Warning: Could not open " << metal_lib_path << ". Custom kernels disabled." << std::endl;
        library_ = nullptr;
    } else {
        // Compute hash of source file for cache validation
        std::string source_hash = cache::compute_file_hash(metal_lib_path);
        std::string short_hash_str = cache::short_hash(source_hash);
        
        // Setup cache paths
        std::filesystem::path cache_dir = cache::ensure_cache_dir("metal");
        std::filesystem::path cached_lib_path = cache_dir / ("kernels_" + short_hash_str + ".metallib");
        std::filesystem::path metadata_path = cache_dir / ("kernels_" + short_hash_str + ".meta");
        
        id<MTLLibrary> lib = nil;
        
        // Try to load from cache first
        if (std::filesystem::exists(cached_lib_path)) {
            // Validate cache metadata
            std::string cached_hash = cache::read_metadata(metadata_path);
            if (cached_hash == source_hash) {
                // Cache hit - load pre-compiled library
                NSString* cached_path_ns = [NSString stringWithUTF8String:cached_lib_path.c_str()];
                NSURL* cached_url = [NSURL fileURLWithPath:cached_path_ns];
                lib = [dev newLibraryWithURL:cached_url error:&error];
                
                if (lib) {
                    std::cout << "\033[92m  ✓ \033[0m\033[90m[Metal] \033[0mLoaded cached library: \033[2m" 
                              << short_hash_str << "\033[0m" << std::endl;
                } else {
                    std::cerr << "Warning: Failed to load cached library, recompiling..." << std::endl;
                }
            }
        }
        
        // Cache miss or invalid - compile from source
        if (!lib) {
            std::cout << "\033[94m  ⚡ \033[0m\033[93m[Metal] \033[0mCompiling kernels..." << std::endl;
            
            std::ifstream file(metal_lib_path);
            std::stringstream buffer;
            buffer << file.rdbuf();
            NSString* source = [NSString stringWithUTF8String:buffer.str().c_str()];
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            
            lib = [dev newLibraryWithSource:source options:options error:&error];
            
            if (!lib) {
                std::cerr << "Error compiling Metal library: " << [[error localizedDescription] UTF8String] << std::endl;
                library_ = nullptr;
            } else {
                // Try to cache the compiled library using metal command-line compiler
                // This creates a .metallib file that can be loaded directly
                std::string air_path = cached_lib_path.string() + ".air";
                std::string metal_cmd = "xcrun -sdk macosx metal -c " + metal_lib_path + 
                                       " -o " + air_path + " 2>/dev/null";
                std::string metallib_cmd = "xcrun -sdk macosx metallib " + air_path + 
                                          " -o " + cached_lib_path.string() + " 2>/dev/null";
                
                // Execute compilation commands
                int ret1 = system(metal_cmd.c_str());
                if (ret1 == 0) {
                    int ret2 = system(metallib_cmd.c_str());
                    if (ret2 == 0) {
                        // Save metadata
                        cache::write_metadata(metadata_path, source_hash);
                        std::cout << "\033[92m  ✓ \033[0m\033[90m[Metal] \033[0mCached compiled library: \033[2m" 
                                  << short_hash_str << "\033[0m" << std::endl;
                        
                        // Clean up intermediate .air file
                        std::filesystem::remove(air_path);
                    }
                }
            }
        }
        
        if (lib) {
            library_ = (__bridge_retained void*)lib;
            
            // Helper to load a kernel and handle errors
            // Helper to load a kernel and handle errors
            auto load_kernel = [&](const std::string& name) {
                id<MTLFunction> func = [lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
                if (func) {
                    kernels_[name] = (__bridge_retained void*)func;
                } else {
                    std::cerr << "Warning: Function " << name << " not found in library" << std::endl;
                }
            };

            load_kernel("rms_norm");
            load_kernel("rope");
            load_kernel("rope_batched");
            load_kernel("rope_ragged");
            load_kernel("silu");
            load_kernel("mul");
            load_kernel("add");
            load_kernel("copy_kv");
            load_kernel("copy_kv_batched");
            load_kernel("gemv_q4_0");
            load_kernel("gemm_q4_0");
            load_kernel("gemm_f32");
            load_kernel("gqa_attention_prefill");
            load_kernel("gqa_attention_prefill_ragged");
            load_kernel("gqa_attention_prefill_int8");
            load_kernel("paged_attention");
            load_kernel("paged_attention_int8");
            load_kernel("quantize_store_kv");

            // MoE kernels
            load_kernel("moe_gate_softmax");
            load_kernel("moe_topk_experts");
            load_kernel("moe_aggregate");
            load_kernel("moe_swiglu_ffn");
            load_kernel("softmax");
        }
    }
}

MetalBackend::~MetalBackend() {
    if (device_) CFRelease(device_);
    if (command_queue_) CFRelease(command_queue_);
    if (library_) CFRelease(library_);
    if (last_command_buffer_) CFRelease(last_command_buffer_);
    if (active_command_buffer_) CFRelease(active_command_buffer_);
    for (auto& pair : kernels_) {
        CFRelease(pair.second);
    }
    for (auto& pair : pso_cache_) {
        CFRelease(pair.second);
    }
    for (auto& pair : mps_matmul_cache_) {
        CFRelease(pair.second);
    }
}

std::unique_ptr<DeviceBuffer> MetalBackend::allocate(size_t bytes, DType dtype) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
    id<MTLBuffer> buffer = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!buffer) {
        throw std::runtime_error("Failed to allocate Metal buffer");
    }
    return std::make_unique<MetalDeviceBuffer>((__bridge_retained void*)buffer, bytes, dtype);
}

void MetalBackend::copy_to_device(DeviceBuffer* dst, const void* src, size_t bytes) {
    if (dst->device_type() != DeviceType::METAL) {
        throw std::runtime_error("Destination buffer is not a Metal buffer");
    }
    memcpy(dst->raw_ptr(), src, bytes);
}

void MetalBackend::copy_to_device(DeviceBuffer* dst, size_t dst_offset, const void* src, size_t bytes) {
    if (dst->device_type() != DeviceType::METAL) {
        throw std::runtime_error("Destination buffer is not a Metal buffer");
    }
    // MetalDeviceBuffer uses shared memory (StorageModeShared), so raw_ptr() returns a valid host pointer
    // We can just add the offset to the pointer
    uint8_t* dst_ptr = static_cast<uint8_t*>(dst->raw_ptr());
    memcpy(dst_ptr + dst_offset, src, bytes);
}

void MetalBackend::copy_to_host(void* dst, const DeviceBuffer* src, size_t bytes) {
    if (src->device_type() != DeviceType::METAL) {
        throw std::runtime_error("Source buffer is not a Metal buffer");
    }
    synchronize(); 
    memcpy(dst, src->raw_ptr(), bytes);
}

void MetalBackend::copy_device_to_device(DeviceBuffer* dst, size_t dst_offset, const DeviceBuffer* src, size_t src_offset, size_t bytes) {
    if (dst->device_type() != DeviceType::METAL || src->device_type() != DeviceType::METAL) {
        throw std::runtime_error("Buffers must be Metal buffers");
    }
    // Ensure GPU has finished writing to src (if applicable)
    synchronize();
    
    // Metal uses shared memory, so raw_ptr() is valid on host
    uint8_t* dst_ptr = static_cast<uint8_t*>(dst->raw_ptr());
    const uint8_t* src_ptr = static_cast<const uint8_t*>(src->raw_ptr());
    memcpy(dst_ptr + dst_offset, src_ptr + src_offset, bytes);
}

void MetalBackend::execute_kernel(
    const std::string& kernel_name,
    const std::vector<DeviceBuffer*>& inputs,
    const std::vector<DeviceBuffer*>& outputs,
    const KernelConfig& config,
    const std::vector<KernelParam>& params
) {
    if (kernel_name == "matmul" || kernel_name == "matmul_trans_b") {
        // ...
    } else if (kernels_.find(kernel_name) != kernels_.end()) {
        // Custom kernel execution
        // std::cout << "Executing kernel: " << kernel_name << std::endl;
        id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)command_queue_;
        
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        // active_command_buffer_ logic removed
        
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        id<MTLFunction> func = (__bridge id<MTLFunction>)kernels_[kernel_name];
        id<MTLComputePipelineState> pso = nil;
        
        auto it = pso_cache_.find(kernel_name);
        if (it != pso_cache_.end()) {
            pso = (__bridge id<MTLComputePipelineState>)it->second;
        } else {
            std::cout << "\033[94m  ⚡ \033[0m\033[93m[Metal] \033[0mCompiling: \033[1m" << kernel_name << "\033[0m" << std::endl;
            NSError* error = nil;
            pso = [dev newComputePipelineStateWithFunction:func error:&error];
            if (!pso) {
                throw std::runtime_error("Failed to create PSO for " + kernel_name + ": " + [[error localizedDescription] UTF8String]);
            }
            pso_cache_[kernel_name] = (__bridge_retained void*)pso;
        }
        
        [encoder setComputePipelineState:pso];
        
        // Bind buffers
        for (size_t i = 0; i < inputs.size(); ++i) {
            MetalDeviceBuffer* buf = static_cast<MetalDeviceBuffer*>(inputs[i]);
            [encoder setBuffer:(__bridge id<MTLBuffer>)buf->metal_buffer() offset:0 atIndex:i];
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
            MetalDeviceBuffer* buf = static_cast<MetalDeviceBuffer*>(outputs[i]);
            [encoder setBuffer:(__bridge id<MTLBuffer>)buf->metal_buffer() offset:0 atIndex:inputs.size() + i];
        }
        
        // Handle params
        size_t param_idx = inputs.size() + outputs.size();
        for (const auto& param : params) {
            std::visit([&](auto&& arg) {
                [encoder setBytes:&arg length:sizeof(arg) atIndex:param_idx++];
            }, param);
        }
        
        // Handle scalars/constants if any?
        // Our current interface doesn't support passing scalars easily except via config or buffers.
        // For RMSNorm, we need N and epsilon.
        // For RoPE, head_dim.
        // We'll assume these are passed as 1-element buffers or we hack it for now.
        // Or we use setBytes if we can extract from config?
        // The config struct has grid/block/shared_mem.
        // Let's assume for this demo that scalars are passed as the LAST input buffers (inefficient but works).
        // OR we can encode them into the command buffer using setBytes if we knew the values.
        // But we don't have the values here, only DeviceBuffer*.
        // So the user must have put scalars into DeviceBuffers.
        
        // Dispatch
        MTLSize gridSize = MTLSizeMake(config.grid.x, config.grid.y, config.grid.z);
        MTLSize threadGroupSize = MTLSizeMake(config.block.x, config.block.y, config.block.z);
        
        // Metal requires threadGroupSize to be <= maxTotalThreadsPerThreadgroup
        // and dimensions to be supported.
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        
        // std::cout << "  [Metal] Dispatched " << kernel_name << std::endl;
        
        if (last_command_buffer_) CFRelease(last_command_buffer_);
        last_command_buffer_ = (__bridge_retained void*)commandBuffer;
        
    } else {
        std::cerr << "Warning: Unknown kernel " << kernel_name << ", ignoring." << std::endl;
    }
    
    synchronize(); // Force sync for debugging/stability
}

void MetalBackend::all_reduce(DeviceBuffer* /*buffer*/, ReduceOp /*op*/) {
    // Not implemented for single device
}

void MetalBackend::all_gather(DeviceBuffer* /*send*/, DeviceBuffer* /*recv*/) {
    // Not implemented for single device
}

void MetalBackend::synchronize() {
    // active_command_buffer_ logic removed

    if (last_command_buffer_) {
        // std::cout << "Synchronizing..." << std::endl;
        id<MTLCommandBuffer> cmdBuf = (__bridge id<MTLCommandBuffer>)last_command_buffer_;
        [cmdBuf waitUntilCompleted];
        
        if (cmdBuf.status == MTLCommandBufferStatusError) {
            std::cerr << "  [Metal] Command buffer error: " << [[cmdBuf.error localizedDescription] UTF8String] << std::endl;
        }
    }
}

DeviceType MetalBackend::type() const {
    return DeviceType::METAL;
}

std::string MetalBackend::device_name() const {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
    return std::string([[dev name] UTF8String]);
}

size_t MetalBackend::total_memory() const {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
    return [dev recommendedMaxWorkingSetSize];
}

} // namespace infra
} // namespace freellm

#endif
