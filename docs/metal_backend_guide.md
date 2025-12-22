# Metal Backend Implementation Guide

This guide provides a detailed overview of the Metal backend implementation in FreeLLM. It is designed to help new contributors understand the architecture, memory management, and kernel execution model, enabling them to quickly add new features or optimize existing ones.

## 1. Architecture Overview

The Metal backend is implemented as a subclass of the `ComputeBackend` interface, which provides a unified abstraction for device-agnostic operations.

-   **`ComputeBackend` (`include/infra/compute_backend.hpp`)**: The abstract base class defining the interface for memory allocation, data transfer, and kernel execution.
-   **`MetalBackend` (`src/infra/metal_backend.mm`)**: The concrete implementation for Apple Silicon GPUs using the Metal API.
-   **`MetalDeviceBuffer`**: Represents a memory buffer on the GPU. It wraps an `id<MTLBuffer>`.

### Key Components

-   **Device**: The `MTLDevice` (e.g., the GPU).
-   **Command Queue**: The `MTLCommandQueue` used to submit work to the GPU.
-   **Library**: The `MTLLibrary` containing compiled Metal kernels (`kernels.metal`).
-   **PSO Cache**: A cache of `MTLComputePipelineState` objects to avoid recompiling kernels at runtime.

## 2. Memory Management

Memory management in Metal is handled through `MetalDeviceBuffer`.

### Allocation
`MetalBackend::allocate` creates a new `MTLBuffer` with `MTLResourceStorageModeShared`.
-   **Shared Memory**: On Apple Silicon, memory is unified. `StorageModeShared` allows both the CPU and GPU to access the memory.
-   **`raw_ptr()`**: Returns a CPU-accessible pointer to the buffer contents. This allows for direct memory access without explicit staging buffers, though synchronization is required.

### Data Transfer
-   **`copy_to_device`**: Copies data from a host pointer to a `DeviceBuffer`. Since we use shared memory, this is often a `memcpy` to the buffer's `contents` pointer.
-   **`copy_to_host`**: Copies data from a `DeviceBuffer` to a host pointer. Requires `synchronize()` to ensure the GPU has finished writing.
-   **`copy_device_to_device`**: Copies data between two `DeviceBuffer`s. Currently implemented via CPU `memcpy` for simplicity (leveraging shared memory), but requires `synchronize()` to ensure data consistency.

**Important**: Always call `synchronize()` before reading data written by the GPU or before writing data that the GPU is about to read if you are bypassing the command buffer dependency chain.

## 3. Kernel Execution

Kernels are written in Metal Shading Language (MSL) and stored in `kernels/metal/kernels.metal`.

### Execution Flow (`execute_kernel`)
1.  **Lookup/Compile**: The backend looks up the kernel name in its `pso_cache_`. If not found, it compiles the function from the library and creates a new `MTLComputePipelineState`.
2.  **Encode**: A `MTLComputeCommandEncoder` is created.
3.  **Bind Resources**:
    -   Input buffers are bound to indices `0` to `N-1`.
    -   Output buffers are bound to indices `N` to `N+M-1`.
    -   Scalar parameters (if any) are encoded using `setBytes` starting at index `N+M`.
4.  **Dispatch**: The kernel is dispatched using `dispatchThreads:threadsPerThreadgroup:`.
    -   **Grid Size**: Total number of threads to launch.
    -   **Threadgroup Size**: Number of threads per group (block).
5.  **Commit**: The command buffer is committed to the queue.

### Adding a New Kernel

1.  **Write the Kernel**: Add your MSL function to `kernels/metal/kernels.metal`.
    ```metal
    kernel void my_new_kernel(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint& size [[buffer(2)]],
        uint id [[thread_position_in_grid]])
    {
        if (id < size) {
            output[id] = input[id] * 2.0f;
        }
    }
    ```
2.  **Register the Kernel**: Add the kernel name to the `kernel_names` list in `MetalBackend::MetalBackend` constructor in `src/infra/metal_backend.mm`.
    ```cpp
    std::vector<std::string> kernel_names = {
        ...,
        "my_new_kernel"
    };
    ```
3.  **Call from C++**: Use `backend->execute_kernel`.
    ```cpp
    KernelConfig config;
    config.grid = Dim3(size, 1, 1);
    config.block = Dim3(256, 1, 1);
    backend->execute_kernel("my_new_kernel", {input_buf}, {output_buf}, config, {(uint)size});
    ```

## 4. Key Kernels

### Attention Mechanisms
-   **`gqa_attention_prefill`**: Optimized for the prefill phase. It processes the entire prompt in parallel. It supports **Paged KV Cache** by reading from non-contiguous memory blocks using a block table.
-   **`paged_attention`**: Optimized for the decode phase. It generates one token at a time, attending to the history stored in the Paged KV Cache.

### Matrix Multiplication
-   **`gemm_q4_0`**: Block-quantized matrix multiplication (4-bit weights). Used for linear layers.
-   **`gemv_q4_0`**: Quantized matrix-vector multiplication (batch size 1).

### KV Cache Management
-   **`copy_kv_batched`**: Copies Key/Value pairs from temporary buffers to the KV cache.
-   **`rope_batched`**: Applies Rotary Positional Embeddings (RoPE) to Q and K vectors.

## 5. Paged KV Cache Integration

The Metal backend is fully integrated with the `PagedKVCache` system (`include/core/paged_kv_cache.hpp`).

-   **`KVCacheManager`**: Manages the global pool of memory blocks. When initialized with a `MetalBackend`, it allocates `global_keys` and `global_values` on the GPU.
-   **`PagedKVCache`**: Represents a sequence's cache. It maintains a `block_table` mapping logical blocks to physical blocks.
-   **Update Process**:
    1.  Compute Q, K, V for the current token(s).
    2.  Call `PagedKVCache::update` to copy K/V into the allocated blocks on the GPU.
    3.  Pass the `block_table` to the attention kernel.

## 6. Debugging Tips

-   **`synchronize()`**: If you see garbage output or race conditions, ensure you are calling `backend->synchronize()` before reading back results or between dependent operations that are not implicitly synchronized by the command buffer.
-   **`capture_gpu_frame`**: Xcode's Metal Frame Capture is invaluable for inspecting buffer contents and kernel execution timing.
-   **Print Debugging**: You can use `printf` inside Metal kernels (output appears in the console/Xcode), but it can be slow.
-   **Immediate Verification**: For critical memory operations, you can map the buffer on the CPU (via `raw_ptr()`) and verify contents immediately after a `synchronize()`.

## 7. Common Pitfalls

-   **SIMD Reductions**: When performing reductions (e.g., sum, max) inside a kernel, remember that `simd_sum` only works within a SIMD group (32 threads on Apple Silicon). If your reduction size exceeds 32 (e.g., `head_dim=64`), you must perform a second reduction step using shared memory (`threadgroup` memory).
-   **Threadgroup Sizes**: Metal has limits on threadgroup sizes (usually 1024 threads max). Ensure your `config.block` dimensions don't exceed this.
-   **Buffer Alignment**: Metal buffers should generally be aligned to page boundaries (4KB) for best performance, though `newBufferWithLength` handles this. Structs passed to kernels should be aligned to their largest member.
