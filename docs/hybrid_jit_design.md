# RFC: Unified Multi-Backend Inference Engine

| Metadata | Details |
| :--- | :--- |
| **Status** | Draft v4 |
| **Target System** | FreeLLM (C++ Inference Engine) |
| **Proposed Tech** | Triton (NVIDIA), Pallas/StableHLO (TPU/AMD), PJRT, CUDA, NCCL |
| **Goal** | Unified inference across NVIDIA GPUs and Google TPUs with AOT-compiled quantized kernels, multi-device tensor parallelism, and flexible architecture support (MoE, Mamba). |
| **Build Dependency** | Python 3.10+ with Triton and JAX/Pallas (build-time only) |
| **Runtime** | Pure C++ (no Python required at runtime) |
| **Hardware Targets** | NVIDIA GPUs (direct CUDA), Google TPUs (PJRT), AMD GPUs (PJRT) |

## 1. Executive Summary

FreeLLM currently relies on hand-written AVX2/NEON intrinsics for CPU inference (see `include/kernels/tensor_ops.hpp`). While performant for CPU, this approach:
- Limits adaptability to new architectures (MoE routing, State Space Models)
- Makes operator fusion (e.g., `SiLU(Gate * Up)`) difficult to implement manually
- Does not leverage GPU/TPU tensor cores for maximum throughput

This RFC proposes a **Unified Multi-Backend Architecture** where:
- **NVIDIA Path**: Triton kernels compiled to PTX, direct CUDA execution, NCCL for multi-GPU
- **TPU/AMD Path**: Pallas kernels compiled to StableHLO, executed via PJRT plugin system
- **Host (C++)**: Unified runtime with `ComputeBackend` interface abstracting device differences
- **Quantization**: AOT-compiled Q4_K/Q8_0 kernels matching GGUF formats on all platforms

**Competitive Moat**: First embeddable C++ runtime supporting both NVIDIA GPUs and Google TPUs with optimized quantized kernels, no Python runtime dependency, deployable from edge to cloud. Unlike vLLM/SGLang (Python ecosystems, NVIDIA-only), FreeLLM ships as a native library targeting multiple hardware platforms.

### 1.1 Why Not Just Use vLLM/SGLang?

| Aspect | vLLM | SGLang | FreeLLM |
|--------|------|--------|---------|
| Runtime | Python + CUDA | Python + Triton | **Pure C++ (runtime)** |
| NVIDIA GPU | Yes | Yes | **Yes (Triton)** |
| Google TPU | No | No | **Yes (PJRT)** |
| AMD GPU | Limited | No | **Yes (PJRT)** |
| Quantized kernels | Marlin (GPTQ/AWQ) | Limited | **Q4_K/Q8_0 (GGUF)** |
| Startup time | Heavy (Python) | Heavy | **Fast (native)** |
| Embeddable | No | No | **Yes** |
| Edge deployment | No | No | **Yes (CPU fallback)** |

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Python Kernel Frontend                             │
│  ┌─────────────────────────────┐  ┌─────────────────────────────────────┐   │
│  │    Triton Kernels (NVIDIA)  │  │    Pallas Kernels (TPU/Portable)    │   │
│  │    • Maximum GPU perf       │  │    • TPU-optimized (8x128 tiles)    │   │
│  │    • Q4_K/Q8_0 quantized    │  │    • StableHLO export               │   │
│  │    • Tensor core utilization│  │    • AMD/Intel via PJRT             │   │
│  └─────────────┬───────────────┘  └───────────────┬─────────────────────┘   │
│                │ PTX/cubin                        │ StableHLO               │
└────────────────┼──────────────────────────────────┼─────────────────────────┘
                 ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        C++ Runtime (Unified Host)                           │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      ComputeBackend Interface                          │  │
│  │    virtual execute_kernel(name, inputs, outputs, config) = 0           │  │
│  │    virtual allocate(bytes, dtype) = 0                                  │  │
│  │    virtual all_reduce(buffer, op) = 0                                  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│        │                                              │                      │
│        ▼                                              ▼                      │
│  ┌─────────────────────┐                    ┌─────────────────────────┐     │
│  │    CUDABackend      │                    │     PJRTBackend         │     │
│  │    • Direct PTX load│                    │     • Plugin loader     │     │
│  │    • NCCL collectives│                   │     • StableHLO exec    │     │
│  │    • cuBLAS fallback │                   │     • DLPack interop    │     │
│  └──────────┬──────────┘                    └───────────┬─────────────┘     │
│             │                                           │                    │
│  ┌──────────▼───────────────────────────────────────────▼────────────────┐  │
│  │                         Inference Engine                               │  │
│  │  • Scheduler (multi-device aware)                                      │  │
│  │  • ShardedPagedKVCache                                                 │  │
│  │  • Sampling/Tokenization                                               │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                 │                                           │
                 ▼                                           ▼
    ┌────────────────────────┐               ┌────────────────────────────┐
    │   NVIDIA GPUs (CUDA)   │               │   PJRT Plugin System       │
    │   ┌──────┐  ┌──────┐   │               │   ┌──────────────────────┐ │
    │   │GPU 0 │◄►│GPU 1 │   │               │   │ libtpu.so (TPU)      │ │
    │   └──────┘  └──────┘   │               │   │ pjrt_gpu.so (AMD)    │ │
    │       ▲ NCCL ▲         │               │   │ pjrt_cpu.so (CPU)    │ │
    └───────┴──────┴─────────┘               │   └──────────────────────┘ │
                                             └────────────────────────────┘
```

### 2.2 Component Responsibilities

| Component | Responsibility | Tech Stack | Location |
| :--- | :--- | :--- | :--- |
| **Triton Frontend** | Define NVIDIA-optimized kernels | Python, Triton | `kernels/triton/` |
| **Pallas Frontend** | Define TPU/portable kernels | Python, JAX Pallas | `kernels/pallas/` |
| **Build System** | Compile Triton→PTX, Pallas→StableHLO | CMake, Python | `CMakeLists.txt`, `scripts/` |
| **ComputeBackend** | Abstract device interface | C++ | `include/infra/compute_backend.hpp` |
| **CUDABackend** | NVIDIA-specific execution | C++, CUDA, NCCL | `include/infra/cuda_backend.hpp` |
| **PJRTBackend** | TPU/AMD/portable execution | C++, PJRT | `include/infra/pjrt_backend.hpp` |
| **Weight Manager** | Load quantized weights, device offload | C++ | `include/infra/weight_manager.hpp` |
| **Core Engine** | Scheduling, KV cache, sampling | C++ | `include/core/` |

### 2.3 Data Flow

1. **Offline (Build Time)**:
   - **NVIDIA path**: Triton kernels compiled to PTX/cubin for SM80, SM86, SM89, SM90
   - **TPU/AMD path**: Pallas kernels compiled to StableHLO bytecode
   - Both paths generate kernel manifest files for C++ loader

2. **Startup**:
   - Detect available hardware (CUDA GPUs, TPU, AMD)
   - Select appropriate `ComputeBackend` implementation
   - **CUDABackend**: Load PTX via CUDA Driver API, initialize NCCL
   - **PJRTBackend**: dlopen plugin (libtpu.so, pjrt_gpu.so), load StableHLO executables
   - Weights loaded to device memory

3. **Inference**:
   - Scheduler selects batch of sequences
   - For each transformer layer:
     - `backend->execute_kernel("flash_attention", ...)`
     - `backend->execute_kernel("swiglu_ffn", ...)`
     - `backend->all_reduce(...)` for tensor parallelism
   - Sample next tokens, update KV cache
   - **Same C++ code works for both CUDA and PJRT backends**

## 3. Detailed Design

### 3.1 GPU Backend: Triton

We use **Triton** instead of raw MLIR→NVVM because:
- Better tensor core support out of the box
- Proven in production (vLLM, xformers)
- Easier to write and maintain kernels
- Auto-tuning capabilities

**Triton Kernel Library:**
```
kernels/triton/
├── attention/
│   ├── flash_attention.py      # Fused QKV attention with RoPE
│   ├── paged_attention.py      # For KV cache
│   └── gqa_attention.py        # Grouped Query Attention
├── positional/
│   └── rope.py                 # Rotary Position Embeddings
├── ffn/
│   ├── swiglu.py               # Fused SwiGLU FFN
│   └── moe_router.py           # MoE gating
├── quantized/
│   ├── q8_matmul.py            # Q8_0 GEMM
│   ├── q4k_matmul.py           # Q4_K GEMM
│   └── mixed_precision.py      # FP16 Q, Q8 K, Q4 V
└── normalization/
    ├── rmsnorm.py              # Fused RMSNorm
    └── layernorm.py            # LayerNorm
```

**Attention with RoPE (Critical for LLaMA-style models):**

RoPE (Rotary Position Embeddings) must be applied to Q and K before attention. For efficiency, we fuse this into the attention kernel:

```python
@triton.jit
def apply_rope(
    x,          # [seq_len, n_heads, head_dim] - Q or K tensor
    cos,        # [seq_len, head_dim/2] - precomputed cos(m*theta)
    sin,        # [seq_len, head_dim/2] - precomputed sin(m*theta)
    HEAD_DIM: tl.constexpr,
):
    """
    Apply rotary embeddings: rotate pairs of dimensions.

    For each pair (x_i, x_{i+d/2}):
        x_i'      = x_i * cos - x_{i+d/2} * sin
        x_{i+d/2}' = x_i * sin + x_{i+d/2} * cos
    """
    half_dim = HEAD_DIM // 2

    # Split into first half and second half
    x_first = x[:, :, :half_dim]
    x_second = x[:, :, half_dim:]

    # Apply rotation
    x_rotated_first = x_first * cos - x_second * sin
    x_rotated_second = x_first * sin + x_second * cos

    # Concatenate
    return tl.cat([x_rotated_first, x_rotated_second], axis=-1)


@triton.jit
def flash_attention_with_rope(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    cos_ptr, sin_ptr, positions_ptr,
    seq_len, n_heads, head_dim,
    stride_qm, stride_qh, stride_qd,
    # ... other strides
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Flash Attention with fused RoPE application.

    1. Load Q, K blocks
    2. Apply RoPE to Q and K using position-dependent cos/sin
    3. Compute attention scores: softmax(Q @ K.T / sqrt(d))
    4. Compute output: scores @ V
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Load position indices for this block
    positions = tl.load(positions_ptr + pid_m * BLOCK_M + tl.arange(0, BLOCK_M))

    # Load Q block
    q_block = tl.load(Q_ptr + ...)  # [BLOCK_M, head_dim]

    # Load cos/sin for these positions
    cos_block = tl.load(cos_ptr + positions[:, None] * head_dim // 2 + ...)
    sin_block = tl.load(sin_ptr + positions[:, None] * head_dim // 2 + ...)

    # Apply RoPE to Q
    q_rotated = apply_rope(q_block, cos_block, sin_block, head_dim)

    # Attention loop over K,V blocks
    acc = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_start in range(0, seq_len, BLOCK_N):
        # Load K block and apply RoPE
        k_block = tl.load(K_ptr + ...)
        k_positions = tl.arange(0, BLOCK_N) + k_start
        k_cos = tl.load(cos_ptr + k_positions[:, None] * head_dim // 2 + ...)
        k_sin = tl.load(sin_ptr + k_positions[:, None] * head_dim // 2 + ...)
        k_rotated = apply_rope(k_block, k_cos, k_sin, head_dim)

        # Compute attention scores
        scores = tl.dot(q_rotated, tl.trans(k_rotated)) / tl.sqrt(float(head_dim))

        # Online softmax update (Flash Attention algorithm)
        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        l_i = alpha * l_i + beta * tl.sum(tl.exp(scores - m_ij[:, None]), axis=1)

        # Load V and accumulate
        v_block = tl.load(V_ptr + ...)
        acc = alpha[:, None] * acc + beta[:, None] * tl.dot(
            tl.exp(scores - m_ij[:, None]), v_block
        )
        m_i = m_new

    # Normalize and store
    o_block = acc / l_i[:, None]
    tl.store(O_ptr + ..., o_block.to(tl.float16))
```

**C++ RoPE Cache Management:**
```cpp
class RoPECache {
    CUdeviceptr cos_cache_;  // [max_seq_len, head_dim/2]
    CUdeviceptr sin_cache_;  // [max_seq_len, head_dim/2]
    int max_seq_len_;
    int head_dim_;
    float theta_base_;       // Usually 10000.0, or 500000.0 for extended context

public:
    RoPECache(int max_seq_len, int head_dim, float theta_base = 10000.0f);

    // Precompute cos/sin for all positions up to max_seq_len
    void initialize(cudaStream_t stream);

    // Get cache pointers for kernel invocation
    void* cos_ptr() const { return reinterpret_cast<void*>(cos_cache_); }
    void* sin_ptr() const { return reinterpret_cast<void*>(sin_cache_); }
};
```

### 3.2 Quantized Kernels (Key Differentiator)

This is where FreeLLM differentiates: GGUF-compatible quantization formats with efficient GPU dequantization.

**Quantized Kernel Types:**
```
├── Q4_K GEMM (4-bit weights × FP16 activations → FP16 output)
├── Q8_0 GEMM (8-bit weights × FP16 activations → FP16 output)
├── Mixed Attention (FP16 Q, Q8 K, Q4 V → FP16 output)
└── Fused Dequant-Compute (on-the-fly dequantization)
```

**Q4_K Format (GGUF-compatible):**

Each Q4_K block encodes 256 weights in 144 bytes:
```
┌────────────────────────────────────────────────────────────┐
│ Q4_K Block Structure (256 weights → 144 bytes)            │
├────────────────────────────────────────────────────────────┤
│ bytes 0-1:   d     (FP16) - super-block scale             │
│ bytes 2-3:   dmin  (FP16) - super-block minimum           │
│ bytes 4-15:  scales_and_mins (12 bytes, 6-bit packed)     │
│              - 8 sub-block scales (6 bits each)           │
│              - 8 sub-block mins (6 bits each)             │
│ bytes 16-143: qs (128 bytes) - 256 4-bit quantized values │
└────────────────────────────────────────────────────────────┘

Dequantization formula:
  weight[i] = d * scale[i/32] * q[i] - dmin * min[i/32]

Where:
  - i/32 selects one of 8 sub-blocks
  - scale[j], min[j] are 6-bit values unpacked from scales_and_mins
```

**Q4_K Triton Kernel:**
```python
@triton.jit
def q4k_matmul_kernel(
    # Inputs
    X_ptr,              # [M, K] FP16 activations
    W_ptr,              # [num_blocks, 144] Q4_K packed blocks
    Y_ptr,              # [M, N] FP16 output
    # Dimensions
    M, N, K,
    # Strides
    stride_xm, stride_xk,
    stride_ym, stride_yn,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Q4_K GEMM with hierarchical dequantization.

    Each GPU thread block processes BLOCK_M × BLOCK_N output tiles.
    Dequantization happens in registers, compute uses tensor cores.
    """
    Q4K_BLOCK_SIZE = 256  # Elements per Q4K block
    Q4K_BYTES = 144       # Bytes per Q4K block

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension in Q4K block-sized chunks
    for k_start in range(0, K, Q4K_BLOCK_SIZE):
        block_idx = k_start // Q4K_BLOCK_SIZE

        # Load Q4K block header for each column
        for n_idx in range(BLOCK_N):
            col = offs_n[0] + n_idx
            block_base = (block_idx * N + col) * Q4K_BYTES

            # Load d, dmin (FP16)
            d = tl.load(W_ptr + block_base).to(tl.float16)
            dmin = tl.load(W_ptr + block_base + 2).to(tl.float16)

            # Load and unpack 6-bit scales/mins (bytes 4-15)
            scales_packed = tl.load(W_ptr + block_base + 4,
                                    mask=tl.arange(0, 12) < 12)

            # Unpack 8 scales and 8 mins from 12 bytes (6-bit each)
            # ... (detailed unpacking logic)

            # Load 4-bit quantized values (bytes 16-143)
            qs = tl.load(W_ptr + block_base + 16,
                         mask=tl.arange(0, 128) < 128)

            # Dequantize 256 weights
            for sub_block in range(8):
                sub_start = sub_block * 32
                scale = scales[sub_block]
                min_val = mins[sub_block]

                for i in range(32):
                    byte_idx = (sub_start + i) // 2
                    q_val = (qs[byte_idx] >> (4 * (i % 2))) & 0xF
                    w_dequant[sub_start + i] = (
                        d * scale * q_val - dmin * min_val
                    )

        # Load FP16 input tile
        x_tile = tl.load(X_ptr + offs_m[:, None] * stride_xm +
                         (k_start + tl.arange(0, Q4K_BLOCK_SIZE))[None, :])

        # Tensor core matmul
        acc += tl.dot(x_tile, w_dequant)

    # Store FP16 output
    tl.store(Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
             acc.to(tl.float16))
```

> **Note**: The above is simplified for clarity. Production kernels need optimized 6-bit unpacking,
> vectorized loads, and careful tiling to match tensor core requirements.

### 3.3 C++ ↔ Triton Integration

Since Triton compiles to PTX, we load and call kernels from C++:

**Kernel Loader (`include/infra/kernel_loader.hpp`):**
```cpp
class TritonKernelLoader {
public:
    TritonKernelLoader();
    ~TritonKernelLoader();

    // Load pre-compiled Triton kernel (.cubin or .ptx)
    void load_kernel(const std::string& name, const std::string& path);

    // Invoke with CUDA driver API
    void invoke(
        const std::string& name,
        const std::vector<void*>& args,
        dim3 grid,
        dim3 block,
        size_t shared_mem,
        cudaStream_t stream
    );

    // Check if kernel exists
    bool has_kernel(const std::string& name) const;

private:
    std::unordered_map<std::string, CUfunction> kernels_;
    std::unordered_map<std::string, CUmodule> modules_;
};
```

**Zero-Copy Integration:**
```cpp
// C++ Tensor holds CUDA device pointer
class CUDATensor {
    CUdeviceptr data_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    DType dtype_;

public:
    void* raw_ptr() { return reinterpret_cast<void*>(data_); }
};

// Invoke Triton kernel with C++ tensors
void invoke_q4k_matmul(
    TritonKernelLoader& loader,
    const CUDATensor& input,      // FP16 [M, K]
    const Q4KWeight& weight,      // Quantized [K, N]
    CUDATensor& output,           // FP16 [M, N]
    cudaStream_t stream
) {
    int M = input.shape()[0];
    int K = input.shape()[1];
    int N = weight.N();

    // Grid/block configuration
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    dim3 grid((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
    dim3 block(128);  // Triton manages thread layout

    std::vector<void*> args = {
        weight.data_ptr(),
        weight.scales_ptr(),
        weight.mins_ptr(),
        input.raw_ptr(),
        output.raw_ptr(),
        &M, &N, &K,
        // ... strides
    };

    loader.invoke("q4k_matmul", args, grid, block, 0, stream);
}
```

### 3.4 ComputeBackend Interface (Unified Abstraction)

The `ComputeBackend` interface abstracts device-specific operations, allowing the same inference code to run on NVIDIA GPUs, TPUs, or AMD GPUs.

```cpp
// include/infra/compute_backend.hpp

enum class DeviceType { CUDA, PJRT_TPU, PJRT_GPU, PJRT_CPU };

class DeviceBuffer {
public:
    virtual ~DeviceBuffer() = default;
    virtual void* raw_ptr() = 0;
    virtual size_t size_bytes() const = 0;
    virtual DeviceType device_type() const = 0;

    // DLPack interop for zero-copy with PyTorch/JAX
    virtual DLManagedTensor* to_dlpack() = 0;
    static std::unique_ptr<DeviceBuffer> from_dlpack(DLManagedTensor* tensor);
};

class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;

    // Factory: auto-detects best available backend
    static std::unique_ptr<ComputeBackend> Create(DeviceType type, int device_id = 0);

    // Memory management
    virtual std::unique_ptr<DeviceBuffer> allocate(size_t bytes, DType dtype) = 0;
    virtual void copy_to_device(DeviceBuffer* dst, const void* src, size_t bytes) = 0;
    virtual void copy_to_host(void* dst, const DeviceBuffer* src, size_t bytes) = 0;

    // Kernel execution (dispatches to Triton PTX or PJRT StableHLO)
    virtual void execute_kernel(
        const std::string& kernel_name,
        const std::vector<DeviceBuffer*>& inputs,
        const std::vector<DeviceBuffer*>& outputs,
        const KernelConfig& config
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
```

### 3.5 PJRT Backend Implementation

For TPU, AMD, and other non-NVIDIA hardware, we use PJRT (Portable JAX Runtime):

```cpp
// include/infra/pjrt_backend.hpp

class PJRTBackend : public ComputeBackend {
    void* lib_handle_;           // dlopen handle to plugin
    const PJRT_Api* api_;        // Function pointer table from GetPjrtApi()
    PJRT_Client* client_;        // Device client

    // Pre-compiled StableHLO executables (from Pallas kernels)
    std::unordered_map<std::string, PJRT_LoadedExecutable*> executables_;

public:
    PJRTBackend(const std::string& plugin_path) {
        // Load plugin dynamically
        lib_handle_ = dlopen(plugin_path.c_str(), RTLD_NOW);
        auto get_api = (const PJRT_Api* (*)())dlsym(lib_handle_, "GetPjrtApi");
        api_ = get_api();

        // Create client
        PJRT_Client_Create_Args args = {.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE};
        api_->PJRT_Client_Create(&args);
        client_ = args.client;
    }

    void load_stablehlo_kernels(const std::string& hlo_dir) {
        // Load each .hlo file and compile
        for (const auto& entry : std::filesystem::directory_iterator(hlo_dir)) {
            if (entry.path().extension() == ".hlo") {
                auto hlo_bytes = read_file(entry.path());

                PJRT_Client_Compile_Args compile_args = {
                    .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
                    .client = client_,
                    .program = hlo_bytes.data(),
                    .program_size = hlo_bytes.size(),
                };
                api_->PJRT_Client_Compile(&compile_args);

                std::string kernel_name = entry.path().stem();
                executables_[kernel_name] = compile_args.executable;
            }
        }
    }

    void execute_kernel(
        const std::string& kernel_name,
        const std::vector<DeviceBuffer*>& inputs,
        const std::vector<DeviceBuffer*>& outputs,
        const KernelConfig& config
    ) override {
        PJRT_LoadedExecutable* exec = executables_.at(kernel_name);

        // Convert DeviceBuffers to PJRT_Buffer*
        std::vector<PJRT_Buffer*> pjrt_inputs;
        for (auto* buf : inputs) {
            pjrt_inputs.push_back(static_cast<PJRTDeviceBuffer*>(buf)->pjrt_buffer());
        }

        // Execute
        PJRT_LoadedExecutable_Execute_Args exec_args = {
            .struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE,
            .executable = exec,
            .argument_lists = &pjrt_inputs,
            // ... other args
        };
        api_->PJRT_LoadedExecutable_Execute(&exec_args);
    }
};
```

### 3.6 Pallas Kernel Compilation Script

```python
#!/usr/bin/env python3
# scripts/compile_pallas.py
"""
Compile Pallas kernels to StableHLO for PJRT execution.
"""

import jax
from jax.experimental import pallas as pl
import sys
import os

def export_kernel_to_hlo(kernel_fn, block_spec, input_shapes, output_path):
    """Export a Pallas kernel to serialized StableHLO."""

    # Create Pallas call with block spec
    pallas_fn = pl.pallas_call(
        kernel_fn,
        out_shape=jax.ShapeDtypeStruct(output_shapes, jax.numpy.float16),
        grid=grid_spec,
        in_specs=[block_spec for _ in input_shapes],
        out_specs=block_spec,
    )

    # JIT and lower to StableHLO
    jitted = jax.jit(pallas_fn)
    lowered = jitted.lower(*[jax.numpy.zeros(s) for s in input_shapes])
    hlo_module = lowered.compiler_ir()

    # Serialize
    bytecode = hlo_module.as_serialized_hlo_module_proto()

    with open(output_path, 'wb') as f:
        f.write(bytecode)

    print(f"Exported {kernel_fn.__name__} → {output_path}")


# TPU-specific block sizes (must align to MXU: 128x128)
TPU_BLOCK_SPEC = pl.BlockSpec((128, 128), lambda i, j: (i, j))

if __name__ == "__main__":
    from kernels.pallas.attention import flash_attention_kernel
    from kernels.pallas.ffn import swiglu_kernel

    os.makedirs("compiled/stablehlo", exist_ok=True)

    export_kernel_to_hlo(
        flash_attention_kernel,
        TPU_BLOCK_SPEC,
        input_shapes=[(1, 32, 4096, 128), ...],  # Q, K, V shapes
        output_path="compiled/stablehlo/flash_attention.hlo"
    )
```

### 3.7 The ABI Layer (MemRef Descriptors)

For MLIR interop (CPU fallback path), we maintain MemRef compatibility:

```cpp
template <typename T, int N>
struct MemRefDescriptor {
    T* allocated;       // Base pointer (for free())
    T* aligned;         // Aligned pointer (for computation)
    int64_t offset;     // Offset from aligned
    int64_t sizes[N];   // Dimension sizes
    int64_t strides[N]; // Stride sizes
};

// Convert CUDA tensor to MemRef (for CPU fallback)
template <typename T, int N>
MemRefDescriptor<T, N> cuda_tensor_to_memref(const CUDATensor& tensor) {
    MemRefDescriptor<T, N> desc;
    desc.allocated = reinterpret_cast<T*>(tensor.raw_ptr());
    desc.aligned = desc.allocated;
    desc.offset = 0;
    for (int i = 0; i < N; i++) {
        desc.sizes[i] = tensor.shape()[i];
        desc.strides[i] = tensor.strides()[i];
    }
    return desc;
}
```

### 3.5 Fusion Policy

Explicit fusion annotations guide kernel selection:

```python
# Python frontend marks fusion boundaries
@fusion_boundary("swiglu_ffn")
def ffn(x, w_gate, w_up, w_down):
    gate = silu(x @ w_gate)
    up = x @ w_up
    return (gate * up) @ w_down
```

The JIT recognizes this pattern and emits a SINGLE fused kernel that:
1. Computes `x @ w_gate` tile
2. Applies SiLU in-register (no memory write)
3. Computes `x @ w_up` tile
4. Multiplies in-register
5. Writes to output (single memory write)

**Fusion Benefits:**
- Reduces memory bandwidth by 3x for FFN
- Keeps intermediate values in L2/registers
- Critical for memory-bound inference

## 4. Cross-GPU Compatibility

### 4.1 Hardware Differences

| Feature | RTX 3090/4090 | A100 | H100 |
|---------|---------------|------|------|
| VRAM | 24GB GDDR6X | 80GB HBM2e | 80GB HBM3 |
| Tensor Cores | FP16, TF32 | FP16, TF32, INT8 | FP16, FP8, INT8 |
| NVLink | No (PCIe) | Yes (600GB/s) | Yes (900GB/s) |
| Memory BW | 1TB/s | 2TB/s | 3.35TB/s |
| Compute Cap | 86 (3090), 89 (4090) | 80 | 90 |

### 4.2 Adaptive Kernel Strategy

```python
@triton.jit
def matmul_kernel(
    ...,
    USE_FP8: tl.constexpr,
    COMPUTE_CAP: tl.constexpr,
):
    if USE_FP8 and COMPUTE_CAP >= 90:
        # Hopper path: FP8 tensor cores
        acc = tl.dot(a_fp8, b_fp8, out_dtype=tl.float32)
    else:
        # Ampere/Ada path: FP16 tensor cores
        acc = tl.dot(a_fp16, b_fp16, out_dtype=tl.float32)
```

**Runtime GPU Detection (`include/infra/gpu_features.hpp`):**
```cpp
struct GPUCapabilities {
    int compute_capability;     // 80=A100, 86=3090, 89=4090, 90=H100
    bool has_fp8_tensor_cores;  // H100 only
    bool has_int8_tensor_cores; // A100, H100
    bool has_nvlink;
    size_t vram_bytes;
    size_t memory_bandwidth_gbps;
    std::string device_name;

    static GPUCapabilities detect(int device_id = 0);
};
```

### 4.3 Memory-Adaptive Loading

**For Consumer GPUs (24GB VRAM):**
```cpp
class StreamingWeightManager {
    // Full model on CPU (pinned memory)
    std::vector<QuantizedWeight> cpu_weights_;

    // Double-buffered GPU workspace
    CUdeviceptr gpu_buffer_[2];
    size_t buffer_layers_;  // Layers that fit per buffer
    int active_buffer_ = 0;

public:
    // Async prefetch next layer while computing current
    void prefetch_layer(int layer_idx, cudaStream_t prefetch_stream);

    // Get pointer to current layer's weights
    void* get_layer_weights(int layer_idx, cudaStream_t compute_stream);
};
```

**For Datacenter GPUs (80GB VRAM):**
```cpp
void load_model(const GPUCapabilities& caps, const ModelConfig& config) {
    size_t model_size = estimate_model_size(config);

    if (caps.vram_bytes >= model_size * 1.2) {
        // Load everything to GPU
        weight_manager_ = std::make_unique<FullWeightManager>();
    } else {
        // Layer-by-layer streaming
        weight_manager_ = std::make_unique<StreamingWeightManager>();
    }
}
```

## 5. Multi-GPU Tensor Parallelism

### 5.1 Parallelism Strategy

| Strategy | How It Works | Best For |
|----------|--------------|----------|
| **Tensor Parallel (TP)** | Split weight matrices across GPUs | Large layers, fast interconnect |
| **Pipeline Parallel (PP)** | Different layers on different GPUs | Many layers, slow interconnect |
| **Sequence Parallel (SP)** | Split sequence dimension | Very long sequences |

**Recommendation:** Start with **Tensor Parallelism** (most common for inference).

### 5.2 Tensor Parallel Architecture (Megatron-Style)

```
                         Tensor Parallel Forward Pass

Layer Input (broadcast to all GPUs)
         │
         ├─────────────────────────────────────────────────────────┐
         ▼                                                         ▼
      GPU 0                                                     GPU 1
         │                                                         │
    ┌────┴────┐                                               ┌────┴────┐
    │ Q,K,V   │ (column-parallel)                            │ Q,K,V   │ (column-parallel)
    │ proj    │ W_qkv[:, :D/2]                               │ proj    │ W_qkv[:, D/2:]
    └────┬────┘                                               └────┬────┘
         │                                                         │
         │              ← NO AllReduce here                        │
         │                                                         │
    ┌────┴────┐                                               ┌────┴────┐
    │ Attn    │ (local heads 0 to H/2)                       │ Attn    │ (local heads H/2 to H)
    └────┬────┘                                               └────┬────┘
         │                                                         │
    ┌────┴────┐                                               ┌────┴────┐
    │ O proj  │ (row-parallel)                               │ O proj  │ (row-parallel)
    │         │ W_o[:D/2, :]                                 │         │ W_o[D/2:, :]
    └────┬────┘                                               └────┬────┘
         │                                                         │
         └──────────────── AllReduce (sum) ────────────────────────┘
                              │
                         + Residual
                              │
         ┌────────────────────┴────────────────────┐
         ▼                                         ▼
    ┌────┴────┐                               ┌────┴────┐
    │ FFN     │ (column-parallel)            │ FFN     │ (column-parallel)
    │ gate/up │ W_gate[:, :D_ff/2]           │ gate/up │ W_gate[:, D_ff/2:]
    └────┬────┘                               └────┬────┘
         │                                         │
         │              ← NO AllReduce here        │
         │                                         │
    ┌────┴────┐                               ┌────┴────┐
    │ FFN     │ (row-parallel)               │ FFN     │ (row-parallel)
    │ down    │ W_down[:D_ff/2, :]           │ down    │ W_down[D_ff/2:, :]
    └────┬────┘                               └────┬────┘
         │                                         │
         └──────────────── AllReduce (sum) ────────────────────────┘
                              │
                         + Residual
                              │
                         Layer Output
```

**Key insight**: AllReduce happens after **row-parallel** operations (O projection, FFN down), NOT after column-parallel operations. This minimizes communication while maintaining mathematical correctness.

### 5.3 NCCL Communicator (`include/infra/nccl_comm.hpp`)

```cpp
class NCCLCommunicator {
    ncclComm_t comm_;
    int world_size_;
    int rank_;
    cudaStream_t stream_;

public:
    NCCLCommunicator(int world_size, int rank);
    ~NCCLCommunicator();

    // Collective operations
    void all_reduce(
        CUdeviceptr buf,
        size_t count,
        ncclDataType_t dtype,
        ncclRedOp_t op = ncclSum
    );

    void all_gather(
        CUdeviceptr send_buf,
        CUdeviceptr recv_buf,
        size_t send_count,
        ncclDataType_t dtype
    );

    void broadcast(
        CUdeviceptr buf,
        size_t count,
        ncclDataType_t dtype,
        int root
    );

    // Synchronize
    void sync();

    // Getters
    int rank() const { return rank_; }
    int world_size() const { return world_size_; }
};
```

### 5.4 Tensor Parallel Config

```cpp
struct TensorParallelConfig {
    int tp_degree;      // Number of GPUs for tensor parallelism
    int pp_degree;      // Number of GPUs for pipeline parallelism (future)
    int rank;           // This GPU's rank in TP group
    int local_rank;     // Rank within node

    // Weight sharding info
    struct ShardInfo {
        int64_t start_idx;
        int64_t end_idx;
        int64_t shard_size;
    };

    ShardInfo get_column_shard(int64_t total_cols) const {
        int64_t shard_size = total_cols / tp_degree;
        return {
            rank * shard_size,
            (rank + 1) * shard_size,
            shard_size
        };
    }

    ShardInfo get_row_shard(int64_t total_rows) const {
        int64_t shard_size = total_rows / tp_degree;
        return {
            rank * shard_size,
            (rank + 1) * shard_size,
            shard_size
        };
    }
};
```

### 5.5 Sharded KV Cache

```cpp
class ShardedPagedKVCache {
    TensorParallelConfig tp_config_;

    // Each GPU holds 1/N of the KV heads
    // For GQA with 8 KV heads on 2 GPUs: 4 heads per GPU
    int local_n_kv_heads_;

    // Local page pool
    CUdeviceptr local_keys_;   // [max_blocks, block_size, local_n_kv_heads, head_dim]
    CUdeviceptr local_values_;

    // Block table (same on all GPUs for simplicity)
    std::vector<int32_t> block_table_;

public:
    ShardedPagedKVCache(
        const KVCacheConfig& config,
        const TensorParallelConfig& tp_config
    );

    // Update with new KV (only local heads)
    void update(
        const CUDATensor& new_keys,    // [seq_len, local_n_kv_heads, head_dim]
        const CUDATensor& new_values,
        int start_pos,
        cudaStream_t stream
    );

    // Get KV for attention (only local heads)
    std::pair<CUDATensor, CUDATensor> get_kv(
        const std::vector<int>& positions,
        cudaStream_t stream
    );
};
```

### 5.6 Tensor Parallel FFN

```python
@triton.jit
def tp_ffn_kernel(
    x_ptr,              # Input [seq, D]
    w_gate_shard_ptr,   # [D, D_ff/TP]
    w_up_shard_ptr,     # [D, D_ff/TP]
    w_down_shard_ptr,   # [D_ff/TP, D]
    out_ptr,            # Output [seq, D]
    seq_len, D, D_ff_shard,
    ...
):
    """
    Each GPU computes with its weight shard.
    Output is partial sum, needs AllReduce after kernel.
    """
    # Local compute
    gate = silu(x @ w_gate_shard)    # [seq, D_ff/TP]
    up = x @ w_up_shard              # [seq, D_ff/TP]
    local_out = (gate * up) @ w_down_shard  # [seq, D]

    # Store partial result (AllReduce handled by C++ host)
    tl.store(out_ptr + ..., local_out)
```

**C++ Host Orchestration:**
```cpp
void TransformerBlock::forward_tp(
    CUDATensor& hidden_states,
    const TensorParallelConfig& tp_config,
    NCCLCommunicator& nccl,
    cudaStream_t stream
) {
    // 1. Attention (local compute, heads already sharded)
    auto attn_out = attention_->forward(hidden_states, stream);

    // 2. AllReduce attention output
    nccl.all_reduce(attn_out.raw_ptr(), attn_out.numel(), ncclFloat16);

    // 3. Residual + norm
    hidden_states = add_and_norm(hidden_states, attn_out);

    // 4. FFN (local compute with sharded weights)
    auto ffn_out = ffn_->forward(hidden_states, stream);

    // 5. AllReduce FFN output
    nccl.all_reduce(ffn_out.raw_ptr(), ffn_out.numel(), ncclFloat16);

    // 6. Residual
    hidden_states = add(hidden_states, ffn_out);
}
```

### 5.7 Embedding and LM Head Parallelism

The embedding and LM head matrices are often the largest in the model (`vocab_size × d_model`, e.g., 128K × 4096 = 500MB+ each). These require special handling for tensor parallelism.

**Parallelism Strategy:**
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Embedding Layer (Row-Parallel)                   │
├─────────────────────────────────────────────────────────────────────┤
│ GPU 0: embed[0 : V/2, :]        GPU 1: embed[V/2 : V, :]           │
│                                                                     │
│ Token IDs → lookup on owner GPU → AllReduce (sum zero-padded)      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    LM Head (Column-Parallel)                        │
├─────────────────────────────────────────────────────────────────────┤
│ GPU 0: lm_head[:, 0 : V/2]      GPU 1: lm_head[:, V/2 : V]         │
│                                                                     │
│ Hidden → local matmul → AllGather logits → full vocab logits       │
└─────────────────────────────────────────────────────────────────────┘
```

**Implementation:**
```cpp
class ParallelEmbedding {
    CUdeviceptr local_embed_;  // [vocab_size/TP, d_model]
    int64_t vocab_start_;      // First vocab index on this GPU
    int64_t vocab_end_;        // Last vocab index (exclusive)
    TensorParallelConfig tp_config_;

public:
    CUDATensor forward(
        const std::vector<int32_t>& token_ids,
        NCCLCommunicator& nccl,
        cudaStream_t stream
    ) {
        // 1. Create output tensor (zeros)
        CUDATensor output({token_ids.size(), d_model_}, DType::FP16);
        cuda_memset(output.raw_ptr(), 0, output.size_bytes());

        // 2. Lookup tokens that belong to this GPU's shard
        for (size_t i = 0; i < token_ids.size(); i++) {
            int32_t token = token_ids[i];
            if (token >= vocab_start_ && token < vocab_end_) {
                int32_t local_idx = token - vocab_start_;
                // Copy embedding row to output[i]
                cuda_copy_row(output, i, local_embed_, local_idx);
            }
        }

        // 3. AllReduce to combine (other GPUs have zeros for our tokens)
        nccl.all_reduce(output.raw_ptr(), output.numel(), ncclFloat16, ncclSum);

        return output;
    }
};

class ParallelLMHead {
    CUdeviceptr local_weight_;  // [d_model, vocab_size/TP]
    TensorParallelConfig tp_config_;

public:
    CUDATensor forward(
        const CUDATensor& hidden_states,  // [seq, d_model]
        NCCLCommunicator& nccl,
        cudaStream_t stream
    ) {
        // 1. Local matmul: [seq, d_model] × [d_model, V/TP] → [seq, V/TP]
        CUDATensor local_logits = matmul(hidden_states, local_weight_);

        // 2. AllGather to get full vocab logits
        CUDATensor full_logits({hidden_states.shape()[0], vocab_size_}, DType::FP16);
        nccl.all_gather(
            local_logits.raw_ptr(),
            full_logits.raw_ptr(),
            local_logits.numel(),
            ncclFloat16
        );

        return full_logits;
    }
};
```

> **Note**: For models with tied embeddings (embed == lm_head.T), store weights once and transpose as needed.

## 6. Implementation Strategy

### 6.1 Shape Specialization (Bucketing)

Triton compilation is fast (~100ms), but we still cache for repeated shapes.

**Enhanced Cache Key:**
```cpp
struct KernelCacheKey {
    std::string kernel_name;
    std::vector<int64_t> shapes;
    int compute_capability;     // SM version
    QuantType quant_type;       // NONE, Q8_0, Q4_K
    int tp_degree;              // Tensor parallel degree

    bool operator==(const KernelCacheKey& other) const;
    size_t hash() const;
};

// Bucketing logic
int64_t bucket_seq_len(int64_t seq_len) {
    // Power of 2 buckets: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
    if (seq_len <= 1) return 1;
    return 1 << (64 - __builtin_clzll(seq_len - 1));
}
```

### 6.2 Async Compilation

Don't block inference on first request with new shape:

```cpp
class KernelCache {
    std::unordered_map<KernelCacheKey, CUfunction, KeyHash> cache_;
    std::unordered_set<KernelCacheKey, KeyHash> compiling_;
    ThreadPool compilation_pool_;
    std::mutex mutex_;

public:
    // Non-blocking: returns nullopt if compiling, triggers async compile
    std::optional<CUfunction> try_get(const KernelCacheKey& key);

    // Blocking: waits for compilation if needed
    CUfunction get_or_compile(const KernelCacheKey& key);

    // Trigger async compilation
    void compile_async(const KernelCacheKey& key);
};
```

### 6.3 Implementation Phases

**Phase 1: GPU Foundation**
1. CUDA memory management (adapt PagedKVCache to CUDA)
2. Triton kernel library (attention, FFN, norms)
3. C++ ↔ Triton integration layer (PTX loading, dispatch)
4. Basic FP16 inference end-to-end
5. Single GPU working prototype

**Phase 2: Quantized JIT (Differentiator)**
1. Q8_0 Triton kernel with tensor core utilization
2. Q4_K Triton kernel with block dequantization
3. Fused dequant-matmul patterns
4. Mixed-precision attention (FP16 Q, Q8 K, Q4 V)
5. Performance validation vs llama.cpp CUDA

**Phase 3: Multi-GPU**
1. NCCL communicator wrapper
2. Weight sharding for tensor parallelism
3. Sharded PagedKVCache
4. AllReduce integration in forward pass
5. Multi-node support (future)

**Phase 4: Production Hardening**
1. Kernel caching and async compilation
2. Auto-tuning framework (tile sizes, warp configs)
3. CPU fallback path (existing SIMD kernels)
4. Profiling and instrumentation
5. Error handling and recovery

## 7. Build System Integration

### 7.1 CMake Configuration

```cmake
cmake_minimum_required(VERSION 3.24)
project(FreeLLM LANGUAGES CXX CUDA)

# CUDA
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")  # A100, 3090, 4090, H100

# NCCL
find_package(NCCL REQUIRED)

# Triton (build kernels at configure time)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Compile Triton kernels to PTX
set(TRITON_KERNELS
    kernels/triton/attention/flash_attention.py
    kernels/triton/ffn/swiglu.py
    kernels/triton/quantized/q4k_matmul.py
    kernels/triton/quantized/q8_matmul.py
)

foreach(kernel ${TRITON_KERNELS})
    get_filename_component(kernel_name ${kernel} NAME_WE)
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/kernels/${kernel_name}.ptx
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/compile_triton.py
                ${CMAKE_SOURCE_DIR}/${kernel}
                ${CMAKE_BINARY_DIR}/kernels/${kernel_name}.ptx
        DEPENDS ${kernel}
        COMMENT "Compiling Triton kernel: ${kernel_name}"
    )
    list(APPEND COMPILED_KERNELS ${CMAKE_BINARY_DIR}/kernels/${kernel_name}.ptx)
endforeach()

add_custom_target(triton_kernels DEPENDS ${COMPILED_KERNELS})

# Main library
add_library(freellm_cuda
    src/infra/kernel_loader.cpp
    src/infra/nccl_comm.cpp
    src/infra/cuda_memory.cpp
    src/core/cuda_tensor.cpp
)

target_link_libraries(freellm_cuda PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
    NCCL::nccl
)

add_dependencies(freellm_cuda triton_kernels)

# Optional: MLIR for CPU fallback
option(FREELLM_ENABLE_MLIR "Enable MLIR for CPU fallback" OFF)
if(FREELLM_ENABLE_MLIR)
    find_package(MLIR REQUIRED CONFIG)
    target_link_libraries(freellm_cuda PRIVATE
        MLIRExecutionEngine
        MLIRIR
        MLIRLinalg
    )
    target_compile_definitions(freellm_cuda PRIVATE FREELLM_HAS_MLIR)
endif()
```

### 7.2 Triton AOT Compilation Script

Triton 2.x uses a different compilation model than shown in many tutorials. For AOT compilation, we use `triton.compile()` with explicit specializations:

```python
#!/usr/bin/env python3
# scripts/compile_triton.py
"""
AOT compilation of Triton kernels to cubin/PTX for C++ runtime loading.

Usage: python compile_triton.py <kernel_module.py> <output_dir>

Triton kernels must define:
  - KERNEL_CONFIGS: List of (constexpr_dict, signature) tuples
  - Main kernel function decorated with @triton.jit
"""

import sys
import os
import importlib.util
import json
from pathlib import Path

import triton
from triton.compiler import ASTSource, compile as triton_aot_compile


def load_kernel_module(source_path: str):
    """Load a Python module containing Triton kernel definitions."""
    spec = importlib.util.spec_from_file_location("kernel_module", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_triton_kernels(module):
    """Find all @triton.jit decorated functions in a module."""
    kernels = {}
    for name in dir(module):
        obj = getattr(module, name)
        if hasattr(obj, 'fn') and hasattr(obj.fn, '__triton_meta__'):
            kernels[name] = obj
    return kernels


def compile_kernel_configs(
    kernel_fn,
    configs: list,  # List of (constexpr_values, signature)
    output_dir: str,
    target_archs: list = [80, 86, 89, 90],
):
    """
    Compile a Triton kernel for multiple configurations and architectures.

    Each config is a tuple of:
      - constexpr_values: dict mapping constexpr param names to values
      - signature: dict mapping param names to types (e.g., "*fp16", "i32")
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest = {"kernels": []}

    for config_idx, (constexprs, sig_dict) in enumerate(configs):
        # Build signature string
        sig_parts = []
        for param_name, param_type in sig_dict.items():
            sig_parts.append(param_type)
        signature = ",".join(sig_parts)

        for arch in target_archs:
            # Create specialization key for caching
            spec_key = f"config{config_idx}_sm{arch}"

            try:
                # Triton AOT compilation
                compiled = triton_aot_compile(
                    kernel_fn,
                    signature=signature,
                    constants=constexprs,
                    target=("cuda", arch),
                )

                # Write cubin (preferred) or PTX
                cubin_path = os.path.join(output_dir, f"{kernel_fn.fn.__name__}_{spec_key}.cubin")
                with open(cubin_path, "wb") as f:
                    f.write(compiled.asm["cubin"])

                # Also write PTX for debugging
                ptx_path = os.path.join(output_dir, f"{kernel_fn.fn.__name__}_{spec_key}.ptx")
                with open(ptx_path, "w") as f:
                    f.write(compiled.asm["ptx"])

                # Record in manifest
                manifest["kernels"].append({
                    "name": kernel_fn.fn.__name__,
                    "config_idx": config_idx,
                    "arch": arch,
                    "cubin": os.path.basename(cubin_path),
                    "ptx": os.path.basename(ptx_path),
                    "constexprs": constexprs,
                    "signature": sig_dict,
                    "shared_mem": compiled.metadata.shared,
                    "num_warps": compiled.metadata.num_warps,
                    "num_stages": compiled.metadata.num_stages,
                })

                print(f"✓ Compiled {kernel_fn.fn.__name__} config={config_idx} SM{arch}")

            except Exception as e:
                print(f"✗ Failed {kernel_fn.fn.__name__} config={config_idx} SM{arch}: {e}")

    # Write manifest for C++ loader
    manifest_path = os.path.join(output_dir, "kernel_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# Example kernel configuration file (kernels/triton/quantized/q4k_matmul.py):
"""
import triton
import triton.language as tl

@triton.jit
def q4k_matmul_kernel(
    X_ptr, W_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # ... kernel implementation ...
    pass

# Configurations for AOT compilation
KERNEL_CONFIGS = [
    # Config 0: Small batches (M <= 16)
    (
        {"BLOCK_M": 16, "BLOCK_N": 64},
        {"X_ptr": "*fp16", "W_ptr": "*i8", "Y_ptr": "*fp16",
         "M": "i32", "N": "i32", "K": "i32",
         "stride_xm": "i32", "stride_xk": "i32",
         "stride_ym": "i32", "stride_yn": "i32"}
    ),
    # Config 1: Medium batches (M <= 64)
    (
        {"BLOCK_M": 64, "BLOCK_N": 64},
        {"X_ptr": "*fp16", "W_ptr": "*i8", "Y_ptr": "*fp16",
         "M": "i32", "N": "i32", "K": "i32",
         "stride_xm": "i32", "stride_xk": "i32",
         "stride_ym": "i32", "stride_yn": "i32"}
    ),
    # Config 2: Large batches (M > 64)
    (
        {"BLOCK_M": 128, "BLOCK_N": 128},
        {"X_ptr": "*fp16", "W_ptr": "*i8", "Y_ptr": "*fp16",
         "M": "i32", "N": "i32", "K": "i32",
         "stride_xm": "i32", "stride_xk": "i32",
         "stride_ym": "i32", "stride_yn": "i32"}
    ),
]
"""


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compile_triton.py <kernel_module.py> <output_dir>")
        sys.exit(1)

    source_path = sys.argv[1]
    output_dir = sys.argv[2]

    module = load_kernel_module(source_path)
    kernels = find_triton_kernels(module)

    if not kernels:
        print(f"No Triton kernels found in {source_path}")
        sys.exit(1)

    # Get kernel configurations from module
    configs = getattr(module, "KERNEL_CONFIGS", None)
    if configs is None:
        print(f"Warning: No KERNEL_CONFIGS found in {source_path}, using defaults")
        configs = [({"BLOCK_M": 64, "BLOCK_N": 64}, {})]

    for kernel_name, kernel_fn in kernels.items():
        print(f"\nCompiling kernel: {kernel_name}")
        compile_kernel_configs(kernel_fn, configs, output_dir)
```

**C++ Kernel Loader with Manifest:**
```cpp
class TritonKernelLoader {
    struct KernelVariant {
        CUfunction function;
        int shared_mem;
        int num_warps;
        std::map<std::string, int64_t> constexprs;
    };

    // kernel_name -> arch -> config_idx -> variant
    std::map<std::string, std::map<int, std::map<int, KernelVariant>>> kernels_;
    int current_arch_;

public:
    void load_from_manifest(const std::string& dir) {
        auto manifest = json::parse(read_file(dir + "/kernel_manifest.json"));

        for (const auto& entry : manifest["kernels"]) {
            std::string name = entry["name"];
            int arch = entry["arch"];
            int config_idx = entry["config_idx"];

            // Load cubin
            CUmodule module;
            std::string cubin_path = dir + "/" + entry["cubin"].get<std::string>();
            cuModuleLoad(&module, cubin_path.c_str());

            CUfunction func;
            cuModuleGetFunction(&func, module, name.c_str());

            kernels_[name][arch][config_idx] = {
                .function = func,
                .shared_mem = entry["shared_mem"],
                .num_warps = entry["num_warps"],
            };
        }
    }

    // Select best config based on problem size
    const KernelVariant& select_variant(
        const std::string& name,
        int M, int N, int K
    ) {
        auto& arch_variants = kernels_.at(name).at(current_arch_);

        // Simple heuristic: pick config based on M
        int config_idx = (M <= 16) ? 0 : (M <= 64) ? 1 : 2;
        return arch_variants.at(config_idx);
    }
};
```

## 8. Decision Matrix: C++ vs. Triton vs. MLIR

| Logic Type | Example | Implementation | Reason |
| :--- | :--- | :--- | :--- |
| **System Logic** | KV Paging, Sampling, Scheduling | C++ | Logic-heavy, not compute-heavy |
| **Memory Management** | Allocation, Streams | C++ | CUDA Driver API |
| **Communication** | AllReduce, Broadcast | C++ (NCCL) | Library call |
| **Dense Math (GPU)** | FFN, Projections | **Triton** | Tensor core utilization, fusion |
| **Attention (GPU)** | Flash Attention, GQA | **Triton** | Fused QKV, memory efficient |
| **Quantized Ops (GPU)** | Q4_K/Q8_0 GEMM | **Triton** | On-the-fly dequant, tensor cores |
| **Norms (GPU)** | RMSNorm, LayerNorm | **Triton** | Fused with adjacent ops |
| **Dense Math (CPU)** | FFN fallback | MLIR JIT | Vectorization, tiling |
| **Quantized Ops (CPU)** | Q4_K Decoding | C++ | Hand-tuned bit manipulation |

## 9. Profiling and Instrumentation

```cpp
struct KernelStats {
    std::string name;
    uint64_t invocation_count = 0;
    double total_time_ms = 0.0;
    double min_time_ms = std::numeric_limits<double>::max();
    double max_time_ms = 0.0;
    size_t total_bytes_read = 0;
    size_t total_bytes_written = 0;
    size_t total_flops = 0;

    double avg_time_ms() const {
        return invocation_count > 0 ? total_time_ms / invocation_count : 0;
    }

    double achieved_bandwidth_gbps() const {
        if (total_time_ms == 0) return 0;
        return (total_bytes_read + total_bytes_written) / (total_time_ms * 1e6);
    }

    double achieved_tflops() const {
        if (total_time_ms == 0) return 0;
        return total_flops / (total_time_ms * 1e9);
    }
};

class KernelProfiler {
    std::unordered_map<std::string, KernelStats> stats_;
    bool enabled_ = false;

public:
    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }

    // RAII timer
    class ScopedTimer {
        KernelProfiler& profiler_;
        std::string name_;
        cudaEvent_t start_, stop_;
    public:
        ScopedTimer(KernelProfiler& p, const std::string& name);
        ~ScopedTimer();
    };

    void print_report() const;
    void export_json(const std::string& path) const;
};
```

## 10. Quantization Strategy

### 10.1 Why Custom Triton Kernels (Not bitsandbytes)

We evaluated several quantization libraries and chose custom Triton kernels:

| Library | Format | Verdict | Reason |
|---------|--------|---------|--------|
| **bitsandbytes** | NF4/FP4 | ❌ Don't use | Python-centric, NF4 ≠ Q4_K, training-focused |
| **Marlin (vLLM)** | GPTQ/AWQ | ⚠️ Optional | Good for HuggingFace models, different format |
| **CUTLASS** | INT4/INT8 | ⚠️ Complex | Maximum performance, steep learning curve |
| **Custom Triton** | Q4_K/Q8_0 | ✅ Primary | Matches GGUF, full control, good performance |

**Why NOT bitsandbytes:**
1. **Wrong formats**: NF4 (Normal Float 4) ≠ Q4_K (asymmetric k-quant). No path to GGUF compatibility.
2. **Python dependency**: Requires PyTorch 2.3+ at runtime, defeating our "pure C++ runtime" goal.
3. **Training-focused**: Designed for QLoRA fine-tuning, not inference optimization.
4. **No C++ API**: Would need to reverse-engineer their CUDA kernels.

**Our approach:**
```
┌─────────────────────────────────────────────────────────────────────┐
│                     Quantization Format Support                     │
├─────────────────────────────────────────────────────────────────────┤
│ Primary (Day 1):                                                    │
│   • Q4_K: Custom Triton kernel with hierarchical dequantization     │
│   • Q8_0: Custom Triton kernel, simpler structure                   │
│   • Matches existing FreeLLM CPU implementation exactly             │
├─────────────────────────────────────────────────────────────────────┤
│ Secondary (Future):                                                 │
│   • GPTQ/AWQ: Consider Marlin kernel extraction (Apache 2.0)        │
│   • FP8: Native Triton support on H100/Blackwell                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 10.2 Data Type Strategy (BF16 vs FP16)

Different GPUs have different optimal data types:

| GPU | Tensor Core Support | Recommended Activation Type |
|-----|---------------------|----------------------------|
| RTX 3090 (SM86) | FP16, TF32 | **FP16** |
| RTX 4090 (SM89) | FP16, BF16, TF32 | FP16 or BF16 |
| A100 (SM80) | FP16, BF16, TF32 | **BF16** |
| H100 (SM90) | FP16, BF16, FP8 | **BF16** (or FP8) |

**Implementation:**
```cpp
enum class DType {
    FP16,
    BF16,
    FP32,
    FP8_E4M3,  // H100 only
    FP8_E5M2,  // H100 only
};

class DTypeSelector {
public:
    static DType select_activation_dtype(const GPUCapabilities& caps) {
        if (caps.compute_capability >= 80) {
            // Ampere+ prefers BF16 for numeric stability
            return DType::BF16;
        }
        return DType::FP16;
    }

    static DType select_accumulation_dtype() {
        // Always accumulate in FP32 for precision
        return DType::FP32;
    }
};
```

**Triton kernels support both:**
```python
@triton.jit
def matmul_kernel(
    ...,
    ACTIVATION_DTYPE: tl.constexpr,  # "fp16" or "bf16"
):
    if ACTIVATION_DTYPE == "bf16":
        x = tl.load(X_ptr + ...).to(tl.bfloat16)
    else:
        x = tl.load(X_ptr + ...).to(tl.float16)

    # Accumulate in FP32 regardless
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # ...
```

### 10.3 Error Handling and Recovery

Robust error handling for production deployments:

```cpp
// Error categories
enum class InferenceError {
    CUDA_OUT_OF_MEMORY,
    CUDA_KERNEL_LAUNCH_FAILED,
    NCCL_TIMEOUT,
    NCCL_COMM_ERROR,
    INVALID_INPUT,
    MODEL_LOAD_FAILED,
    KERNEL_NOT_FOUND,
};

class InferenceEngine {
public:
    struct Result {
        std::vector<int32_t> tokens;
        std::optional<InferenceError> error;
        std::string error_message;
    };

    Result generate(const std::vector<int32_t>& input_ids, const GenerateConfig& config) {
        try {
            return generate_impl(input_ids, config);
        } catch (const CUDAException& e) {
            return handle_cuda_error(e);
        } catch (const NCCLException& e) {
            return handle_nccl_error(e);
        }
    }

private:
    Result handle_cuda_error(const CUDAException& e) {
        if (e.code() == CUDA_ERROR_OUT_OF_MEMORY) {
            // Attempt recovery: clear KV cache, reduce batch size
            kv_cache_.clear();
            return Result{{}, InferenceError::CUDA_OUT_OF_MEMORY,
                         "GPU OOM - cleared KV cache, retry with smaller batch"};
        }
        return Result{{}, InferenceError::CUDA_KERNEL_LAUNCH_FAILED, e.what()};
    }

    Result handle_nccl_error(const NCCLException& e) {
        // NCCL errors often require full restart
        if (e.code() == ncclSystemError) {
            nccl_comm_.abort();
            return Result{{}, InferenceError::NCCL_COMM_ERROR,
                         "NCCL communication failed - requires restart"};
        }
        return Result{{}, InferenceError::NCCL_TIMEOUT, e.what()};
    }
};

// CUDA error checking macro
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        CUresult err = call;                                                \
        if (err != CUDA_SUCCESS) {                                          \
            const char* errStr;                                             \
            cuGetErrorString(err, &errStr);                                 \
            throw CUDAException(err, std::string(#call) + ": " + errStr);   \
        }                                                                   \
    } while (0)

// NCCL error checking
#define NCCL_CHECK(call)                                                    \
    do {                                                                    \
        ncclResult_t err = call;                                            \
        if (err != ncclSuccess) {                                           \
            throw NCCLException(err, ncclGetErrorString(err));              \
        }                                                                   \
    } while (0)
```

**Graceful degradation strategies:**
1. **OOM**: Clear KV cache, reduce batch size, fall back to streaming weights
2. **NCCL timeout**: Log, abort communicator, require manual restart
3. **Kernel not found**: Fall back to generic kernel or CPU path

## 11. Future Extensions

### 11.1 Speculative Decoding
- Draft model runs on CPU or smaller GPU
- Verification batch on main GPU
- Requires careful memory management

### 11.2 MoE (Mixture of Experts)
- Expert routing via Triton kernel
- Load balancing across GPUs
- Expert parallelism in addition to tensor parallelism

### 11.3 Continuous Batching Enhancements
- Preemption support for long sequences
- Priority queues
- SLA-aware scheduling

### 11.4 Additional Quantization Formats
- FP8 (H100/Blackwell)
- GPTQ/AWQ (via Marlin extraction)

---

## Summary

This RFC proposes a **unified multi-backend architecture** for FreeLLM that:

1. Uses **Triton** for NVIDIA GPU kernels with maximum tensor core utilization
2. Uses **Pallas/StableHLO** for TPU and AMD GPU kernels via PJRT plugin system
3. Provides **ComputeBackend interface** abstracting device differences in C++
4. Implements **GGUF-compatible quantized kernels** (Q4_K, Q8_0) on all platforms
5. Supports **multi-device tensor parallelism** (NCCL for NVIDIA, PJRT collectives for TPU)
6. Keeps **C++ as the runtime** with Python only at build-time
7. Maintains **CPU fallback** via existing SIMD kernels + optional MLIR

### Hardware Support Matrix

| Hardware | Backend | Kernel Source | Status |
|----------|---------|---------------|--------|
| NVIDIA GPU (SM80+) | CUDABackend | Triton → PTX | Primary |
| Google TPU v4/v5 | PJRTBackend | Pallas → StableHLO | Primary |
| AMD GPU (MI300+) | PJRTBackend | Pallas → StableHLO | Planned |
| Intel GPU | PJRTBackend | Pallas → StableHLO | Future |
| CPU (AVX2/NEON) | Native | Hand-tuned SIMD | Fallback |

**Competitive Position**: First **embeddable C++ runtime** supporting both NVIDIA GPUs and Google TPUs with optimized quantized kernels, no Python runtime dependency, deployable from edge to cloud. Unlike vLLM/SGLang (Python-only, NVIDIA-only), FreeLLM ships as a native library targeting multiple hardware platforms.
