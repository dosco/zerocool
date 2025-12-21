# Agent Instructions for FreeLLM

This document contains high-level guidelines for AI coding agents working on the FreeLLM project.

## Project Overview

FreeLLM is a fast CPU-only LLM inference engine written in modern C++23. The project aims to be:
- **Educational**: Code should be easy to follow, learn from, and understand
- **Evolving to production**: Starting simple but with an eye toward performance optimization
- **Modern C++**: Using C++23 features throughout

## Code Philosophy

### 1. Educational First, Performance Second

- **Write clear, well-commented code** that explains what's happening and why
- **Include extensive documentation** in headers explaining concepts (e.g., "What is RoPE?", "Why SwiGLU?")
- **Progressive optimization**: Start with naive implementations, then add optimized versions alongside
  - Keep `_naive` implementations as reference
  - Add optimized versions (e.g., `_avx2`) in separate functions
  - Use runtime dispatch to select best implementation

Example structure:
```cpp
// Naive implementation (educational reference)
Tensor matmul_naive(const Tensor& a, const Tensor& b);

// Optimized implementation (performance)
Tensor matmul_avx2(const Tensor& a, const Tensor& b);

// Smart dispatcher (automatically picks best)
Tensor matmul(const Tensor& a, const Tensor& b);
```

### 2. C++23 Modern Practices

- **Use `std::print` and `std::println`** instead of `std::cout` for all output
- **Use modern C++ features**: `std::format`, structured bindings, concepts, etc.
- **Prefer standard library** over external dependencies when possible
- **Move semantics**: No unnecessary copies, use move constructors/assignment
- **RAII**: Proper resource management, smart pointers where appropriate

### 3. Code Organization

- **Headers in `include/`**: All public APIs and implementations (header-only for simplicity)
- **Source in `src/`**: Main entry point and test code
- **Separate concerns**: One major concept per file
  - `tensor.hpp` - Core tensor class
  - `rope.hpp` - Rotary position embeddings
  - `attention.hpp` - Attention mechanism
  - etc.

### 4. Infrastructure & Capabilities

**CRITICAL**: Before implementing any new feature, check `include/infra` and `include/kernels` for existing capabilities. Do not reinvent the wheel.

- **`include/infra/`**: Core infrastructure components
  - `compute_backend.hpp`: Abstract interface for compute backends (CPU, Metal, CUDA)
  - `metal_backend.hpp`: Metal-specific backend implementation
  - `safetensors_loader.hpp`: Efficient weight loading
- **`include/kernels/`**: Optimized kernel implementations
  - `quantiz/`: Unified quantization library (Q8_0, Q4_K, Q4_0)
  - `tensor_ops.hpp`: Standard tensor operations
- **`include/core/`**: Core data structures
  - `tensor.hpp`: The fundamental Tensor class
  - `model_config.hpp`: Model configuration structures

**Rule**: If you need a new capability (e.g., a new quantization type), add it to the existing library (e.g., `include/kernels/quantiz`) rather than creating a standalone header.

### 5. Documentation Standards

Every major class/function should have:
- **Purpose**: What does this do?
- **Architecture**: How does it work? (diagrams in comments welcome)
- **Design decisions**: Why this approach?
- **Shape transformations**: For tensor operations, document input/output shapes
- **Examples**: Show typical usage
- **Complex Features**: When a complex feature is implemented, its explanation should be added to a relevant doc or a new doc under `./docs`.

Example:
```cpp
/**
 * @brief Rotary Position Embeddings (RoPE)
 *
 * RoPE encodes position information by rotating token embeddings.
 * Instead of adding position embeddings, we rotate the query/key vectors
 * by an angle that depends on their position.
 *
 * Why RoPE?
 * - Better extrapolation to longer sequences
 * - Relative position encoding (naturally handles distance between tokens)
 * - Used in modern LLMs (LLaMA, GPT-NeoX, PaLM)
 *
 * @param seq_len Maximum sequence length
 * @param dim Dimension (must be even)
 */
```

### 5. Target Model

- **Focus on TinyLLaMA 1.1B** as the practical target model
- Keep code general enough to support similar architectures (LLaMA-style models)
- Configuration should match TinyLLaMA by default

### 6. Performance Targets

- **CPU architecture**: x86-64 with AVX2/AVX-512, ARM with NEON
- **Runtime dispatch**: Detect CPU features and use best implementation
- **Memory efficiency**: Use memory mapping, SIMD-aligned allocations
- **Future optimizations** (not yet implemented):
  - KV cache (10-100x speedup for generation)
  - Quantization (INT8/INT4)
  - Flash Attention

### 7. Testing Philosophy

- **Comprehensive test suite** in `src/main.cpp`
- Test each component individually before integration
- Include performance benchmarks (GFLOPS, tokens/sec)
- Real-world validation with actual model weights

### 8. Build System

- **CMake** with C++23 support
- **Compiler flags**: `-mavx2 -mfma` for x86-64 optimizations
- **Always use the build script**: Run `./build.sh` for building (do not run cmake commands directly)
- **The build script** handles compiler selection (GCC 15) and configuration
- **Strict warnings**: Catch issues early

### 9. Dependencies

- **Minimize external dependencies**: Prefer standard library when possible
- **Use CMake FetchContent for third-party libraries**: Never require manual installation
  - All libraries must be automatically downloaded and built via FetchContent
  - Example: RE2, SentencePiece, etc.
  - Ensures ABI compatibility (everything uses same compiler)
  - Makes project self-contained and easy to build
- **System dependencies only for basics**: POSIX APIs for memory mapping, SIMD intrinsics
- **No JSON library**: Write simple parsers for specific formats (e.g., safetensors)

### 10. Git Workflow

- **Descriptive commit messages**: Explain what and why
- **Small, focused commits**: One logical change per commit
- **No emojis** in code or commits unless explicitly requested
- **Include context**: Generated with Claude Code attribution when appropriate
- [x] Follow Design Docs: All work should follow the designs in `docs/hybrid_jit_design.md`. Ensure your implementation aligns with the unified multi-backend architecture.

### 11. Documentation Lookup

When working on specific technologies or languages, **YOU MUST** look up the relevant documentation files to ensure you are following the project's standards and best practices.

- **Metal**: Look up `docs/metal.md` and `docs/metal_backend_guide.md`
- **C++**: Look up `docs/c++.md`
- **Triton**: Look up `docs/triton.md`
- **Continuous Batching**: Look up `docs/continuous_batching_design.md`

Example: If you are asked to write a new Metal kernel, you should first read `docs/metal.md` to understand the "Pure C++, Zero Obj-C" philosophy and `docs/metal_backend_guide.md` for integration details.

## Current Architecture

```
FreeLLM/
├── include/
│   ├── tensor.hpp              # Core tensor class with SIMD alignment
│   ├── tensor_ops.hpp          # Naive implementations + smart dispatch
│   ├── tensor_ops_simd.hpp     # AVX2/NEON optimized operations
│   ├── cpu_features.hpp        # Runtime CPU feature detection
│   ├── rope.hpp                # Rotary position embeddings
│   ├── attention.hpp           # Multi-head attention mechanism
│   ├── transformer_block.hpp   # FeedForward + TransformerBlock
│   ├── llm_model.hpp           # Complete LLM pipeline
│   ├── model_config.hpp        # TinyLLaMA configuration
│   ├── model_loader.hpp        # Weight loading interface
│   ├── safetensors_loader.hpp  # Safetensors format parser
│   ├── sampling.hpp            # Greedy, Top-K, Top-P sampling
│   └── generation.hpp          # Autoregressive generation loop
├── src/
│   └── main.cpp                # Test suite and examples
├── scripts/
│   └── download_tinyllama.sh   # Download TinyLLaMA weights
└── build.sh                    # Simple build script
```

## Future Roadmap

1. **KV Cache**: Major performance improvement for generation
2. **Tokenizer Integration**: SentencePiece/BPE for text ↔ token conversion
3. **Quantization**: INT8/INT4 support for reduced memory
4. **Flash Attention**: More efficient attention computation
5. **Multi-threading**: Parallelize layer computation
6. **Metal/CUDA backends**: GPU acceleration (while keeping CPU as primary)

## Common Patterns

### Adding a New Operation

1. Write naive implementation with extensive comments
2. Add optimized SIMD version if performance-critical
3. Add dispatcher function that auto-selects best implementation
4. Add test case in main.cpp
5. Document shape transformations and complexity

### Adding a New Model Component

1. Create dedicated header file (e.g., `new_component.hpp`)
2. Include architecture diagram in comments
3. Explain the "why" behind design decisions
4. Add accessor methods for weight loading
5. Integrate into LLMModel if needed
6. Add focused test in main.cpp

### Debugging Performance Issues

1. Add timing measurements (use `std::chrono`)
2. Print GFLOPS or tokens/sec for comparison
3. Profile with perf/Instruments to find bottlenecks
4. Consider SIMD optimization for hot paths
5. Document performance characteristics in comments

### Debugging GPU Kernels (Metal, CUDA, etc.)

When GPU kernel output is incorrect (NaN, garbage, wrong values), follow this **systematic isolation methodology**:

#### Step 1: Verify Data Integrity on CPU Before GPU Transfer

Before suspecting kernel bugs, confirm that the data being *sent* to the GPU is valid.

```cpp
// Example: Check for NaNs in weights before upload
size_t nan_count = 0;
for (size_t i = 0; i < tensor.size(); ++i) {
    if (std::isnan(tensor.data()[i])) nan_count++;
}
std::cout << "[DEBUG] " << name << ": " << nan_count << " NaNs\n";
```

**Checkpoints to add:**
- After loading weights from safetensors (BF16→F32 conversion)
- After quantization (check scales for NaN in Q4_0 blocks)
- After copying embeddings to input buffer

#### Step 2: Add GPU Checkpoint After First Kernel

Once CPU data is verified clean, add a checkpoint *immediately after* the first GPU kernel:

```cpp
// After first rms_norm kernel in layer 0
if (layer == 0) {
    backend_->synchronize();  // Wait for GPU completion
    std::vector<float> host_buffer(size);
    backend_->copy_to_host(host_buffer.data(), gpu_buffer, size * sizeof(float));
    
    size_t nans = 0;
    for (auto v : host_buffer) if (std::isnan(v)) nans++;
    std::cout << "[DEBUG] After kernel: " << nans << " NaNs\n";
}
```

**Interpretation:**
- If input is clean but output has NaNs → Kernel bug
- If input already has NaNs → Problem is earlier (data transfer or previous kernel)

#### Step 3: Trace Kernel Dispatch Decisions

A common bug is **wrong kernel selection**. Add debug prints to trace which kernel is being dispatched:

```cpp
// In run_linear_batched or similar dispatch function
bool is_fp32 = fp32_weight_names_.count(weight_name) > 0;
std::cout << "[DEBUG-GEMM] " << weight_name << " -> " 
          << (is_fp32 ? "gemm_f32" : "gemm_q4_0") << "\n";
```

**Common dispatch bugs:**
- Using FP32 kernel on quantized data (interprets Q4_0 bytes as floats → garbage/NaN)
- Using quantized kernel on FP32 data (tries to decode floats as packed nibbles)
- Weight name mismatch between tracking set and actual weights

#### Step 4: Binary Search Through Pipeline

If NaN appears somewhere in a multi-layer pipeline:

1. Add checkpoint after layer N/2
2. If NaN present → problem is in layers 0..N/2, recurse left
3. If clean → problem is in layers N/2..N, recurse right
4. Narrow down to exact layer and kernel

#### Common GPU Kernel Bugs and Fixes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| All outputs are NaN | Wrong kernel for data type | Verify dispatch logic matches weight type |
| Random NaNs | Division by zero, sqrt of negative | Add epsilon guards: `rsqrt(x + epsilon)` |
| Garbage values | Index out of bounds | Verify grid/block dimensions match data size |
| Consistent wrong values | Incorrect dequantization | Check scale/offset extraction from Q4_0 blocks |
| First token correct, rest wrong | Off-by-one in position encoding | Verify RoPE position indices |

#### Metal-Specific Debugging

```cpp
// Metal kernel argument validation
// Ensure buffer bindings match kernel signature order:
// kernel void foo(device float* A [[buffer(0)]], device float* B [[buffer(1)]], ...)
backend_->execute_kernel("kernel_name", 
    {buffer_for_arg0, buffer_for_arg1, ...},  // Order matters!
    {}, config);
```

**Metal gotchas:**
- `thread_position_in_grid` is 3D `uint3`, not scalar `uint`
- Shared memory (`threadgroup`) must be sized correctly
- No runtime bounds checking—silent corruption on OOB access

#### CUDA-Specific Debugging

```cpp
// Always check for launch errors
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";
}
```

**CUDA gotchas:**
- Grid/block dimensions silently truncated if too large
- Shared memory size must be specified at launch for dynamic allocation
- Warp divergence can cause subtle correctness issues

#### Weight Loading and Caching Bugs

When implementing caching systems for weights or compiled kernels:

1. **Empty tensor list bug**: If cache loader takes `tensor_names` as input, caller must populate it. An empty list loads nothing, causing silent fallback to full reload.

2. **Dual-map confusion**: If loader returns both `fp32_tensors` and `quant_tensors`, a tensor may appear in BOTH maps. The engine must skip duplicates:
   ```cpp
   for (const auto& [name, tensor] : fp32_tensors) {
       if (quant_tensors.find(name) != quant_tensors.end()) {
           continue;  // Skip—will load quantized version instead
       }
       fp32_weight_names_.insert(name);  // Track for dispatch
   }
   ```

3. **Stale cache detection**: Always verify cache validity with source hash AND config hash. Quantization config changes should invalidate cache.

## Key Principles Summary

- **Clarity over cleverness**: Code should be understandable
- **Education over optimization**: Unless optimization is the goal
- **Modern C++**: Use C++23 features throughout
- **Practical focus**: TinyLLaMA 1.1B is the target
- **Progressive enhancement**: Naive → Optimized → Production-ready
- **Comprehensive documentation**: Every major concept explained
- **Documentation**: Verify documentation is up-to-date with code changes.
- **Gotchas**: When you find a subtle bug or "gotcha" during implementation, YOU MUST add it to `docs/backend_gotchas.md` with an explanation and fix. This legacy helps future backend implementations.
- **Model Management**: Use the `huggingface-cli` to download and manage models when needed.
- **Real-world validation**: Test with actual model weights

## Questions to Ask Before Major Changes

1. Does this make the code easier or harder to understand?
2. Is this optimization premature, or does it address a real bottleneck?
3. Are we maintaining both naive and optimized versions?
4. Have we documented the "why" as well as the "what"?
5. Does this work with TinyLLaMA specifically?
6. Are we following C++23 best practices?

---

When in doubt, prioritize clarity and educational value. Performance optimizations should be added alongside clear implementations, not at their expense.
