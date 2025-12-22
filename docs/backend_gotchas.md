# Backend Implementation Gotchas

This document serves as a "lessons learned" repository for implementing compute backends (Metal, CUDA, Triton, etc.) in FreeLLM. When you encounter a subtle bug or "gotcha," specifically one that might recur in other backends, verify it, fix it, and **document it here**.

## 1. Kernel Grid & Block Dimensions

### The "Truncated Head Dimension" Bug
**Symptoms:**
- The model produces garbage output (random tokens, repetition) despite weight loading and basic kernels appearing correct.
- "Simple" unit tests (e.g., identity checks) might pass if they use small dimensions, but full inference fails.

**Cause:**
Calculations involving `head_dim` (the size of a single attention head vector) require a sufficient number of threads in the threadgroup (block).
If the kernel code expects one thread per vector element (or handles tiling based on `threads_per_group`), hardcoding the block size (e.g., to `32`) for a model with a larger `head_dim` (e.g., `64` or `128`) results in SILENT failure. Data beyond index 31 is simply ignored or zeroed out.

**Fix:**
**NEVER** hardcode block dimensions for dynamic model parameters. Always calculate them at runtime based on the config.

**Incorrect (Metal Example):**
```cpp
// BAD: Hardcoded block size of 32
grid.block = infra::Dim3(32, 1, 1); 
```

**Correct (Metal Example):**
```cpp
// GOOD: Round up head_dim to nearest warp/simdgroup size
int attn_block_size = (config.head_dim + 31) / 32 * 32;
grid.block = infra::Dim3(attn_block_size, 1, 1);
```

**Applies to:**
- Attention Kernels (`paged_attention`, `gqa_attention`)
- RMSNorm (if reducing across `head_dim` or `d_model`)
- RoPE (Rotary Embeddings)

### The "DispatchThreads vs. DispatchGroups" Trap
**Symptoms:**
- Kernel executes but produces results only for a small subset of the data (e.g., only the first few rows or heads).
- Performance is suspiciously low.

**Cause:**
APIs like Metal's `dispatchThreads` take the **total number of threads** in the grid, whereas CUDA/HIP (and Triton) usually take the **number of blocks (groups)**.
If you pass `n_heads` (e.g., 32) as the grid size to Metal, expecting 32 *groups*, you will instead get 32 *threads* total (likely 1 group), processing only 1/32th of the work.

**Fix:**
Know your backend API's convention.

- **CUDA/Triton:** Grid = `(n_blocks_x, n_blocks_y, n_blocks_z)`
- **Metal (`dispatchThreads`):** Grid = `(n_blocks_x * block_size_x, n_blocks_y * ...)`

## 2. Kernel Parameter Passing

### The "Scalar by Value" Trap in Metal
**Symptoms:**
- Integer or Float parameters passed to kernels read as `0` or random values.
- `setBytes` works for some indices but fails for others.

**Cause:**
Passing scalars by value to `constant` references in Metal kernels via `setBytes` index binding can be fragile depending on argument alignment and compiler padding.

**Fix:**
Allocate small `DeviceBuffer`s for ALL scalar parameters (constants) and pass them as standard buffers. This guarantees memory layout visibility.

```cpp
auto b_head_dim = backend->allocate(sizeof(int), DType::INT32); // Safe
```
