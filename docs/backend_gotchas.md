# Backend Implementation Gotchas

## GDN normalization can cross BF16 boundaries (2026-09-08)

After fixing PLE, mixed inference matched the first six layers exactly but
diverged in layer 6's query/key normalization. Two query values differed while
the incoming convolution values were identical. The old kernel summed strided
groups of four squares and used the approximate `rsqrt`; the pinned MLX code
uses adjacent groups of four FP32 squares and `precise::rsqrt`. Matching both
restores all 48 layer outputs and all 248,320 logits exactly for the saved
five-token fixture. `tests/fixtures/qwen/gdn-qk-mixed` reproduces the failure
before the correction and passes afterward. Keep the reduction fixed across
native chunk sizes; changing precision alone is not a numerical validation.

## PLE gates require BF16 partial sums (2026-09-08)

The mixed artifact exposed a PLE reduction mismatch that the original short Q4
fixture did not. Key/query projections and normalization were bit-identical,
but a float accumulation rounded a product sum to -314 instead of MLX's -312.
That changed a sigmoid gate and grew to 0.0986 relative logit L2 across 48 layers.
The fixed 2,560-wide reduction now mirrors MLX 0.31.1: 640 threads each add four
adjacent BF16 products with BF16 rounding, then twenty rounded SIMD partials and
a rounded final sum. The real activation fixture under
`tests/fixtures/qwen/ple-gate-mixed` preserves the failing boundary.

## Mixed artifact selection and input embedding (2026-09-08)

Supporting Q8 matrix kernels does not by itself make the complete mixed artifact
work. The ordinary forward path still dispatched a Q4-only embedding kernel,
while panels used the format-aware embedding operator. Both paths now use the
same declared-format operator. Artifact selection also binds the file lock,
route traces, API model ID, and session state; a state from another artifact is
rejected before mutation. Prepared Q4 reuse requires complete expert/ngram byte
equivalence. That comparison has now passed, and `mixed-payload-reuse.lock.json`
binds the prepared manifest and both complete source locks to the evidence.

## Qwen audit: Q8 rounding, failed state, and replay validation (2026-09-08)

Affine Q8 cannot reuse affine Q4's bias-correction arithmetic unchanged. MLX
0.31.1 Q4 sums four BF16 inputs with BF16 rounding at each addition; Q8 sums
individual inputs in FP32. The native Q8 path follows the latter and keeps
the QMV partition fixed across token counts. The real mixed-checkpoint fixture
generator compares native operators with independent per-token MLX QMV;
ordinary batched MLX QMM is reported separately because it changes arithmetic.

Recurrent and attention buffers are mutated before a forward pass produces
logits. A read, GPU, cancellation, or allocation failure must invalidate that
state, including when a native-library caller bypasses ModelSession. StateUpdate
marks it invalid before the first mutation and commits history only after all
work and output allocation succeed. Rebuilding is required after failure.

Recorded-route diagnostics must validate nonempty token counts, complete
48-layer passes, integral expert IDs and positions, and consistent build and
pass metadata before allocating GPU work. Converting JSON directly to int can
truncate fractions or wrap large IDs. Paired summaries must retain every
recorded pass and compare ordered expert-output hashes across layouts and
schedules; silently selecting runs[0] loses later passes.

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

## 3. Algorithm Implementation Nuances

### The "Normalization Mismatch" in MoE Gating
**Symptoms:**
- Model output is repetitive, garbage, or numerically unstable.
- Logits seem "reasonable" but downstream activations explode or vanish.

**Cause:**
Some MoE models (like OLMoE) depend on **raw probabilities** from the gating mechanism and explicitly disable post-TopK normalization.
- **Normalized Top-K:** `softmax(topk_logits)` -> Sums to 1.0. (Standard)
- **Raw Top-K:** `softmax(all_logits)[topk_indices]` -> Sums to < 1.0. (OLMoE style)

If you force normalization on a model trained without it, you artificially amplify the signal from the selected experts, breaking the model's internal scale expectations.

**Fix:**
Check the model's configuration for flags like `norm_topk_prob`. Do not assume standard Softmax behavior.
```cpp
// Correct logic:
if (config.norm_topk_prob) {
    normalize(probs); // Sum to 1.0
} else {
    // Keep raw probabilities from global softmax
}
```

### The "RoPE Rotation Style" Mismatch
**Symptoms:**
- Attention mechanism fails to attend to correct previous tokens.
- Output resembles random sampling from the vocabulary distribution.

**Cause:**
Rotary Positional Embeddings (RoPE) have two common implementation styles for applying coefficients to the head vector:
1. **Half Rotation (LLaMA style):** Pairs `x[i]` with `x[i + half_dim]`.
   `[-x[i+half], x[i]]`
2. **Interleaved Rotation (GPT-NeoX/OLMoE style):** Pairs `x[2i]` with `x[2i+1]`.
   `[-x[2i+1], x[2i]]`

Applying the wrong rotation style scrambles the positional information, making the attention mechanism ineffective.

**Fix:**
Verify the model architecture lineage.
- **LLaMA / Mistral / Gemma:** Use Half Rotation.
- **GPT-NeoX / OLMoE / Qwen:** Use Interleaved Rotation.

### The "Missing QK-Normalization" Bug
**Symptoms:**
- Model produces semantically wrong output (e.g., "The capital of France is" → "amongst" instead of "Paris")
- Embeddings and early layer outputs match reference implementation
- Hidden states diverge progressively through layers
- Final logits have completely wrong distribution

**Cause:**
Some models (OLMoE, Qwen2, Gemma2) apply RMSNorm to Q and K vectors **after** linear projection but **before** computing attention scores:

```
Q_proj = x @ W_q
K_proj = x @ W_k
Q = RMSNorm(Q_proj, q_norm_weight)  // Often missed!
K = RMSNorm(K_proj, k_norm_weight)  // Often missed!
scores = Q @ K.T / sqrt(head_dim)
```

Without this normalization, attention scores are computed on unnormalized vectors, causing the model to attend to wrong positions and produce garbage output.

**Detection:**
- Check for `self_attn.q_norm.weight` and `self_attn.k_norm.weight` in weight files
- These weights are NOT indicated in config.json for most models - must detect from weight names
- If these weights exist, QK-normalization is required

**Fix:**
1. Map HuggingFace weight names to internal names:
   - `model.layers.{i}.self_attn.q_norm.weight` → `layers.{i}.attn.q_norm_weight`
   - `model.layers.{i}.self_attn.k_norm.weight` → `layers.{i}.attn.k_norm_weight`
2. Load weights when present and set a flag to enable QK-norm
3. Apply RMSNorm to Q and K after linear projection, before RoPE and attention

**Applies to:**
- OLMoE (all variants)
- Qwen2 / Qwen2.5
- Gemma2
- Any model with `q_norm` / `k_norm` weights in attention layers


## Qwen residency and temporary-buffer lifetimes

Residency follows an allocated Metal buffer, not the expert currently stored in
it. Registering only initial state is insufficient: convolution updates replace
state buffers. Last-owner destruction may run on an I/O thread; queue retirement
for the inference coordinator and retain accounting through residency removal.
Explicit residency never replaces GPU completion or cache leases.

A copied `State` shares its `Buf` fields. Diagnostic replay must deep-copy all
state data and restore validity/history/positions. Double-workspace prefill must
exclude convolution outputs from temporary pools and preserve CPU sparse-mask
and router visibility waits. Pool reset requires the last submitted user to
complete, including commands submitted by an internal visibility boundary.

A completed Metal process can return its exit status before its working set is
fully reflected as reclaimable memory. The first 12GiB residency screen observed
this after the core arm. Benchmark admission now retries metadata after 2 and
5 seconds, preserving rejected reports; it never reduces the configured budget
or retries a partially executed inference request.

## Held-out operator measurements must remain paired

Do not calculate candidate/reference latency ratios by iterating only over
available reference samples. That silently discards an unmatched candidate
sample, which can hide a slow run. The held-out validator now requires
contiguous repetition IDs starting at zero and identical repetition coverage
for every reported variant of a selected shape. Missing pairs fail validation
before confidence bounds are computed. Regression tests include an unmatched
slow candidate and gaps in repetition numbering.

## M1 dispatch attribution requires diagnostic compute passes

The M1 Pro reports stage-boundary timestamp support but no dispatch-boundary
sampling. Per-dispatch attribution therefore uses separate compute encoders
inside the existing command buffers. Retain each sample buffer until command
completion, resolve only completed samples, and correlate CPU/GPU clocks before
converting durations. Fixed sample capacity must fail explicitly on overflow.
This instrumentation changes encoder overhead; compare performance separately
with counters disabled. Summed GPU passes and host waits overlap and cannot be
added as independent wall-time costs.
