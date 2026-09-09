# Cached decode computation stage

The previous mixed-artifact screen reached 2.846 tokens/s at a 12GiB budget.
Its 2K cached-token diagnostic still took a median 274.90ms with 480 ready
expert hits and no model reads. The next experiment measures and reduces that
computation path while retaining the artifact, exact arithmetic, router choices,
and persistent state. A 150ms cached-token result is an engineering target,
not an established hardware limit or a normal-request performance claim.

## Measurement

`bench --cached-token-replay --dispatch-profile FILE` adds GPU timestamp
measurements to each dispatch. On M1 this requires a separate compute pass per
dispatch; command submission and cache-lease completion boundaries are retained.
This changes encoder overhead and is explicitly diagnostic. The normal cached
timing validator rejects instrumented reports.

Each live command group owns its counter buffer through GPU completion. Each
buffer has 4096 samples (32KiB); capacity exhaustion fails instead of silently
dropping dispatches. Two submitted groups and the current encoded group bound
counter storage within the existing driver/bookkeeping reserve. Samples resolve
only after completion. Paired CPU/GPU clock samples convert GPU durations to
nanoseconds, following Apple's [timestamp conversion guidance](https://developer.apple.com/documentation/metal/converting-gpu-timestamps-into-cpu-time).

`scripts/qwen/summarize_decode_profile.py REPORT PROFILE --output FILE` excludes
warmup, verifies all-hit/exact execution and complete dispatch coverage, and
groups measured GPU passes by kernel, stage, and layer. GPU pass durations and
CPU waits must not be added together as a wall-time breakdown.

`--cached-compare` with `--q8-decode-rows` alternates the current control and
the specified packed-Q8 candidate in the same cached-token process. Both arms
restore every persistent state buffer before each forward, execute all routers,
and require exact logits/routes/state and zero model reads. Warmup is excluded.
`summarize_cached_comparison.py` requires at least five complete alternating
pairs, matching kernel, execution and memory configurations apart from the Q8
candidate, and no profiling.

## Packed Q8 candidate

`--q8-decode-rows 0|2|4|8` defaults to zero (existing kernels). Candidate mode
allows packed 32-bit weight loads and compile-time lane widths for single-token
affine Q8 matrices, reusing input loads and bias sums across independent output
rows. Width selection follows the original kernel's partition. Scalar product
accumulation, metadata grouping, BF16 output rounding and SIMD reduction order
are unchanged. Multi-token computation and Q4 tensors retain their existing
paths. Unaligned weight bindings fall back to the existing Q8 implementation.

Captured operator replay with a nonzero `--q8-decode-rows` compares that variant
to the previous two-row implementation. Its separate report kind prevents the
ordinary shape selector from treating these as token-tile measurements.
`scripts/qwen/benchmark_q8_decode.py` runs these comparisons serially and rejects
missing, duplicate, inexact or unpaired samples. Capture separate tuning and
held-out token windows with `capture_phase_inputs.py`; do not relabel a tuning
manifest or change its build identity to make it qualify as fresh evidence.

## Experiment and acceptance

Use the largest measured cost to select one bounded exact-arithmetic candidate.
Keep the previous best configuration as its experimental control. Preserve the
12GiB total budget, mixed artifact, prepared records, panel/microchunk settings,
and all router-selected experts.

Validate numerical edge cases and GPU lifetime behavior, then complete cached
full-state replay. Compare unprofiled normal generation and sample-free retained
history appends using identical workloads. A single screen can reject a
candidate; production promotion still requires the existing paired-workload,
long-context, sustained-memory, and coding/recovery gates. Record negative and
inconclusive results alongside improvements.
