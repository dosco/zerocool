# FreeLLM: quality-preserving inference beyond RAM on a 32GB M1 Pro

Approved September 7, 2026. This is the implementation contract; status and
measured evidence belong in [qwen_engine.md](qwen_engine.md) and the benchmark
reports. Planned capabilities must not be presented as implemented.

## Direction and success criteria

Build one native C++23/Metal engine for Qwen3.8-Flash-Next. Coordinate weight
precision, SSD layout, caching, and execution order to reduce the time spent
waiting for each token's required weights while preserving useful coding
quality. Promote optimizations using complete requests at equal memory budgets.

| Requirement | Target |
|---|---|
| Machine | Actual 32GB M1 Pro and internal SSD |
| Engine memory | At most 22GiB, reduced by system limits |
| Context | 8192 tokens, including output |
| Concurrency | One active conversation |
| Generation | At least 5 tokens/s, aim for 8 |
| Initial 2K prompt | First token within 60 seconds |
| 128-token append to retained 4K history | First token within 10 seconds |
| Sustained use | 20-minute coding session without progressive memory or sustained swap growth |

The quality target is the mixed 4/8-bit artifact. Retain the existing Q4 artifact
as the unchanged numerical and performance control. Publisher-reported lower
quantization error motivates evaluation; it does not establish native coding
quality or M1 performance. [Published comparison](https://huggingface.co/pipenetwork/Qwen3.8-Flash-Next-MLX-4bit#quality)

| Purpose | Repository | Revision |
|---|---|---|
| Q4 control | pipenetwork/Qwen3.8-Flash-Next-MLX-4bit | aa7c790e804bbf9d491ddb109c3d61bc4a555f7c |
| Quality reference | pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit | b2c422f3c643e36f04227a64d61796b44a4b1029 |
| Original source | Qwen/Qwen3.8-Flash-Next | de4b8e4d43b917e7706784d8bb445c9af86a3540 |

Within an artifact, execute every router-selected expert with fixed arithmetic
and reduction order. Cache state and read-completion order cannot alter results.
Different quantizations may produce different routes. Never change precision or
OS memory limits automatically. Changing artifacts requires new session state.

The next exact-arithmetic performance stage is specified in
[qwen_next_stage.md](qwen_next_stage.md). It preserves this plan's artifact and
quality boundaries while measuring prompt and generation latency separately.

## Integrated architecture

### Prepared storage

Create a versioned artifact before lower-bit kernels. Place each expert's packed
projections, scales, and biases in one aligned contiguous record, preserving
source bytes. Record formats, dimensions, offsets, lengths, alignment, and
hashes. Reuse the index design for Q4 and later mixed formats.

Interleave each ngram row's existing 80+10+10 bytes into a 100-byte record without
changing its hash-to-row address. Coalesce duplicate rows and storage pages,
including outstanding reads. Store decoded cached values as BF16: the current
decoder already rounds to BF16, so this introduces no additional loss. Keep the
packed ngram tables unchanged and SSD-backed; their roughly 32GB size alone
prevents full residency.

### Completion pipeline

One inference coordinator owns expert-cache mutation and Metal encoding. Reader
and GPU callbacks publish completions. Use a sliding window of at most 32 leases
and two live command groups, admitted under the byte budget. Compute any ready
experts, scatter to original token/expert positions, and retain the existing
final reduction. Submit ready work before waiting; begin with groups of four
and sweep 1, 2, 4, 8. Release resources after their final GPU use, then admit more
reads without a batch barrier.

Use one bounded scheduler with eight workers initially. Current dependencies
outrank future reads. Reserve half the queue for demand, cap prefetch, and prevent
large ngram lookups from filling the demand queue before routed experts.

This applies PowerInfer's completion-to-task mechanism at expert granularity,
without its model-specific neuron skipping or moved router.
[Implementation](https://github.com/Tiiny-AI/PowerInfer/blob/8bd56d69906c9d2dba4d3bf6899763401e01a9a4/smallthinker/powerinfer/moe_sparse_pipeline/expert_cache.cpp#L119-L128)

### Request schedules

| Schedule | Behavior |
|---|---|
| Generation | Route one token, overlap misses with ready/shared work, reduce |
| Retained-history append | Group appended tokens by expert, retain cache and state; explicitly cover 128-token follow-ups |
| Large prefill | Take a bounded panel through one layer at a time, grouping expert work across the panel |

Start prefill with a 1024-token panel and 128-token attention/recurrent
microchunks. Stream each selected expert once per panel where possible, compute
its rows in bounded microbatches, and never require a complete expert layer in
RAM. Compare panels 256, 512, 1024 against the old chunk-major schedule at equal
total memory. Track absolute positions per layer, commit progress after the
panel, and invalidate partially updated state on failure/cancellation after
draining outstanding users. Keep generation and short append dedicated paths.

### Joint memory and precision planning

Account for actual aligned allocations, metadata, resident weights, recurrent
and attention state, panel activations, expert contributions, both caches,
staging, live GPU resources, and driver reserve. Count shared allocations once.
Use a few expert record-size classes within one byte budget with free-capacity
rebalancing. Start with global CLOCK. Shrink by evicting eligible entries while
retaining useful survivors; reduce panel size before refusing a request when
that makes it fit. Higher-precision resident weights reduce expert-cache space
and must be charged in every comparison.

## Implementation milestones

1. **Measure dependencies.** Replay 48 successive layers with ten selected
   experts each from recorded hit/miss distributions and representative GPU
   work. Measure queue delay, read service, last-required-read latency,
   ready-to-GPU delay, GPU work, and buffer lifetime. Separate ready hits,
   outstanding-read joins, and new misses. Report application bytes separately
   from device counters, which include other processes. Establish normal Q4
   initial, append, and generation baselines; diagnostic trunk streaming cannot
   qualify performance. Budgets are 200ms/token for 5 tokens/s and 125ms for 8.
2. **Improve unchanged Q4.** Implement prepared expert/ngram records, bounded
   I/O priorities, completion-driven submission and release. Measure layout and
   scheduling independently and together. Require unchanged logits and state.
3. **Establish mixed 4/8 reference.** Add checkpoint-specific affine Q8 for
   sensitive non-expert matrices while preserving BF16 tensors and Q4 experts
   and ngrams. Verify tokenizer, architecture, tensor correspondence, and hashes
   before payload reuse. Compare native operators and full inference with an
   independent implementation. Measure actual resident increase/cache decrease.
4. **Reduce prefill rereads.** Implement layer-major panels with the existing
   schedule retained as reference/fallback. Select the fastest measured admitted
   configuration, not merely the largest panel.
5. **Calibrate selective compression.** Gather bounded per-expert activation
   statistics from coding, tools, continuations, and general text. Keep
   calibration, selection, and held-out evaluation disjoint. Implement one first
   lower-bit format: affine Q3, group size 64. Compare Q3 gate/up + Q4 down;
   Q3 routed projections with Q4 restored in sensitive groups; and the unchanged
   mixed 4/8 reference. Uncovered groups remain Q4. Derive new weights from the
   verified original source; Q4-to-Q3 is exploratory only. Gate/up-only Q3 saves
   about 15% of current expert payload, not 25%, after scale/bias overhead.
   Select using quality, latency, and total memory together. Defer Q2.
6. **Trace-justified cache/prefetch.** Compare CLOCK with one probation/protected
   policy at equal bytes; protected entries remain evictable and prefill gives
   no permanent importance. First run expert prediction in shadow mode. Initial
   live prediction uses previous-token routes one layer ahead, at most two
   predicted experts and one active speculative read. Yield to demand, do not
   displace actively needed records, and always fall back to the true router.
   Prediction affects read timing only. [PowerInfer-2](https://arxiv.org/html/2406.06282v3#S4)

Mixed-reference validation and calibration preparation may proceed alongside
the first two milestones. The first deliverable is a dependency-level report
and numerically identical Q4 inference using contiguous records and completions.

## Interfaces and qualification

Keep `inspect`, `run`, `bench`, `serve`. Inspect reports artifact/recipe identity,
precision distribution, actual prepared bytes, and allocations. Bench adds
completion tracing, recorded-route replay, per-layer waits, byte-weighted cache
metrics, prefetch usefulness, and actual computation reuse. Run/serve require
an explicit artifact; API model IDs identify the recipe. Offline developer tools
prepare, calibrate, convert, and evaluate; native inference needs no Python/MLX.

Validate reversed read completion, delayed GPU completion, all-hit/all-miss and
mixed loads, duplicates, eviction, resizing, cancellation, corrupted reads,
irregular panels, sparse boundaries, ngram history/EOS, and continuation versus
fresh replay. Include changed tools, edited history, rollback, and compaction.
Use independent decoders/operators for Q3/Q4 experts and Q8 resident matrices.
Use five alternating paired performance repetitions on identical workloads,
plus freely generated coding sessions whose routes may differ.

Keep runtime correctness and quantization quality separate. Compression must
add at most 0.02 nats/token held-out NLL and lose at most two percentage points
coding success relative to mixed 4/8, using paired confidence bounds.
Inconclusive results fail promotion. Required tool-call and recovery cases must
pass separately. Promotion also requires all latency targets, 7K reporting,
and the sustained coding workflow. Bind reports to native build, artifact,
recipe, and evaluation data. Missing model assets never produce a release pass.

## Lessons retained from ds4 and Lily

Use distinct short-appends, ordinary prefill cache references, direct reads into
reusable shared buffers, overlapped hit/miss work, and bounded resource lifetimes.
Avoid serial duplicate read-ahead before worker submission. Use M1-compatible
Metal rather than M5-only TensorOps. Adapt fusion, packed computation, and fewer
intermediate dispatches to this artifact's affine format; IQ2/Q4_K kernels are
not interchangeable. ds4 comparisons on larger Macs justify experiments, not
M1 performance claims. Preserve MIT/Apache notices when reusing code.

Vision, extra models/backends, neuron pruning, training, speculative decoding,
and lossy KV/recurrent/ngram compression remain outside v1.
