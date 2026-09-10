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
   Tie each expert's gate/up format, group size, and bitrate so its fused kernel
   remains valid; down precision may differ. Calibration captures exact templated
   model-visible token streams, including generated reasoning, coding, tool calls,
   tool results, continuations and recovery. Keep calibration, candidate selection,
   and final evaluation conversations disjoint. Evaluate actual quantized
   candidates rather than importing EXL3's noise-sensitivity estimates, whose
   optimizer is explicitly untested on sparse models. Measure padded record bytes,
   resulting cache capacity, exposed read waits, decoding computation and complete
   requests together. Tool parser/recovery checks must pass independently.
   EXL3 trellis support is deferred until affine-Q3 results identify a concrete
   need; any later investigation starts with a bounded M1 operator experiment.
6. **Trace-justified cache/prefetch.** Compare CLOCK with one probation/protected
   policy at equal bytes; protected entries remain evictable and prefill gives
   no permanent importance. First run expert prediction in shadow mode. Initial
   live prediction uses previous-token routes one layer ahead, at most two
   predicted experts and one active speculative read. Yield to demand, do not
   displace actively needed records, and always fall back to the true router.
   Prediction affects read timing only. [PowerInfer-2](https://arxiv.org/html/2406.06282v3#S4)
   The offline `query_evidence.py cache` query now provides equal-byte CLOCK,
   probation/protected SLRU and future-aware MIN curves from saved routes, under
   explicit fixed-order, immediate-release assumptions. It reports application
   read bytes, not a native speedup. Existing five-token prefill and cancellation
   fixtures do not establish normal generation locality; do not change the live
   policy from those curves. First obtain a deadline-limited 32-token normal
   continuation with complete routes and explicit phase/session/reset boundaries.
   Extend promising evidence to 256 generated tokens and retained-history append,
   then screen a single policy at equal admitted memory before full qualification.
   `capture_routes.py` now implements the first capture with a 180-second default
   deadline, the mixed artifact and unchanged reference kernels at 12GiB. Native
   committed-route markers distinguish prefill, decode, session changes and aborts;
   no partial forward is admitted to the cache simulation. Instrumented timings
   remain ineligible for performance qualification.
   The [first complete normal capture](benchmarks/2026-09-10-normal-routes/README.md)
   recorded all 32 decode steps in 34 seconds at the fixed budget. At its actual
   1,848 slots, SLRU simulated 10.71% fewer decode expert reads than CLOCK, but no
   latency gain is established. This supports the longer locality capture and
   append before implementing a policy; native CLOCK remains the default.
   The [extended capture](benchmarks/2026-09-10-extended-routes/README.md) completed
   256 decode steps, a 128-token append and 32 further steps in 167 seconds at the
   same budget, verifying 328 tokens of actual state reuse. SLRU simulated 5.72%
   fewer first-request decode reads and 9.46% fewer append reads, but 0.42% more
   reads during generation after the append: 4.95% fewer across the conversation.
   Instrumented native generation was 2.47 tokens/s and append TTFT 24.13 seconds;
   neither proves acceptance. Next implement this one policy behind an experimental
   option, verify lifetimes and exact state, then run a short equal-budget paired
   request screen including append. Keep CLOCK until measured latency supports a
   change; the modest simulated read benefit does not establish a speedup.
   SLRU is now implemented behind `bench`/`inspect --cache-policy slru`; CLOCK
   remains the production default. The [native screen](benchmarks/2026-09-10-slru-screen/README.md)
   passed exact all-layer logits/routes/state and failure/cancellation checks under
   forced eviction. In two alternating short-history pairs, SLRU issued 9.72% fewer
   expert reads, but whole-conversation time was 15.97% lower in one pair and 5.08%
   higher in the other. The predeclared repeatability gate failed; no five-pair or
   long qualification followed. Preserve this inconclusive result in the ledger.
   Use existing memory/dependency evidence to investigate variable initial decode
   latency before tuning policies or rerunning expensive validation. Do not assign
   the observed timing variance to compression without testing that explanation.
   The [bounded startup diagnostic](benchmarks/2026-09-10-decode-startup/README.md)
   now links per-token counters to two alternating off/core residency pairs at
   the same 12GiB allocation. Core reduced the first four decode forwards from
   3.72/4.15s to 1.69/1.71s, with decompressions falling from 478,188/446,245 to
   1/54 and identical output tokens. This supports the startup-memory hypothesis;
   all runs are instrumented and do not qualify normal latency. Next screen
   off/core without diagnostics, including retained-history append, before
   longer qualification or more cache tuning. Keep existing defaults; observed
   core generation remains about 2.57 tokens/s, below the 5 tokens/s target.
   The [normal residency screen](benchmarks/2026-09-10-residency-screen/README.md)
   then passed its predeclared gate: conversation time fell 4.94% and 6.06% in
   two alternating pairs, with identical outputs, 104-token follow-up reuse,
   memory plans and expert-read counts. Diagnostics were disabled. Next use five
   paired normal repetitions with uncertainty estimates before longer
   qualification; keep defaults unchanged. Core still generates about 2.5
   tokens/s, and the short-history follow-up still waits about 24s for its first
   token. This does not meet the product targets.

Mixed-reference validation and calibration preparation may proceed alongside
the first two milestones. The first deliverable is a dependency-level report
and numerically identical Q4 inference using contiguous records and completions.

## Current exact-attention experiment

The [sparse-attention stage](qwen_sparse_attention_stage.md) specifies CPU-equivalent
GPU selection, wholly masked score-tile skipping, heavily selected expert and
execution-lifetime regression coverage, and isolated paired measurements. This
stage changes no artifact bytes or production defaults. Its numerical and timing
qualification does not substitute for quantization quality or product acceptance.

The next deliverable is [screen-first selector evaluation](qwen_selector_qualification_stage.md):
CPU/full versus GPU/full only at the fixed 12GiB budget. Verify quick operator
and saved-input correctness, then use `triage_selector.py` for five cached 4K
pairs with a 600-second total deadline. Stop weak or inconclusive candidates;
timeouts and resource blocks remain unfinished. A promising result proceeds to
a small normal append screen (`triage_requests.py`, one pair with a 900-second
total deadline), then full 7K/session/recovery
validation and the established paired normal-performance gate. The existing
all-phase qualifier retains its old order and is reserved for survivors. Reuse
only original, revalidated evidence. No default or precision change is implied.

## Interfaces and qualification

Use the [offline evidence queries and experiment ledger](qwen_evidence_queries.md)
to choose experiments and retain negative or blocked outcomes. Start with existing
reports and verified paired comparisons. The bounded cache simulation answers
read-volume what-ifs while keeping incomplete trace coverage explicit. Critical-path
ranking, shared runtime event IDs and new lifecycle probes follow only when a
concrete decision cannot be answered from current evidence. The index remains rebuildable;
raw reports and ledger JSON retain their original provenance.

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
