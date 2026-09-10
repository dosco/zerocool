# Native Qwen engine: implementation and qualification

This is the specialized engine authorized in the September 2026 plan. It is
experimental. The presence of all four commands does **not** mean that the
numerical, coding-quality, or M1 performance acceptance gates have passed.
Current evidence is under `docs/benchmarks/2026-09-08-validation/`; the earlier
Q4 baseline and audit are under `docs/benchmarks/2026-09-08-audit-q8/`.

The [sparse-attention experiment](qwen_sparse_attention_stage.md) adds exact GPU
block selection and wholly masked score-tile skipping, with production expert
microbatch and resource-lifetime regression coverage. Its [current evidence](benchmarks/2026-09-09-sparse-attention/README.md)
separates cached-token comparisons from normal requests and promotion gates.
Both attention candidates remain opt-in for benchmarking.

The [exact-arithmetic performance stage](qwen_next_stage.md) now implements
benchmark-selectable GDN gate preparation/shared staging and Q4/Q8 token tiles,
fresh-process comparisons, sample-free session priming, bounded phase traces,
and finite tuning/qualification tools. Its [separate report](benchmarks/2026-09-08-exact-kernels/README.md)
binds each result to its build and distinguishes screens from promotion evidence.
Automatic dispatch still selects the original kernels.

The [approved implementation plan](qwen_plan.md) uses mixed 4/8-bit weights as
the quality target, retaining Q4 as the unchanged control. The complete mixed
artifact is integrated behind explicit `--artifact mixed-4_8bit --model DIR`.
Its 24 required files are hash-verified. A complete byte comparison qualifies
reuse of the Q4 prepared expert/ngram records, with both source locks and
the equivalence report pinned in `mixed-payload-reuse.lock.json`.
Full-model comparison exposed PLE and GDN normalization rounding discrepancies.
After correcting both, all 48 layers and all 248,320 logits match the independent
five-token reference exactly for Q4 and mixed precision. Source/prepared layouts,
cache sizes, batching and short session continuation preserve all retained state.
The mixed runtime is explicitly selectable; Q4 stays the default. Long-context
correctness, coding quality, calibration/Q3, cache-policy experiments and
prediction remain pending.
Layer-major panels are implemented behind explicit `--panel 256|512|1024`;
the default remains `--panel 0` while panel qualification continues.
The [latest report](benchmarks/2026-09-08-validation/README.md) records 30 native
tests/896 assertions, 53 full-model fixture/state checks per artifact and
20 normal session/panel/failure checks per artifact. The small session fixture
has a five-token prompt, three-token append including EOS, and two continuations;
requested panels 512 and 1024 are admitted as 256 at its context limit of 256.
Earlier full 48-layer Q4 panel checks also passed:
257-token prompt, 129-token append, two continuations, and identical retained
state versus fresh replay. Panel 512 reduced expert read bytes by 59.4% at
32 cache slots. Those longer runs precede the normalization changes and must
be requalified; paired equal-memory normal-request panel timing remains unmeasured.
One normal mixed 2K/256-output request with panel 512 and a 12GiB budget completed
at 470.63s to first token and 1.985 tokens/s. Memory stayed within budget and
system swap decreased. Both latency targets fail; this is one baseline, not
the required paired performance or coding-quality qualification.

The [storage/scheduling foundation report](benchmarks/2026-09-07-foundation/README.md)
records the September 7 foundation build. It implements lossless prepared expert/ngram records,
BF16 decoded ngram caching, demand-priority reads, and completion-driven expert
execution. All 1553 routed expert records in the saved fixture match source bytes;
all 48 layers' expert contributions and the ngram cache replays match bit for bit.
That suite passed 22 tests and 741 assertions with Metal API/shader validation,
and 11 altered/truncated prepared-artifact cases were rejected.
The [September 8 audit](benchmarks/2026-09-08-audit-q8/README.md) then passed 25
tests/855 assertions and requalified the five-token full model, including every
retained state buffer across source/prepared and batched/completion schedules.
A normal 2K/256-output run completed at 461.55s to first token and 1.753 tokens/s.
These fail the latency targets. Memory availability varies substantially while
other applications run; admission continues to enforce live limits.

The following full-model results belong to the earlier control build
`8c56e73ec097e9520f95e63a7bfe97c9b381cd5f94a233da95e1f7686b044107`:

The saved five-token full-model fixture now matches the original MLX reference
bit for bit: all 48 layer outputs and all 248320 final logits are identical.
The same logits are produced with chunk=8/cache=32 and chunk=1/cache=64.
All 290 independent CPU operator checks pass. These results qualify this
fixture, not longer contexts or coding quality.

The fixes cover router matrix accumulation, expert reduction order, BF16
attention score/probability boundaries, and the recurrent state's ordinary
four-value sum. The native suite passes 18 tests and 599 assertions with Metal
API and shader validation. Normal performance and the sustained coding-session
acceptance gates remain open; memory admission has also refused normal startup
when other applications leave too little reclaimable memory.

## Scope and preserved work

The default target is C++23/Metal on a 32GiB Apple M1 Pro, one active
conversation, an 8192-token input-plus-output limit, and a maximum 22GiB
engine budget. The old implementation remains in place behind
`FREELLM_BUILD_LEGACY=ON`. No existing model or backend was deleted. The
initial uncommitted work was also backed up before the new implementation.

`freellm_lib` provides the native library. `include/qwen/` exposes checkpoint,
model, tokenizer, session, and local-server interfaces. The production path
has no Python or MLX dependency. Independent reference tooling uses Python.
`cmake --install build/qwen --prefix /your/prefix` installs the static library,
public headers (including its JSON header dependency), CLI, and third-party
notices. A native client links `freellm_lib` and the Foundation, Metal, and
IOKit frameworks using C++23.

Metal source and the model lock are embedded in the binary, so installation
does not require running from the source directory.

## Pinned artifact

`--artifact q4-control` is the default. `--artifact mixed-4_8bit` selects the
experimental complete checkpoint in `mixed-models.lock.json` at
`b2c422f3c643e36f04227a64d61796b44a4b1029`. Always supply the matching `--model`
directory. Startup rejects a receipt from the other artifact. The API model ID,
statistics, route replay and session state bind to the selected revision;
state from another artifact is rejected before any computation. The production
runtime has no Python or MLX dependency for either selection.

The mixed artifact keeps Q4 routed experts and Q4 ngrams and uses 498 resident
affine Q8 matrices. Existing BF16 tensors are preserved. Its resident allocation
is 5,362,515,968 bytes, an increase of 2,424,832,000 bytes (2.258GiB) over Q4.
That increase is included in memory admission and reduces expert-cache capacity.
The unchanged Q4 prepared sidecar is accepted only through the pinned complete
payload-equivalence proof; its source and consuming revisions are both reported.

`models.lock.json` pins the model repository
`pipenetwork/Qwen3.8-Flash-Next-MLX-4bit` at
`aa7c790e804bbf9d491ddb109c3d61bc4a555f7c`, its tokenizer and template, all
required file sizes and SHA256 hashes, and the consulted Slotstream,
Slotpack, and ds4 revisions. The downloaded artifact has 11 safetensors
shards and 3215 tensors, with 103,769,745,912 bytes of tensor payload.

| Payload | Bytes | Treatment |
|---|---:|---|
| Routed experts | 67,947,724,800 | SSD-backed, packed affine Q4 |
| Ngram tables | 32,000,153,600 | SSD-backed, indexed Q4 row reads |
| Remaining tensor payload | 3,821,867,512 | Includes tensors outside the resident forward path |
| Resident forward allocations | 2,937,683,968 | Packed trunk weights, including alignment |

No full-shard FP32 conversion or mapping is used. Each expert retains its
original codes, scales, and biases. One expert occupies 2,764,800 payload
bytes in a 2,768,896-byte aligned cache slot. Its three projections each
require code, scale, and bias reads in the source layout: nine reads per expert
and three reads per ngram row. An explicit `--prepared DIR` selects the lossless
sidecar with one expert read and one 100-byte row read. Requests starting on the
same 4KiB page coalesce into one bounded range; singleton rows still read only
100 bytes. The sidecar contains 100,048,541,696 bytes, including alignment, and
requires that much additional disk space while preserving the source checkpoint.

`models.lock.json` pins the prepared manifest SHA256. Startup validates that pin,
source hashes, ranges/formats, payload receipts, and current file identities.
`prepare_storage.py --verify` rehashes every prepared payload. The manifest
contains source hashes and each prepared file's hash; the verification receipt
contains local file fingerprints and is regenerated after a full verification.

```sh
.cache/qwen-reference-venv/bin/python scripts/qwen/prepare_storage.py \
  --output .cache/prepared/q4-records-v1
.cache/qwen-reference-venv/bin/python scripts/qwen/prepare_storage.py \
  --output .cache/prepared/q4-records-v1 --verify
build/qwen/bin/freellm inspect --prepared .cache/prepared/q4-records-v1
```

Preparation requires NumPy, uses bounded panels, resumes completed unchanged
files, and publishes each file atomically. Production inference stays native.

`verify_checkpoint.py` hashes every required file and writes a receipt.
Native startup checks that receipt against the embedded lock and current
file identity, size, and modification/change times. A release check rehashes
all files; the receipt is a startup optimization, not an independent trust
boundary against a local user modifying both receipt and files.

## Memory and ownership

Admission uses the smallest of the requested budget, physical memory minus
8GiB, Metal's recommended working set, and currently reclaimable memory
minus 1.5GiB. It includes resident weights, recurrent and attention state,
expert slots, scratch, a separate 64MiB ngram cache, and a 1GiB CPU/driver
reserve. Buffers visible to both the CPU and GPU are counted once.

The expert cache defaults to global CLOCK replacement. Loading, ready, and leased
records have explicit ownership. A slot cannot be overwritten while an I/O
job or encoded GPU operation still uses it. Prefill references receive the
same CLOCK treatment as generation references. Tests exercise eviction,
resizing, and pinned leases. Shrinkage evicts eligible entries and keeps survivors.
`bench` and `inspect` accept experimental `--cache-policy slru`. New entries enter
probation; a hit moves an entry to protected MRU. Protected capacity targets
floor(75% of slots); excess entries move back to probation. Eviction prefers the
oldest eligible probation entry, then the oldest eligible protected entry if
probation is busy. Neither queue can evict a loading or leased buffer. Shrinkage
preserves surviving queue order and restores the protected target; growth retains
all entries. Clear empties both queues. The coordinator owns all queue changes.
Links are embedded in the existing entry metadata, with the same allocation layout
for CLOCK and SLRU. No extra record or queue buffers are allocated. Metadata stays
within the common CPU/driver reserve; reported `entry_metadata_bytes` covers entry
objects only, excluding allocator, lookup and future overhead. This estimate is
not added to shared weight allocations or used to enlarge the admitted cache.
The policy and probation/protected counts appear in native statistics. SLRU remains
an explicit experiment; `run` and `serve` retain CLOCK.
Decoded ngram cache rows now hold BF16 values (the existing decoder already
rounds to BF16), with an explicit allowance for index overhead. Duplicate rows,
including queued ones in a lookup, share the same read and decoded result.

The fixed allocation budget is enforced at each Metal allocation. Process
physical footprint, compressed memory, swap counters, and live/peak Metal
allocations are also reported; the CPU/driver reserve is an estimate that
still requires the sustained-session acceptance test. No code changes
macOS wired-memory settings or closes other applications.

## Forward pass and arithmetic

The runtime implements all 48 layers: Gated DeltaNet, sparse attention,
four-stream gated residuals, PLE/ngram lookup, shared experts, and the ten
selected routed experts. The architecture and tensor shapes are fixed and
validated at load time. Attention uses bounded score buffers and exact causal block masks, including
each query's partial tail. Scores and probabilities have explicit BF16 rounding
boundaries, matching MLX's prefill fallback for 256-wide heads and GQA=12.
The native engine keeps this arithmetic fixed across chunks; MLX can select a
different fused attention path for one- or two-token calls.

Activations are stored in float buffers with explicit BF16 operation
boundaries. Packed affine Q4 computation follows MLX's GEMV input-sum
rounding; it does not use a GGUF dequantization recipe. BF16 sigmoid,
softplus, normalization, and four-way reduction boundaries are explicit.
The recurrent memory dot uses the original MLX four-value sum; compensated
summation changed a BF16 midpoint decision in a real fixture. The recurrence
retains FP32 state across native chunks. This is a deliberate
arithmetic choice inherited from the Slotstream design; the Python MLX
oracle can truncate state at its call boundaries, so session comparisons
must state their arithmetic and chunking.

The native router evaluates every router logit, selects exactly ten experts,
and uses descending score with expert ID as the tie-break. Its FP32 matrix
multiply uses M1 SIMD 8x8 fragments and sixteen fixed K partitions, matching
MLX 0.31.1 small-prefill arithmetic for this checkpoint. Keeping that geometry
fixed avoids chunk-dependent routing arithmetic. Expert sums follow MLX's
eight-part reduction: combine positions 0/8 and 1/9 before positions 2–7. Expert execution
order can change with cache availability. Contributions are stored by
selection position and reduced in a fixed order. There is no pruning,
expert prediction, speculative decoding, or lossy storage rearrangement.

Numerical agreement is checked separately from coding quality. Small
operator checks cannot establish full-model agreement, and coherent text
cannot establish correct logits. The release threshold is relative logit
L2 <= 0.02 and cosine >= 0.9998 against the independent full-model fixture;
passing one fixture is necessary but does not cover all contexts or sessions.

## Scheduling

| Path | Current behavior |
|---|---|
| One-token generation | Compact router readback, direct expert input, missing reads overlap available/shared GPU work |
| Short append | Group selected token rows by expert; consume live hits first |
| Large prefill | Default chunk-major forward pass, or explicitly requested layer-major panels |

The provisional short-append threshold is 32 tokens; it is configurable and
has not been qualified as the M1 crossover. Chunk sizes are limited to
1–256. Eight persistent I/O workers are the default. Fused gate/up activation
avoids full dequantized matrices and intermediate gate/up buffers. Metal
commands are grouped, with waits at routing, sparse-selection, and resource
reuse dependencies rather than after every kernel.

This incorporates ds4's cache-lifetime, ordinary prefill-seeding, and bounded
I/O lessons. It does not claim ds4's measured gains on another model and
machine. Whole-layer lookahead and larger matrix tiling remain measured
optimization candidates; they are not silently enabled.

The default expert executor admits at most 32 leases and two GPU groups. Readers
and GPU callbacks publish completion events; the inference coordinator encodes
ready experts, reaps completed buffers/leases, and admits replacements without
waiting for a whole batch. `--ready-group` accepts 1, 2, 4 (default), or 8.
`--legacy-schedule` keeps the previous batch schedule available for comparisons.
The one eight-worker read pool reserves half its queue for demand and always
services queued demand before future ngram/prefetch work. Outstanding work is
drained before cancellation can release its resources.

`--dependency-trace FILE` appends bounded per-layer JSON records including actual
routes, queue/read/encode/submit/release times, ready hits/loading joins/new misses,
and GPU group timings. GPU timestamps describe groups, not individual kernels;
overlapping group durations must not be summed as exclusive token latency.
`bench --replay-routes FILE --replay-hits N` uses the same executor and real expert
GPU operations, with explicit seeded hit counts for controlled experiments.
It excludes the rest of the model and is marked ineligible for latency promotion.

`--panel 256|512|1024` groups a panel's selected experts once per layer. Attention,
Gated DeltaNet and PLE run in bounded `--chunk` units first; expert token rows
also execute in bounded microbatches. GPU copies preserve panel activations
without CPU readback. Each layer has its own absolute position, and global
session progress commits only after the entire panel finishes. Failure or
cancellation invalidates partial state and drains outstanding users.

Panel scratch is an additional admitted allowance covering full activations,
router partials, expert contributions and shared work. Admission reduces the
requested panel to 512, 256 or the existing schedule before refusing a request.
The admitted size is reported. State allocations count actual 16KiB-aligned
buffers; truncated diagnostics count only their loaded layers.

With panels explicitly enabled, a retained-history append requiring 129 new
computations can run as one grouped panel, rather than reloading experts for
128 tokens and then one more. The default 32-token hit-ordering threshold still
needs request-level crossover measurements. Larger-panel correctness and
qualification status is recorded in the [panel report](benchmarks/2026-09-08-panels/README.md).

## Sessions and API

A session retains recurrent state, convolution history, attention/index
state, ngram history, and the token sequence actually consumed by the model.
An exact continuation reuses that state. An earlier edit, rollback, or
compaction clears and rebuilds it. Reports count actual reused computation,
not merely matching text. Context overflow returns an error.

The server binds to `127.0.0.1`, handles one active inference request, and
provides `/v1/models` and `/v1/chat/completions`. Chat completions support
SSE, greedy/stochastic sampling, reasoning text, and parsed tool calls.
Tool execution belongs to the client. Unsupported request features are
rejected explicitly. Disconnects and termination stop new work and drain
outstanding GPU/I/O uses before releasing resources.

## Reproducible checks

```sh
./build.sh
ctest --test-dir build/qwen --output-on-failure
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 build/qwen/test_qwen
python3 scripts/qwen/verify_checkpoint.py
build/qwen/bin/freellm inspect --json inspect.json
build/qwen/bin/freellm bench --storage --repetitions 3 --json storage.json
```

The model-free suite requires a real Metal device and fails when it is
unavailable. Configure `FREELLM_REAL_MODEL_TESTS=ON` to add a real-generation
CTest smoke case. That smoke case is not the release gate.

Run `bash scripts/qwen/setup_reference.sh` to install pinned diagnostic dependencies
in their own environment. For numerical diagnosis, provide a JSON array of token IDs:

```sh
build/qwen/bin/freellm bench --tokens-file tokens.json --chunk 8 \
  --context 256 --expert-slots 32 --memory-gb 3 --stream-trunk \
  --logits-file native.f32 --trace-dir native-trace --json native.json
.cache/qwen-reference-venv/bin/python scripts/qwen/reference_mlx.py \
  --model .cache/models/qwen38-flash-next --tokens tokens.json \
  --trace reference-trace --out reference.json
.cache/qwen-reference-venv/bin/python scripts/qwen/compare_logits.py \
  --native native.f32 --reference reference-trace/logits.f32 \
  --native-report native.json --reference-report reference.json --out logits.json
```

`--stream-trunk` is diagnostic only: one layer's resident weights are loaded
at a time so parity checks can run under current memory pressure. It is
rejected for normal generation and serving, and cannot satisfy performance
acceptance. `--probe-layers N --trace-dir DIR` supports smaller real-weight
operator fixtures. `reference_numpy.py` independently evaluates the CPU
operator equations and can run the complete model on short fixtures.

The small replay tool isolates routing and the expert sum without loading the
resident trunk. It links the production executor and caps GPU allocations at
64MiB. It requires a five-token trace; recorded expert outputs are inputs to
the check, so a pass is not evidence of full-forward agreement.

```sh
build/qwen/qwen_moe_replay .cache/models/qwen38-flash-next \
  native-trace/step_0 replay-output > replay-native.json
.cache/qwen-reference-venv/bin/python scripts/qwen/check_moe_replay.py \
  --model .cache/models/qwen38-flash-next --trace native-trace/step_0 \
  --replay replay-output --replay-report replay-native.json --out moe-replay.json
```

Attention can likewise be checked using recorded native operator inputs:

```sh
build/qwen/qwen_attention_replay .cache/models/qwen38-flash-next \
  native-trace/step_0 attention-output > attention-native.json
.cache/qwen-reference-venv/bin/python scripts/qwen/check_attention_replay.py \
  --replay attention-output --replay-report attention-native.json \
  --out attention-replay.json
.cache/qwen-reference-venv/bin/python scripts/qwen/test_release_check.py
```

Both replay tools are diagnostics and are not installed as user-facing commands.

For performance qualification:

```sh
.cache/qwen-reference-venv/bin/python scripts/qwen/prepare_benchmarks.py --out workloads.json
build/qwen/bin/freellm bench --workload-file workloads.json \
  --context 8192 --repetitions 3 --json performance.json
.cache/qwen-reference-venv/bin/python scripts/qwen/api_check.py --out api.json
.cache/qwen-reference-venv/bin/python scripts/qwen/release_check.py \
  --evidence docs/benchmarks/2026-09-07 --out release.json
```

The generated repeated-code workloads are controlled fixtures, not a
coding-quality score. Acceptance also requires representative coding
prompts, alternating comparison order, retained 128-token appends, 7K
context reporting, a 20-minute session, and independently checked
read/edit/test/recovery results. Missing evidence fails the release check. Numerical, cache, performance, API,
soak, and coding reports must identify the current native source fingerprint;
older successful reports cannot qualify a changed engine.

## Measured storage foundation on this M1 Pro

These are medians of three repetitions using real nine-piece expert reads
with `F_NOCACHE`, not a simulated memory cap on a newer Mac. GB/s is decimal.

| I/O workers | Expert GB/s | Optimistic minimum hit rate at 5 tokens/s | At 8 tokens/s |
|---:|---:|---:|---:|
| 1 | 1.28 | 80.7% | 88.0% |
| 4 | 3.75 | 43.6% | 64.7% |
| 8 | 4.88 | 26.4% | 54.0% |
| 16 | 5.62 | 15.3% | 47.1% |
| 32 | 5.82 | 12.3% | 45.2% |

A token selects 480 expert records, or 1,327,104,000 bytes before cache hits.
The lower bound is `max(0, 1 - measured_Bps / (bytes_per_token * target_tps))`.
It leaves no time for compute, routing, ngrams, or synchronization. Decode's
small sequential layer batches can achieve much less bandwidth. Therefore
this table does not establish achievable generation speed or justify
changing the default worker count.

At eight workers, cold ngram reads cost about 0.88ms per token in the indexed
read probe. Device counters are recorded separately from application read
bytes and include other processes; they are not attributed solely to
FreeLLM. Per-layer cache hits and misses accompany inference measurements.

## Qualification limitations

The [offline evidence query tool](qwen_evidence_queries.md) indexes existing
JSON/JSONL reports and experiment decisions in disposable SQLite. It exposes
paired comparisons, recorded memory boundaries and bounded dependency traces
with original source hashes. It runs no inference and makes no promotion claim.

Candidate iteration now starts with [bounded cached triage](qwen_selector_qualification_stage.md#fast-iteration-is-the-entry-point).
It checks five exact 4K CPU/GPU pairs within a 600-second total deadline and
never starts full qualification automatically. A cached speedup alone cannot
qualify normal requests. See the [current evidence](benchmarks/2026-09-09-selector-qualification/README.md)
for interrupted and resource-blocked attempts.

The published Slotstream v0.2.11 binary was attempted on this exact laptop.
Its bundled metallib requires Metal language 4.0, which macOS 15.6 rejects.
`slotstream.txt` records the failure. No comparable Slotstream throughput
was obtained. An OS-compatible build is required to complete that baseline;
substituting a different metallib would need a separately identified run.

The early native eight-token generation smoke measured 1.41 tokens/s and a
2.5-second first token with only 64 expert slots. It produced coherent text
and showed no swap increase during that short run. It predates subsequent
arithmetic corrections and is retained as historical evidence only.

Normal full-model startup has also been refused when other applications
leave too little reclaimable memory. Diagnostic layer streaming permits
numerical work under that condition, but cannot resolve the performance
qualification requirement. Consult the latest evidence status before
making any speed, fidelity, coding-quality, or release claim.
