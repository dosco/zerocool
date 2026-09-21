# Residency and execution stage

This stage implements benchmark-only memory-residency and execution candidates
for the pinned Q4 and mixed 4/8-bit artifacts. Production defaults remain on the
reference paths. Weight bytes, router decisions, numerical rounding and the
final expert reduction are unchanged. The 22GiB ceiling, lower live admission
limits, 8192-token context, and one active conversation still apply.

Measured results and their qualification limits are retained in the
[M1 stage report](benchmarks/2026-09-08-residency/README.md).

## Native candidates

`--residency off|core|core-cache` requests Metal residency for no additional
allocations, resident weights/session state, or core plus expert-cache buffers.
Allocation identity survives cache-slot content replacement. State replacements
are registered individually. Last-owner destruction queues retirement; the
coordinator removes residency before releasing the allocation/accounting.
GPU and I/O ownership still determine when storage may be reused. Requested
residency bytes are a subset of existing allocation bytes, not extra payload.
Set overhead is bounded to 64MiB within the existing driver reserve. Explicit
unsupported modes fail; no OS memory limit is changed.

`--decode-path reference|direct|grouped` selects original per-expert output,
direct output into the router-assigned contribution slot, or two grouped Q4
projection dispatches. Direct output removes the ten scatter operations per
layer for single-token decode. Grouped execution uses `--ready-group 1|2|4|8`,
flushes partial ready groups immediately, and retains the existing 32-lease and
two-command-group limits. Two bounded activation buffers are reused only after
their GPU groups complete. Multi-token expert execution retains its original
path and reduction.

`--prefill-pipeline serial|double` selects the original microchunk drain or two
bounded, reusable temporary allocation pools. The double path waits before
reusing a pool. Persistent convolution state bypasses the pools; panel outputs
remain live through their final use. CPU router and sparse-mask visibility
boundaries remain. Both pools are reserved before cache admission; cancellation
invalidates partial session state and drains their users.

New controls are accepted by `bench` and `inspect`. `run` and `serve` reject
experimental execution selectors until normal-request qualification promotes a
configuration. `inspect --expert-slots` now reports the actual requested cap.
Reports include execution identity, allocation count, temporary-pool reuse,
residency coverage/overhead, phase process decompressions, faults/pageins,
compression and memory peaks. Unavailable OS counters remain null. System swap
and device traffic include other applications.

## Complete cached-token diagnostic

`bench --cached-token-replay --tokens-file FILE --memory-gb 12 --expert-slots 480`
primes all but the final input token without sampling. It deep-copies all six
possible state-buffer fields per layer, plus history, validity and positions.
The last token first runs with original kernels to obtain real routes, logits
and resulting state. Those exact expert records and ngram rows are preloaded;
the same token then executes with the selected candidate after restoring the
snapshot. Every router still executes. Snapshots are separately charged before
memory admission; ordinary shared-pointer copies are never used as snapshots.

An untimed warmup precedes restored repetitions. Each repetition requires exact
logits, all state bytes and route hashes, 480 ready hits, zero expert/ngram
misses, and zero checkpoint/prepared reads. Setup, restore and output hashing
are outside timing. These reports explicitly cannot qualify normal throughput.

```sh
python3 scripts/qwen/benchmark_cached_tokens.py --residency core-cache \
  --output .cache/benchmarks/cached-core-cache
```

The default cases follow 2048, 4096 and 7168 prompt tokens, with five repetitions
each, 8192 context capacity, 480 slots and a strictly checked 12GiB budget. Use
`--artifact q4-control --model .cache/models/qwen38-flash-next` for the control.
One process/model runs at a time.

## Captured operators and shape rules

`bench --operator-capture NEW_DIRECTORY` captures at most 32 distinct affine
operator shapes/phase combinations and 256MiB of packed weights, metadata and
real input activations. It is diagnostic profiling and may insert waits.
Outputs are bound to native build/artifact and each payload has a byte count
and SHA-256. Existing nonempty output directories are rejected.

```sh
build/qwen/bin/zerocool bench --operator-fixtures CAPTURE/manifest.json \
  --artifact mixed-4_8bit --repetitions 5 --json operators.json
python3 scripts/qwen/select_shape_rules.py operators.json --output shape-policy.json
```

The fixture runner verifies identity, file bounds and hashes, then alternates
tiles 1/2/4/8 while requiring bitwise output equality. Selection requires five
complete repetitions and an improvement bound for every captured input case
of a shape. Single-token operations keep their reference tile. A selected
policy is forced with `--kernel-policy candidate --shape-policy FILE`, without
an additional forced token tile. Untested shapes use the reference kernel.
A stale build/artifact policy is rejected. Operator selection is never runtime
promotion; it still needs normal-request and full-state checks.

## Finite normal-request experiments

```sh
python3 scripts/qwen/benchmark_execution.py --experiment residency --memory-gb 12 \
  --cases prompt_2k append_128 --output .cache/benchmarks/residency-12g
python3 scripts/qwen/benchmark_execution.py --experiment decode --memory-gb 12 \
  --output .cache/benchmarks/decode-paths
python3 scripts/qwen/benchmark_execution.py --experiment prefill --memory-gb 12 \
  --output .cache/benchmarks/prefill-pipeline
```

Residency comparison fixes the same expert capacity across all three arms.
Decode compares reference, direct and grouped sizes 1/2/4/8. Prefill compares
serial and double workspaces. `--base-config FILE` supplies one previously
checked configuration. Final comparisons keep equal total budgets and include
workspace costs in cache admission. The existing panel/chunk sweep remains
available after selecting the execution path. Use separate directories for
18GiB experiments, only when live admission permits. Process-exit memory cleanup
can lag the prior process exit: admission is retried after 2 and 5 seconds,
retaining every rejected metadata report. Inference is never retried silently
and the requested budget/panel is never reduced.

The normal runner records all new execution selectors and rejects unexpected
policies, diagnostic profiling, snapshot replay, changed cache capacity,
reduced panels/budgets, different artifact/build, sampling or output tokens.
Five alternating paired 2K+256, 4K+256 and retained-4K+128+256 comparisons remain
the promotion gate, extended to ten when inconclusive. Improvement is required
in at least one primary latency measure; the upper paired 95% bound must stay
within 3% regression for every other measure.

## Verification

```sh
./build.sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 build/qwen/test_qwen
.cache/qwen-reference-venv/bin/python -m unittest discover -s scripts/qwen -p 'test_*.py'
python3 scripts/qwen/qualify_exact_sessions.py --case short --memory-gb 8 \
  --residency core-cache --decode-path grouped --prefill-pipeline double \
  --output .cache/benchmarks/execution-short
```

The native cases exercise direct-output sentinels, grouped partial batches,
reversed read readiness, eviction, cancellation, failed reads, snapshot alias
rejection, replacement state, executor teardown, pool reuse, captured file
corruption, shape-policy identity and memory admission. Long session cases
`boundary`, `append`, and `7k` remain mandatory for promotion. Independent
saved-reference checks, 7K performance, the 20-minute memory check and real
coding/tool recovery qualification remain distinct from short native parity.
The session qualifier also accepts `--ready-group` and `--shape-policy` so the
same selected configuration is checked; a shape policy requires `--token-tile 1`.
The bounded `microchunks` case uses a 257-token prefix plus a 129-token append
to exercise 128-token workspaces, an irregular tail, reuse and fresh replay.
It does not replace the long sparse-attention boundary cases.

Product targets remain at least 5 tokens/s, at most 60s initial 2K TTFT, and
at most 10s TTFT after a 128-token append to retained 4K history. Stage test
passes and isolated timing gains do not establish those targets. Q3 conversion,
cache-policy changes, prediction and speculation are outside this stage.

The [memory and computation stage](qwen_memory_compute_stage.md) adds explicit
prompt-workspace reclamation, filtered operator capture, and exact affine
output-row/gate-up candidates. Fixed allocation and existing kernel policies
remain the defaults.
