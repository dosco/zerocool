# Exact-arithmetic performance stage

Approved September 8, 2026. Prompt latency and generation speed have equal
priority. Every candidate must preserve bit-for-bit logits, selected routes,
and all persistent state within each artifact. The Q4 control and mixed 4/8
reference, prepared storage, 8192-token context, and 22GiB maximum remain fixed.

## Implementation and evidence

- Preserve the original native control and source archive under `.cache/controls`.
- Keep original GPU kernels selectable; automatic dispatch remains the reference
  until paired normal-request evidence qualifies a rule.
- Precompute GDN decay/beta once per token/head using the existing rounding.
  Charge two aligned live gate buffers in admission. Experiment separately with
  4/8 value rows and 4/8/16-token shared staging, preserving the 32-lane reduction.
- Calculate 2/4/8 independent token accumulators per SIMD output row, reusing
  packed Q4/Q8 weights without changing the scalar accumulation order. Cover
  ordinary linear, fused gate/up, gathered rows and irregular tails. One-token
  linear execution keeps the original kernel.
- Benchmark-only flags force candidates. Report their identity and actual
  matrix row counts. Runtime chat APIs and artifact identities are unchanged.

## Measurement contract

Each independent conversation and repetition runs in a fresh process. Live
prime/append/decode phases retain caches and state. This establishes empty
runtime caches, not cold OS/device caches. Ordinary CLI repetitions explicitly
report retained caches instead of pretending that clearing history evicts them.

Priming ingests 4096 tokens without sampling, so the next 128-token append
measures exactly 128 new tokens. Ordinary generation also reports whether it
must ingest the pending final emitted token. Keep original forward-only decode
throughput and report wall-clock throughput separately; paired comparisons use
wall-clock generation latency.

Prepare GPU pipelines before request timing and report initialization separately.
Normal reports contain phase snapshots. Optional phase profiling annotates existing
command groups without changing submissions, buffers events until after the
request, and bounds detailed capture. Groups can contain several kinds of work;
GPU time and CPU/I/O wait overlap. Full counters remain separate from sampled
traces. Tensor dumps and legacy synchronous dependency traces are diagnostic.

## Finite experiments

1. Run exact operator tests and real-weight screening for GDN and token tiles.
2. Run normal 2K +64-token comparisons of reference, gate preparation, and combined
   candidates at one common admitted budget. Screen append behavior separately.
3. Sweep panels 0/256/512/1024, chunks 32/64/128, ready groups 1/2/4/8, and I/O
   workers 2/4/8 one dimension at a time; recheck the combined winner.
4. Keep 12-versus-18GiB cache experiments separate and only run them when current
   admission permits those budgets. Initial screening uses a common whole-GiB
   budget no larger than 12GiB. Reject reduced panels or mid-request cache resizing.
5. Compare shortlisted candidates with five alternating paired normal requests
   at 2K+256, 4K+256, and retained-4K+128+256; use ten pairs if inconclusive.
   Paired 95% bounds must show an improvement in at least one primary latency
   measure and no more than 3% regression in the others.

## Qualification

Keep independent five-token checks for both artifacts. Compare all 48 layers,
logits, route hashes, history, absolute positions and state at 2053+129, 4096+128,
and 7K context, including continuation versus fresh replay. Never load two
models together. Long native-reference agreement proves optimization parity;
it does not replace an independent model-quality reference.

Run irregular tails, nonzero recurrent state, forced eviction, reversed reads,
cancellation, corrupted reads, and Metal validation. Require exact requested
panel admission instead of counting a silently reduced panel as coverage.

Report 7K performance and a 20-minute coding-session memory check. Full product
acceptance still requires >=5 tokens/s, <=60s initial 2K TTFT, <=10s append TTFT,
and the independently checked coding workflow. No measured kernel improvement
alone qualifies the engine. ANE/CPU offload, new quantization, prediction and
speculation remain outside this stage.

## Running the implemented tools

The native candidates and measurement tools are implemented. Original kernels
remain the runtime default. See the [stage evidence](benchmarks/2026-09-08-exact-kernels/README.md)
for measured results and qualification that is still outstanding.

```sh
./build.sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 build/qwen/test_qwen
.cache/qwen-reference-venv/bin/python -m unittest discover -s scripts/qwen -p 'test_*.py'

# Normal fresh-process screen: reference, GDN preparation, tiled + GDN.
python3 scripts/qwen/benchmark_exact.py --mode screen --pairs 1 --memory-gb 8 \
  --cases prompt_2k --output /tmp/freellm-exact-screen

# Real mixed-weight operators with five alternating repetitions.
build/qwen/bin/freellm bench --kernel-bench --artifact mixed-4_8bit \
  --model .cache/qwen-mixed-reference --prepared .cache/prepared/q4-records-v1 \
  --repetitions 5 --json /tmp/freellm-operators.json

# All-layer native parity, including fresh replay, cancellation and priming.
python3 scripts/qwen/qualify_exact_sessions.py --case short --memory-gb 8 \
  --output /tmp/freellm-exact-session
```

Use `--case boundary`, `append`, or `7k` for the long session cases. These
execute reference and candidate sequentially and can take substantially longer
than a normal request because they replay the whole history. The short case
also checks `Session::prime` through the CLI, with no sampled token or pending
token silently included in append timing.

`benchmark_exact.py --mode paired --pairs 5` requires all three primary
workloads before its latency gate can pass. A restricted case list remains
partial evidence. `--cases prompt_7k` adds the 7K report. Output directories must
be new, keeping failed and successful runs distinguishable. Disable Metal
validation and profiling during normal timing.

`tune_exact.py --config candidate.json --memory-gb 8 --output DIR` takes one
configuration object with `name`, `kernel_policy`, `token_tile`, `gdn_path`,
`panel`, `chunk`, `ready_group`, and `io_workers`. It runs the finite four-axis
sweep, writes each selection, and never promotes defaults. Staged GDN additionally
accepts `gdn_rows` and `gdn_block`. Recheck a selected combination, then use the
paired runner before proposing a dispatch change.

`benchmark_cache_sizes.py --output DIR` runs the separate 12/18GiB capacity
screen with the same combined candidate and 2K+64 workload. Both budgets must
pass live admission. `--resume` can revalidate a completed native report against
current sources before running a missing budget; it does not treat an incomplete
native request as a result. Cache-hit gains must be considered alongside process
compression, GPU time, and complete-request latency.

For a diagnostic phase trace, add `--phase-profile FILE` to a native benchmark.
It records current command groups, actual matrix dimensions/row counts, and
bounded expert-read details. Phase dependency totals continue after detailed
capture fills. Profiled runs are rejected by the normal performance checker.

For a controlled memory soak, pass a coding conversation workload to native
`bench --workload-file FILE --soak-seconds 1200 --repetitions 1 --json REPORT`.
It completes each conversation before repeating, so elapsed time may exceed
20 minutes. Validate with `check_soak.py REPORT --output SUMMARY`. This checks
memory behavior; it does not substitute for independently scoring a real
read/edit/test/recovery workflow.

The next implementation extends this work with [bounded residency and execution
candidates](qwen_residency_stage.md). Its controls remain benchmark-only and
retain all exact-arithmetic and normal-request promotion requirements above.
