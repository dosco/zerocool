# Layer-major panels: implementation and bounded validation

Native build: `86d30a8a70f848c55d18d9ceeb77750a918fee90ad8f55e8b2da2a600a6b2cc9`.
Actual 32GiB M1 Pro, internal SSD, unchanged pinned Q4 artifact and lossless
prepared records. This implements the panel experiment; it does **not** qualify
full-model panel correctness or normal-request performance.

## Implemented behavior

Explicit `--panel 256|512|1024` runs one bounded panel through each layer,
grouping its routed token rows so an expert is acquired once per panel.
Attention, Gated DeltaNet, PLE, and each expert's token rows still run in bounded
microchunks. GPU copies preserve activation bits and avoid CPU readback.
Single-token generation retains its existing schedule. The default is `--panel 0`.

Each layer tracks its own absolute position. Only a completed panel commits
global history; cancellation or failure invalidates partial state and drains
GPU/I/O users. Memory admission includes panel activations, router partials,
expert contributions, shared work, and individually aligned state buffers.
It reduces panel size before refusing admission. Diagnostic accounting now
counts only actually loaded layers; the driver reserve and system headroom are
unchanged.

## Correctness evidence

- [Native suite](native-tests.txt): **27 tests, 873 assertions**, with Metal API
  and shader validation enabled. Includes panel admission/fallback, aligned
  state sizing, and bit-preserving GPU copy/slice bounds.
- [Short real fixture](four-layer-small.json): **16 checks passed** across
  panel 0/256/512/1024, one-token microchunks, irregular tails, EOS-containing
  history, retained append, and continuation.
- [Sparse-boundary fixture](sparse-boundary.json): **16 checks passed** with a
  2,053-token prefix, 129-token append, two continuation tokens, and 32-token
  microchunks. All requested panel sizes were admitted. This crosses the 2,048
  sparse-attention boundary.
- Both real fixtures compare retained convolution, recurrent, attention,
  index, PLE, history, and position state exactly against panel 0, and against
  fresh replay. Injected failure and cancellation occur after partial layer
  work; state becomes unusable and outstanding GPU users drain.
- [Storage replay](storage-correctness.json): all **1,553** selected expert
  records match their original bytes; all **48 isolated MoE outputs** and ngram
  cache replays match the saved control exactly on this build.
- [Mixed Q8 operators](q8-operators.json): **35 cases, 110,566 values** match
  the fixed per-token MLX reference exactly. [Four invalid Q8 fixtures](q8-rejections.json)
  are rejected. This does not qualify the full mixed artifact.
- Release-evidence rejection, dependency-summary, and paired-comparison
  harness tests pass: [four](release-tests.txt), [two](dependency-tests.txt),
  and [two](comparison-tests.txt) cases respectively.

The panel fixtures execute the first four real layers, including Gated
DeltaNet, PLE, and sparse attention, with diagnostic trunk streaming. They
compare persistent state, **not full-model logits or the truncated final
activation**. Full 48-layer panel comparison remains required. Historical
five-token full-model parity belongs to the [preceding build](../2026-09-08-audit-q8/README.md).

## Expert reads

The sparse-boundary fixture used a fixed 32-slot expert cache. The following
counters cover its continued prefix/append/continuation sequence, before fresh
replay; they are application expert bytes, not physical device traffic.

| Panel | Planned allocation, GiB | Peak Metal allocation, MiB | Expert read bytes |
|---|---:|---:|---:|
| 0 (32-token chunks) | 1.683 | 536.75 | 97,857,331,200 |
| 256 | 1.755 | 562.36 | 30,636,748,800 |
| 512 | 1.827 | 617.38 | 19,854,028,800 |
| 1,024 | 1.971 | 727.39 | 12,900,556,800 |

Panel 1,024 performed **86.8% fewer expert read bytes** in this diagnostic.
Panel scratch increases the allocation, so this is not an equal-total-memory
latency comparison. It provides evidence that grouping reduces repeated expert
loads; it does not establish a full-model speedup or the generation target.

## Remaining qualification and reproducibility

The [normal paired attempt](paired-attempt.json) stopped before inference:
normal execution required at least 5,275,336,704 engine bytes, but live system
availability admitted only 1,644,642,304. The [full-model diagnostic attempt](full-model-admission.txt)
also failed admission (2,280,062,976 required, 1,961,426,944 available).
These are incomplete checks, not passing results. No memory limit was raised.

The preceding normal 2K/256-output baseline remains **461.55s to first token,
1.753 tokens/s**, failing the 60s/5-tokens/s targets. Panels must first pass
full-model numerical/state comparison, then five alternating normal-request
comparisons with identical artifacts, tokens, and total admitted memory.
4K append, 7K reporting, sustained coding, and mixed/Q3 quality gates remain open.

```sh
# Real panel correctness: fail if assets or admission are unavailable.
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 build/qwen/qwen_panel_check \
  .cache/models/qwen38-flash-next .cache/prepared/q4-records-v1 \
  tests/fixtures/qwen/panel-small.json full-panel-check.json
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 build/qwen/qwen_panel_check \
  .cache/models/qwen38-flash-next .cache/prepared/q4-records-v1 \
  tests/fixtures/qwen/panel-sparse-boundary.json boundary-check.json

# Timing: disable validation. Output must be a new directory.
# 7.5GiB is a requested common budget, not a guarantee of current admission.
python3 scripts/qwen/benchmark_panels.py \
  --prepared .cache/prepared/q4-records-v1 \
  --workload docs/benchmarks/2026-09-08-audit-q8/normal-2k-workload.json \
  --memory-gb 7.5 --output panel-comparison
```

The paired runner alternates panel order, requires at least five repetitions,
and rejects changed builds, artifacts, generated tokens, reuse, admitted panel
size, budget, or cache resizing under pressure. It records admission failures
and never treats diagnostic streaming as normal performance.

[Evidence identity](evidence-identity.json) binds reports and checker sources
to this native build. Raw reports retain machine, prepared artifact, memory,
cache, and state identities.
