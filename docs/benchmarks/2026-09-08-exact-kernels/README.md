# Exact-arithmetic kernel stage — September 8, 2026

Packed-weight reuse across prompt tokens substantially improved the first normal
mixed-artifact screen. Generation did not improve. Original kernels remain the
runtime default; no dispatch rule or product acceptance gate is promoted here.

## Normal request screen

Actual 32GiB M1 Pro/internal SSD; mixed artifact
`b2c422f3c643e36f04227a64d61796b44a4b1029`; unchanged prepared records
`c4bb4db3220a0de2218128738b2a6ac650a3d65086690c8dc2e8c23d097160da`.
Each configuration ran in a fresh process with the same 2048-token prompt,
64 greedy output tokens, 8192 context capacity, 8GiB budget, panel 512,
microchunk 128, eight I/O workers and ready groups of four experts.

| Configuration | First token | Generation wall rate | Expert slots |
| --- | ---: | ---: | ---: |
| Original kernels | 402.974s | 1.642 tokens/s | 297 |
| Precomputed GDN gates | 388.410s | 1.635 tokens/s | 297 |
| Token tile 8 + precomputed gates | 231.154s | 1.633 tokens/s | 297 |

The combined candidate reduced first-token time by 42.6% in this screen.
All generated token IDs were identical. This is one repetition in the order
shown, with no confidence bound; it does not establish full-logit/state parity
at 2K. All-hit, cold, and warm OS-cache behavior are not implied by a fresh
runtime cache. Initialization/pipeline creation is reported separately.

All three runs read the same 169.916GB during prompt ingestion and 83.608GB
during generation (application bytes, decimal GB). There were no expert-cache
hits in either phase. The cache held 297 experts, fewer than the 480 selected
across 48 layers for one token. This constrains the 8GiB generation result.

The sum of GPU command intervals during ingestion fell from 394.140s to
224.805s. During generation it stayed near 23.8s. GPU intervals, CPU waits,
and I/O service times overlap; adding them would double-count elapsed time.

Evidence: [`screen-8g/summary.json`](screen-8g/summary.json), individual native
reports/admission records/workloads beside it. Screening build:
`1b0d41a9e9a8a8f469fe1a81c7ddc4010e4cbf68a6fe36465c59fb8a676ae69c`.
Its binary, source archive and exact runner are saved under
`.cache/controls/1b0d41a9/`. Later reporting changes use a different build;
the historical results are not relabeled as measurements of that build.

## Implemented and checked

Current native build:
`402c3a651b2f720b2ae48f9319408415c8746b566dfbcba5f6f5a40671d183b8`.
[`current-build.json`](current-build.json) records its binary/archive hashes;
the local preserved control is under `.cache/controls/402c3a65/`.
The final change after measurement build `4a19a77e...` corrects scalar precision
labels in profiling. The Metal kernel source and inference arithmetic are unchanged.

- GDN gate preparation and six shared-staging geometries preserve the original
  32-lane/four-adjacent-value reduction and BF16 boundaries. Gate scratch is
  charged before expert-cache admission.
- Q4/Q8 token tiles 2/4/8 reuse packed weights across independent accumulators,
  covering ordinary, fused gate/up and gathered expert work. One-token linear
  execution uses the original kernel.
- Native tests: 34 tests, 1043 assertions, no skips, with Metal API and shader
  validation. See [`native-tests.txt`](native-tests.txt).
- Offline tools: 37 tests covering comparison rejection, sampling/reuse,
  confidence bounds, tuning, session checks and soak evidence. See
  [`tool-tests.txt`](tool-tests.txt).
- The Q4 real-weight sweep passed all 960 bitwise comparisons on build
  `4a19a77e...`. The earlier mixed real-weight sweep also passed 960 comparisons.
  Each uses five alternating operator repetitions, pinned real weights, and
  deterministic synthetic BF16 activations. See
  [`q4-operator-screen.json`](q4-operator-screen.json) and
  [`mixed-operator-screen.json`](mixed-operator-screen.json).
- Both artifacts passed short original-versus-candidate native session checks:
  48 layers, route hashes, all retained buffers and final logits; fresh replay;
  partial-panel cancellation/failure drain; and CLI priming/append versus fresh
  generation. Final-build results are
  [`mixed-session-final-build/summary.json`](mixed-session-final-build/summary.json)
  and [`q4-session-final-build/summary.json`](q4-session-final-build/summary.json).

The first prime harness attempt used an incomplete chat header and terminated
at EOS before exercising decode. It was rejected, retained under
`mixed-session-final/`, and rerun using the established raw-token fixture.
This was an insufficient test input, not an output mismatch.

An earlier mixed five-token candidate probe preserved saved reference logits
and persistent state (`mixed-full-probe.json`, build `ae99dcf3...`). Independent
five-token model-reference evidence is kept separate from native optimization
parity. None of these short checks establishes long-context correctness or
coding quality.

The current build also rechecked both saved independently verified five-token
fixtures: all 248,320 logits and every persistent-state buffer are identical.
See [`final-fixture-recheck.json`](final-fixture-recheck.json).
The independent reference computation was not rerun; its saved payload and
reference-report hashes are recorded.

## Diagnostic phase trace

Build `4a19a77e...`'s bounded mixed 2K+64 trace retained 7,761 command groups,
17,322 operation families and 96 expert-dependency passes. Detailed capture
reached its bound; aggregate operation and dependency counters continued.
Generated tokens matched the normal screen. See
[`profile-summary.json`](profile-summary.json) and
[`profile-shapes.json`](profile-shapes.json).
The full [trace is losslessly compressed](mixed-2k-profile.json.gz); its raw
copy remains under `.cache/benchmarks/exact-kernels-4a19a77e/`.

Resident matrices commonly processed 128 or 512 rows. Of 62,615 routed-expert
microbatches, 38,208 (61.0%) had at most eight rows. This supports testing kernel
choices against actual row distributions rather than a single large matrix.

The profiled run was slower (306.7s first token, 0.816 tokens/s), so it is not
used to claim a speedup or isolate an operator's cost. Group timings and
per-expert waits overlap, and the instrumentation changes host timing. Normal
comparisons reject this report's `profiling_enabled: true` flag.

The raw historical trace used a default `bits` value for some unquantized
matrices. The shape summary identifies them from their kernel names without
inventing a scalar dtype. The final profiler emits explicit affine/BF16/F16/F32
metadata, covered by the new native regression check.

## Separate cache-capacity screen

Both budgets passed initial admission. Each ran the same combined candidate,
mixed artifact, panel/chunk/scheduler configuration, and 2K+64 workload in a
fresh process on build `4a19a77e...`; 12GiB ran first. All generated tokens match.

| Engine budget | Expert slots | Generation hit rate | Generation read bytes | First token | Generation wall rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 12GiB | 1,848 | 54.5% | 38.038GB | 334.152s | 1.385 tokens/s |
| 18GiB | 4,175 | 73.7% | 22.022GB | 299.914s | 1.744 tokens/s |

Neither run meets the latency targets. End-of-run process footprints were
10.43GiB and 16.43GiB, including approximately 5.6GiB reported as compressed
in both runs. System swap was unchanged at 12GiB and grew by 1.73GiB during
the 18GiB run. Other processes also ran on this laptop, so system counters
cannot attribute the growth to FreeLLM alone. These observations do not qualify
18GiB for sustained use, or isolate cache capacity as the only timing cause.

Evidence: [`cache-12-18g/summary.json`](cache-12-18g/summary.json) and
[`memory-observations.json`](cache-12-18g/memory-observations.json).
The first checker rejected the native binary32 serialization of `top_p=0.95`.
That checker was corrected and regression-tested; the complete 12GiB native
report was revalidated without rerunning or changing its measurements. The
original failed checker summary remains in its `12g` directory. Only the
missing 18GiB workload was executed on resume.

The next memory investigation should distinguish an engine cache hit from
data that macOS can supply without decompression or swap. Initial admission
and a high cache-hit percentage are insufficient performance evidence.

## Outstanding qualification

The finite schedule sweep, five alternating paired comparisons at 2K/4K/append
(ten if inconclusive), exact long session cases at 2053+129, 4096+128 and 7K,
7K performance, and sustained coding/memory qualification remain separate gates.
The tools reject missing workloads, reduced budgets/panels, changed builds,
changed outputs, profiling, insufficient outputs, or changed sampling.

The performance targets remain >=5 tokens/s, <=60s first token for 2K input,
and <=10s for a 128-token append to retained 4K history. This screen fails the
first two and does not measure the third. No ANE offload, new quantization,
pruning, prediction, or speculative decoding was introduced.

See [the implemented stage and commands](../../qwen_next_stage.md).
