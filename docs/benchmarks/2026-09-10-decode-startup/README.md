# Generation startup and core residency

Requesting Metal residency for the core weights and session state substantially
reduced the observed generation startup stall in both alternating pairs. The
first four decode forwards fell from **3.72 / 4.15 seconds to 1.69 / 1.71 seconds**,
while their recorded process decompressions fell from **478,188 / 446,245 to
1 / 54**. This supports a memory-compression explanation for the recent startup
variability and a short normal timing screen of core residency.

This is a completed diagnostic, not performance qualification. All four runs
have per-step observation enabled. CLOCK and residency-off remain the defaults;
no weights, precision, arithmetic or OS memory limits changed.

## Method and scope

The actual 32GiB M1 Pro ran two alternating fresh-process pairs, off/core then
core/off. Only requested residency changed. Each run used the same 72-token
coding prompt, 33 greedy outputs, reference kernels, CLOCK, 12GiB allocation,
1,848 expert slots, 512-token panel, 128-token microchunks, four ready experts
per group and eight I/O workers. Context capacity was 8,192 tokens. The experiment
completed in **125.83 seconds**, within its 360-second deadline.

The existing Metal residency implementation was reused. The new opt-in
`bench --workload-file FILE --decode-diagnostics` records process, GPU, expert-read
and dependency counters before and after each of the first 32 committed decode
forwards. It adds no GPU submission or wait. Reports identify the token, absolute
position, timing interval and captured/omitted counts. All 128 expected intervals
were captured here; prefill and sampling have no per-step coverage.

The first-four split was selected after inspecting the earlier
[SLRU screen](../2026-09-10-slru-screen/README.md), before this experiment.
The [earlier residency screen](../2026-09-08-residency/README.md) used a 2K prompt,
candidate kernels and recorded zero decompressions. It does not resolve the
compression episode measured here, nor does this short experiment supersede its
longer-prompt observations.

## Observed timings and memory activity

| Pair / residency | Complete request | First token | First 4 forwards | Next 28 forwards | First 4 decompressions |
|---|---:|---:|---:|---:|---:|
| 0 / off | 31.38 s | 16.18 s | 3.72 s | 11.45 s | 478,188 |
| 0 / core | 26.57 s | 14.12 s | 1.69 s | 10.75 s | 1 |
| 1 / core | 26.92 s | 14.49 s | 1.71 s | 10.70 s | 54 |
| 1 / off | 29.35 s | 14.48 s | 4.15 s | 10.69 s | 446,245 |

Core/off complete-request ratios were **0.8468 and 0.9173**, or 15.32% and 8.27%
less elapsed time. Decode wall time fell 18.03% and 16.40%. First-token latency
improved in one pair and was effectively unchanged in the reverse pair. There
are only two pairs and no confidence bounds. These instrumented timings cannot
be used as normal request acceptance evidence.

The startup difference aligns much more closely with CPU waiting and process
decompression than with reported GPU command duration:

| Pair / residency | First 4 CPU waits for GPU | First 4 GPU command duration |
|---|---:|---:|
| 0 / off | 2.731 s | 1.088 s |
| 0 / core | 0.883 s | 1.000 s |
| 1 / core | 0.879 s | 0.999 s |
| 1 / off | 3.086 s | 1.105 s |

All four first-four intervals issued exactly 997 expert reads and 2,756,505,600
application bytes. Across all 32 forwards, misses were 6,534 / 6,533 / 6,533 /
6,532 in run order. Completion timing can alter cache turnover; that tiny read
difference does not explain the startup gap. GPU command durations, CPU waits
and concurrent I/O service totals overlap. They must not be added or subtracted
to manufacture a critical-path breakdown. The counters support the hypothesis;
they do not attribute every millisecond or prove which specific pages decompressed.

Core registered 5,933,858,816 bytes: 5,362,515,968 resident matrix bytes and
571,342,848 state/control bytes. No expert buffers were registered. These are
existing allocations, not extra payload or guaranteed physical residency.
The admitted memory plan and expert capacity matched in every run.

After the requests, process compressed memory was 6.05 / 5.60GB with residency
off, versus 10.96 / 106.45MB with core. The largest sampled physical footprint
was approximately 10.42GiB. Observed system swap usage stayed at 1,891,368,960
bytes across all captured boundaries. These are boundary observations, not
continuous peak measurements or sustained-session proof; system counters include
other processes.

Measured counter-snapshot work totaled 1.83–2.34ms per request. That excludes
outer sample assembly and report serialization. Request wall time includes the
observation work performed during generation; this experiment does not measure
instrumentation overhead against an uninstrumented arm.

## Verification and evidence

- **56 native tests / 45,682 assertions passed**, with Metal API and shader
  validation and zero skips. The new observer test verifies that reading counters
  does not submit a pending GPU encoder. Existing residency lifecycle tests pass.
- **168 Python tests passed**, including bounded coverage, missing/reset counters,
  invalid token/timing identity, incomplete evidence and exclusion of instrumented
  requests from normal timing validation. Two existing SQLite resource warnings
  were emitted; there were no test failures.
- All four real requests produced exactly the same 33 output token IDs as each
  other and the prior uninstrumented CLOCK control. They used the same artifact,
  allocation and reference arithmetic. This is output parity, not an independent
  full-logit/state comparison or coding-quality evaluation.
- The frozen source/artifact identity, all four request reports, all 128 samples
  and both original/copied raw seals were revalidated offline after completion.

Native build:
`968132061556ac0077ff41b7c922b5b7c632bb12e21b8ea8e2c5f82f10bb3c73`.
Mixed artifact: `b2c422f3c643e36f04227a64d61796b44a4b1029`, with the unchanged
prepared Q4 expert records.

[Raw summary](raw/summary.json), [historical analysis](raw/historical.json),
[derived observations](observations.json), [native tests](native-tests.log),
[Python tests](python-tests.log), [ledger result](ledger-result.json) and
[verification hashes](verification.json) retain the evidence and its limitations.

## Next decision

Run a bounded, uninstrumented paired screen changing only off/core residency,
with a short initial request and retained-history append at the same budget.
Check exact outputs, memory and complete conversation latency before longer
qualification. Keep CLOCK and avoid further cache-policy tuning until the startup
variance is controlled or explicitly accounted for.

Core runs reached approximately **2.57 tokens/s**, still below the 5 tokens/s
target. Their remaining 28 forwards took about 0.38 seconds each, so reducing
startup stalls alone does not solve sustained generation. This experiment does
not qualify 2K/4K/7K contexts, a 4K-history append or the 20-minute coding session.
