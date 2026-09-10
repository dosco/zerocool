# Normal request screen of core residency

Core residency passed the predeclared short screen with diagnostics disabled:
complete-conversation time was **4.94% and 6.06% lower** in the two alternating
pairs. All output tokens, computation reuse, memory plans and expert-read counts
matched. This supports five paired repetitions; it does not qualify production
latency or change defaults.

The existing native engine was used unchanged from the
[startup diagnostic](../2026-09-10-decode-startup/README.md). The new
`scripts/qwen/screen_residency.py` provides a bounded, reproducible normal timing
comparison. CLOCK and residency-off remain defaults.

## Method and timing

The actual 32GiB M1 Pro ran four fresh processes, ordered off/core then core/off.
Each used the pinned mixed artifact, reference arithmetic, CLOCK, 12GiB budget,
1,848 expert slots, 512-token panels, 128-token microchunks, four ready experts per
group and eight I/O workers. Only requested residency changed. Route capture,
per-step diagnostics, profiling and Metal validation were disabled.

Each process ingested a 72-token coding prompt and generated 33 outputs. It then
appended 128 tokens to the live history and generated 33 more. The follow-up
reused 104 computed tokens and ingested the pending output plus 128 new inputs.
There are 32 measured generation intervals per request. The experiment finished
in **273.52 seconds**, inside its 480-second deadline.

| Pair / residency | Initial request | Follow-up request | Conversation | Initial generation | Follow-up generation |
|---|---:|---:|---:|---:|---:|
| 0 / off | 29.01 s | 37.99 s | 67.00 s | 2.45 tokens/s | 2.35 tokens/s |
| 0 / core | 26.78 s | 36.91 s | 63.69 s | 2.53 tokens/s | 2.50 tokens/s |
| 1 / core | 26.69 s | 36.84 s | 63.54 s | 2.54 tokens/s | 2.51 tokens/s |
| 1 / off | 30.50 s | 37.13 s | 67.64 s | 2.00 tokens/s | 2.44 tokens/s |

Core/off conversation ratios were **0.950602 and 0.939421**. Their median was
0.945011. The gate required both ratios below one, at least 1% median improvement,
and no per-request metric median regression beyond 3%. All conditions passed.
There are only two pairs, with no confidence bounds. Do not pool these runs with
the instrumented diagnostic as if their observation modes were identical.

Core reduced initial request time by 7.70% and 12.48%; follow-up request time fell
2.83% and 0.78%. Follow-up first-token times were 24.40 / 24.10 seconds in the
first pair and 24.08 / 24.00 seconds for core/off in the reverse pair. Residency
does not solve the follow-up prompt-processing delay or the 10-second target.

## Memory and read evidence

All four conversations issued the same **30,264 expert misses**, with identical
application bytes in each phase: 7,688 misses during initial ingestion, 6,533
during initial generation, 8,527 during follow-up ingestion and 7,516 during
follow-up generation. The observed latency difference was not a cache-read
reduction in this experiment.

Initial generation recorded 89,563 and 742,444 process decompressions with
residency off, versus zero and one with core. The first four forward times were
2.21 / 5.24 seconds off versus 1.63 / 1.69 seconds core. Those token times are
existing normal timing fields; decompression counters cover the whole phase and
cannot locate decompression at a particular token. Host memory pressure remains
an uncontrolled influence. The earlier per-step diagnostic supplies separate
evidence for the startup hypothesis.

Core registered 5,933,858,816 existing resident/state/control bytes, no expert
buffers, and no pending residency retirements at observed boundaries. Registration
requests residency from Metal; it does not guarantee it. No OS limits changed.
The largest sampled physical footprint across all requests was approximately
10.44GiB, within the 12GiB allocation limit. Observed system swap usage remained
1,891,368,960 bytes at all captured boundaries. These samples do not establish a
continuous peak or a sustained-session result; system counters include other
processes. Lower compressed-memory counters are not reduced total model storage.

## Verification

- **173 Python tests passed**. New cases cover the controlled configuration,
  inconsistent pairs, follow-up regression, invalid/missing timings, residency
  enrollment, outstanding retirements, extra profiling and unfinished evidence.
  Two existing SQLite resource warnings were emitted, with no test failures.
- The native source/build is unchanged from the prior **56-test, 45,682-assertion**
  Metal validation run. Native tests were not rerun for this Python-only addition;
  that earlier [verification](../2026-09-10-decode-startup/verification.json) remains
  bound to the same build.
- All eight normal requests matched the previous CLOCK control's output tokens.
  Every follow-up reported the expected reuse and pending-token ingestion. This
  is output parity, not new full-logit/state proof or independent coding quality.
- After completion, source/artifact identity, all four raw conversation reports,
  memory/residency checks, original/copied seals and the decision were rechecked
  offline. Raw files remain unchanged.

Native build:
`968132061556ac0077ff41b7c922b5b7c632bb12e21b8ea8e2c5f82f10bb3c73`.
Mixed artifact: `b2c422f3c643e36f04227a64d61796b44a4b1029`, using unchanged
prepared Q4 expert records. Native sources were committed locally at `dd8229b`.

[Raw summary](raw/summary.json), [derived observations](observations.json),
[Python tests](python-tests.log), [ledger result](ledger-result.json) and
[verification hashes](verification.json) preserve the experiment.

## Next decision

Run five alternating paired normal conversations at the same configuration,
with a declared deadline and paired uncertainty estimates, before longer
qualification. Retain the short-screen and diagnostic results separately.
Continue to report startup variability and memory activity rather than treating
a faster isolated kernel or fewer reads as sufficient evidence.

Core generation remains about **2.5 tokens/s**. The short history here does not
qualify 2K/4K/7K workloads, a 4K-history append, or the 20-minute coding workflow.
Those targets and production promotion remain unproven.
