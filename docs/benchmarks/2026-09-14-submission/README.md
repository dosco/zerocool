# GPU submission and expert residency on the 32GiB M1 Pro

Two changes failed the short speed gate: removing Metal's duplicate resource
ownership and coalescing the remaining expert reads. Keeping immediate expert
execution and requesting residency for the core plus expert cache passed a
two-pair screen and full-model state validation. Five fresh pairs measured
**4.35% lower complete-conversation latency**, with a paired 95% interval of
**3.21–5.48% lower latency**. The append saving misses the stricter 20ms/token
stage gate. No production default or 5 tokens/s qualification is claimed.

All request comparisons use the pinned mixed 4/8-bit artifact, unchanged Q4
expert/ngram records, 12GiB maximum memory, 8192 context capacity, scratch reuse,
packed Q8 rows 2, SIMD routing, CLOCK, eight readers, panel 512 and microchunks
128. The workload has 72 initial tokens, 33 outputs, a 128-token append and 33
more outputs. Each comparison alternates control/candidate, then
candidate/control. No pairs from different comparisons are pooled.

## Driver scheduling capture

The [new normal/traced capture](profile-01/summary.json) adds commit-return and
Metal CPU scheduling timestamps to existing command groups. It does not add
callbacks, GPU submissions or waits. These timestamps are distinct from GPU
execution timestamps. [Apple's timestamp documentation](https://developer.apple.com/documentation/metal/mtlcommandbuffer/kernelstarttime)
defines the driver interval as CPU scheduling work.

The [reconstructed trace](driver-analysis.json) covers all sixteen decode
forwards per request and joins every layer and selected expert. The first three
rows partition GPU-idle time with submitted work; the commit rows overlap them.

| Interval | Initial ms/token | Append ms/token |
|---|---:|---:|
| Before driver scheduling | 4.00 | 4.27 |
| Within driver scheduling | 61.00 | 59.20 |
| After driver scheduling, before GPU start | 24.55 | 25.08 |
| Commit API overlap with submitted idle time | 1.19 | 1.29 |
| Total CPU commit API duration | 1.66 | 1.75 |

Both 1460-slot processes encountered compression: approximately 5.05GiB in
the normal process and 1.14GiB in the traced process. These observations cannot
qualify speed or establish how much driver work is removable. In particular,
the earlier clean trace's roughly 76ms submitted interval must not be replaced
or adjusted using this disturbed capture.

Subsequent paired comparisons use 1072 slots in both arms, approximately
10GiB planned memory within the same 12GiB maximum. They stop on observed
compression or decompression. This is an experiment allocation, not a new
production cache default.

## Duplicate command ownership: insufficient benefit

FreeLLM already holds every bound buffer and pipeline until its GPU users
finish. A developer-only API tests command buffers without Metal's additional
retained references. It cannot change ownership mode while work is live;
ordinary execution retains its existing Metal references.

The [64MiB probe](ownership-01/summary.json) uses eight real Q4 expert records
and captured inputs, unchanged gate/up/down kernels, and equal preallocated
outputs. It tests one, two and four experts per command group. Hash checks and
warmup occur outside measured intervals; Metal validation runs separately.
This is resident operator work, with no SSD, router, scatter or full-request
speed claim.

The primary one-expert result across five alternating pairs has geometric
candidate/control ratio **0.99726**, with 95% interval **0.98526–1.00941**.
All probe arms were free of compression and outputs matched. Even optimistic
scaling of the median group saving gives only 0.13ms/token. The candidate does
not advance to a request screen.

## Coalesced decode: fewer submissions, slower requests

The benchmark-only `--decode-submission coalesced` experiment submits initial
ready/shared work, then collects all remaining selected experts into one more
command group. It keeps original expert kernels and destination/reduction
order. All ten leases fit the admitted window, and the ordinary cancellation
and resource-release rules still apply. Multi-token schedules are unchanged.

The [two-pair screen](coalesced-01/summary.json) is clean in all four processes,
with exact output tokens and unchanged arithmetic dispatch counts.

| Measurement, mean across two repetitions | Immediate initial | Coalesced initial | Immediate append | Coalesced append |
|---|---:|---:|---:|---:|
| Decode ms/token | 292.05 | 309.38 | 333.29 | 363.07 |
| Submissions/token | 330.45 | 144.28 | 348.58 | 144.69 |
| Expert application reads, MiB/token | 643.32 | 643.28 | 736.47 | 736.47 |
| Mean expert-ready-to-encode delay, ms/expert | 0.074 | 0.328 | 0.080 | 0.418 |

Decode is approximately **5.9% slower initially and 8.9% slower after the
append**. Complete-conversation ratios are 1.02540 and 1.02574. Grouping reduces
CPU encoding and summed GPU command durations but delays ready expert work;
those overlapping metrics cannot be added into a predicted request saving.
This candidate stops before expensive full-model state or five-pair validation.
Immediate execution remains selected for subsequent work.

## Core plus expert residency

The earlier [five-pair residency experiment](../2026-09-10-residency-paired/raw/summary.json)
compared off versus core-only residency. A separate September 8
[single core-plus-cache observation](../2026-09-08-residency/README.md)
suggested a gain but was not a paired qualification. The driver measurements
justify testing that existing option while preserving immediate expert work.

The [fresh two-pair screen](residency-01/summary.json) changes only
`--residency off` to `--residency core-cache`. Both arms use 1072 slots. Every
candidate enrolls the complete expert cache, all four processes are clean,
and outputs and arithmetic dispatch counts match.

| Pair | Initial saving, ms/token | Append saving, ms/token | Conversation candidate/control |
|---|---:|---:|---:|
| Control then candidate | 51.47 | 48.60 | 0.92910 |
| Candidate then control | 9.91 | 15.27 | 0.96166 |

Median savings are **30.69/31.93ms per token**. Variation is material; these
two pairs establish a promising screen, not a confidence-based gain.

The [state qualification](residency-state-01/summary.json) passed separate
Metal API/shader validation for both arms. All 48 layers have identical logits,
routes, and recurrent/attention state at three checkpoints. Continued sessions
match fresh replay. Forced eviction at 32 slots, cancellation, failure and
recovery checks passed. This compares the unchanged native arithmetic across
residency settings; it is not a new independent framework/model-quality check.

## Verification and scope

[71 native tests / 49,117 assertions](native-tests.log) passed with Metal API
and shader validation, none skipped. Source snapshots preserve the original
stage, timestamp diagnostic, ownership probe and paired runtime revisions.
Raw seals preserve failed or disturbed attempts without relabeling them.

The benchmark comparisons do not exercise 2K/4K/7K acceptance contexts or a
20-minute retained coding conversation. Exact arithmetic is unchanged, and
there is no new quantized artifact. Production defaults remain unchanged.

## Five fresh pairs: a measured gain below the stage target

The [confirmation](residency-confirm-01/summary.json) completed all ten fresh
processes, with no screen samples pooled and no early success stopping. Every
recorded phase was free of compression and had unchanged decompression
counters. Outputs matched the source control and arithmetic dispatch counts
were identical. All five full-conversation candidate/control ratios were
below one.

| Median across five repetitions | Residency off | Core plus expert cache |
|---|---:|---:|
| Initial generation, tokens/s | 3.566 | 3.796 |
| Retained-append generation, tokens/s | 3.188 | 3.309 |
| Initial TTFT, seconds | 14.363 | 13.828 |
| Retained-append TTFT, seconds | 24.873 | 24.008 |

The paired geometric conversation ratio is **0.95645**, with 95% interval
**0.94518–0.96786**. Initial and append decode ratios are 0.92977 and 0.95376;
their separate marginal 95% intervals are 0.88547–0.97630 and 0.91950–0.98930.
These intervals use Student-t over five paired log ratios; correlated host
effects may violate their assumptions. Separate medians in the table are not
the paired effect estimates.

Median paired savings are **22.59ms/token initially and 11.24ms/token after the
append**. Both phases needed at least 20ms to pass the declared stage target,
so automatic advancement remains false. The first collector labeled any
failure of that gate `improvement_not_demonstrated`, which conflated effect
size with statistical evidence. The [corrected interpretation](confirmation-decision.json)
records `gain_below_stage_target`, `request_gain_demonstrated: true`, and
`stage_target_met: false`. Original reports, numerical results and advancement
criteria are unchanged.

The [independent reconstruction](verification.json) verifies every raw seal,
reconstructs native source identities, rechecks output/configuration/dispatch
parity, validates the state proof, and recomputes the paired decisions. It also
preserves the disturbed driver capture and both rejected experiments. The
[247 Python tests](python-tests.log) cover overlap accounting, missing evidence,
pair requirements and the distinction between a gain and a stage target.

```sh
.cache/qwen-reference-venv/bin/python scripts/qwen/verify_submission.py \
  docs/benchmarks/2026-09-14-submission --output /tmp/submission-audit.json
.cache/qwen-reference-venv/bin/python scripts/qwen/verify_stage200.py \
  docs/benchmarks/2026-09-13-stage200 \
  --source-snapshot docs/benchmarks/2026-09-14-submission/pre-change-sources \
  --output /tmp/stage200-audit.json
```

Core-plus-expert residency remains an explicit option. The next useful memory
experiment is 1072 versus 1460 expert slots with that same residency setting,
immediate execution, and a fixed 12GiB maximum. Admit the larger allocation
before inference and stop on compression or swap disturbance. This tests a
new condition; it does not erase the earlier failures with residency off.
Long-context, 5 tokens/s and sustained coding acceptance remain open.

The agent's experiment ledger records the rejected
[ownership](../../experiments/242273a01dfcec2af0f7b5cff302be2335e6ab597517be03793762cceaef16db.json)
and [coalescing](../../experiments/8556caf1c8573c4822f3093a73bf6fcbd0b26dc9a52b6162c32227b2dd7c6c62.json)
hypotheses and the promising but below-target
[residency result](../../experiments/5977e9dc6a408b015a18dc5de76d162176fb475c48c697538e7452abe3a5372d.json).
Raw evidence remains authoritative; the SQLite index is rebuildable.
