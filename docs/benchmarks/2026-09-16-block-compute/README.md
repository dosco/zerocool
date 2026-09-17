# Four-token compute profile: complete, exact and memory-clean

The [third attempt](capture-03/summary.json) completes both profiling modes in
54.43 seconds with exact logits, routes, persistent state and initial cache.
The [independent audit](audit-03.json) passes. Seven focused tests pass. Each
process has zero observed compression, stable decompressions/swap, nominal
thermal state, AC power and Low Power Mode off. Peak physical footprint is
9.523GiB in both modes. Production sources, executable and defaults are unchanged.
This is a completed diagnostic, not a new throughput result.

Follow-up: the [actual-input row-pair screen](../2026-09-16-block-gdn/README.md)
now passes exact validation and measures a 9.73% isolated GPU reduction, but its
2.94ms/token projection misses the 10ms gate. No full-verifier run follows.

The [protocol](protocol.md) keeps CLOCK at 1,460 slots and the same 12GiB budget.
It captures all four blocks using the existing command boundaries, followed by
a separate per-operation counter capture. Each mode validates all vocabulary
logits, routes, persistent state and initial cache against the sealed same-capacity
reference. It records actual per-expert row counts and preserves mixed command
classes. Counter timings cannot be subtracted from normal latency.

## Findings and coverage

Each mode captures all sixteen inputs in four complete blocks: 192 expert
dependency passes and 21,747 GPU dispatches. Routes, operation/shape populations
and per-expert row counts agree. The command capture has 3,542 submissions; the
counter capture has 2,861. Completion-driven grouping can differ with timing:
the scheduler policy is preserved, not an identical realized grouping.

Command capture partitions the observed forward interval as follows. These
exclusive buckets sum to 256.76ms per input token **under instrumentation**:

| Observed interval | ms/input token |
|---|---:|
| GPU active, union of intervals | 160.46 |
| GPU idle with submitted work | 27.64 |
| GPU idle with a ready expert | 20.84 |
| GPU idle with a pending expert read | 22.39 |
| GPU idle during callback completion | 3.00 |
| Other GPU-idle time | 22.44 |

Bucket precedence and overlapping waits are documented in the source. These are
observed concurrency classes, not proof of why the coordinator waited or savings
that can be subtracted from the uninstrumented verifier.

The separate per-operation capture identifies three recurring Q8 GDN matrices
that must finish before the layer's routing/expert reads:

| Input/output dimensions | Captured GPU pass ms/input token |
|---|---:|
| 2560 / 10240, four rows | 19.47 |
| 2560 / 6144, four rows | 10.71 |
| 6144 / 2560, four rows | 11.00 |

They total 41.17ms per input token in this counter capture. The one-row routed
Q4 gate/up operation is individually larger at 20.85ms, but occurs alongside
expert reads and has a different optimization history. The selected next
experiment is [actual-input Q8 GDN row pairing](../2026-09-16-block-gdn/protocol.md):
one existing exact kernel option on these three shapes. No kernel is promoted.
The output head also costs 16.20ms in the counter capture and remains a separate
future hypothesis. Counter capture changes pass structure; its costs and the
command timeline are not one shared execution trace.

## Preserved incomplete attempts

[Capture 01](capture-01/summary.json) has `complete: false`, `status: failed`.
Its exact native error is `expert slots exceed admitted capacity or are below 32`.
The request is the fixed valid 1,460-slot configuration; system headroom reduces
the admitted capacity below that value. The host-only preflight reports
10.449GiB reclaimable, AC power, Low Power Mode off and nominal thermal state.
The model process exits at an 8.28MiB physical peak, with no compression.
No weights were loaded. A later [host-only check](headroom-02.json) reports
9.975GiB reclaimable, so no unchanged model retry was attempted.

The native admission calculation leaves 1.5GiB of currently available memory
outside the engine. A fixed 12GiB experiment therefore needs at least 13.5GiB
available before loading, with additional headroom helpful for clean measurement.
These are time-specific host readings, not the engine's measured inference size.

The original attempt stays sealed with its original `failed` classification.
The runner now recognizes this exact pre-forward error as `resource_blocked`
for future attempts; errors after block execution and unrelated Metal failures
are not reclassified. The [independent audit](audit-01.json) passes and preserves
the attempt's incomplete status. Frozen original tools are in `screen-sources/`;
the narrow offline review is preserved in `review-sources/`.

[Capture 02](capture-02/summary.json) admits after headroom rises to 13.6GiB and
completes the command capture with exact mathematics, but its startup compresses
163.92MiB of the process. It remains resource-blocked and its counter process
does not run. Those timings contribute nothing to the clean capture. After a
[native check](headroom-04.json) shows 18.15GiB reclaimable, one fresh unchanged
retry completes. Its earlier [headroom check](headroom-03.json) and both prior
attempts stay separate. The [capture-02 audit](audit-02.json) passes.

Offline review fixed tuple/list serialization in route summaries. A JSON
round-trip regression check now protects independent revalidation. This changes
no native capture, arithmetic or stored measurements. Pre-fix tools remain in
`capture-sources/`; the corrected tools used by capture 03 are in `initial-sources/`.

## Useful offline finding

Revalidation of both complete prior cache traces gives identical expert row
geometry across all sixteen inputs. The [source-bound calculation](route-geometry.json)
contains 4,937 expert calls and 7,680 token/expert contributions:

| Rows processed by an expert call | Calls | Share |
|---|---:|---:|
| 1 | 3,110 | 62.99% |
| 2 | 1,143 | 23.15% |
| 3 | 452 | 9.16% |
| 4 | 232 | 4.70% |

Block verification does not turn all expert work into four-row matrix operations.
The current kernel selector already chooses a one-token tile for single-row
experts; this is not evidence of a missing tile clamp. The profile must distinguish
these shapes, including gathered gate/up inputs, before selecting an optimization.
This finding gives no kernel cost or speed prediction. Previous single-token
packed-Q4 request regressions remain valid for those experiments.

## Next experiment

The selected row-pair experiment is complete; see its linked report. Next test
packed Q8 word loading for four-token execution on its verified inputs, retaining
one output row and exact accumulation order. Only material isolated benefit
advances to a fresh verifier comparison. Preserve all startup blocks and checkpoints.
The existing 4.07 tokens/s serial observation and 4.67 optimistic verifier
observation remain the latest clean short timing evidence.
