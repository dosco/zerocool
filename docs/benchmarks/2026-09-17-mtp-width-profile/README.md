# Current-width target profiling

The interval-merging workload now has **clean command and per-dispatch captures**,
with exact full-request logits, tokens and persistent state. GPU execution is
the largest exclusive overlap bucket in the command capture. The per-dispatch
capture identifies multi-row routed gate/up work as a useful next operator
experiment. Neither capture measures a normal-request speed improvement.

The LRU retry passed admission and finished exact numerical capture, but 4.0625MiB
of process compression keeps it resource-blocked. Its timing is excluded from
optimization selection. Both the original admission stop and this disturbed
capture remain unchanged.

## Scope

The [protocol](../2026-09-17-mtp-widths/next-protocol.md) selects width-one LRU
and width-four interval merging from the completed fixed-width screens. The
new isolated C++/Metal producer retains full-replay recovery, direct output,
lazy ngram initialization, reference Q4, packed Q8, scratch-off controls,
1,460 target expert slots, 32 draft slots and the same checkpoint capacity.
Its 512MiB trace allowance is admitted inside the unchanged 12GiB total.
Production sources, defaults and executable are unchanged.

Each capture executes the complete original 128-token request. Profiling begins
at the first cycle boundary at or after 32 committed output tokens and covers
at least 16 committed tokens, retaining the final whole cycle. Reports separate
committed output tokens, verified input rows and omitted positions. Full-request
logits, token IDs, proposal/rejection decisions and final target/draft state must
match the sealed numerical reference. Historical timing is never reused.

Only target verification is profiled. Draft, checkpoint, recovery and prefix
work remain outside the window. Existing submission boundaries are preserved;
profiling adds no per-kernel GPU waits. Every captured call must reconcile all
48 layers, selected-expert records, GPU dispatches and command ownership.
The native metadata cap is 20,000 operations per call, with at most 32MiB of
serialized data. Truncation, unexpected outside work and partial coverage fail.

The analysis now counts actual expert down projections as well as scatter
operations. Four-token direct output legitimately removes single-row scatters;
requiring one scatter per expert would incorrectly reject the current engine.
The old four-token analyzer retains its existing default contract.

## Completed captures

[Interval command capture](merge-01/summary.json) runs the original 128-token
request and traces output positions **33–48**: five complete target calls,
20 verified rows, 16 committed outputs and 240 layer passes. All **23,087
dispatches** and 4,322 command groups reconcile. The process peaks at
**9.7436GiB**, with zero process compression/decompression, unchanged swap,
AC power, Low Power Mode off and nominal thermals.

Exclusive observed intervals per committed output in this instrumented window:

| Interval | ms/output |
|---|---:|
| GPU active | 120.60 |
| GPU idle with required expert reads pending | 34.04 |
| Other GPU idle | 29.11 |
| GPU idle after command submission | 26.39 |
| GPU idle with an expert ready | 17.24 |
| GPU idle awaiting a completion callback | 3.82 |

These partition the traced target interval; they do not identify removable
latency. Drafting, checkpointing and recovery are outside target profiling.

The [separately declared counter capture](counter-protocol.md),
[raw stage](merge-counters-01/summary.json), retains the same complete request,
coverage and 23,087 dispatches. It also passes exactness and all memory/host
checks, peaking at **9.7462GiB**. Counter sampling splits compute passes; readiness
and grouping naturally differ (3,980 commands). Its timings are not compared
as a speedup against the command capture.

| Selected per-dispatch scope | Summed instrumented ms/committed output |
|---|---:|
| Routed Q4 gate/up, 2–4 token rows | 27.48 |
| Routed Q4 gate/up, one token row | 15.24 |
| Remaining generic Q8 hyper projections, both shapes and all stages | 14.50 |
| GDN recurrent scan | 2.12 |

There are **2,606 overlapping adjacent counter intervals**. Their summed
durations rank experiments; they are not exclusive costs, predicted savings
or quantities to add to the command buckets. The recurrent scan does not
explain the largest resident command groups. The two generic Q8 hyper shapes
are smaller than the multi-row gate/up scope; defer a full-model Q8 extension
while the [next bounded expert experiment](next-protocol.md) is screened.

## Verification and preserved resource stops

- The profiler builds as a separate executable with SHA256
  `94e6d4712b6a632fe697ca16a37c8a29eeed73026f7f8753c8eaaff0e7088bc3`.
- All ten clean numerical prerequisite reports were independently rechecked.
  They establish the parent width implementation, not a completed profiler run.
- **515 Python tests pass.** Coverage includes whole-cycle window boundaries, partial acceptance, direct-output
  geometry, missing operations, early buffer release, altered outputs and
  admission, incomplete-stage query handling, explicit counter identity, complete
  dispatch counters, and charging rejected verified rows to committed outputs.
- [The first capture attempt](lru-01/summary.json) reports `resource_blocked`.
  Its host probe measured **5.743GiB available**, below the existing **13.5GiB**
  minimum. AC power, Low Power Mode and thermal checks passed. No inference ran.
  A later read-only refresh measured 6.104GiB; it is not a new capture.
- The [LRU retry](lru-02/summary.json) passed admission with **16.009GiB available**.
  It completed 128 exact outputs and captured all 768 layer passes in its
  16-call window, peaking at 9.7421GiB. Compression first appears after output
  20, before profiling begins at output 32. Its 4.0625MiB compression peak and
  54 lifetime decompressions fail the unchanged resource gate, despite unchanged
  swap and clean power/thermals. No cause or speed conclusion follows.

The [first sealed review](review-01/summary.json) retains the earlier implementation
and admission stop. The [capture review](review-02/summary.json) binds the new raw
stages, recomputed queries, source copies, tests and unchanged native identity.
Production sources, executable and defaults remain unchanged.

## Developer tools

```sh
PYTHONDONTWRITEBYTECODE=1 .cache/qwen-reference-venv/bin/python \
  scripts/qwen/build_mtp_width_profile.py --output .cache/NEW-PROFILE-BUILD

PYTHONDONTWRITEBYTECODE=1 .cache/qwen-reference-venv/bin/python \
  scripts/qwen/capture_mtp_width_profile.py --build .cache/NEW-PROFILE-BUILD \
  --case lru_cache --output docs/benchmarks/NEW-LRU-CAPTURE
```

Use `--case merge_intervals` for the second bounded capture. Each output directory
must be new. The runner uses the existing exclusive GPU lease, frozen source and
artifact identities, phase progress, deadlines, and host/memory gates. A later
attempt requires a changed resource condition, not an unchanged retry loop.

For the separately declared kernel-timing diagnostic, build with
`build_mtp_width_counters.py --output NEW_BUILD` and capture with
`capture_mtp_width_profile.py --mode dispatch --build NEW_BUILD
--case merge_intervals --output NEW_CAPTURE`. The mode is explicit in native
reports, producer receipts, stage summaries and offline validation; command
captures cannot be relabeled as counter captures.

Import a completed or blocked stage through `query_evidence.py import DIRECTORY`,
then use `query_evidence.py next DIRECTORY/summary.json`. Completed stages are
revalidated from raw profiles before returning compact coverage, overlap buckets
and command classes. Incomplete or disturbed stages select no optimization.
`timeline` can inspect individual captured profile files by layer and absolute
input-token position; those positions can include rejected verification rows.

An instrumented bucket measures observed overlap. It does not establish a
causal stall, recoverable latency, normal throughput, or a production decision.
Choose the next intervention only after a complete clean trace, then test it
with fresh ordinary-request timing.
