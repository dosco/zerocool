# Exact cache replay: neither bounded tweak earns a GPU experiment

The [clean capture](capture-02/summary.json) completes both cache sizes and selects
**no cache candidate**. At 1,460 slots, SLRU slightly increases simulated reads.
Increasing CLOCK to 1,536 slots saves only 0.13–0.18%. Both fall short of the
predeclared 10% read-reduction screen. There is no new latency or tokens/s result.

## Native evidence and coverage

Each fresh process captures the complete 72-token prime followed by four blocks
of four fixed continuation tokens: 240 layer passes, 38,134 events and seven
complete cache snapshots. Trace sizes are 4,434,894 and 4,773,196 bytes. Every
selected expert is admitted in actual execution order; pin and release events
retain the real lease lifetime. The maximum live lease count is 32.

The independent replay matches every CLOCK hit/miss, selected slot, victim,
reference bit, pin count and cache snapshot/hash. It reconciles per-layer and
aggregate counters with the native reports. Full vocabulary logits, routes,
persistent state and initial cache identity match the same-capacity untraced
correctness references. No old timing is reused. The [audit](audit-02.json) and
[stricter follow-up audit](audit-02-reviewed.json) pass.

Both processes have zero observed compression, unchanged decompressions/swap,
nominal thermal state, AC power and Low Power Mode off. Peak physical footprint
is 8.522GiB at 1,072 slots and 9.523GiB at 1,460. The 12GiB admission includes
an additional conservative 32MiB trace workspace bound. Capture and offline
analysis finish in 53.60 seconds. These are diagnostics, not performance tests.

| Block offset | Distinct expert records | Live misses, 1,072 slots | Live misses, 1,460 slots |
|---|---:|---:|---:|
| 72 | 1,154 | 1,154 | 1,107 |
| 76 | 1,130 | 1,130 | 901 |
| 80 | 1,283 | 1,283 | 1,031 |
| 84 | 1,370 | 1,370 | 832 |

Every block exceeds the smaller cache. The larger cache retains useful records,
but this does not imply that small additional capacity keeps improving reads.

## Conditional offline comparisons

For each source trace, replay all policies/capacities against the same captured
admission and release order, starting from empty and including all priming work.
The table reports decode misses over sixteen inputs; each miss reads 2,764,800
application bytes. Alternative I/O completion, GPU scheduling and policy-specific
hit-first ordering are **not** simulated. Keep both source orders separate.

| Replay policy | Slots | Misses using 1,072-slot source order | Misses using 1,460-slot source order |
|---|---:|---:|---:|
| CLOCK | 1,072 | 4,937 | 4,937 |
| SLRU | 1,072 | 4,937 | 4,937 |
| CLOCK | 1,460 | 3,881 | 3,871 |
| SLRU | 1,460 | 3,892 | 3,884 |
| CLOCK | 1,536 | 3,876 | 3,864 |
| SLRU | 1,536 | 3,804 | 3,793 |

Against CLOCK at 1,460, SLRU at equal capacity increases misses by 0.28–0.34%.
The extra 76 CLOCK slots save only five or seven reads across all sixteen inputs.
Neither meets the 10% screen in either observed order. The combined policy and
capacity change is reported for completeness, but was not eligible for selection;
its small savings also fall below the screen.

The [lease ablation](lease-ablation.json) shows why exact replay matters at this
scale. Ignoring pins while keeping the larger-cache trace's demand order predicts
3,877 misses rather than the native 3,871. That six-read discrepancy is comparable
to the entire proposed capacity saving. The new replay reproduces the native
count exactly. This does not claim that leases explain the token-latency gap.

## Joint memory planning

The offline capacity filter keeps the existing draft dense/state/scratch allowances
and reserves 128 draft expert slots, then requires another 128MiB of planned
headroom. At 1,460 target slots it leaves 361,759,232 bytes; at 1,536 it leaves
151,323,136 bytes. Both fit this allocation calculation. Neither has been tested
with a real draft forward pass or measured draft-cache traffic. The fully resident
512-expert MTP plan still does not fit beside the larger target cache at 12GiB.

## Implementation and preserved attempts

New developer tools are `build_block_cache_trace.py`, `block_cache_trace.hpp`,
`block_cache_replay.py` and `capture_block_cache.py`. The builder instruments
copies of the existing native verifier and cache; production sources, executable,
model weights and defaults stay unchanged. Original admission and eviction
decisions remain in the native code. Trace write failures reject the capture;
lease destructors retain their noexcept cleanup behavior.

Seven focused tests pass, including agreement with the existing independent
unleased simulators, pinned-reference behavior, SLRU fallback to eligible protected
entries, complete/truncated traces, changed victims/routes/snapshots and selection
gates. The [review](review.json) adds strict integer/type checks and fixes the
exact 10% threshold to use integer arithmetic. Re-auditing the native evidence
produces the same result; no additional GPU run was needed.

The [first capture](capture-01/summary.json) remains resource-blocked at 47.328MiB
process compression after its first process. Its native replay matched, but it
could not select a candidate. After a [native check](retry-headroom.json) showed
reclaimable memory rising from 15.87GiB to 17.38GiB, one fresh unchanged retry
completed cleanly. Attempts remain separate. `screen-sources/` contains captured
tools; `review-sources/` preserves the stricter offline validators.

The native base fingerprint remains
`51877f96f015386dd642fd4e63fbc39ff2e75c1156f5ae1bcba764f546963594`.
The production executable SHA256 remains
`05495d0baf993b1f7f1bc4b9365753e88603909ac66db3309a1380ea8c7c519f`.
Each sealed report binds its separate instrumented executable and source identity.

## Next step

Close these two cache hypotheses without another policy/capacity timing sweep.
Keep CLOCK at 1,460 as the experimental four-token reference. Next measure the
four-token GPU work and its overlap with reads, using a bounded full-block profile
and actual inputs before choosing one compute change. Prior single-token kernel
results do not qualify that workload. Retain the free-proposal 5-token/s gate,
all startup work, exactness, recovery and the same total memory budget. Do not
integrate real MTP drafting until the verifier can justify its added cost.
