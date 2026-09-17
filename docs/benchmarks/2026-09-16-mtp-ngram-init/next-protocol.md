# Next bounded verifier trial: direct output for single-row experts

Status: [implemented and measured](../2026-09-16-mtp-direct-output/README.md).
Both short pairs pass, at about 5.04 tokens/s and 3.14% lower aggregate latency.
The three 128-token cases are flat / 2.15% faster / 2.41% faster, all below
5 tokens/s. Full numerical, lifetime and memory checks pass. Keep the candidate
experimental; the protocol below records the original trial and its gates.

The scratch-reuse trial reduces allocation count by 70.37% but misses the 2%
ordinary latency gate. Target verification still costs 187–190ms/token in its
short pair. The audited current-verifier profile has 3,110 reference `q4_mm`
single-row down projections and 1,827 tiled down projections, followed by 4,937
`scatter_experts` dispatches in total. This suggests a smaller change that removes
actual dispatches and copies as well as allocations. Counts do not establish cost.

`encode_expert_rows` currently writes directly into the contribution buffer only
when the entire forward has one token. In four-token verification, a routed expert
can still have just one row (`n == 1`). Its final destination is already known from
`positions`. `Metal::linear` itself calls `linear_into`, so the existing reference
down kernel can write to that destination without creating a down-output buffer
and then scattering it. This changes destination binding, not Q4 arithmetic,
expert selection, row order or the final selected-expert reduction.

1. Use an isolated off/on developer build. Enable only for a four-token target
   decode call whose individual expert has one row. Keep multirow experts,
   priming, shared experts and single-token rejection replay on their existing
   paths. Keep scratch reuse off in both arms so this experiment is independent.
   Share the validated lazy ngram baseline, original reference Q4 kernels, Q8
   selection, cache slots, two GPU groups and 12GiB admission including reserves.
2. Compare complete contribution buffers under Metal validation with real Q4
   records and nonzero destination offsets across all four input rows. Check
   untouched destinations using sentinels, mixed one-/multirow work, delayed
   completion, forced eviction, cancellation and failure draining. Invalid
   destinations must still fail before writing. Then compare full-model logits,
   target/draft state and forced-rejection boundaries against the qualified
   reference. Count exercised direct writes explicitly.
3. Run a fresh sixteen-token normal pair with identical workloads and no detailed
   profiling. Require clean memory/host conditions and at least 2% lower cycle
   latency before a reverse-order pair. Both directions must improve and their
   geometric-mean ratio must be at most 0.98. Record copied bytes, dispatches,
   allocations, submissions and complete-request latency; do not substitute a
   dispatch count for time saved. Preserve the earlier rejected scratch result.
4. Only a survivor gets the three 128-token coding comparisons, then longer
   qualification if warranted. All-proposals-accepted short results cannot
   establish general 5-token/s performance. Acceptance-aware block selection is
   still a separate response to the longer LRU/retry regressions.

This does not reopen packed-Q4 arithmetic, larger caches or expert-tail overlap.
The earlier packed-Q4 diagnostic that tested scatter/direct conditions changed
the Q4 kernel itself; this trial retains reference arithmetic in both arms.
