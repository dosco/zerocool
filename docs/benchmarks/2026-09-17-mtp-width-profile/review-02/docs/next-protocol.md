# Next screen: share gate/up input work for multi-row routed experts

## Why this candidate

The clean current-width interval-merging counter capture records 2,124
`q4_gate_up_t4` calls: 1,175 with two input rows, 566 with three, and 383 with
four. Their summed instrumented intervals are 27.48ms per committed output.
This ranks a scope; overlapping counter intervals and instrumentation prevent
using it as an attainable saving. GDN scan is only 2.12ms in the same ranking.
The remaining generic Q8 hyper projections sum to 14.50ms, a smaller scope.

`affine_multi` in the existing four-token gate/up kernel processes gate and up
separately. Both traverse the same gathered inputs and calculate the same
four-value BF16-rounded affine bias sums. Test sharing those input loads and
bias sums while keeping separate gate/up dot products and accumulators.

This changes neither quantization nor routing. Preserve each projection's lane
partition, scalar addition order, SIMD reduction, BF16 boundaries and activation.
Retain one output channel per SIMD group. The earlier single-token packed-Q4
output-row pairing result remains rejected; do not combine or repeat it here.
Single-row experts, down projections, reduction and scheduling stay unchanged.

## Cheap rejection before runtime integration

1. Build one isolated developer shader/probe. Reuse only independently rehashed
   original expert records and saved real input bytes; no historical timings.
   Cover two, three and four rows, contiguous and nontrivial gathered orders,
   the unchanged scalar reference and existing tile-four path. Require exact
   hidden outputs, intact input/destination guards and Metal API/shader validation.
2. Admit at most 256MiB of operator buffers. Measure all three row counts in
   five alternating pairs, with equal explicit warmups and fixed repetitions.
   Bind source/compiler/binary, prepared records, input hashes and host counters.
3. Weight each row count using the recorded calls per 16 committed outputs
   (1175/16, 566/16, 383/16). This is a screening projection for this window,
   not measured request savings or a general model routing distribution.
   Require at least 10ms median projected GPU saving and a paired upper 95%
   ratio below one. Keep the temporal-dependence limitation of five pairs in
   one process explicit. Reject weak or inconclusive candidates immediately.

## Only a survivor receives request work

Add the selector only to an isolated current-width producer, limited to the
measured multi-row routed shapes. Retain the 12GiB total, 1460/32 expert slots,
reference Q4, packed resident Q8, full-replay recovery, direct output and lazy
ngram controls. Recheck full-model logits, rejection and continued state, then
run one fresh normal interval-merging pair. Stop if latency reduction is below
2% or any exactness/resource check fails. Reverse order and additional ordinary
coding workloads follow only a survivor; five-pair/long-context/sustained
qualification remains a later gate.

No production default changes or performance claims follow from profiling.
The LRU capture remains resource-blocked and contributes no timing to this
decision. Another unchanged LRU capture is not a prerequisite for the small
operator screen and should not be retried solely because the other workload
completed cleanly.
