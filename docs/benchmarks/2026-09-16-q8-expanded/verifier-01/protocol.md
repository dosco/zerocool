# Expanded four-token packed-Q8 experiment

Test one expanded scope of the existing word-load design: the output vocabulary
projection, attention query/output projections and the three GDN projections.
Keep one output row, four token accumulators, width-eight lanes, scalar addition
order, affine Q8/64 semantics, final SIMD reduction and BF16 rounding. Scope and
per-block frequencies are fixed in `q8_expanded_contract.py`; verify those counts
against the complete clean four-block native counter profile before timing.
Do not combine row pairing, change precision or introduce other kernel variants.

Capture actual first-block inputs after the fixed 72-token prime, using layer 0
for GDN, layer 3 for attention and the final head. Capture at most six inputs and
1MiB total input payload. Hash already loaded weights, scales and biases; save
no weight copies. Match all captured weight hashes to selected ranges of the
pinned mixed checkpoint, including dtype/dimensions, metadata hashes and its
existing verification receipt. Replay loads only those ranges, one case at a
time. The wide output head requires a separately declared 1GiB Metal-buffer and
2GiB physical-footprint operator allowance. This does not change the full-model
12GiB budget, 1,460 expert slots, 512-token panel or 8,192-token context limit.

The capture must reproduce full logits, routes, persistent state and initial
cache identity against the existing fixed-token control. Capture timing is never
performance evidence. If complete numerical capture is memory-disturbed, preserve
its blocked status; a separately recorded operator process may reuse only the
verified bytes, after revalidating the entire numerical and payload proof. No
timing or memory observations carry over. Incomplete numerical captures cannot
be used. Bound the capture process to 150 seconds and its stage to 210 seconds.

Use fresh native AC-power/thermal preflight, Low Power Mode off, and an exclusive
GPU lease. Run Metal API/shader validation, then five alternating pairs of 32
dispatches per arm and case, with one excluded warmup for each arm. Run the wide
head first; measure all six cases together in the same process, with original
four-token kernels as control. Record native dispatch counts and compare every
output bit with the separate scalar reference. Validation/timing reference
hashes must agree. Require zero observed process compression, stable system
decompression/swap counters and the stated allocation bounds. Stop on disturbed
operator measurements; keep those attempts separate. Bound each operator process
to 90 seconds and the operator stage to 420 seconds.

Advance only if the median frequency-weighted projection is at least 10ms per
input token and the summed paired GPU ratio upper 95% bound is below 1. Weights
are 1 head, 12 of each attention projection and 36 of each GDN projection per
four-token block. This projection is a screening heuristic, not predicted request
latency. Do not pool previous GDN timings or infer gains for untested shapes.

A survivor advances to a fresh full-verifier comparison at the unchanged 12GiB
budget. Validate all logits, routes, persistent state, causal prefixes, rollback
and replay with native Metal validation before normal timing. Compare unchanged
and expanded four-token paths in two alternating fresh-process rounds, keeping
16 continuation inputs and identical deterministic priming. A clean new timing
result must improve both paired request times and every candidate run must
reach 5 verified tokens/s. Stop early if a candidate misses that absolute floor.
Preserve all startup work, cache identity and resource controls. No result
establishes production promotion, long-context qualification, real draft cost,
acceptance rate or speculative generation throughput.
