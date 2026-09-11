# Packed Q8 improves generation on the original schedule

The existing packed-Q8 two-row kernel passed the short request screen on the
current build. Two alternating pairs reduced complete conversation time by
**10.7% and 9.2%**, with a median reduction of **10.0%**. No other kernel option,
cache capacity, precision, or scheduling policy changed. Defaults remain unchanged.

| Measured request | Original generation, median | Packed Q8 generation, median | Original request, median | Packed Q8 request, median |
|---|---:|---:|---:|---:|
|72-token initial prompt + 33 outputs|2.55 tok/s|3.50 tok/s|26.49s|22.95s|
|128-token retained append + 33 outputs|2.48 tok/s|3.20 tok/s|36.57s|33.82s|

Generation improved by 27–42% across the individual paired requests. These
rates measure the 32 decode steps after the first output. First-token times
remained approximately 13.8–14.0 seconds initially and 23.6–24.0 seconds after
the append; this change does not resolve ingestion latency.

| Pair | Order | Original conversation | Packed Q8 conversation | Candidate/control |
|---|---|---:|---:|---:|
|0|Original, packed|63.327s|56.530s|0.892665|
|1|Packed, original|62.790s|57.005s|0.907867|

Both conversation ratios and all secondary medians passed the predeclared
screen. Two pairs provide no confidence bound. The next experiment is five
fresh alternating pairs of this same configuration, without pooling these
screening runs. Production qualification still requires longer prompts,
7K-context reporting, sustained memory checks, and a real coding workflow.
The 5 tok/s target remains unmet by these measurements.

## Fixed execution and exactness

Both arms used the actual 32GiB M1 Pro, 12GiB engine ceiling, 1848 CLOCK expert
slots, residency off, serial prefill, fixed phase memory, reference expert
execution, panel 512, chunk 128, four ready experts, eight I/O workers, context
limit 8192, and the pinned mixed 4/8-bit artifact with unchanged Q4 records.
The candidate enables `kernel_policy=candidate` and `q8_decode_rows=2`; other
kernel options retain their original values. This does not quantize any weights.

Normal requests excluded Metal validation, profiling, captures and GPU boundary
probes. Both follow-ups reused 104 computed tokens and ingested 129 tokens,
including the pending previous output. All generated token sequences matched.
After folding packed Q8 dispatch counts into the replaced Q8 operation, every
other kernel dispatch count matched across all four conversations. Total expert
application reads were 77.9274GiB per conversation in every arm; tiny differences
in phase allocation cancelled over the conversation. These are application
reads, not physical device traffic.

All admitted memory-plan categories matched. Observed lifecycle footprints
ranged from 5.11GiB to 10.44GiB, within the 12GiB ceiling. These endpoint samples
do not establish continuous memory peaks or sustained swap behavior.

Short all-48-layer original/candidate checks with 32 expert slots forced eviction
and preserved logits, expert routes, and persistent state exactly. Continued
state matched fresh replay; cancellation and injected failure checks passed
under Metal API/shader validation. These checks compare native implementations;
they are not a new independent model oracle or a coding-quality evaluation.

**57 native tests / 45,704 assertions** and **188 Python tests** passed. The
post-run evidence review also tightened exact panel admission and normalized
dispatch-count checks. The original reports pass these stricter checks without
changes. The harness used during measurement is preserved in
[harness-snapshots](harness-snapshots); the native build did not change.

## Early screening and the noisy pilot

The [five-pair pilot](pilot/summary.json) stopped after **35.70s**. Every captured
output was exact and five projections passed, but the smallest projection's
CPU wall-clock ratio had a 95% upper bound of 1.236 despite lower GPU time in all
five pairs. No state check or request timing ran in that pilot.

One separately declared follow-up used fresh captures and 20 operator pairs,
with unchanged candidate, deadlines, and per-case criterion. It completed in
**318.92s**, including operator capture/replay, state checks, and four normal
conversations. All six projections passed:

| Capture | Matrix K×N | Median wall ratio | Bootstrap 95% upper ratio |
|---|---|---:|---:|
|Layer 0 GDN|2560×10240|0.363|0.378|
|Layer 0 GDN|2560×6144|0.448|0.454|
|Layer 0 GDN|6144×2560|0.390|0.399|
|Layer 31 attention|2560×12288|0.348|0.359|
|Layer 31 attention|2560×512|0.905|0.918|
|Layer 31 attention|6144×2560|0.399|0.403|

Operator timing uses the previous two-row kernel as control; request timing
uses the original reference kernel. Both operator variants are checked bitwise
against that original reference. Operator intervals are screening statistics
from within-process repetitions, not independent request confirmation. Neither
pilot samples nor historical bundled-kernel timings were pooled into the result.

## Reproducible evidence

Native build: `1ddec378e0a2550eda51c351e5e9315863c15b61773617c457b4d8e2f8df9189`.
Mixed artifact: `b2c422f3c643e36f04227a64d61796b44a4b1029`.

[Protocol](../../qwen_q8_steady_stage.md), [raw summary](raw/summary.json),
[request observations](observations.json), [revalidated comparison](comparison.json),
[native tests](native-tests.log), [Python tests](python-tests.log).

The offline query reopens all eight original correctness/operator/request
reports by hash and reconstructs the same decision. Metadata and logs are
archived here with fresh archive inventories. The original full inventories
are retained as `original-full-seal.json`. Binary captures remain in the two
local `.cache/benchmarks/q8-steady*20260911` directories, listed with hashes in
each archive's `local-payloads.json`; they are not duplicated in git. Both full
local seals verified before archival. Historical harness snapshots preserve
the stopped pilot and the measured follow-up exactly.
