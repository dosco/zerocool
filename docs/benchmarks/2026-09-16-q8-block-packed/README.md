# Packed Q8 loads for four-token execution

The isolated candidate cuts the summed GPU time for three recurring GDN shapes
by **24.49%**, preserving every output bit. Its median shape-frequency projection
is **7.56ms per input token**, below the predeclared 10ms advancement threshold.
The screen is complete; no full-verifier timing or production promotion follows.
The previous 4.065 serial / 4.6655 free-perfect-proposal verifier measurements
remain unchanged. This experiment does not measure real speculative decoding.

## Measurement

[The protocol](protocol.md) changes byte loads to aligned 32-bit weight loads,
keeping four token accumulators, one output row, the width-eight lane partition,
scalar accumulation order, SIMD reduction and BF16 rounding. It does not combine
the earlier output-row-pair experiment. Production sources and defaults are
unchanged; the developer binary appends one shader to a copied Metal backend.

| Input / output dimensions | Control GPU ms / four rows | Packed GPU ms / four rows |
|---|---:|---:|
| 2,560 / 10,240 | 1.6046 | 1.1420 |
| 2,560 / 6,144 | 0.8311 | 0.6801 |
| 6,144 / 2,560 | 0.9606 | 0.7423 |

These are means over five alternating pairs of 32 dispatches per arm and shape,
after one explicit warmup per arm. The summed paired GPU ratio is **0.75507**,
with paired 95% interval **0.74386–0.76645**. Projected savings across the five
pairs are 7.9171, 6.9308, 7.4528, 7.5601 and 7.5781ms per input token, assuming
36 calls of each shape per four-token block. This is an isolated screening
projection, not a measured request latency improvement. Temporal dependence
within one process limits the paired confidence interval.

## Correctness and resource checks

The [completed screen](screen-01/summary.json) took 4.89 seconds. Fresh Metal API
and shader validation, all warmups and all timing outputs match the unchanged
scalar reference bit for bit. Native dispatch counts confirm exactly the intended
control/candidate kernel. Reference hashes also match between fresh validation
and timing processes. Eight focused tests and the [independent evidence
recomputation](audit-01.json) pass.

Only verified input bytes are reused from the earlier memory-disturbed capture;
its timings, memory samples and incomplete status are not reused. All nine
weight/scale/bias payloads are rehashed against the pinned mixed checkpoint;
source capture logits, routes, persistent state and initial cache identity are
revalidated. The new processes have clean native power/thermal checks, zero
observed process compression and unchanged decompression/swap counters.

Peak Metal allocation is 26.92MiB; peak physical footprint is 95.33MiB during
validation and 49.25MiB during timing. The production build fingerprint remains
`51877f96f015386dd642fd4e63fbc39ff2e75c1156f5ae1bcba764f546963594`.

## Next experiment

Keep this implementation as a measured component, while preserving its failed
advancement decision. The next useful extension is the same packed-load design
for attention projections and the final vocabulary projection, with independently
verified actual inputs and a new declared scope. The clean counter profile shows
the vocabulary projection at 16.20ms/token and the largest attention projection
at 7.40ms/token; these are instrumented costs, not predicted savings. Do not
transfer the GDN speedup to those matrices without measurement.

Test the wide vocabulary projection first with an explicit bounded fixture budget
(its packed weights alone exceed the present 256MiB probe allowance). If the
expanded scope merits a verifier test, remeasure all selected shapes together
in fresh alternating pairs; do not add gains from separate experiments. Keep
the existing 12GiB full-model budget, 1,460 slots, exactness/recovery checks and
normal-request performance gate. Do not repeat the unchanged GDN-only screen.
