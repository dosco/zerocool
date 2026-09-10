# Resumed selector qualification

The disk and native memory checks admitted this run with the original frozen
identity and fixed 12GiB engine budget. **The run stopped during full-state
qualification; no normal-request speedup is qualified.**

Completed so far:

- All 50 native tests / 6,888 assertions passed with Metal API and shader
  validation; all 83 Python tests passed.
- Both real-model cached-replay cancellation cases passed at exactly 1 and 97
  completed layer records. Settings were restored, GPU users drained, and the
  same model successfully ran again. [Recovery report](recovery/report.json).
- All eight fresh 4K, retained-append and 7K attention cases replayed exactly.
  Their original capture payload totals 181,934,336 bytes; the
  [sealed inventory](capture-evidence-files.json) records every hash.
- The retained append reused 4,096 tokens of computation and ingested exactly
  128 new tokens. The 7K capture peaked at 10.403GiB of Metal buffers and
  10.394GiB process footprint, below the 12GiB budget.

The original-reference boundary harness passed its nine checks, including fresh
replay and failure/cancellation recovery. The candidate process failed a later
model allocation: 8,774,254,592 bytes were required, but only 1,417,199,616 bytes
were admitted. It did not produce a completed candidate report. See the
[reference report](qualify/boundary/reference.json),
[candidate log](qualify/boundary/candidate.log) and
[outer log](qualify/boundary.log).

The outer runner incorrectly reported this nested resource-admission exception
as a generic failure. The original [summary](summary.json) preserves that result
unchanged; the [checkpoint](checkpoint.json) records the copied evidence. Raw
reports and capture payloads remain in
`/repo/.cache/benchmarks/selector-qualification/2026-09-09-run-02`.
The recovery directory preserves its original filenames and sealed inventory.
Captured tensor payloads remain in the source directory rather than being
duplicated in Git documentation.

Investigation reproduced physical GPU-buffer retention after C++ ownership and
allocation accounting had returned to zero. Scoping autorelease cleanup around
Metal command submission fixes the bounded reproduction. The new regression
failed before the fix and passed after it; all 51 native tests / 6,931 assertions
and 84 Python tests pass. Nested resource-blocked classification now also has a
regression test. The [cleanup report](../submit-cleanup/README.md) records the
scope and limitations. The changed build must start a fresh qualification; this
run's evidence cannot qualify it.

Full-model state comparisons, normal timing, and final profiling must finish
before this stage is complete. Recovery and attention replay are correctness
results, not normal-request latency or product acceptance.
