# Bounded allocation investigation

The allocation investigation identifies live CPU owners but does not establish a
fix for intermittent compression. It is closed for this iteration; do not retry
the unchanged recovery qualification or repeatedly tune pipeline preparation.

## Evidence

[Allocation diagnostic 03](allocation-diagnostic-03/summary.json) reproduced
compression with stack logging enabled. Its first heap filter selected only
untyped allocations, so it did not identify the typed owners.

[Diagnostic 04](allocation-diagnostic-04/summary.json) corrected the filter to
include typed allocations of at least 32KiB. Its owned native child was paused
for VM mapping, heap enumeration and allocation stack collection. This changes
execution and can touch compressed pages: none of its timing qualifies.

The retained allocation sizes include:

| CPU owner | Allocated bytes / MiB | Interpretation |
|---|---:|---|
| Target rollback buffers | 112.875MiB | Required checkpoint capacity, including 36 recurrent snapshots; not a new leak |
| Ngram table bookkeeping and capacity | 50.734MiB | Allocation capacity, not physical residency; entries are lazily constructed |
| Large CFString and C++ string allocations | 21.875MiB | Allocation stacks include Metal pipeline preparation and driver metadata |

These are lower bounds by owner from the >=32KiB filter, not an exhaustive heap
breakdown. The observed VM regions and allocation stacks do not map every
compressed page to a particular object. Diagnostic 04's VM snapshot was clean,
but its full process later recorded 50.781MiB peak compression. It cannot prove
that the shader strings were the pages compressed in another run.

## Tested intervention

The isolated [pipeline preparation comparison](prewarm-01/summary.json) prepares
either all 77 kernels or the 39 used by the recorded workload/configuration.
It changes neither kernel code nor arithmetic. Both arms use the same binary,
cache, checkpoint capacity, Metal validation and 12GiB admission. Counters
require zero pipeline creation after preparation, preventing compilation from
silently moving into measured execution.

All logits, committed tokens, proposal/rejection boundaries and final target and
draft state match exactly. Both arms recorded zero late pipeline creations.
The all-kernel control peaked at 9.787GiB with zero compression; the selected
arm peaked at 9.766GiB but recorded 13.156MiB compression. This single ordered,
instrumented diagnostic does not establish a causal physical-memory saving or
a latency effect. It fails as a way to eliminate compression and is not adopted.

## Next experiment

Continue fixed widths 1/2/4 using the retained full-replay recovery path and
original all-kernel preparation. A bounded numerical diagnostic may compare
all logits and persistent state while reporting disturbed resources explicitly.
Every performance sample still requires zero compression/decompression and
unchanged swap. Recovery, selected preparation and width policy remain separate;
none is promoted by numerical equality alone.
