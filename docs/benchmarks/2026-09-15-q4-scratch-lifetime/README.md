# Shared and routed temporary-buffer lifetime

The 48-pass lifetime diagnostic is implemented and its exact-output checks
pass. The planned latency comparison remains unqualified: both forward-scope
attempts stopped after Metal validation recorded process compression, before
timing. The fresh batch control completed, but one retained slow observation
left its group-four wall interval inconclusive. No normal-request improvement,
allocator change, or production promotion follows.

## Controlled change and coverage

[Protocol declared before timing](protocol.md). Both conditions execute six
cycles of eight existing fixture batches: 48 actual shared Q8/BF16 chains and
384 routed Q4 experts per arm. Eight readers, eight fixed expert slots, two
ready hits/six SSD-backed misses per pass, and at most two live GPU groups
remain fixed. Only scratch reset frequency changes.

| Scratch scope | Reset boundaries per arm | Retained physical buffers | Retained bytes |
|---|---:|---:|---:|
| Batch | 48 | 27 | 442,368 (0.421875MiB) |
| Forward | 1 | 1,296 | 21,233,664 (20.25MiB) |

Complete-scope warmup occurs before every measured arm. Both scopes reuse
1,296 buffers per arm with zero new measured Metal allocations. File preparation
runs only after GPU/I/O drain and uses the fixed expert pool, preserving the
arena's active prefix. Shared output views clear after their final GPU use.

The native full request retains 3,201 scratch buffers /74.390625MiB at its short
initial context. This replay covers 40.5% of the count and 27.2% of those bytes.
It omits two experts per layer plus attention, recurrent, residual, routing,
embedding and logits temporaries. It repeats four shared layers and eight
expert records rather than executing 48 distinct layers or a complete token.
[Source-bound full-request allocation breakdown](full-request-context.json)
separates actual measured counters from longer-context shape estimates.

## Original qualification attempts

| Condition | Complete | Recorded disposition | Audit |
|---|---|---|---|
| [forward-01](forward-01/summary.json) | No | resource_blocked after check | [Audit](forward-01-audit.json) |
| [forward-02](forward-02/summary.json) | No | resource_blocked after check | [Audit](forward-02-audit.json) |
| [batch-01](batch-01/summary.json) | Yes | no_clear_arrival_gain | [Audit](batch-audit.json) |

[Source-bound comparison](comparison.json),
[experiment ledger](../../experiments/dbc6a7fa3dc750078a29502b77ad27b52eff13186a5321a9a446c2c45035a456.json).

The first forward attempt held 27.7–28.0MiB of compressed process memory across
check boundaries. The single unchanged retry held 27.6–27.9MiB. Each records 23
counter increments across its recorded check span, including intervening
warmups; measured-arm decompression deltas total 16 and 15 respectively.
Compression already existed
before the first arm and then declined; these samples cannot identify which
pages were compressed. Both check processes passed the independent 32-case
CPU oracle and byte-identical shared/routed native comparisons. They provide
no normal timing or detailed forward-lifetime trace in the original stage.

Final scratch release returns all three processes to the same charged Metal
allocation, 5,385,306,112 bytes. The forward/batch retained-allocation difference
is exactly the declared 20,791,296 bytes. Outstanding command users are drained.
This does not show an accumulating native buffer-ownership leak; physical
compression and driver behavior remain distinct from allocation accounting.

## Fresh batch timings

Five fresh alternating pairs at each group cap, validation and profiling off.
Ratios are packed/reference paired geometric means with two-sided 95% Student-t
intervals over log ratios. Do not pool prior shared-arrival runs or exclude
observations based on latency.

| Group cap | Combined GPU ratio [95% interval] | Coordinator wall ratio [95% interval] |
|---|---:|---:|
| 1 | .62064 [.61279, .62860] | .94014 [.92033, .96038] |
| 4 | .63031 [.59063, .67265] | 1.13099 [.67177, 1.90414] |

The GPU gain survives in this control. Group-four pair two records 419.83ms
packed wall versus 175.29ms reference; preparation also takes longer in that
arm. Its cause is unestablished and it remains in the analysis. The group-four
upper wall bound exceeds the 1.03 guard, so the overall result remains
no_clear_arrival_gain despite lower medians. Neither command duration nor
coordinator wall is a complete-token latency or tokens/s measurement.

All timing arms pass memory/host observations, with zero observed process
compression or decompression. Device/application read ratios are 1.0012–1.0145.
Every arm reads 796,262,400 demand bytes plus 265,420,800 preparation bytes and
performs 384 selected-range invalidations. Device counters include other
processes and preparation; they are not per-process attribution or proof of
NAND-cold storage. Coordinator wall excludes preparation.

## Ownership trace and verification

The original batch trace audits all 192 passes and 1,536 routed expert records
across both variants and group caps. It verifies 27 retained physical buffers,
1,296 reuses per arm, zero allocation deltas, unchanged arena state during
preparation, exact shared-prefix command joins, and drained final release.
Per-pass snapshots run outside the measured coordinator windows. Used-prefix
buffer counts are inferred from allocated minus unused bytes and the verified
16KiB charge, not from a new production cursor interface.

Routed native output SHA256 remains
`05b73dca9c175709ede2bf65096b68e9193d32277a3d95f97d142adfe7728d45`;
shared native output SHA256 remains
`155e75079cfaf0a49d1ba20bf26da5fbb522fafefe50b94a303ad97f4dc57d9e`.
The independent CPU reference and actual selected tensor bytes remain bound
to their existing manifest and source identity. All 318 Python tests pass,
including shortened/reordered trace coverage, preparation mutation, cursor
discontinuity, incorrect allocation counts and altered resource observations.
All six preceding arrival/shared audits reproduce their sealed decisions.
[Tests](python-tests.log), [native build](build.log). Current tooling and the
native source basis are archived in `screen-sources` and `initial-sources`.

The native library fingerprint remains
`51877f96f015386dd642fd4e63fbc39ff2e75c1156f5ae1bcba764f546963594`.
Changes are confined to developer diagnostics, audits, and evidence. Reference
Q4 and 1072 expert slots remain the experimental control; the prior normal
request rejection and 5 tokens/s /2K /4K /7K /sustained-coding gates remain open.

## Separate memory investigation

The [memory protocol](memory-protocol.md) declares one separately sealed,
validation-off forward trace to inspect physical memory and the complete
48-pass arena ownership. It does not resume either blocked qualification or
provide performance evidence. Validation and profiling differ from the check
processes, so even a clean result cannot establish a validation-specific cause.

[forward-memory-01](forward-memory-01/summary.json) completed in 4.88 seconds
with status `memory_observed`: all boundary samples report zero compression
and zero decompressions. The reported process-lifetime physical peak is
5.0674GiB; the highest individual boundary observation is 5.0602GiB. Its
192 captured passes verify the active prefix growing by 27 buffers each pass,
with 1,296 physical buffers /20.25MiB retained through each 48-pass scope and
zero measured allocations. Both variants preserve the independent CPU oracle
and native bytes. GPU/I/O users drain before preparation and final release;
the final charged allocation again equals 5,385,306,112 bytes.
[Independent audit](forward-memory-audit.json).

This is evidence that the larger partial arena can execute with clean observed
memory under this capture configuration. It does not determine whether Metal
validation caused the earlier compression, and it provides no uninstrumented
timing comparison. The original lifetime stage remains unqualified.

## Next bounded step

Add a small memory attribution check at setup, reference creation, complete
warmup, drained execution, and scratch release. Use the same forward work with
profiling off in fresh validation-on/off processes to isolate instrumentation
more directly. Preserve all existing resource gates and failed reports. Once
the resource difference is understood, complete fresh uninstrumented lifetime
timing; retain the slow batch observation rather than silently replacing it.
Do not redesign buffer ownership or infer a full-request speedup from these
diagnostics. A short normal request must still improve before expensive
qualification or a production change.
