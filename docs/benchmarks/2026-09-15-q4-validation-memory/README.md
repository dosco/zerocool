# Validation memory and completed forward-lifetime timing

The earlier compression did not recur in four fresh validation-off/on/on/off
processes. All used identical native work and completed with zero observed
current or peak compression, zero decompressions, and stable swap/host state.
Validation adds memory overhead, but these runs do not reproduce the prior
resource block or establish its cause.

The admitted fresh full forward-lifetime sequence then passed validation,
five paired timing repetitions, and complete ownership tracing. Packed Q4
reduces its shared/routed coordinator window by 7.1–8.3%. Retaining this
partial 20.25MiB scratch arena therefore does not remove the replay gain.
The fresh batch control completed with clean memory but inconclusive group-one
wall timing. Keep all earlier results and the full-request rejection unchanged.
No production change or normal-request throughput improvement is claimed.

## Same-work memory attribution

[Predeclared protocol](protocol.md), [sealed four-process report](attribution-01/summary.json),
[recomputed audit](attribution-audit.json), [independent review](independent-review.json).

Each process runs reference Q4 with group cap one, one 48-pass warmup, and
one 48-pass checked execution. Command and hardware counter profiling are off
in both validation settings. Existing aggregate counters remain enabled.
The resident weights, inputs, CPU oracle, prepared bytes, shared chain, eight
fixed cache slots, two ready hits/six SSD misses, and native allocation
signatures match across all four processes.

| Process order | Validation | Reported physical peak | Compression peak |
|---:|---|---:|---:|
| 0 | Off | 5.0568GiB | 0 |
| 1 | On | 5.1073GiB | 0 |
| 2 | On | 5.1072GiB | 0 |
| 3 | Off | 5.0566GiB | 0 |

All 228 lifecycle observations are present: before Metal creation, after
pipelines, resident loading, fixtures, shared reference, routed reference,
expert-pool preparation, warmup, every observed pass, and scratch release.
Validation-on footprint is about 28MiB higher by pipeline setup and about
51MiB higher after warmup/execution, while charged engine allocations match.
Device allocation-size differences are recorded separately; they are not
direct driver physical-memory attribution. Two observations per setting do not
provide a statistical causal estimate or identify individual compressed pages.

The developer observer flushes each phase before checking hard bounds.
Unknown or excessive process footprint, changed swap/power, thermal pressure,
low-power mode, or outstanding GPU users stops the diagnostic. Compression
would remain an explicitly dirty observation in this attribution-only protocol.
Existing inference and timing clean-memory criteria were not relaxed.
The four-process stage completed in 12.28 seconds.

## Fresh full diagnostic sequences

Because all four memory processes were clean, the protocol selected the
ordinary check/timing/trace runner for both scopes. No timing-only shortcut
ran. Each condition has five fresh alternating reference/packed pairs at caps
one/four, six cycles per arm, 48 shared chains, and 384 routed expert chains.
Validation, uninstrumented timing, and detailed capture use separate processes.

Ratios below are packed/reference paired geometric means with two-sided 95%
Student-t intervals over log ratios. Conditions use separate processes; their
absolute-time differences are not a paired estimate of scratch-lifetime cost.
No earlier samples are pooled, removed, or replaced.

| Scratch scope | Group cap | Shared/routed GPU ratio [95% interval] | Coordinator wall ratio [95% interval] |
|---|---:|---:|---:|
| Forward | 1 | .58281 [.52459, .64748] | .91722 [.88504, .95056] |
| Forward | 4 | .61999 [.60665, .63364] | .92922 [.89513, .96460] |
| Batch | 1 | .68593 [.51603, .91177] | .89225 [.74164, 1.07345] |
| Batch | 4 | .62273 [.61028, .63544] | .93792 [.90508, .97195] |

[Forward report](forward-01/summary.json), [forward audit](forward-audit.json),
[batch report](batch-01/summary.json), [batch audit](batch-audit.json),
[independent lifetime review](independent-lifetime-review.json),
[source-bound comparison](comparison.json),
[experiment ledger](../../experiments/6664a3d862dfead4539d48e9173f570d86bf6c811c533ca90830443daad297f6.json).
Forward completes in 30.66 seconds with diagnostic_gain. Batch completes in
31.62 seconds with no_clear_arrival_gain: its group-one wall upper bound
exceeds the unchanged 1.03 guard. The low and high observations remain in
the result; lower medians do not override the confidence interval.
In that batch group, the first reference GPU observation is unusually fast
(59.35ms versus roughly 99–100ms later), and the final reference wall/preparation
observation is slow (251.46/164.07ms). Work counts, memory and byte checks remain
valid; the measurements do not establish why those observations differ.

At forward cap four, median GPU time is 98.69 to 61.66ms and coordinator wall
is 183.05 to 170.92ms per observed arm. These values describe repeated
shared/routed fixtures, not a complete token, coding request, or tokens/s.
Coordinator wall excludes explicit file preparation. GPU durations include
mixed shared/routed commands and cannot be split into independent expert costs.

Every timing arm meets actual-device-read coverage. Device/application ratios
are 1.00155–1.00237 for forward and 1.00146–1.00310 for batch. Each arm performs
384 selected-range invalidations and reads 796,262,400 demand bytes plus
265,420,800 preparation bytes. Systemwide counters include other processes and
preparation; they do not prove NAND-cold storage or per-process attribution.

## Correctness and scope

The same 32 independent CPU shared-reference cases and native byte checks
pass throughout. Shared/native oracle and routed output hashes remain those
of the preceding stage. The new forward trace verifies all 192 captured passes
and 1,536 routed expert records, a used prefix growing by 27 buffers per pass,
1,296 retained buffers /20.25MiB, no observed-work allocations, and complete
scratch release. The batch trace verifies the same work with 27 retained
buffers /0.421875MiB. Every arm records 1,296 scratch reuses.

This remains a partial replay: 20.25MiB is 27.2% of the measured 74.390625MiB
full-token scratch. It repeats four shared layers and eight experts and omits
other operators, two selected experts per layer, complete recurrent/attention
state, and the normal expert-cache footprint.
[Full-request allocation context](../2026-09-15-q4-scratch-lifetime/full-request-context.json).

All 327 Python tests pass, including missing/reordered phases, mismatched
work, hidden compression peaks, hard resource guards, intermediate scratch
counters, changed references, and false promotion. The native checker rebuilds
successfully. [Tests](python-tests.log), [build](build.log). Source snapshots
are in screen-sources and initial-sources. The native inference library remains
51877f96f015386dd642fd4e63fbc39ff2e75c1156f5ae1bcba764f546963594; only developer
measurement code, tests, and evidence changed.

## Decision and next step

Close the validation-memory attribution attempt without assigning a cause:
the original compression was not reproduced. The tested temporary lifetime
does not reproduce packed Q4's earlier full-request slowdown. Keep reference
Q4 and 1072 slots as the experimental control.

Further fixture expansion is unlikely to be the fastest path to 5 tokens/s.
Return to actual requests at 1072 slots on the same current build: a short
initial prompt and retained 128-token append, limited to 17 outputs per phase.
The resulting 32 decode forwards /101,600 dispatches fit the existing command
profile cap. Run fresh uninstrumented reference/packed controls first. If the
slowdown recurs, capture the existing command/dependency trace in reverse order
with hardware counter profiling off; bound all four processes to about four
minutes in total. Preserve exact outputs/state and reject dirty, mismatched,
or truncated captures. These short histories do not qualify 2K/4K performance.

Rank whole-token GPU work and blocking time against the approximately
100ms/token gap to the 5 tokens/s target, alongside the packed-Q4 regression.
Mixed command durations remain mixed; do not mislabel them as isolated kernel
costs. This also permits dropping the packed-Q4 lead if its attainable benefit
is too small. No unchanged candidate should enter expensive long-context or
sustained-session qualification.
