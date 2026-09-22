# Q4 gains with verified prepared-record read arrivals

The packed Q4 gain survives actual device-read arrivals through the production
expert coordinator. With two ready experts and six misses per batch, GPU time
falls about 47%, while coordinator completion time falls only 6.6–7.0%.
These small replays still do not reproduce the packed kernel's previously
measured full-request regression. Keep reference Q4 and the 1072-slot
experimental control; no inference default or quantization changes.

## Results

Each condition has five fresh alternating reference/packed pairs at group caps
one and four. Each measured arm executes 256 expert chains from the same eight
original Q4 expert records and 64 saved input cases. Ratios are packed/reference
paired geometric means with two-sided 95% intervals; lower is better. Timing,
Metal validation and detailed event capture run in separate processes.

| File-cache preparation | Ready experts | Group cap | GPU ratio [95% interval] | Coordinator wall ratio [95% interval] | Disposition |
|---|---:|---:|---:|---:|---|
| Retained | 8 | 1 | .851 [.620, 1.168] | .925 [.832, 1.029] | Inconclusive; storage unexercised |
| Retained | 8 | 4 | .882 [.758, 1.026] | .923 [.834, 1.021] | Inconclusive; storage unexercised |
| Retained | 2 | 1 | .853 [.580, 1.254] | .983 [.886, 1.090] | Inconclusive; storage unexercised |
| Retained | 2 | 4 | .877 [.724, 1.062] | .922 [.850, 1.000] | Inconclusive; storage unexercised |
| Selected ranges invalidated | 8 | 1 | .531 [.531, .532] | .689 [.668, .711] | Diagnostic gain |
| Selected ranges invalidated | 8 | 4 | .524 [.499, .551] | .639 [.615, .664] | Diagnostic gain |
| Selected ranges invalidated | 2 | 1 | .532 [.526, .537] | .930 [.902, .958] | Diagnostic gain |
| Selected ranges invalidated | 2 | 4 | .531 [.528, .534] | .934 [.915, .954] | Diagnostic gain |

The two verified-storage stages finish in 15.56 and 12.26 seconds. For the mixed
condition at cap four, median GPU time is 53.49→28.36ms per 256 expert chains;
median coordinator wall time is 117.49→108.92ms. Reads and computation overlap,
so saved GPU time cannot be added directly to saved wall time. These are replay
milliseconds, not full-token latency or tokens/s.

Each condition is independent. Do not pool samples or treat differences between
conditions as paired causal estimates. Wider variance in the retained-cache
runs does not identify a clock, thermal or cache-coherence cause. The exact
[declared initial protocol](protocol.md) and subsequent
[storage protocol](storage-protocol.md) are preserved with each sealed run.

## Verifying storage rather than counting application reads

The first two conditions each requested 14,155,776,000 application bytes across
all timed arms including preparation. Observed whole-device read deltas were
only 6,049,792 and 684,032 bytes respectively. Those reports remain sealed with
`no_clear_arrival_gain`; they cannot establish an SSD-read result.

A small CPU-only preflight checks scoped invalidation before another GPU run.
The first preflight found pages already absent and therefore could not establish
an eviction effect. The second explicitly warmed the eight record ranges with
buffered reads, then compared native uncached reads before/after invalidation.
Every range changed from 169 resident pages to zero. For 66,355,200 application
bytes per phase, device reads rose from 274,432 to 66,514,944 bytes. All returned
hashes and prepared-file size, inode, modification and change times matched.
Its elapsed time includes hashing and is not an SSD service-time measurement.
[First preflight](cache-preflight-01/probe.json),
[warmed preflight](cache-preflight-02/probe.json),
[source metadata and driver record](cache-preflight-02/summary.json).

This behavior is consistent with the running XNU kernel's
[uncached-read path consulting existing UBC pages](https://github.com/apple-oss-distributions/xnu/blob/xnu-11417.140.69/bsd/vfs/vfs_cluster.c#L5011-L5022).
The explicit developer option uses read-only mappings and
[range-scoped invalidation](https://github.com/apple-oss-distributions/xnu/blob/xnu-11417.140.69/bsd/kern/kern_mman.c#L1073-L1087),
without modifying source bytes or purging global caches. Mapping cleanup covers
errors, offsets are checked, and remaining resident pages fail the run.

The new runs invalidate only those eight ranges after draining all GPU and I/O
users, before priming each batch. Invalidation is excluded from coordinator
wall time, included in preparation time, and separately counted: 256 calls per
timing arm, about 29ms total. Preparation also reports its own read bytes and
logical load calls. An eight-hit arm performs all reads during preparation;
a two-hit arm performs 530,841,600 demand bytes and 176,947,200 preparation bytes.

Every timed arm must observe device bytes within 90–110% of its total application
bytes. The measured ranges were 100.25–100.61% for eight hits and 100.00–100.35%
for two hits. Aggregate device totals were 14,215,962,624 and 14,176,858,112 bytes.
Counters include other processes; matching coverage supports exercised device
reads, not per-process attribution, cold controller caches or physical NAND
access. The auditor returns `unqualified_storage` if any arm fails this coverage.
Missing counters stay unavailable.

## What the separate trace establishes

In the verified mixed-read trace at cap four, reference emits 53 commands for
64 expert chains: 43 commands contain one expert, nine contain two and one
contains three. Packed also emits 53 commands: 42 contain one expert and eleven
contain two. Thus roughly 79–81% of commands contain one expert. Small uneven
groups remain present while the GPU gain survives. Timing submissions also
obey the cap; trace occupancy is not substituted for uninstrumented occupancy.

Detailed records join original layer/expert IDs and output positions with
admission, queue, read, encode, submission, GPU and release events. Their
per-hit/per-miss readiness intervals distinguish already-ready work from new
reads. Each expert executes once, and buffers release after their GPU users
complete. Overlapping command intervals are reported without treating their
sums as exclusive elapsed time. This trace covers eight declared batches per
variant/group; it is not a whole-request trace.

## Correctness, memory and review

All four conditions pass separate Metal-validation checks, source/prepared
record equivalence and exact output checks on saved inputs. Timing checks its
actual final batch and verifies all cases outside measurement. Destination
canaries remain untouched. The common output SHA256 is
`05b73dca9c175709ede2bf65096b68e9193d32277a3d95f97d142adfe7728d45`.
The same-build full-model state/failure proof is revalidated from sealed prior
evidence; no new full-model qualification is claimed.

All runs retain 4.994GiB of actual resident weights, eight reusable expert
buffers, eight readers, at most two live command groups and a 12GiB budget.
No measured arm allocates a new physical Metal buffer. Maximum sampled physical
peak is 5.052GiB; sampled process compression/decompression and host checks pass,
with no observed swap growth at those boundaries. Sampling does not establish
sustained-session memory behavior.

The native library fingerprint remains
`51877f96f015386dd642fd4e63fbc39ff2e75c1156f5ae1bcba764f546963594`.
Only developer replay/evidence tooling changes. All 294 Python tests pass.
Source snapshots retain the first condition tooling under `screen-sources`,
the invalidation extension under `capture-sources`, CPU preflight variants
under `cache-preflight-sources`, and native inputs under `initial-sources`.

Original reports and reconstructed audits:

- [Retained eight-hit report](hits8-01/summary.json), [audit](hits8-audit.json).
- [Retained two-hit report](hits2-01/summary.json), [audit](hits2-audit.json).
- [Verified-storage eight-hit report](storage-hits8-01/summary.json), [audit](storage-hits8-audit.json).
- [Verified-storage two-hit report](storage-hits2-01/summary.json), [audit](storage-hits2-audit.json).
- [Source-bound condition comparison](comparison.json).
- [Independent raw-evidence review](independent-review.json).
- [Experiment ledger](../../experiments/126fdc7140d50d611162a1aca9f6abc34543a0105d07013e03aa20b1ff85b859.json).

## Next bounded experiment

The next replay should queue actual shared-expert resident operations before
`execute_experts`, matching the first production submission in `Model::moe`
(`src/qwen/model.cpp`). Preserve these same records, native cache/coordinator,
read coverage, kernel variants, and separate timing/trace processes. Compare
that condition against a fresh matching replay without the preceding work.
Measure the combined interval as well as expert GPU time; an isolated expert
gain can be offset elsewhere. Do not insert an extra global wait that removes
the production dependency boundary. Validate saved resident outputs independently.

If preceding computation still does not reproduce the request regression,
expand scratch-buffer lifetime from one expert batch to a full token as a
separate controlled change. Keep byte-budget admission and final-user release
checks. Neither mechanism is established as the cause by this report.

Do not change the read scheduler merely because the group cap rarely fills:
this diagnostic gain survives that distribution, and the previous coalescing
request experiment already made full requests slower. Return to a short normal
request comparison only after reproducing a relevant condition and validating
the narrow change. The 5 tokens/s, 2K/4K, 7K and sustained coding targets remain
open; these diagnostic results do not qualify any of them.
