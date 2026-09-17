# Packed Q4 through native submission and the expert coordinator

The packed-Q4 GPU gain survives the native scatter path, real resident weights,
and the actual all-hit expert coordinator. These bounded replays do not reproduce
the normal request regression. Existing request traces expose a useful mismatch:
most expert-bearing commands contain one expert, even with a group limit of four.
Next test real read arrivals and preceding resident computation; the native Q4
default and the rejected normal-request result remain unchanged.

## Independent conditions

Each row is five alternating pairs in one process, with 256 expert executions
per measured arm. Ratios below are paired geometric means, packed/reference,
with two-sided 95% intervals. Conditions are separate; do not pool their samples
or treat differences between rows as a paired causal comparison.

| Condition | Group | GPU ratio [95% interval] | Wall ratio [95% interval] |
|---|---:|---:|---:|
| Native scatter | 1 | .565 [.558, .571] | .943 [.855, 1.040] |
| Native scatter | 4 | .534 [.529, .538] | .674 [.657, .691] |
| Direct output, synthetic bridge | 1 | .510 [.447, .582] | .893 [.814, .979] |
| Direct output, synthetic bridge | 4 | .537 [.482, .597] | .665 [.625, .707] |
| Scatter plus resident allocations | 1 | .541 [.491, .596] | .884 [.871, .896] |
| Scatter plus resident allocations | 4 | .530 [.529, .531] | .663 [.654, .671] |
| Actual coordinator, all hits, resident allocations | 1 | .559 [.552, .565] | .902 [.841, .968] |
| Actual coordinator, all hits, resident allocations | 4 | .529 [.528, .530] | .669 [.656, .683] |

The scatter condition retains `no_clear_native_gain`: its group-one wall upper
bound exceeds the predeclared 1.03 guard. The other three conditions pass the
diagnostic gate. This is not normal-request qualification. In the all-hit
coordinator, median GPU time falls from 17.02 to 9.51ms per 256 experts at group
one, while median wall time falls only from 33.81 to 29.93ms. Faster arithmetic
does not translate proportionally into completion latency for small groups.

The stages, including separate operator validation, finish in 2.12, 1.47, 3.43
and 3.57 seconds respectively. The measured loops contain neither Metal
validation nor per-dispatch profiling. Complete reports, original logs and seals:
[scatter](scatter-01/summary.json), [direct](direct-01/summary.json),
[resident](resident-01/summary.json), [coordinator](coordinator-01/summary.json).
Read-only reconstructions: [scatter audit](scatter-audit.json),
[direct audit](direct-audit.json), [resident audit](resident-audit.json),
[coordinator audit](coordinator-audit.json).

## What the request traces add

The separately sealed, older command trace covers 16 decode forwards in each
phase. Its expert-bearing command histogram is:

| Phase | 1 expert | 2 experts | 3 experts | 4 experts | Single-expert share |
|---|---:|---:|---:|---:|---:|
| Initial generation | 3,487 | 465 | 217 | 653 | 72.31% |
| Generation after append | 3,945 | 457 | 167 | 580 | 76.62% |

Of those commands, 706 initially and 690 after append also contain shared-expert
work. The rest contain only routed-expert work. `ready_group=4` limits group size;
it does not wait to collect four experts. A separate per-dispatch profile changes
the single-expert share to approximately 52–55%, demonstrating that this extra
instrumentation changes the scheduling distribution. Keep those observations
separate. These traces use earlier native build `81e6f5...`, reference Q4 and
core-plus-expert residency. They establish a hypothesis, not the current packed
kernel's request-level behavior. The detailed read records cover only the first
initial decode token at offset 72; command coverage includes both longer windows.

[Group occupancy evidence](group-occupancy.json) includes source hashes, build,
exact token/layer coverage and limitations. It verifies ten expert chains for
each captured layer/token. No overlapping GPU durations are summed or assigned
as exclusive costs.

## Correctness, memory and provenance

Native library fingerprint remains
`51877f96f015386dd642fd4e63fbc39ff2e75c1156f5ae1bcba764f546963594`.
This stage adds only developer replay/evidence tools. Packed kernel arithmetic,
artifact bytes and production settings are unchanged. Source snapshots preserve
the first three conditions under `screen-sources`, the coordinator extension
under `capture-sources`, and the native sources under `initial-sources`.

All conditions use eight verified original Q4 expert records and 64 saved real
expert/input cases. Separate native operator validation checks BF16/FP32
outputs and destination guards under Metal validation. Replay checks preserve
all output bytes, including the actual final timed batch and untouched output
slots. Their common output SHA256 is
`05b73dca9c175709ede2bf65096b68e9193d32277a3d95f97d142adfe7728d45`.
The previously sealed full-model state proof is revalidated for the same native
build; this stage does not rerun a full-model correctness workload.

All 80 timed arms have exact dispatch counts, no new physical GPU allocations
after warmup, at most two live command groups, and drained users at observations.
Resident conditions allocate 5,362,515,968 bytes (4.994GiB) of actual mixed-artifact
resident weights. Maximum sampled physical peak is 5.049GiB. All runs observe
zero process compression, unchanged decompression/swap counters, nominal thermal
state and unchanged power source. Boundary samples can miss brief disturbances.

The coordinator uses real cache leases and I/O workers, with 256 ready hits per
arm and zero timed reads, misses or joins. Original fixture GPU copies are freed
after cache priming. Its Metal CPU-wait counter stays zero because waiting occurs
in the coordinator instead: median waits are 30.54/26.48ms at group one and
22.35/14.57ms at group four for reference/packed. Those waits overlap GPU work;
do not add them to GPU time. Reporting one wait counter alone would be misleading.

The developer checker rebuild passes, and all **280 Python tests** pass, including
15 targeted replay/evidence tests. Independent review reconstructs the reported
ratios, counts and seals. No release, 2K/4K or sustained-coding gate is implied.

## Next bounded experiment

Preserve the previous normal-request rejection. The native buffer/scatter path,
resident allocation alone and an all-hit coordinator do not reproduce it. Real
read arrivals and preceding resident computation remain absent here, as does the
whole-token scratch lifetime.

Extend the diagnostic through `execute_experts` with real prepared-record reads
and an explicitly declared hit/miss pattern. Compare reference and packed on
the same records/inputs, retain every selected expert and its destination, keep
the memory/worker/group limits fixed, and count actual group occupancy plus
read-completion-to-submission delay. Reuse the present all-hit path as the control;
verify output bytes and resource lifetimes before timing. Capture detailed events
in a separate run so instrumentation does not determine the timing outcome.

Do not turn a diagnostic gain into a new full-request test until a concrete
scheduling or kernel change explains the mismatch. If read arrivals alone still
preserve the gain, add the normal preceding resident computation as a separately
declared condition. Keep the larger-cache/append-first-use investigation separate.

## Reproduce

The original [protocol](protocol.md), [resident decision](resident-decision.md)
and [coordinator decision](coordinator-decision.md) were written before their
respective timing conditions. Use a new output directory for each attempt:

```sh
cmake --build build/qwen --target qwen_q4_check -j 3
.cache/qwen-reference-venv/bin/python scripts/qwen/native_q4_replay.py run \
  --condition coordinator --output docs/benchmarks/NEW/coordinator-01
.cache/qwen-reference-venv/bin/python scripts/qwen/native_q4_replay.py verify \
  docs/benchmarks/NEW/coordinator-01 --output docs/benchmarks/NEW/coordinator-audit.json
```

The runner takes the shared GPU lease, enforces existing resource admission,
freezes source/build/artifact identity, runs separate operator validation, then
collects five alternating pairs. The fixed ceiling is 12GiB, with a 60-second
process deadline and 120-second stage deadline. Interrupted, blocked and partial
attempts retain their terminal status and do not become passing evidence.
