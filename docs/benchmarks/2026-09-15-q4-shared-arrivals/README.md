# Shared-expert work before routed experts

The packed-Q4 gain survives the actual shared-expert operations queued ahead
of routed work. Combined shared/routed GPU command time falls 37.3–38.0%, and
the replay's combined work window falls 6.9–8.2%. This condition also fails to
reproduce the earlier full-request slowdown. Keep the normal-request rejection,
reference Q4 and the 1072-slot experimental control. No production source,
inference default, model byte or quantization changes.

## Fresh paired results

Both conditions use two ready experts and six SSD-backed misses per batch,
eight fixed buffers, eight readers, two live command groups and the same actual
resident weights under a 12GiB budget. Each is five alternating reference/packed
pairs at group caps one/four, with 256 routed expert chains per measured arm.
The shared condition adds one three-operation shared chain per batch.

Ratios are packed/reference paired geometric means with two-sided 95% intervals.
Conditions have independent samples; do not pool them or treat their absolute
time difference as a paired estimate of shared-work cost.

| Condition | Group cap | GPU command ratio [95% interval] | Combined wall ratio [95% interval] |
|---|---:|---:|---:|
| Fresh routed-only control | 1 | .533 [.523, .542] | .915 [.885, .947] |
| Fresh routed-only control | 4 | .531 [.527, .535] | .912 [.879, .946] |
| Shared plus routed | 1 | .627 [.609, .644] | .918 [.879, .959] |
| Shared plus routed | 4 | .620 [.609, .630] | .931 [.907, .955] |

Both conditions pass the predeclared diagnostic gate. At group cap four in the
shared condition, median combined GPU time is 66.10→40.64ms and median combined
wall time is 120.99→112.28ms per 256 routed expert chains plus 32 shared chains.
The wall measurement includes shared encoding, expert admission, overlapping
reads/computation and final drain; it excludes explicit preparation. The first
command mixes shared and routed work, so its GPU duration cannot be apportioned
into individual expert costs. Pure routed GPU time is unavailable in that mixed
command. These are replay measurements, not full-token latency or tokens/s.

The shared stage finishes in 13.75 seconds; its fresh control finishes in 12.63.
[Protocol declared before timing](protocol.md), [shared report](shared-01/summary.json),
[control report](control-01/summary.json), [source-bound comparison](comparison.json).
Earlier arrival results remain separate and sealed.

## What was added

The developer checker queues the same three operations as `Model::moe`:
Q8 gate/up (2560→640), Q8 down (640→2560), and the BF16 shared gate (2560→1).
There is no added submit or wait before the actual `execute_experts` coordinator.
Each batch uses one shared layer, rotating through 0,16,32,47 twice, and that
layer's saved input row at offsets 72–79. The routed fixtures span four layers;
this is a declared eight-expert diagnostic, not a real layer's ten routed experts.
The final MoE reduction and shared-gate weighting remain outside this replay.

An independent CPU tool reads only the selected mixed-checkpoint tensor ranges:
20,910,080 bytes total. It computes FP64 decoded-weight dot products with the
model's BF16 projection and activation boundaries. The reference manifest binds
all 40 tensor hashes, source file identity, the generator, BF16 helper, artifact
lock and original inputs. The native checker hashes the actual resident tensor
buffers against those identities before using the reference.

The [CPU reference](reference/manifest.json) contains 32 layer/row cases and
409,728 bytes of activation, down and scalar-gate outputs. Across the native
checks, worst vector relative L2 is 0.001119 and minimum cosine is 0.99999938,
within the fixed 0.01 / 0.99995 limits. All scalar-gate values match the CPU
reference exactly, within the declared 1e-6+abs(reference)/128 bound.

The native implementation independently saves its own output bytes before
timing. Both Q4 variants must then preserve those bytes, including the actual
final timed batch. This exactness check is separate from CPU numerical tolerance.
The shared native-reference SHA256 is
`155e75079cfaf0a49d1ba20bf26da5fbb522fafefe50b94a303ad97f4dc57d9e`
in all three processes. Routed outputs retain the prior SHA256
`05b73dca9c175709ede2bf65096b68e9193d32277a3d95f97d142adfe7728d45`.
Destination guards remain intact.

## Scheduling and storage evidence

Check, timing and trace are separate native processes. Only check enables Metal
validation. Shared trace enables existing command-group profiling, without
per-dispatch compute-pass timing; normal timing enables neither. Every captured
batch joins dependency events to actual command submission times and proves
that exactly three shared operations precede routed operations in the first
command. All command and read coverage is complete, with no extra submission
or global wait. Instrumented trace latency is not used as performance evidence.

At group cap four, the shared reference trace emits 52 commands for 64 routed
experts: 41 single-expert, ten two-expert and one three-expert command. Packed
emits 54 commands: 44 single-expert and ten two-expert. Small groups remain
present while the combined gain survives. This trace is bounded to eight batches
per variant/group, and its occupancy is not substituted for timing occupancy.

Selected file ranges are invalidated after GPU/I/O drain and before hit priming.
Every timed arm requests 530,841,600 demand bytes plus 176,947,200 preparation
bytes. Observed device/application ratios are 100.14–100.31% with shared work
and 100.07–100.40% in the fresh control, within the per-arm 90–110% gate. Each
condition totals 14,155,776,000 application bytes across its timing arms.
Systemwide counters include other processes and preparation; they establish
neither per-process attribution nor NAND/controller-cold state.

## Memory and verification

The shared path adds three reused temporary buffers per batch: 48KiB of Metal
allocation capacity, plus 819,456 bytes of retained CPU/native reference payload.
Shared output views remain owned until the GPU completes and clear before
scratch reuse. Each shared timing arm reports 864 scratch reuses, 32 of each
shared dispatch and 256 of each routed dispatch, with zero new physical Metal
allocations. The control has 768 reuses and no shared dispatches.

Peak sampled physical footprint is 5.0534GiB with shared work and 5.0513GiB in
the control. All observed process compression/decompression and host/resource
gates pass. These small, short replays do not establish sustained memory behavior.
The native library fingerprint remains
`51877f96f015386dd642fd4e63fbc39ff2e75c1156f5ae1bcba764f546963594`.
The existing same-build full-model state proof is revalidated; a new full-model
correctness or release qualification is not claimed.

All 311 Python tests pass, including CPU decoding/reference checks, altered
shared-output/provenance cases, dispatch counts, mixed-command joins and trace
instrumentation checks. Historical arrival audits still reproduce their original
results. [Test log](python-tests.log), [native build log](build.log),
[shared audit](shared-audit.json), [control audit](control-audit.json),
[experiment ledger](../../experiments/37f15924dfa97cfb232c9c891983ed3016d74ad42421a24ccf8d8c4893def1d6.json).
The [independent raw-evidence review](independent-review.json) also passes. It
recomputes the paired intervals and command joins, checks all frozen sources
and asset metadata, and rehashes the selected real weight and fixture bytes.
Source snapshots are under `screen-sources` and `initial-sources`; raw reports
and their seals remain the source of truth.

## Next bounded step

Test temporary-buffer lifetime as a separate controlled change: the current
replay reuses its scratch arena after each eight-expert batch, while a real
token keeps more layer temporaries alive through the forward pass. Preserve
the shared prelude, real read coverage, arithmetic, inputs, memory admission
and completion rules. Measure the resulting allocation/residency pattern and
combined latency against a fresh matching control. Do not infer that lifetime
is the cause merely because these other conditions did not reproduce the issue.

Continue to require short normal-request improvement before any long validation
or production selection. The 5 tokens/s, 2K/4K, 7K and sustained coding targets
remain open. This investigation narrows an unexplained regression; it adds no
new normal-request performance claim.
