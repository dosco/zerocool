# Packed Q4 and cache capacity: measure their combined request effect

This is the next implementation stage, following the
[resident breakdown and exact Q4 operator screens](benchmarks/2026-09-14-q4-packed/README.md).
Native integration is now implemented behind `--q4-decode packed-r2`; the
default remains `reference`. All 72 native tests and 64 real expert/input checks
pass under Metal validation. Both full-model state arms pass, including exact
continuation versus fresh replay at 32 expert slots. The two-by-two request
screen completed and failed: the combined change was 1.98% slower for whole
conversations; packed Q4 was slower at both cache sizes. No five-pair or long
qualification follows for this candidate. Keep 1072 slots and reference Q4
as the experimental control. See
[the integration report](benchmarks/2026-09-14-combined-q4/README.md).

Packed Q4 cuts isolated expert GPU duration by 43–47%, with a rough frequency
projection near 14ms/token. The earlier one-GiB cache increase saved about
13–15ms/token in short conversations. Both failed their individual 20ms gates.
Preserve those failures. Test whether these changes work together; their
savings cannot be added because computation, reads, and cache behavior overlap.

## 1. Integrate the exact single-token kernel behind an explicit option

- Add `q4_decode=reference|packed-r2` to the native kernel configuration,
  benchmark interface and reported identity. Keep `reference` as the default.
- Copy the proven kernel arithmetic into the canonical native Metal source.
  Limit selection to decode, one token, affine Q4/group64, contiguous inputs,
  gate/up 2560→640 and down 640→2560. Keep gathered ingestion, other shapes,
  Q8 residents, routing, caches, scheduling and reduction order unchanged.
- Preserve packed codes, metadata, all selected experts, activation rounding
  and per-lane sums. This is a computation change, not quantization.
- Bind the integration to the probe shader hash and archive the preceding
  native source. New binaries require new evidence; earlier timings stay separate.

## 2. Prove the runtime path before measuring requests

Run the real-record operator checks on the native dispatch path, including the
partially populated last K iteration of the 640-input down projection. Verify
fallback selection for gathered rows, ingestion, Q8 and unsupported shapes.
Use the existing all-48-layer fixture to compare logits, routes, convolution,
recurrent, attention and index state with the option off/on. Force eviction
at 32 slots, exercise retained continuation versus fresh replay, and check
cancellation/failure drains under Metal validation. Missing assets cannot pass.

## 3. Run one bounded two-by-two request screen

All arms use the same mixed artifact, prepared records, 12GiB maximum, 8192
context, core-cache residency, CLOCK, Q8 rows two, SIMD routing, original GDN,
reused scratch, immediate submission, panel512/chunk128, eight readers and
ready groups of four.

| Arm | Expert slots | Q4 decode | Planned allocation |
|---|---:|---|---:|
| A | 1072 | reference | about 10GiB |
| B | 1072 | packed-r2 | same as A |
| C | 1460 | reference | about 11GiB |
| D | 1460 | packed-r2 | same as C |

Verify actual admitted bytes and identical fixed allocations; do not rely on
the rounded estimates. Run two fresh rounds, `A B C D`, then `D C B A`: eight
complete conversations, no prior samples pooled. Use the existing 72-token
initial prompt plus 33 outputs, then a retained 128-token append plus 33 outputs,
temperature zero and fixed seed. Cap the stage at 900 seconds and each process
at 150 seconds. Stop on missing admission, compression, decompression or swap
growth; retain incomplete evidence without shrinking an arm to make it pass.

Report B/A and D/C to isolate the kernel effect; C/A and D/B show the cache
effect. D/A is the primary combined comparison. Include initial and append
decode wall time, whole-conversation latency, TTFT, output identity, actual
reuse, application bytes and measured memory. Keep profiling, Metal validation,
and synthetic copy traffic out of all request timing.

Advance only if D/A saves at least 20ms per decode token in both phases,
both combined conversation pairs improve, their median ratio is at most .99,
and all secondary request/TTFT medians remain within 1.03. The packed kernel
must also improve median decode time at each cache size; a cache-only gain
does not justify keeping a slower kernel. These are two-pair screening
decisions, not confidence-qualified claims. Do not run long qualification for
a losing or memory-disturbed configuration.

## 4. Confirm the survivor, then test the real target

A survivor gets five fresh alternating pairs and the existing confidence and
state gates. Product acceptance still requires at least 5 tokens/s for 256
generated tokens at 2K and 4K prompts, the initial/append TTFT limits, 7K
reporting and the 20-minute coding workflow. Neither a 47% operator gain nor a
successful short combined screen establishes that outcome.
