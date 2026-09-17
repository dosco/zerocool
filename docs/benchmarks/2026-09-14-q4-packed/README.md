# Exact packed Q4: faster operators; request benefit remains unmeasured

The previously blocked full-model GPU breakdown completed. It led to a
two-row packed-Q4 developer kernel that reduces isolated expert GPU duration
by **43–47%**, preserving output bytes. It is not integrated into the engine.
Both operator experiments still miss the declared **20ms/token** projection
gate, with savings near **14ms**. No request-speed or production claim follows.

The [next stage](../../qwen_combined_decode_stage.md) measures Q4 alone and
together with the previously screened cache increase. This is a new combined
experiment, not a retroactive pass for either individual screen.

## Completed resident breakdown

[Raw capture](../2026-09-14-resident-operators/capture-02/summary.json),
[per-dispatch counters](../2026-09-14-resident-operators/capture-02/dispatch.json),
and [reconstructed request audit](resident-audit.json).

Same 1072 slots, core-cache residency, 12GiB maximum, artifact and native
fingerprint as the preceding stage. The capture took **93.57 seconds** for a
normal and a traced conversation. All output tokens, actual reuse, memory
allocation and all 48 layers match the declared workload. Both runs have clean
sampled memory and no observed swap growth. The earlier admission failure
remains incomplete in its original directory.

Coverage is four decode forwards after the 72-token prompt and four after the
retained append, at absolute positions **72–75** and **205–208**. Each phase has
12,700 dispatches. The normal four-step rates were **3.07/3.15 tokens/s**; these
are a diagnostic control, not a new steady-generation qualification or a
controlled comparison with older longer runs. Traced decode takes **34.5/38.7%**
longer than its matching normal phase. No overhead correction is applied.

Selected summed instrumented pass durations, milliseconds per token:

| Operation | Initial | Append |
|---|---:|---:|
| Routed Q4 gate/up | 83.85 | 92.78 |
| Routed Q4 down | 59.22 | 62.73 |
| Two narrow resident injection projections per layer | 31.62 | 31.26 |
| Three GDN Q8 projections | 31.77 | 30.29 |
| GDN recurrent update | 4.63 | 5.52 |

These counters are **inclusive intervals, not additive exclusive kernel
costs**. There are 1,123/1,128 overlapping adjacent pass intervals in the two
decode phases. Summed pass durations are 333.28/345.03ms per token, while
summed command durations are 321.22/327.82ms. Instrumentation also changes
execution. Use the ranking to select an operator experiment, not to subtract
its duration from request latency. The recurrent update alone is too small
to explain the large resident command class; projections dominate it.

## Packed expert candidate

The probe appends `probe_q4_packed.metal` to the unchanged native shader.
Each SIMD group computes two independent output rows, sharing input loads
and affine bias sums. It loads whole packed words once, preserving the
four-value BF16 sum rounding, scalar accumulation order, SIMD reduction and
gate activation. Gate/up and down use their original lane partitions. There
is **no new quantization, pruning, route change, or approximate reduction**.

Eight original Q4 expert records from layers 0,16,32,47 and eight saved real
inputs per layer give **64 expert/input cases**. Both hidden activations and
final outputs match the reference byte-for-byte, with BF16 and FP32 down
outputs. Separate Metal API/shader validation passes. The record bytes were
rehashed against the pinned prepared artifact, and all fixture payload hashes
were verified before execution. Old capture producer identity is preserved.

The original operator arm is compared with the new packed kernels in five
alternating pairs. An existing paired-gate/row-two combination is measured
third as a diagnostic; it is not the primary candidate. Each timed arm executes
four cycles through all 64 cases. Groups of one and four retain the same
arithmetic and buffers.

| Screen / expert group | Original GPU µs/expert | Packed GPU µs/expert | Paired GPU ratio, 95% interval | Rough saving/token |
|---|---:|---:|---|---:|
| Resident fixtures / 1 | 64.50 | 35.05 | .535 [.519, .552] | 14.11ms |
| Resident fixtures / 4 | 62.25 | 32.21 | .529 [.496, .564] | 14.44ms |
| Interleaved copy / 1 | 65.81 | 37.32 | .569 [.564, .573] | 13.57ms |
| Interleaved copy / 4 | 63.04 | 34.04 | .539 [.537, .541] | 13.92ms |

The third-arm existing kernels save about 6ms/token in the resident-fixture
projection, less than the new candidate. Each projection multiplies measured
per-expert GPU savings by 480. It omits SSD waits, dependency overlap and the
model's complete working set; it is not measured request latency. The five
pairs are correlated within a process, and the two screens are not pooled.

[Resident-fixture screen](operator-01/summary.json) finished in **4.09 seconds**
with **24.06MiB** of shared buffers. [Locality screen](stream-01/summary.json)
finished in **14.37 seconds** with **280.06MiB**, within its separately declared
384MiB bound. The latter copies 128MiB on the GPU before each expert group,
then measures only the expert group. Copy time is excluded. This perturbs
locality; it neither guarantees a cache flush nor represents a normal request.
Both screens have clean sampled process memory and no within-sample swap change.

## Verification and reproduction

[Q4 evidence audit](verification.json), [request audit](resident-audit.json),
and [Python checks](python-tests.log). Source snapshots retain the first probe,
the separate locality intervention, and the native build. The native engine
fingerprint is unchanged:
`81e6f5ced54d2802716bd5dd211e47428da0d39d65d55b8fe7564ce94bd26b79`.

```sh
clang++ -std=c++23 -O2 -fobjc-arc -framework Foundation -framework Metal \
  scripts/qwen/probe_q4_packed.mm -o /tmp/freellm-q4-packed
python3 scripts/qwen/screen_q4_packed.py --binary /tmp/freellm-q4-packed \
  --output FRESH_DIRECTORY
```

Use `--stream` for the distinct locality intervention. It has its own
allocation and cannot be pooled with the resident-fixture screen. Both runners
own the GPU lease, freeze artifacts and sources, enforce time/byte bounds,
and retain incomplete attempts. Neither changes the native engine.
