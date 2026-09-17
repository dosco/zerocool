# Four-token GDN Q8: exact row pairing is faster, but too small to advance

The [independent operator replay](replay-01/summary.json) completes in 4.36 seconds.
Metal API/shader validation and all five alternating timing pairs preserve every
output bit. Both processes have clean memory and host observations. The
[audit](audit-replay-01.json) and [reviewed audit](audit-replay-01-reviewed.json)
pass. No full-verifier or production performance change follows.

Computing two output rows together reduces the summed isolated GPU duration by
9.73% (paired ratio 0.90268, 95% interval 0.89093–0.91458). The median
shape-frequency projection is only **2.94ms per input token**, below the
predeclared 10ms screen. The candidate therefore does not advance to an expensive
full-model correctness/timing run. This is a small measured operator improvement,
not an end-to-end speedup or a numerical failure.

| Captured Q8 dimensions, four input rows | Mean baseline GPU ms | Mean two-output-row GPU ms |
|---|---:|---:|
| 2560 / 10240 | 1.614 | 1.501 |
| 2560 / 6144 | 0.823 | 0.718 |
| 6144 / 2560 | 0.957 | 0.844 |

Each sample repeats one operation 32 times; table values are per operation.
Projection multiplies each shape by 36 recurrent layers and divides by four input
tokens. Five pairs within one process are correlated, and layer-0 inputs do not
cover every layer or context. These estimates are screening evidence, not request
savings. Isolated repeated-weight durations are distinct from the earlier GPU
counter profile's 41.17ms ranking opportunity.

## Correctness and resource scope

The three fixtures contain 61,456,384 bytes. All payload hashes are checked, and
all nine packed-weight, scale and bias tensors are rehashed against the pinned
mixed checkpoint. Input tensors come from layer 0, decode offset 72, four rows.
The source capture matches every vocabulary logit, route, persistent state and
initial cache boundary across all sixteen continuation inputs. Production model
source, Metal kernels, artifacts and executable remain unchanged.

The small probe admits 256MiB of GPU buffers and peaks at 26.9MiB. Timing-process
physical memory peaks at 49.0MiB. Both variants are also checked against the
existing scalar kernel, with one declared warmup per arm before paired timing.
Eleven focused tests across the profile and operator tools pass, as does the
native capture-configuration self-test. Offline review adds rejection of zero or
impossible GPU durations without changing any recorded result.

## Preserved attempts and tensor-only reuse

[Screen 01](screen-01/summary.json) and [screen 02](screen-02/summary.json) fail
before priming due to diagnostic capture configuration errors. Priming now clears
both the capture directory and filters while profiling is disabled. Their
[audits](audit-01.json) and [audit](audit-02.json) pass, retaining the failed status.

[Screen 03](screen-03/summary.json) captures all three exact fixtures and completes
the full numerical replay, but observes 167.55MiB of process compression during
startup. It remains resource-blocked; neither operator process runs within that
stage. Its [audit](audit-03.json) preserves that disposition. Those timings and
memory samples are not reused.

The subsequent [separate replay protocol](protocol-replay.md) uses only verified
tensor bytes from that capture. It revalidates the complete numerical identity
and source weights before running fresh small validation/timing processes. This
avoids another full-model load while preserving the original blocked stage and
all whole-verifier memory/performance gates. The earlier clean block profile,
not the disturbed capture timing, selected the three operator shapes.

## Next bounded hypothesis

Investigate packed 32-bit Q8 weight loads for the existing four-token kernel on
these same fixtures. The current generic multi-token kernel loads Q8 codes one
byte at a time; the packed single-token path already exists, but its benefit
does not establish a four-token gain. Keep one output row initially so packing
is the sole change. Preserve scalar accumulation and SIMD reduction order.
Use the same cheap exactness/timing gate before any native request integration.

Do not run a broader row-count/tile sweep, repeat the rejected cache changes,
integrate this weak candidate, or claim 5 tokens/s from projected operator costs.
The latest clean short results remain 4.07 tokens/s serial and 4.67 verified
tokens/s with free, perfectly accepted proposals.
