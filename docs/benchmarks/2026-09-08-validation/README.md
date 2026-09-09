# Full-model validation after freeing memory

Closing Brave made 23,476,404,224 bytes (21.86GiB) reclaimable and allowed the
normal 12GiB engine budget. The blocked full-model checks were retried on the
actual 32GiB M1 Pro. After fixing one additional GDN normalization discrepancy,
**both Q4 and mixed 4/8-bit match their independent five-token references exactly**:
all 48 layer outputs and all 248,320 final logits.

Native build:
`4504da7cab2fe9be2e18967ffacec516c32468120c3611606d9450b68faa9c02`.
Production remains C++23/Metal. No weights or OS memory limits were changed.
Q4 remains the default; mixed precision is explicitly selectable.

## Reproduced error and correction

The first retry, on build `40e554b6...`, confirmed the earlier PLE fix through
layers 0–5 but still failed final-logit agreement: relative L2 0.1053389.
Tracing the first differing layer found two BF16 query values changed by GDN
normalization, with identical incoming convolution values. The native kernel
summed strided groups of four squares and used approximate `rsqrt`. The pinned
MLX implementation uses adjacent groups of four FP32 squares and
`precise::rsqrt`. Matching that arithmetic fixed the complete fixture.

The saved real fixture under `tests/fixtures/qwen/gdn-qk-mixed` fails before
the correction and passes afterward. [before-gdn-fix-parity.json](before-gdn-fix-parity.json)
and [gdn-regression-before.txt](gdn-regression-before.txt) preserve the failure.
The final [native suite](native-tests.txt) passes **30 tests / 896 assertions**
with Metal API and shader validation. The [developer-tool suite](tool-tests.txt)
passes **27 tests**, including rejection of changed golden logits,
cross-artifact replay evidence and different retained state.

## Complete model and session checks

Each artifact passes **53 full-model checks** comparing its independent reference,
native source storage, and native prepared storage. The compared native runs use:

| Configuration | Storage | Chunk | Expert slots | Expert schedule |
|---|---|---:|---:|---|
| Prepared | Lossless contiguous records | 8 | 32 | Completion-driven, up to two GPU groups |
| Source | Original checkpoint ranges | 1 | 64 | Batched control |

Both execute the normal resident-weight path at context 256. The five prompt
tokens are `[760, 6511, 314, 9338, 369]`. All 109 recurrent, attention and PLE
state buffers match exactly across configurations, along with every final logit.
Q4 matches its saved original oracle; mixed matches its separately pinned oracle.
See [mixed-full-state-parity.json](mixed-full-state-parity.json),
[q4-full-state-parity.json](q4-full-state-parity.json) and
[mixed-logit-parity.json](mixed-logit-parity.json).

The prepared mixed run planned 6.39GiB with its deliberately small cache and
reported 5.34GiB final process footprint; the Q4 run planned 4.13GiB and reported
3.08GiB. These are short correctness runs, not the maximum working set required
for an 8K conversation. [memory-after-brave.json](memory-after-brave.json) shows
the full context-8192 mixed allocation and current admission.

Both artifacts also pass **20 normal session/panel/failure checks**: five initial
tokens, a three-token append including EOS, then two single-token continuations.
Continued state and logits match fresh replay. Failed or cancelled panels
invalidate partially updated state and drain GPU work; state from the other
artifact is rejected before mutation. These checks use actual prepared records
and resident weights, not diagnostic trunk streaming.

Requested panel sizes are 0, 256, 512 and 1024. At this fixture's context limit,
the latter two are admitted as 256; these results do not qualify long 512/1024
panels. See [session-mixed-4_8bit.json](session-mixed-4_8bit.json) and
[session-q4-control.json](session-q4-control.json).

## Normal 2K request timing

One complete mixed-artifact request ran with Metal validation disabled for
timing, a 12GiB budget, 1,848 expert slots, panel 512, microchunks of 128,
eight I/O workers and ready groups of four. It used the same 2,048 prompt token
IDs as the earlier Q4 baseline and generated all 256 requested output tokens
with temperature zero and seed zero.

| Measurement | Result | Target |
|---|---:|---:|
| Time to first token | 470.627s | ≤60s |
| Generation throughput | 1.985 tokens/s | ≥5 tokens/s |
| Generation duration, 255 intervals | 128.450s | — |
| Median / p95 token latency | 485.5 / 660.0ms | — |
| Peak Metal allocation | 11,304,058,880 bytes (10.53GiB) | Within 12GiB plan |
| Final process footprint | 11,194,030,848 bytes (10.43GiB) | Within 12GiB plan |
| System swap, before → after | 1,810,432,000 → 1,709,768,704 bytes | No growth observed |
| Application read bytes | 314,893,710,900 | — |

Both latency targets fail. Request timing starts after the resident model
loads. This is a single normal request, not performance promotion or a
20-minute memory-soak result. The mixed 2K output has not been numerically
compared with the independent oracle or independently scored for coding quality.

The aggregate expert hit rate was 38.05%, combining prefill and decode.
Expert GPU work totaled 131.923s and coordinator expert waits 141.872s across
the request. These overlap and exclude other operators, so they cannot be
added to attribute the entire 599-second request to SSD or expert execution.
The next performance investigation should split prefill and decode dependency
traces and measure resident-matrix and attention work as well as reads.

Raw evidence: [mixed-normal-2k.json](mixed-normal-2k.json),
[summary](mixed-normal-2k-summary.json), [workload](workload-2k.json).

## Remaining qualification

Exact agreement is established for the saved five-token references and native
short-session consistency. Longer prompts, sparse-attention boundaries,
129-token retained-history appends and long panels must still be requalified
after the normalization corrections. Coding quality, paired quantization
confidence bounds and the sustained 20-minute coding workflow are separate
open gates. Affine Q3 calibration has not started, and no precision switching
has been introduced.

The older Q4 2K baseline used a different artifact, earlier native build and
lower admitted memory. It cannot serve as an equal-budget paired comparison
with a new mixed-artifact run. Performance promotion still requires the complete
five alternating paired repetitions, 4K and 7K workloads, retained-history
append latency, and sustained-session checks.
