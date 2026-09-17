# Expanded packed Q8 crosses the short perfect-verifier gate

The full-model four-token verifier now records **5.2814 and 5.3016 verified
tokens/s**, versus **4.6298 and 4.5894** for the unchanged four-token path in two
fresh alternating pairs. Both candidates pass the predeclared 5-token/s floor,
with **14.07% and 15.52% higher throughput**, respectively. This is a completed
positive verifier screen, not production speculative generation: proposals remain
free and perfectly accepted, and only sixteen continuation inputs after a
72-token prompt are timed.

## Full-model evidence

The [sealed verifier run](verifier-01/summary.json) completes in 208.41 seconds.
The [independent audit](audit-verifier-01.json) reconstructs every observation,
source identity, selection count, exactness comparison and paired decision.

| Actual order | Arm | Verified tokens/s | ms/token | GPU ms/token | Peak physical GiB |
|---|---|---:|---:|---:|---:|
| 1 | Control, pair 0 | 4.6298 | 215.991 | 125.515 | 9.5310 |
| 2 | Packed, pair 0 | 5.2814 | 189.344 | 99.438 | 9.5226 |
| 3 | Packed, pair 1 | 5.3016 | 188.623 | 98.596 | 9.5228 |
| 4 | Control, pair 1 | 4.5894 | 217.893 | 127.234 | 9.5224 |

Mean time drops from 216.942 to 188.983ms/token. Paired candidate/control wall
ratios are 0.87663 and 0.86567. Two pairs are a screening result; no confidence
interval or long-context qualification is claimed. Every timing process retains
all sixteen inputs, including the first block. Checkpoint, forward and greedy
acceptance work are timed; setup, priming and evidence hashing remain separate.

Every arm has **21.5921% expert-cache hits and 637.921MiB of application expert
reads per token**. The speedup therefore does not depend on omitting SSD work or
using a different initial cache. Device-level storage traffic is not measured by
these application counters.

Three fresh Metal API/shader-validation processes cover serial control, unchanged
four-token control, and packed four-token execution. Every vocabulary logit,
router selection, persistent-state boundary, causal-prefix check, zero-accept
rollback and partial-prefix recovery matches. Native counts show zero packed
dispatches during priming, exactly 399 through candidate validation/replay, and
532 in each candidate timing process: 133 intended operations per block.

All seven verifier processes have zero observed process compression, unchanged
decompression/swap counters, nominal thermal state, AC power and Low Power Mode
off. Timing peaks are 9.522–9.531GiB; validation peaks at 9.618GiB. The admitted
target/checkpoint/logit plan remains **11.1245GiB inside 12GiB**, with 1,460 expert
slots and 8,192-token state capacity. This does not demonstrate 8K-context speed.

## Isolated screen and implementation

The [operator screen](screen-01/summary.json) first passes six real input cases,
including the wide output head, in 26.90 seconds. All cases are measured together
in five alternating pairs of 32 dispatches, with one excluded warmup per arm.
The summed frequency-weighted GPU ratio is **0.583989**, paired 95% interval
**0.581480–0.586508**. Its 25.945ms/token median projection passes the declared
10ms threshold. It is a screening estimate; the full-model measurements above
provide the actual verifier outcome.

| Operator | Calls per block | Control GPU ms / four rows | Packed GPU ms / four rows |
|---|---:|---:|---:|
| Vocabulary head, 2560 → 248320 | 1 | 92.3511 | 27.8059 |
| Attention query, 2560 → 12288 | 12 | 2.0307 | 1.3723 |
| Attention output, 6144 → 2560 | 12 | 0.9546 | 0.7414 |
| GDN QKV, 2560 → 10240 | 36 | 1.5768 | 1.1437 |
| GDN gate, 2560 → 6144 | 36 | 0.8304 | 0.6823 |
| GDN output, 6144 → 2560 | 36 | 0.9545 | 0.7395 |

Only packed weight reads change: four codes are unpacked from each 32-bit load.
One output row, four token accumulators, width-eight lane partition, scalar
addition order, affine Q8/64 metadata, SIMD reduction and BF16 rounding remain
fixed. The developer backend admits only the declared stage/shape combinations
during four-token decode; priming and other operations retain their existing
paths. Control and candidate use the same executable, differing only in the
recorded selector. No quantization or model weights change.

The input capture hashes already resident weights and saves only **360,448 input
bytes**. Replay reads pinned checkpoint ranges without duplicating about 700MiB
of weights. Operator Metal allocation peaks at 651.80MiB within its separately
declared 1GiB allowance; its physical footprint stays under 711MiB. The full
engine budget is unchanged.

The [initial capture](capture-01/summary.json) remains resource-blocked because
startup compressed 85.625MiB. It nevertheless completed exact numerical capture;
only verified bytes and identity were reused under the predeclared protocol.
Its timing and memory samples are excluded. The later small operator processes
and all full-verifier processes are fresh and clean. Earlier GDN-only timings
are not pooled into this result.

Nineteen focused tests, native capture-configuration and checkpoint self-tests,
and capture/operator/verifier evidence audits pass. The tests include wrong
dispatch population, modified priming, corrupted selected weight bytes, malformed
tensor bounds, changed dtypes, disturbed memory and incomplete paired evidence.
The first developer build failed on a C++ JSON/string comparison; build 02 fixes
the comparison and is the sole producer used for inference. Production source
fingerprint and binary remain unchanged; the successful candidate is in the
hashed developer build, not the production CLI.

## Next stage

The optimistic verifier gate is now passed. Proceed to a bounded **real MTP draft
experiment**, keeping the packed verifier as its target implementation. Validate
the complete prepared MTP forward path independently, preserve the target's
10,240-wide hidden stream, and validate private draft attention/index state and
per-prefix rejection recovery before timing.

Begin joint admission with 1,460 target slots and at most 128 draft expert slots.
The existing preparation allowances imply approximately 0.53854GiB of incremental
draft memory and 11.6631GiB combined, leaving about 0.337GiB inside 12GiB. These
are planning estimates; actual metadata, shared ownership, scratch and live GPU
users must be measured before admitting the combined runtime.

At 188.983ms per verified token, only about **11ms/token** remains before the
200ms target, even at perfect acceptance. Measure actual drafting, SSD misses,
accepted lengths, rejection replay and target verification together on ordinary
coding continuations. Do not count rejected proposals as output, omit draft cost,
or assume the prepared head fits fully resident. Keep greedy exactness first;
production sampling/RNG rollback, 2K/4K/7K workloads and sustained coding remain
separate acceptance work. No unchanged cache-policy or GDN-only reruns are needed.
