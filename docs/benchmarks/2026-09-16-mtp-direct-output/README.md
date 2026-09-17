# Direct outputs for single-row experts in four-token verification

Implemented in an isolated native C++23/Metal build. The change passes real-weight
destination/lifetime checks, full-model rejection validation and both alternating
short timing pairs. The three 128-token comparisons are complete: one case is
flat and two improve by about 2–2.4%. All remain below 5 tokens/s. Keep this
candidate experimental; do not spend the long-context/release qualification
budget on it unchanged.

## What changed

When an individual routed expert handles one row within a four-token target
verification call, its existing Q4 down kernel now writes directly into that
row's final expert contribution slot. This removes one temporary output, one
position buffer and one scatter dispatch. `Metal::linear` already calls
`linear_into`; the candidate changes its output binding, not the calculation.
Multirow experts, priming, shared experts and single-token recovery retain their
existing paths. Scope guards reset the switch during normal return and failures.

Both arms use the validated lazy ngram initialization and keep bounded expert
scratch off. They preserve reference Q4 kernels, Q8 selection, weights, router
selection, final reduction order, two live GPU groups and cache capacity. The
12GiB joint admission retains 1,460 target slots, 32 draft slots and the earlier
2MiB reserve. This experiment does not combine the previously rejected scratch
candidate or reopen the packed-Q4 kernel experiment.

The implementation is in `scripts/qwen/mtp_direct_output.hpp` and
`build_mtp_direct_output.py`; `probe_mtp_direct_output.cpp` and
`screen_mtp_direct_output.py` provide the native fixture and staged runner.
The source-copy producer changes only `model.cpp`, `pipeline.cpp` and the developer
harness. Native kernels, public class layouts, production sources and executable
remain unchanged by this stage. Developer executable:
`.cache/mtp-direct-output-build-01/probe-mtp-forward`, SHA256
`178778e06e1850d009114d5484f8df70f6fcfea5907b6d1ea0efbf88bca8ef4c`.

## Correctness completed

- [Real-weight fixture](fixture-01/summary.json): all four destination rows,
  nonzero offsets, untouched sentinels, mixed 1/2/3/4-row experts, invalid
  destinations rejected before any dispatch, reversed reads, forced eviction,
  all hits, delayed GPU completion, cancellation and injected read/encoding
  failures. All Metal buffers are released. Peak charged buffers are 43.64MiB.
  The shared eager/lazy ngram fixture also passes on actual packed tables.
- [Full-model validation](validation-01/summary.json): both arms match the
  qualified lazy control's full logits, committed tokens, future proposals,
  intermediate forced-rejection boundaries and final target/draft state.
  Host and memory observations are clean. Candidate dispatches fall by exactly
  its 1,735 direct writes; these validation times are not speed evidence.
- **42 focused MTP tests pass.** They reject changed outputs, cache identity,
  budgets, uncontrolled scratch use, leaked scopes/GPU users, missing coverage,
  incorrect copy counts and unexplained dispatch differences. The source-copy
  test checks unchanged kernels and layouts. See the
  [fixture audit](audit-fixture-01.json) and [validation audit](audit-validation-01.json).

## Short timing screen

Sixteen generated tokens after the same 72-token prompt, using real MTP proposals,
verification and recovery. All twelve proposals are accepted in each process.
Prompt priming is reported separately. Every pair matches all logits, tokens and
state with zero recorded compression/decompression, unchanged swap, nominal
thermals, AC power and Low Power Mode off.

| Order | Control tokens/s | Direct-output tokens/s | Cycle latency change |
|---|---:|---:|---:|
| Control, candidate | 4.8390 | 5.0408 | -4.0045% |
| Candidate, control | 4.9254 | 5.0394 | -2.2624% |

The geometric-mean cycle-latency reduction is **3.1374%**, passing the declared
2% gate with improvement in both orders. Each candidate removes exactly **3,110
scatter dispatches**: 22,815 to 19,705 total dispatches, 28,000 to 21,780 physical
allocations, and 31,846,400 bytes of logical scatter payload. That payload count
is not measured device-memory traffic. Peak physical footprint is **9.6483GiB**.
GPU submission counts vary with completion order; they are not held artificially
equal. Two pairs provide an initial screen, not a confidence-bounded claim.
See the [raw screen](short-01/summary.json),
[independent audit](audit-short-01.json) and [predeclared protocol](protocol.md).

## Longer comparison

The [three-case 128-token screen](long-01/summary.json) completes in 409.75 seconds
with fresh processes and alternating case order. All logits, tokens and final
target/draft state match between each pair. The independent audit also compares
both arms' target results with the earlier serial reference: all three match.
Historical timing is not reused, and the earlier interrupted suite retains its
original incomplete status.

| Coding case | Control tokens/s | Direct-output tokens/s | Cycle latency change | Draft acceptance |
|---|---:|---:|---:|---:|
| Merge intervals | 4.4128 | 4.4042 | +0.1967% | 86.67% |
| Repair LRU cache | 3.0916 | 3.1595 | -2.1492% | 66.67% |
| Retry/backoff | 3.6269 | 3.7164 | -2.4102% | 82.41% |

The across-case geometric-mean latency reduction is **1.4612%**. Each row is one
pair, not a repeated-measurement confidence bound. Peak physical footprint across
all six processes is **9.7273GiB**, with zero compression/decompression, unchanged
swap and clean power/thermal observations. No missing or blocked processes occur
in this stage. See the [full reconstruction](audit-long-01.json).

The candidate removes 29,529 / 34,490 / 32,115 dispatches respectively, with exactly
two fewer allocations per removed scatter. Cache hits, misses, evictions,
per-layer counts and final cache identity match within each pair. Expert reads
remain **627.91 / 762.11 / 718.90MiB per committed token**. These are application
expert-record bytes, not total measured device traffic. Removing local output
copies does not reduce those reads or change draft acceptance.

## Decision

The implementation is correct and shows modest measured gains, but it does not
establish a broadly faster or 5-token/s coding engine. Do not promote it or extend
this unchanged candidate to 256-token, 2K/4K/7K or sustained-session qualification.
Preserve the code and exactness evidence for a future controlled combination.
The previously rejected scratch and packed-Q4 results remain separate.

The next performance decision needs to address verification/recovery work and
expert reads per committed token. The new candidate still spends approximately
199 / 249 / 219ms per token in verification and another 17 / 54 / 39ms in recovery
on these cases. Dispatch removal alone has not closed that gap. Use the recorded
request evidence to choose the next intervention; do not extrapolate kernel or
allocation counts into a sustained speedup.
