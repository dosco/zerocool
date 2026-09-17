# Next stage: reduce repeated target work after MTP rejection

Status: clean real capture/replay passes all eight prefix checks. All five
full-model numerical diagnostics now match: accepted prefixes 1/2/3/4 and
immediate EOS, including logits, tokens, later proposals and final target/draft
state. Recovery performs zero target forwards and expert reads. Several runs
recorded CPU heap compression, so these diagnostics do not pass the strict
resource gate. The early timing control also hit compression before a usable
pair completed. Full qualification and speed remain unproved. The fixed-width 1/2/4 candidate is
built, with twelve exact synthetic recovery cases. Its full-model comparison
now uses the retained full-replay baseline while recovery remains unqualified. See
[implementation and evidence](benchmarks/2026-09-17-target-recovery/README.md).
Continue from the exact [direct-output experiment](benchmarks/2026-09-16-mtp-direct-output/README.md).
The immediate objective is to stop running accepted tokens through the entire
target again just to reconstruct their persistent state. Then measure whether
choosing a shorter draft on difficult continuations reduces wasted verification.

## Evidence and limits

The current continuation harness restores the pre-block checkpoint and performs
one full target forward per accepted input when a four-token block is partially
rejected. See `scripts/qwen/probe_mtp_continuation.cpp`, around line 112. MTP
state-only catch-up already exists for the **draft**; this proposal concerns the
48-layer **target** and is a separate change.

The completed 128-token candidate runs show:

| Case | Actual tokens/s | Rejected blocks | Extra target input rows replayed | Recovery ms/committed token | If all recovery were free, tokens/s |
|---|---:|---:|---:|---:|---:|
| Merge intervals | 4.4042 | 7 | 14 | 16.85 | 4.7572 |
| LRU repair | 3.1595 | 22 | 46 | 53.91 | 3.8080 |
| Retry/backoff | 3.7164 | 13 | 33 | 38.76 | 4.3419 |

[Reconstruction and source hashes](benchmarks/2026-09-16-mtp-direct-output/recovery-opportunity.json)
bind these calculations to the raw reports. Recovery includes checkpoint restore,
target replay and draft catch-up; it does not isolate their individual costs.
The final column subtracts *all* recorded recovery time from measured wall time.
It is an optimistic arithmetic bound at unchanged remaining costs, not a speed
forecast. Cache behavior will change when replay is removed. This step alone
cannot establish 5 tokens/s on these measurements.

## 1. Prove bounded target state recovery

Use an isolated source-copy build and explicit `full-replay` / `state-only` arms. Keep the mixed
artifact, reference Q4 arithmetic, existing packed Q8 policy, direct-output mode,
lazy ngram initialization and draft catch-up identical between arms. Keep expert
scratch off, two live GPU groups, 1,460 target slots, 32 draft slots, context 8,192
and the total 12GiB admission. Production defaults remain unchanged.

During four-token verification, retain a bounded journal of state-update inputs:

- For each of the 36 Gated DeltaNet layers: projected convolution input,
  normalized Q/K/V and the two per-head gating inputs. Retain exact existing
  values; do not recompute projections or change rounding.
- For the layer-1 ngram convolution: its normalized convolution input.
- Preserve accepted attention key/value/index rows already written by verification.
  Restore only rejected tail rows from the existing checkpoint.
- Keep the pre-block convolution/recurrent checkpoint already allocated by the
  harness. Include layer positions, token count, ngram token history, validity
  and trace identity in the recovery transaction.

On rejection, restore the original recurrent and convolution state, then run
only the existing `gdn_scan` and `conv_update` operations for the accepted prefix.
No target expert reads, router calls, dense projections or vocabulary projection
belong in this recovery path. Reuse the already verified logits and hidden rows.
Keep draft alignment/catch-up unchanged for the first comparison.

Do not save three full recurrent snapshots: each boundary alone is 108MiB before
convolution and metadata. The journal's main raw inputs are about 11.46MiB at
width four. Start with a **16MiB incremental allocation ceiling**, including
alignment, metadata and additional recovery scratch, and prove actual live bytes
before running the model. If it cannot fit, revise the design before timing.
Charge and preallocate the same reserved capacity in both arms. Do not silently
consume the driver reserve or reduce cache capacity in only one arm.

All journal owners must survive their last GPU use. Validate every destination
and complete journal coverage before modifying state. Commit metadata only after
all state updates complete. Cancellation/failure drains users, releases the
journal and leaves partially updated state invalid. The existing full-replay
method remains the reference; an unavailable journal must never produce a
partially committed session.

## 2. Validate first, then screen ordinary requests

1. Small real-weight fixtures: accept 1/2/3/4 inputs; compare every recurrent,
   convolution, attention and index byte, including rejected tails, against full
   replay. Cover attention block boundaries, ngram history, EOS, delayed GPU
   completion, invalid geometry, cancellation and repeated journal reuse.
2. Bounded full-model validation: force each rejection position and compare all
   logits, committed tokens, target/draft state and subsequent proposals with the
   existing replay reference. Include a later continuation to expose stale state.
   Independently completed clean validation runs may be reused from sealed
   stages after rechecking their native producer, source inputs, workload bytes,
   all raw resource observations and numerical results. Combine them only after
   comparing the complete control/candidate pair again, even when its runs came
   from separate attempts. Missing arms still run;
   disturbed runs and all timing samples remain ineligible for reuse. Original
   incomplete stages keep their status. This avoids losing every independent
   correctness check when one later process encounters pressure.
3. Add separate timings for checkpoint work, target recovery and draft catch-up;
   record expert read counts/bytes around target recovery. Require zero target
   expert reads and zero full target forwards in the candidate recovery interval.
   Measure capture overhead on all-accepted blocks too.
4. First timing screen: one fresh 64-token LRU pair, chosen for its previously
   observed rejection cost. Stop if complete-cycle latency fails to improve by
   5%. Repeat in reverse order only after that gate. Both must improve; require
   an across-pair geometric latency reduction of at least 5%. A separate short
   all-accepted pair must not regress by more than 2%.
   An explicitly preliminary screen can reject a weak candidate after clean
   saved-state validation and an exact full-model diagnostic comparison, before
   all clean full-model cases finish. Its timing samples retain the same strict
   resource checks. A promising preliminary result does not permit advancement:
   finish all full-model correctness cases and collect the registered timing
   gates afresh. Never turn resource-disturbed diagnostics into speed evidence.
5. Survivors get fresh 128-token comparisons on all three coding cases, two
   alternating pairs per case. Retain only if each case avoids a greater than
   2% geometric-mean regression and the across-case reduction reaches 5%.
   These are screening gates, not confidence-bounded promotion evidence.

Use the same power, thermal, zero-compression and unchanged-swap controls.
Resource-blocked attempts stay incomplete and separate. Initial cache identity
must match; final cache identity and read counts may legitimately differ because
the candidate removes replay work. Exact numerical state and future outputs
must still match. Count the whole cycle, including journal capture and cleanup.

## 3. Reduce wasted verification with measured draft lengths

Screen fixed widths **1, 2 and 4** with real MTP proposals and the same recovery
path, memory budget and kernels. When recovery qualification remains blocked,
retain full-replay explicitly rather than adopting an unqualified optimization
or repeating unchanged recovery attempts. The bounded
[allocation investigation](benchmarks/2026-09-17-target-recovery/allocation-followup.md)
did not eliminate compression; its selected-pipeline candidate is set aside.
Measure full cycle cost per committed token, accepted-prefix distribution,
expert bytes per committed token, and the cost of checkpoint/journal capture.
The old width-two result predates the current packed-Q8 verifier and is not a
measurement of this configuration.

The width runner compares all logits and every committed target/draft boundary
with a width-one replay, including every possible first accepted prefix at
widths two/four, immediate EOS, and an irregular seven-token output. The serial
reference must also reproduce the established independent full-vocabulary logit
hashes. All widths retain equal four-row checkpoint/journal capacity, and width
one includes draft-state catch-up.

After exact numerical validation, an explicitly preliminary 64-token LRU screen
may reject an unpromising width. Compare four versus two first, and four versus
one separately. Stop a candidate if its first fresh pair does not reduce latency
by 3%; only survivors receive a fresh reverse-order pair. Both pairs must improve
and their geometric-mean reduction must reach 3%. Resource gates are unchanged.
This screening cannot qualify adoption or replace clean full-model correctness;
survivors still need the three longer coding cases and promotion evidence below.

Current [width evidence](benchmarks/2026-09-17-mtp-widths/README.md): all ten
full-model numerical cases pass cleanly. Width two loses the first short LRU
pair by 10.94% latency and stops. Width one's original short reverse remains
resource-blocked; a separate fresh 128-token LRU comparison completes two clean
pairs at 3.83–3.86 versus 3.16–3.17 tokens/s, a 17.77% geometric latency reduction.
It establishes a promising workload-specific result, not a completed short
stage, a general width policy, or 5 tokens/s. The other coding workloads now
complete two fresh alternating pairs each: width one is 9.27% slower for interval
merging and 3.43% faster for retry/backoff. Keep every workload's repetitions
separate. All settings remain below the target, so the next
[bounded profile](benchmarks/2026-09-17-mtp-widths/next-protocol.md) addresses
target verification before an adaptive policy experiment.

Only implement adaptive selection if the fresh results show a useful crossover.
Use completed prior-cycle costs and accepted lengths to choose between measured
widths; optionally evaluate draft confidence in shadow mode. Decisions must be
causal, with a fixed policy calibrated separately from held-out evaluation.
Use hysteresis and bounded exploration. Width one provides a serial fallback
while preserving valid draft continuation state. Compare against both fresh
serial and fixed-width controls, charging policy and state-maintenance costs.

If no measured width approaches 200ms per committed token, stop tuning that
policy and profile the remaining target verification dependency path. Wider
blocks, more cache, compression and new kernels are separate experiments; do
not combine them speculatively or add gains from unrelated runs.

## Promotion remains unchanged

Require five fresh alternating paired repetitions, exact recovery/session tests,
the 2K/4K prompt and append targets, 7K reporting and the sustained coding workflow
before changing production. The short 5.04-token/s result is preserved as a
short-screen result. This proposal does not claim sustained 5 tokens/s.
