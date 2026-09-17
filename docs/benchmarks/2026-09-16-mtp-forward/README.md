# Real trained MTP forward and first joint screen

The trained draft now runs in native C++/Metal with bounded storage, shared
target embedding/output weights, private attention/index state, and exact target
verification/recovery. Production defaults remain unchanged.

The first clean [joint screen](joint-03/summary.json) measures **4.7904 tokens/s**
with actual drafting, versus **3.6393 tokens/s** for its serial control: 24.03%
less wall time, or 31.63% higher throughput. All 12 proposed tokens were accepted;
16 target inputs were committed. Final target state agrees exactly with serial.
This is one short coding continuation and one timing pair, not a confidence
claim, a long-context benchmark, or achievement of the 5-token/s release target.

## Correctness and memory

- [Synthetic fixture](fixture-03/summary.json): 22 saved full-forward boundaries
  agree with an independent NumPy implementation. Maximum relative L2 error is
  0.01460; the full-vocabulary logit error is 0.00962. All expert choices and
  greedy IDs agree. The predeclared tolerance is 0.02 at each boundary.
- [Real target-hidden fixture](joint-03/real-fixture/reference.json): the same
  independent equations pass on actual captured target hidden states.
- Metal API/shader validation checks serial versus block execution, future-token
  independence, forced eviction, rollback and every proper prefix versus fresh
  replay. Those native comparisons require exact bytes. Context overflow and
  invalid rollback geometry are rejected before writes. Pre-update cancellation
  preserves state. Joint validation deliberately rejects a wrong proposal and
  checks recovery against target logits.
- 20 focused Python tests and the existing native target-checkpoint self-test
  pass. The production binary and fingerprint are unchanged.

The head consumes the 10,240-wide pre-final-mixer target hidden stream, uses its
own trained mixer, and applies each original zero-centred norm's `1+w` once.
A dedicated wide-normalization kernel avoids the target RMS kernel's 4,096-wide
limit. Shifted token IDs retain the preceding hidden state's attention position.
Draft priming uses at most 16 rows at once; target hidden capture is capped at
128 rows. No context-wide hidden history is retained.

Both clean timing arms retain 1,460 target expert slots and 32 draft slots under
the same 12GiB admission. The combined plan includes checkpoint storage, driver
reserve and both models' live resources. Peak physical footprint is 9.824GiB
in validation and 9.766/9.696GiB in serial/candidate timing. Compression and
decompressions remain zero, swap usage is unchanged, thermals are nominal, and
AC power is connected with Low Power Mode off.

The [first](joint-01/summary.json) and [second](joint-02/summary.json) 128-draft-slot
attempts passed their executed numerical checks but encountered process
compression. They remain `resource_blocked`; their timings are not pooled. The
32-slot follow-up saves 253.5MiB and commits the admitted host rollback buffer
after priming, when generation needs it. It does not raise system memory limits.

## Remaining cost

Four measured cycles each commit four tokens. Proposal generation takes
33.4–36.7ms per cycle, verification 738.3–806.0ms, and draft state recovery
22.2–25.2ms. Those costs and checkpoint/copy work are included in 4.7904 tokens/s.

## State-only catch-up follow-up

The native draft now has a bounded catch-up path that populates only keys,
values and index from verified target hidden states. It omits unused attention
queries, attention output, routed/shared experts and output mixing. Corrected
rows are processed together, at most four at a time. Initial priming still uses
the complete layer so this experiment isolates catch-up and preserves the
control's initial cache state.

[Fixture 04](fixture-04/summary.json) passes the independent 22-boundary reference
and exact full-layer versus state-only comparisons, including all proper
prefixes. State-only catch-up does not access the expert cache.
The [joint numerical audit](audit-recovery-02-reviewed.json) confirms identical
private and target state at every captured rejection/replay boundary. Numerical
agreement is separate from clean timing: its candidate had 76.219MiB peak
compression and remains `resource_blocked`.

The first fresh [timing attempt](recovery-timing-01/summary.json) also stops at the
first pair's memory gate. Control is clean at 4.8410 tokens/s; candidate records
4.3191 tokens/s with 53.859MiB peak compression and 73 decompressions. Swap usage
is unchanged. The [audit](audit-recovery-timing-01.json) confirms matching
proposals and target state but **zero usable timing pairs**. These values neither
establish a speed gain nor reject the optimization. No validation timing or
earlier pair is pooled, and the planned five pairs were not run after the stop.

The [second attempt](recovery-timing-02/summary.json), after additional host memory
became available, records 54.328MiB peak compression in its control and stops
before the candidate. Its [audit](audit-recovery-timing-02.json) preserves zero
usable timing pairs.

After subsequent memory-clean continuation checks, the fresh
[third attempt](recovery-timing-03/summary.json) completes one clean pair:
**4.8719 tokens/s** for full catch-up and **4.9713 tokens/s** for state-only
catch-up. The latency ratio is 0.980013 (2.00% lower in this one pair).
Catch-up falls from 6.150 to 0.893ms per committed token; complete cycle time
falls from 205.257 to 201.155ms. Both arms accept all twelve proposals and match
target state exactly, with zero compression/decompression, unchanged swap and
nominal AC-powered host readings. Peak footprint is about 9.697GiB.
The [independent audit](audit-recovery-timing-03.json) passes. The predeclared
5-token/s floor still fails, so the runner stops before four more pairs.
This is a directional short screen, not a confidence-bounded improvement or
achievement of the target. Only numerical proof was reused from `recovery-02`;
all timing in this comparison is fresh.

Correction to the initial progress note and timing protocol: 32KiB was the
candidate validation's pre-cycle observation; its eventual peak was 79,921,152
bytes (76.219MiB). The audit now reports the maximum across all lifecycle and
cycle observations explicitly. The original reports and frozen source version
remain preserved in `recovery-sources/`.

The original 4.7904 result remains a separate full-catch-up screen. Removing its
entire measured catch-up cost would still leave about 202.64ms/token. The newer
state-only comparison above is also below 5 tokens/s. The
[longer-continuation stage](../2026-09-16-mtp-continuation/README.md) now measures
128-token coding outputs with exact full-logit and final-state agreement, and
shows that lower proposal acceptance can outweigh the saved verification work.
Use those longer results and the current verifier profile to choose the next
change before wider contexts or production integration. Do not repeat this
unchanged catch-up screen or pool the older blocked timings.

Still unqualified: sustained sessions; normal 2K/4K/7K contexts; sparse draft
attention past 2K; cancellation during GPU/I/O work; tool workflows; non-greedy
exact sampling; and promotion to `run` or `serve`. The draft-only Q4/Q8 recipe's
usefulness is assessed through target acceptance, not claimed BF16 equivalence.

See [protocol](protocol.md), [recovery protocol](recovery-protocol.md), and
[separate timing protocol](recovery-timing-protocol.md) for the gates. Architecture was
checked against the [trained-head implementation](https://docs.vllm.ai/en/latest/api/vllm/models/qwen4_exp/nvidia/mtp/)
and [reference token alignment](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/llm_base_proposer.py).
