# Next experiment: target work after fixed-width screening

The LRU screen measures 258.78–261.02ms per committed token at width one.
Verification accounts for 255.27–257.45ms; eliminating its remaining draft
maintenance alone cannot reach 200ms/token. Both interval-merging pairs favor
width four: width one increases geometric latency by 9.27%. Both retry/backoff
pairs favor width one by 3.43% geometrically. No measured fixed width reaches
5 tokens/s. These are separate preliminary screens, without confidence bounds;
a universal smaller width is not supported.

## Bounded dependency capture

Adapt the existing target profiler to the current fixed-width producer. Its
present 16-token, all-proposals-accepted fixture cannot describe a later LRU
continuation or rejection-heavy verification. Keep the old profiler and its
evidence unchanged.

1. Derive an isolated source-copy diagnostic from the exact width producer,
   preserving packed-Q8, reference Q4, direct output, full-replay recovery,
   lazy ngram initialization and scratch-off controls. Do not combine a new
   kernel, cache policy or recovery method with profiling.
2. Use the sealed 128-token numerical references for width-one LRU and
   width-four interval merging. Reproduce the complete prompt and continuation
   before accepting a trace. Compare all full-vocabulary logits, token IDs,
   proposal/acceptance decisions and final target/draft state. Earlier timing
   remains reference evidence only and is never reused as a new sample.
3. Capture a bounded middle-generation window: begin at the first cycle
   boundary at or after 32 committed output tokens and cover the next 16
   committed tokens, including any final whole cycle crossing that boundary.
   Declare exact covered positions and omitted work. Do not replay a changed
   token order or reset the cache at the window boundary.
4. Profile target verification separately from drafting and recovery. Reconcile
   target dispatches, submissions and all 48 layer dependency records for each
   captured call against before/after counters. Join demand reads, completion,
   GPU work and buffer release using the existing timeline checks. Preserve
   overlap; no per-kernel waits solely for measurement. Never label unexplained
   GPU-idle time as an SSD wait without the matching dependency evidence.
5. Retain 1,460 target slots, 32 draft slots and equal checkpoint capacity.
   Admit the profiler's bounded 512MiB allowance within the same 12GiB total.
   Clear each captured call after draining its users; reject truncation,
   unaccounted dispatches or oversized traces. Existing limits of 20,000
   dispatches and 32MiB serialized data per captured call remain the starting
   bounds. Report physical footprint and trace ownership separately.

Require the same AC, thermal, zero-compression and unchanged-swap conditions.
Use at most one bounded capture per selected workload initially. A failure
preserves partial evidence and identifies the missing prerequisite; it does
not justify repeated unchanged qualification attempts.

## Choose one intervention

Rank opportunities by exposed dependency time and complete-cycle cost. Expert
application bytes, summed GPU work and isolated kernel timings are supporting
measurements, not additive potential savings. A trace may motivate a scheduling,
I/O or kernel experiment only after identifying the concrete blocking work and
the smallest change that could remove it.

Apply the existing quick loop: exact operator/state checks, a short local timing
screen, one fresh normal-request pair with an explicit rejection threshold,
then reverse order and longer workloads for survivors. Retain failed results
in the ledger. Five paired repetitions, 2K/4K prompts, append latency, 7K reporting
and sustained coding remain required for promotion.

An adaptive width policy is a separate candidate. These measured prompts can
motivate it but cannot both calibrate and qualify it. Charge its exploration,
checkpoint and draft-maintenance costs, use only completed prior-cycle evidence,
and evaluate on held-out prompts against fresh fixed-width controls. A policy
does not remove the target-verification floor in the current LRU measurement.
