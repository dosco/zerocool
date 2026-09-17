# Bounded expert temporary reuse

Implemented and numerically validated in an isolated native C++23/Metal build.
The [follow-up with lazy ngram initialization](../2026-09-16-mtp-ngram-init/README.md)
now has a clean ordinary comparison: **5.0355 versus 4.9727 tokens/s** over sixteen
tokens. A 70.37% allocation reduction yields only 1.2465% lower cycle latency,
failing the predeclared 2% advancement gate. The candidate remains isolated;
no reverse pair or longer qualification was run. The first timing control here
switched to battery/Low Power Mode, and two later attempts stopped at memory
admission before model loading. Those unfinished attempts stay separate.

## Change

Four-token target verification can now reuse routed-expert temporaries in two
1MiB pools. Each group submits before closing its pool scope, preserving the
coordinator's completion and expert leases. A pool can only be reset after its
last GPU user finishes. The existing two-group/32-lease limits remain in place.
Persistent activations, expert contributions, state and hidden captures stay
outside these scopes. Errors and cancellation drain GPU and I/O users. Switching
between four-token verification and single-token rejection replay releases the
old pools; a whole four-token forward is never retained in one scratch arena.

Both arms explicitly reserve the extra 2MiB within the same 12GiB admission,
1,460 target slots, 32 draft slots and 8,192-token state capacity. Weights, kernels,
reduction order and sampling are unchanged. The off/on switch exists only in the
developer executable. Production sources and binary are unchanged by this stage.

The source-copy builder, native fixture and runner are
`scripts/qwen/build_mtp_expert_scratch.py`, `mtp_expert_scratch.hpp`,
`probe_mtp_expert_scratch.cpp` and `screen_mtp_expert_scratch.py`.
The candidate is `.cache/mtp-expert-scratch-build-02/probe-mtp-forward`, SHA256
`aa2f5b44bb07bbe70142ab4bc6583b969d2f86920cff7a219f745860e17ae245`.
The original model's class layout remains unchanged when linking the copied
implementation with the native archive.

## Checks completed

- [Native fixture](fixture-01/summary.json): unchanged real Q4 records at 1, 2,
  3 and 4 rows; reversed read completion, forced eviction, all hits, cancellation,
  injected read/encoding failures, delayed GPU completion and complete pool
  release. Metal API and shader validation are enabled.
- [Full-model check](validation-02/summary.json): an eight-token forced-rejection
  continuation matches all full-vocabulary logit hashes, committed tokens,
  intermediate recovery boundaries, future proposals and final target/draft state.
  Both processes have clean memory and host observations, with peak physical
  footprint about 9.82GiB. The control also exactly matches the previously
  qualified continuation producer.
- **33 focused MTP tests pass.** Producer fingerprints, seals, sample observations
  and pair metrics are reconstructed in the [prerequisite audit](audit-prerequisites-01.json).

During the correctness workload, physical allocation count falls from 21,004
to 10,823, with reuse increasing from 6,402 to 16,583. These totals include draft
and rejection recovery work. Metal validation is enabled, so their elapsed times
are **not performance evidence**. The original 26,932-allocation target profile
uses a different, sixteen-token scope and must not be used as this pair's control.

## Preserved stops and next action

- `validation-01`: the control completed numerically, but its compression peak
  reached 82,722,816 bytes. No candidate ran. The fresh `validation-02` pair passes.
- `short-01`: memory stayed clean, but the control ended on battery power with
  Low Power Mode on. No candidate ran.
- `short-02` and `short-03`: after reconnecting power, admission saw 13.321 and
  13.407GiB available, below the unchanged 13.5GiB guard. No model loaded.
- `short-04`: fresh initial availability was 16.79GiB and host observations stayed
  clean, but the control reached 53.094MiB peak compression. No candidate ran.
  This motivated the separately validated lazy-cache baseline used by the later
  complete comparison; no eager timing is pooled with it.
- The two CPU-only setup diagnostics show roughly 184–198MiB of retained harness
  RSS, not the much larger fluctuation in system headroom. The complete setup
  measurement held near 13.2GiB available. No other application was terminated,
  no OS memory limit was raised and no benchmark requirement was weakened.
- The final [host snapshot](headroom-05.json) still reports AC power and Low Power
  Mode off, but only 12.461GiB available and thermal state 1. No further inference
  was launched under those conditions.

The subsequent [fresh pair](../2026-09-16-mtp-ngram-init/short-01/summary.json)
completed under the [predeclared protocol](protocol.md) and failed its 2% gate.
Do not resume missing rounds of this unchanged candidate. Follow the
[recorded next trial](../2026-09-16-mtp-ngram-init/next-protocol.md), which removes
redundant scatter copies while preserving the current reference arithmetic.
