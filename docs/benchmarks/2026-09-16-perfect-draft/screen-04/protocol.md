# Reproducible-prime revision — declared before execution

The clean screen-03 matched two-token logits, routes, recurrent/attention state
and rejected-prefix recovery, but its exact starting expert-cache hash differed
from serial. No timing comparison ran. This revision changes only setup scheduling:
use the existing fixed batches of at most 32 expert leases during prefill, retaining
each batch until its GPU and reads finish. That makes CLOCK admission and release
order deterministic. The completion-driven coordinator is restored on every
forward exit and remains enabled throughout measured decode for every width.

Keep the original 12GiB/1072-slot budget, exact starting-cache gate, arithmetic,
output/state checks and two-round 5 verified tokens/s threshold. Do not pool old
attempts. Require new prime logits/state/routes to match the saved clean screen-03
serial prime; its cache hash may differ because setup scheduling intentionally
changed. Each process must still start measured work with the same full cache hash.
The producer binds the source-copy scheduling change and the runner rejects
a changed setup identity or completion-driven decode being left disabled.

Run the native power/thermal preflight before every model process. Priming and
preflight time remain outside the verifier timing. No claim about normal prefill
speed follows. The prior blocked/failed reports retain their original status.

The unchanged underlying workload and acceptance rules follow. References to
original priming configuration below are superseded only by the setup change above.

# Perfect-draft verifier feasibility screen

Declared before GPU execution. The user authorized this isolated investigation
of exact speculative decoding; production v1 still has no draft model or
speculative interface. Test whether the current multi-token verifier can cross
5 verified tokens/s when proposals are free and always accepted. This is an
optimistic test of this implementation, not an upper bound on future designs.

## Fixed experiment

Use the pinned mixed-4/8 artifact, unchanged Q4 expert records, native base
fingerprint 51877f96f015386dd642fd4e63fbc39ff2e75c1156f5ae1bcba764f546963594,
12GiB total budget, 1072 expert slots, core/cache residency, CLOCK, eight readers,
ready group4, panel512/chunk128, SIMD router, Q8 rows2, reference Q4, immediate
submission and decode scratch reuse. No cache-capacity or precision adaptation.

Build a separate executable from developer copies of model.cpp and storage.cpp.
The production binary remains unchanged. The copy enables existing vectorized
kernels for 2/4 consumed tokens and returns every vocabulary-logit row. Prime
with the original configuration. Hash eviction-relevant expert cache state at
the prime boundary. Record actual binary, copied sources, harness and original
native identities; the native base fingerprint alone does not identify this
executable.

Use the sealed request-context capture-02 normal-A report only for its 72-token
prompt and 17 greedy output IDs. Its containing stage remains resource-blocked;
no earlier timing qualifies or contributes to this experiment. Consume the
first16 output IDs, each predicting the next ID. A width K block consumes K
inputs and earns K verified tokens; no bonus token is credited.

Every process starts a fresh model and repeats the same prime. Require matching
prime logits, all state buffers, routes and exact expert-cache state. Allocate
the same capacity-four host checkpoint in all arms and touch every page before
priming. Copy it only for widths2/4,
within timing. Save complete recurrence/convolution state and only the appended
KV/index rows, plus detached metadata. Charge actual host allocation sizes and
a fixed bound for all live logit vectors against the same12GiB plan; never retain
old GPU owners in a shallow state copy. Cap checkpoint allocation at128MiB.

## Correctness before timing

Run widths1,2,4 separately with Metal API and shader validation. Verify four
consumed tokens, compare all vocabulary logits byte-for-byte at every row and
all persistent state plus exact router choices at matching prefixes. For each
candidate, change its final proposal and prove earlier logits unchanged. Restore
zero accepted tokens and compare full state with the original prime. Restore
and replay each possible accepted prefix serially; compare its state and logits
with the independent width1 process. Return to the correct block and prove
replay equality. These are greedy model-state rollback checks, not production
Session/pending-token/RNG rollback qualification.

Then run two alternating rounds in order1,2,4,4,2,1, each with16 consumed tokens.
Fresh model setup/prime are reported separately. Time checkpoint copies, full
forward and greedy acceptance. Hashes, stats and report serialization are outside
that interval. Full state/route digests and report writes occur at the same
consumed-token boundaries4/8/12/16 in every timing arm; validation retains all
serial prefix states for rollback comparisons. Actual route capture is enabled equally in all arms and is timed
instrumentation. No actual draft, rejection, long-context or normal-session
throughput claim follows from this screen.

## Gates and limits

Require every candidate paired wall time to improve and both candidate runs to
reach5 verified tokens/s. Two pairs only support an early directional decision,
not confidence-bounded qualification. Stop before a real draft implementation
if neither verifier survives. Never relax exactness to pass this gate.

Bound the whole GPU stage to600seconds; validation processes to150seconds and
timing processes to90seconds. Hold the exclusive GPU experiment lease. Preserve
partial results and failed attempts. Stop on changed identity, missing evidence,
compression, decompression growth, swap changes, abnormal thermal/power state,
or failure of fixed memory admission. Check process footprint and lifetime peaks
at block/lifecycle boundaries, with post-destruction measurements. These are
bounded observations, not continuous attribution of all machine activity.

The short prompt covers a small attention boundary but not sparse attention
beyond2048, EOS, long history, cancellation, actual draft acceptance or sustained
coding sessions. A promising result authorizes designing the next experiment;
it cannot promote a production optimization.
