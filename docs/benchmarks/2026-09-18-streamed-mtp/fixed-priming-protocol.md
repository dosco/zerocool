# Fix draft warm-up before measuring embedding storage

Registered after screen-01, before constructing the replacement producer.
Both normal processes in that attempt have clean memory and host observations,
but their initial draft-cache eviction hashes differ. No timing pair is accepted.
Preserve its raw reports; do not rerun the unchanged producer until caches happen
to match, and do not reuse its timings in a new comparison.

Use a separate producer that retains the exact same embedding storage arms,
arithmetic, capacity, memory budget and generation path. During prompt priming
only, the draft holds fixed batches of at most 32 expert leases until all their
GPU users complete. Reuse the existing batched scheduler: experts are acquired
in selected order and ready experts may execute first, with bounded command
submission. Its errors/cancellation path drains GPU and read users before
releasing leases. The measured generation phase retains the original
completion-driven scheduler. Report priming counters before/after generation and
require them to be unchanged, with the fixed-priming scope inactive.

The original target priming already uses fixed lease batches. This change applies
the same cache-seeding discipline to the draft; it is benchmark setup, not a
claimed prompt-latency optimization. A normal pair still requires identical target
and draft eviction hashes before generation. Reject a remaining mismatch.

Run new operator fixtures and the complete six-pair numerical matrix for this
producer. Earlier native samples cannot satisfy its same-producer prerequisites.
The established numerical oracle remains numerical-only. Retain the existing
explicit diagnostic allowance if a clean full-model attempt records compression;
fresh timing still requires zero compression/decompression, unchanged swap,
nominal thermal state and AC power. Then perform the original two alternating
64-token storage pairs per width. No production adoption or historical timing
reuse is permitted.
