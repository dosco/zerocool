# Measure the cache working set before changing capacity

The completed streamed-MTP storage screen saves 644–647MiB of physical peak
memory with essentially unchanged average latency in two alternating pairs per
width. Keep the established 1460 target / 32 draft slots as the control. No
production or 5-token/s claim follows from that short screen.

Capture the complete target cache lifecycle for the known 64-token continuation
at verifier widths four and eight, using four-row compute tiles and exact
streamed embeddings. This is an instrumented perfect-proposal diagnostic, not
real draft generation or a throughput measurement. Keep the 12GiB total budget,
8192 context, quantization and arithmetic unchanged. Reserve 32MiB for trace
bookkeeping in both capture-on and capture-off arms. Write bounded traces with
uncached I/O; fail if any event or snapshot is omitted.

One native producer supplies on/off controls for each width. Use deterministic
target and draft priming. Check every full logit, all persistent state and the
complete cumulative router history between the two arms. Earlier reports supply
only the fixed continuation and numerical oracle, never comparison timings.
Run the existing low-memory embedding/kernel fixtures before model capture.
Require clean host/memory observations for any offline candidate selection.

Record all acquisitions, slots, victims, pins/releases, routed token/expert
positions and drained cache snapshots. Exclude draft-cache events explicitly.
Reproduce every native CLOCK decision, snapshot and application-read count before
simulating 1460, 1909 and 2048 slots. Report each window separately. Track how
many distinct token rows use each expert, so any later bypass policy has evidence.
Counterfactuals preserve the recorded admission/lease order; they predict neither
changed I/O completion order nor latency.

Do not re-run the rejected 1536-slot/SLRU experiments. A 2048-slot candidate needs
its own jointly admitted memory plan, including draft, checkpoint, trace reserve
when present, and driver headroom. Returning prefill-only workspace requires
explicit drain/release/resize and transition correctness checks. No cache growth
or workspace reduction is implemented or qualified by this capture.
