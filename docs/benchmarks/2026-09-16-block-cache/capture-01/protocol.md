# Four-token expert-demand capture and offline replay

Capture the same 72-token prime and sixteen fixed continuation tokens at width4
in two fresh processes, with 1,072 then 1,460 target slots. Use the unchanged
12GiB budget, mixed artifact, prepared Q4 records, deterministic fixed-batch prime
and completion-driven verification. No timing result from this diagnostic
qualifies performance. Production source and binaries remain unchanged.

Instrument developer copies only. Capture every forward/layer boundary, router
selection, actual ordered expert admission, victim slot, pin and release, plus
complete CLOCK snapshots. Include all priming events; never reconstruct panels
as token-major demands. Limit capture to 100,000 events/32MiB and reserve a
conservative extra 32MiB trace workspace in admission. Trace I/O may perturb
completion order. Capture real order rather than assuming an unchanged schedule.

Compare complete row logits, persistent state, routes and starting cache with
the same-capacity clean width4 process from the preceding capacity screen.
Prior timings remain excluded. Require current clean memory/power observations.
Hold the exclusive GPU lease; native host checks precede each process. Bound
each process to 120 seconds and the stage to 300 seconds. Preserve incomplete
or resource-disturbed attempts; they cannot choose a performance candidate.

Replay the captured native CLOCK cache first. Require every hit/miss, slot and
victim decision, pin count, layer count, complete ordered cache snapshot/hash
and raw report boundary to match. Prove all reads have completed at release,
all layer boundaries drain leases, all selected experts are executed once per
layer pass, and native hit-first ordering matches observed cache residency.
Reject truncated/reordered events, unmatched leases, incomplete forwards,
changed routes, or a disagreement with live counters.

Only after that succeeds, compare CLOCK and the existing 75%-protected SLRU
policy at 1,072, 1,460 and 1,536 slots. Preserve the actual acquire/pin/release
order from each capture for every counterfactual. Alternative read completions,
GPU schedules and policy-dependent hit-first ordering are not simulated. Report
both source-order curves separately, including disagreements. These estimates
predict application reads only, not latency, device traffic, quality or tokens/s.

Reserve the existing planned draft dense weights/state/scratch plus 128 draft
expert slots when screening capacity. Require at least 128MiB additional planned
headroom inside 12GiB. Draft allocations and traffic remain planning allowances;
no draft forward pass is implemented or measured here.

Select at most one new GPU experiment. First consider SLRU at unchanged 1,460
slots if it reduces simulated decode misses by at least 10% relative to CLOCK
at 1,460 in both captured orders. Otherwise consider CLOCK at 1,536 with the
same reduction floor and memory gate. Do not combine policy and capacity changes
or tune protected fractions in this stage. If neither passes, record that no
candidate was selected; do not infer a latency benefit from a small read saving.
A selected candidate still needs fresh exactness, recovery and paired native
timing under the original 5-token/s verifier gate before draft integration.
