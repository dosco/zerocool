# Attribute current target GPU work before another kernel experiment

The clean `merge-01` command capture covers five target calls, 20 verified rows
and 16 committed outputs (positions 33 through 48). It matches the entire
128-token numerical reference. Its largest mixed command class combines GDN,
two hyper-connection blocks, residuals and routing. Those command durations
cannot be attributed to a single constituent kernel.

Take one per-dispatch counter capture of the same request and middle window.
Keep the current source-copy producer, original Q4, packed Q8, full-replay
recovery, direct output, lazy ngram initialization, 1,460 target slots, 32 draft
slots, 512MiB trace allowance and 12GiB total admission. The only new native
change enables the existing per-dispatch counter sampling during the target
window. It splits compute passes, preserving command submission boundaries;
its times are not ordinary inference performance. No timing is pooled with
the command capture or the prior 2026-09-16 counter profile.

Require exact complete-request logits, output tokens, proposal/rejection path
and persistent target/draft state, complete 48-layer coverage per target call,
all dispatches and selected-expert lifetimes, valid counters and no truncation.
Retain the same power, thermal, zero-compression/decompression and unchanged
swap gates. Preserve a failed attempt without automatically repeating it.

Rank summed per-dispatch durations per committed output, explicitly charging
rejected verification rows. First answer how much of the mixed groups belongs
to the still-generic Q8 hyper projections (10240→320 and 320→10240), compared
with the already packed resident projections and GDN scan. Frequencies alone
cannot choose a kernel. Do not repeat the rejected single-token hyper fusion,
packed Q4 or unchanged GDN-only experiments.

If the generic hyper projections are material, screen packed loads on their
actual shapes and independently checked weights/inputs before integrating a
selector. Preserve arithmetic and reduction order. Require a newly declared
operator screen and fresh normal-request comparisons; no speed claim or
production promotion follows from this capture.
