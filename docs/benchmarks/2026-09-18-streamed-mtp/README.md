# Exact embedding storage in real speculative generation

The isolated C++/Metal producer now supports explicit `resident` and `rows`
embedding storage with real MTP at widths one and four. Both arms use the same
binary, artifact, arithmetic, 1460/32 expert slots, 12GiB admission and full target
replay on rejection. Production paths and defaults are unchanged by this stage.

The row provider removes 675,446,784 resident bytes and reserves 2MiB for its
bounded host cache: **642.15625MiB less planned memory**. The target and draft
share the provider. Each GPU call owns copied packed rows until completion.

## Completed evidence

**All six original-producer numerical pairs match exactly** in
[`numerical-05`](numerical-05/summary.json): width one, width four with first
accepted prefixes 1/2/3/4, an irregular seven-token tail, and immediate EOS.
This includes every actual draft proposal, all draft and verified-row logit
hashes including rejected rows, primed model state, committed state boundaries,
and final target/draft buffers. Target outputs also match the independent oracle.
Some resident and streamed diagnostics contain compression; this completes
numerical diagnosis, not clean full-model qualification.

One pair has different initial draft-cache eviction hashes despite identical
model state and outputs. Numerical checks now record this difference separately:
inference must be invariant to cache placement. Timing still requires identical
starting target and draft caches. The original `numerical-04` failed status is
preserved, with the correction declared in the
[cache-state protocol](cache-state-protocol.md).

The first fresh normal timing attempt, [`screen-01`](screen-01/summary.json),
completed both processes with clean memory and host conditions. Their physical
peaks are approximately **9.734GiB resident and 9.107GiB streamed**. However,
their starting draft-cache hashes differ, so **no timing comparison is accepted**.
These independent memory observations are consistent with the planned reduction;
the failed pair does not qualify a paired memory or latency result.

## Deterministic benchmark warm-up

A separate native producer now uses the existing bounded batch scheduler during
draft prompt priming. It retains up to 32 leases until their GPU users finish,
so read-completion order cannot change which slots are eligible for eviction.
It returns to the original completion-driven scheduler before generation.
Both storage arms use this setup, and fixed-priming counters must remain unchanged
during generation. Cache equality remains required for every new comparison.
This is benchmark setup, not a claimed prompt-speed optimization.

The replacement producer completes all six numerical pairs in
[`fixed-numerical-04`](fixed-numerical-04/summary.json), including identical
initial target and draft caches. The independent
[`priming-audit-01`](priming-audit-01/summary.json) compares all twelve processes
against the original producer: proposals, logits, state, boundaries and admission
match exactly. Some full-model diagnostics contain compression; clean full
correctness qualification remains open. The earlier memory and thermal stops
are preserved as incomplete attempts.

Its ten real embedding checks pass with Metal validation and clean resources.
They cover repeated IDs, vocabulary boundaries, 128-row gathers, FIFO eviction,
eviction before GPU submission, failed reads and owner destruction.

The fresh normal [`fixed-screen-01`](fixed-screen-01/summary.json) is complete:
two alternating pairs per width, 64 generated tokens, and all eight processes
have clean host/memory observations and identical numerical results. Initial
target and draft caches match within every pair. The execution
[capsule](capsule-protocol.md) preserves the linked experimental executable
despite unrelated changes to the shared build.

| MTP width | Resident / rows tokens/s, pair 0 | Resident / rows tokens/s, pair 1 | Geometric latency ratio, rows/resident |
|---|---|---|---|
| Four | 4.003 / 4.152 | 3.914 / 3.771 | 1.00034 |
| One | 4.032 / 4.236 | 4.326 / 4.105 | 1.00166 |

Peak physical memory falls by **644–647MiB**. Average latency is essentially
unchanged; neither width demonstrates a speedup. These are separate preliminary
comparisons on one short coding workload, without confidence bounds or long-
context qualification. No historical timings are reused.

The evidence query tool now independently recomputes storage comparisons,
verifies their numerical prerequisite, and keeps widths separate. It also
withholds raw-run throughput/opportunity estimates when resource measurements
are missing or disturbed. Production defaults remain unchanged.
The independent evidence query recomputes both completed latency ratios and
their memory savings. It keeps clean correctness, production promotion and
normal-request acceptance unqualified. The earlier
[second audit](review-02/summary.json) remains a record of its original ten
attempt seals and source snapshot, not an audit of the later completed screen.

## Next experiment

`--resume` verifies source seals, producer, workloads and resource checks before
reusing numerical samples. Timing samples are never reused. The replacement
producer requires new native evidence, so the original matrix cannot substitute
for its validation.

Storage-cost measurement is complete. The
[complete cache-window trace](../2026-09-19-horizon-cache/protocol.md) now asks
whether the saved memory can reduce expert rereads at verifier widths four and
eight. It checks trace-on versus trace-off arithmetic, reproduces native CLOCK
decisions, and only then simulates larger capacities. It changes no production
cache size. The compressed numerical diagnostics still cannot authorize promotion.

The query interface supports `compare` with `--control resident --candidate rows
--change embedding_storage`, `cycles`, `opportunity` and `next`. An unfinished
screen remains unfinished; differing widths are separate comparisons.

## Why a larger cache remains a separate experiment

The existing historical four-token traces replay exactly, including native slot
states, acquisition order and leases. Extending their fixed-order CLOCK simulation
gives only about 3.4–3.6% fewer misses at 1909 slots, compared with about
14.3–14.6% at 2048 slots. These are **simulated application reads** over an older
16-token perfect-proposal continuation, not current eight-token or real-MTP
latency. The original 1536-slot/SLRU decisions remain unchanged.

Current streamed allocation arithmetic admits at most 1915 slots at width four
or 1909 at width eight. A hypothetical reduction of generation workspace from
512MiB to 128MiB would admit 2060 / 2054 slots respectively, including the existing
driver reserve. That workspace transition and cache growth are **not implemented
or qualified**. Capture current complete lease traces before testing
phase-specific workspace or admission bypass.

The reproducible offline analysis is
[`cache-budget-verified.json`](cache-budget-verified.json), generated by
`scripts/qwen/analyze_streamed_cache_budget.py`. The earlier exploratory
`cache-budget-analysis.json` remains separate; its calculations agree.

Protocols: [original](protocol.md), [numerical/preliminary boundary](diagnostic-protocol.md),
[cache-state invariance](cache-state-protocol.md), [fixed draft priming](fixed-priming-protocol.md).
Raw attempts: [thermal admission](validation-01/summary.json),
[fixture thermal admission](fixture-01/summary.json),
[clean fixtures and compressed resident](validation-02/summary.json),
[diagnostic thermal admission](numerical-01/summary.json),
[two exact pairs and power transition](numerical-02/summary.json),
[thermal pause](numerical-03/summary.json), [cache-state stop](numerical-04/summary.json),
[complete original numerical matrix](numerical-05/summary.json),
[rejected initial timing comparison](screen-01/summary.json),
[replacement fixture and memory admission](fixed-validation-01/summary.json).
