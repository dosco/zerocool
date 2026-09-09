# Prompt-memory reclamation and exact matrix experiments

This stage remains benchmark-only. The pinned mixed 4/8-bit target and Q4
control retain their existing bytes, router choices, reduction order, and state.
The admitted budget never increases; production defaults remain reference paths.

## Memory ownership

`--phase-memory fixed|reclaim` defaults to `fixed`. Reclamation requires
`--prefill-pipeline double`. Inspection reports prompt and generation plans.
Both plans retain the same context, panel, general scratch, recurrent/attention
state, ngram, driver, and grouped-decode reserves. Only the two prompt-workspace
reservations are reclaimed. An explicit expert-slot cap bounds both plans.

`Model::prepare_ingest(remaining_tokens)` and `finish_ingest()` enclose a whole
ingestion. `Session` uses this scope across every panel, including sample-free
priming. Direct `forward()` callers receive an automatic single-call scope;
callers feeding multiple panels should explicitly enclose the entire sequence.
A short append that never enters the panel path, or a zero-token prefix reuse,
does not shrink the cache or create workspaces.

Entering panel ingestion drains work, shrinks CLOCK while preserving survivors,
and reaps retired resources before allocating pool storage. Finishing ingestion
drains GPU users, drops pool ownership, reaps retirements, and grows cache
capacity lazily. Surviving external buffer views retain their allocation charge.
Errors invalidate partially updated session state. Cleanup drains all users
before the error returns. Emergency pressure reductions are counted, are never
undone by reclamation, and invalidate controlled benchmark comparisons.

Reports expose both plans, bounded transition history (last 256 events plus a
monotonic sequence), cache occupancy, evictions/survivors, transition duration,
workspace capacity/allocated/high-water/unused bytes, reuse and allocation counts.
Workspace backing allocations are separate from deep snapshots and do not change
the residency policy. Transition latency is included in ingestion/first-token time.

## Exact matrix candidates

`--affine-rows 1|2|4` selects independent output accumulators sharing input
loads and affine bias sums. Prompt experiments support Q8 with token tiles 4/8;
T1 uses two rows for either Q4 or Q8. Unsupported multi-token Q4 shapes retain
the existing kernel. `--gate-pair off|on` shares T1 gate/up input loads and bias
sums while retaining independent dot accumulators. Grouped experts use the same
arithmetic and preserve router-assigned destination positions.

These controls require candidate mode. Per-shape policies may specify
`output_rows` and `gate_pair` alongside the existing tile. Uncovered shapes use
the original path. Policies remain bound to native build and artifact identity.

Fixture replay includes the four Q8 prompt candidates and the two applicable
T1 candidates. Every result must match the reference bitwise. Shape selection
requires five complete paired repetitions and improvement for every captured
input of the shape; unresolved comparisons may be extended once to ten.
Isolated winners are not production promotions.

## Capture and qualification

`--capture-phase`, `--capture-layer`, and `--capture-operator` filter
`--operator-capture`. Each run remains limited to 32 cases and 256MiB.
`scripts/qwen/capture_phase_inputs.py` runs serial captures across early/later
GDN, sparse attention, append, decode and routed experts, then replays each
fixture. Its tuning and held-out modes use separate input windows. Requested
coverage with no fixture fails explicitly. Grouped profiling now labels routed
experts and records group size, expert IDs, projection shape and destinations.

`benchmark_exact.py --comparison-purpose experiment` permits a frozen optimized
baseline. The default `promotion` purpose continues requiring the original
control in paired mode. Fixed mode rejects resizing; reclaim mode verifies
complete declared transitions, workspace release and unchanged hard budgets.
Diagnostic replay and profiled timing cannot qualify normal-request latency.

Start with `docs/benchmarks/2026-09-08-memory-compute/memory-configs.json`, then
compare kernels separately with `kernel-configs.json`. Both use 12GiB, panel
512, microchunk 128, eight I/O workers and ready-group two. Run one GPU/model
process at a time. Five alternating pairs use 2K/4K plus 256 output tokens and
a 128-token append to sample-free retained 4K. Require a paired 95% improvement
bound in at least one latency metric and no bound above 1.03 elsewhere.
Inconclusive results do not pass; do not repeatedly sample until a result wins.

Real-model release checks must include both artifacts, irregular panels,
2053+129 and 4096+128 state replay, 7K reporting, and a 20-minute coding recovery
workflow. Missing assets do not pass. Product gates remain 5 tokens/s at 2K/4K,
60s initial 2K first token and 10s retained-history append. Incremental screening
results must not be represented as meeting those targets.
