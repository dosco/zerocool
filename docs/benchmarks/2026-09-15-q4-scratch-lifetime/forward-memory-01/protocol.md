# Separate memory and ownership capture after blocked validation runs

Declared after forward-01 and forward-02 stopped at the memory gate, and after
batch-01 completed with no_clear_arrival_gain. Preserve all three original
reports and their decisions. This follow-up does not resume, replace, qualify,
or reinterpret those timing experiments.

Both blocked forward processes passed their native exact-output checks. They
already held about 28MiB of compressed pages before the first measured check
arm; some early decompression occurred and then the counters stabilized. Both
ended with the same charged Metal bytes as the batch process after scratch
release. Existing observations do not identify the compressed pages or establish
a native ownership leak, validation overhead, or a timing consequence.

Run exactly one independent native forward-scope trace process using the same
binary, sources, artifacts, prepared records, CPU oracle, 48 passes, eight expert
slots, two hits/six misses, explicit range invalidation, and scope geometry.
Disable Metal validation as the existing trace contract requires; enable its
existing command profiling, dependency events, and lifecycle snapshots. This
has one alternating reference/packed pair at each group cap, and is not a timing
sample or a new five-pair qualification. Preserve exact warm and per-pass output
checks and the complete final scratch release.

Use the same exclusive lease, source and asset seals, memory/host checks, 12GiB
ceiling, disk admission, 60-second process limit, and a 90-second stage limit.
Freeze this protocol, the one-off runner, prior blocked-check evidence, and the
existing model-state/control/oracle sources. Validate all 192 captured passes
and 1,536 routed expert records, preparation continuity, 1,296 scratch buffers
retained, growing active prefix, zero measured allocations, exact outputs, and
drained boundaries. A dirty resource observation stops this diagnostic too.

Report memory observations and ownership verification separately from original
qualification. Do not report trace ratios as speedups. Clean trace memory would
show that compression is not inevitable in this separately instrumented run;
it would not prove Metal validation caused the earlier compression, because
process timing and profiling also differ. Compression would warrant setup-phase
memory attribution, not a blind repeat. Keep normal_request_latency_qualified
and production_promoted false, and original_lifetime_stage_qualified false in
either outcome. No production implementation, allocator, precision, cache,
system limit, or qualification threshold changes are authorized by this result.
