# Reference decode diagnostic with observed compression

Declared before execution. This is a new diagnostic protocol; the previous
Q4 request qualification runner, zero-compression gates and blocked reports
remain unchanged. No native source, arithmetic, precision or default changes.

Run one traced reference conversation using native build 51877, the pinned
mixed artifact and prepared Q4 experts, 1072 slots, core-cache residency,
immediate submission, scratch reuse and the existing 12GiB admitted budget.
Use the existing 72-token prompt and retained 128-token append, seventeen
outputs per phase. Compare output IDs and normalized work to the sealed
capture-02 reference, using its tokens only, not its timings. Reuse the sealed
compatible full-model operator/state proof; no new state comparison is claimed.

Enable existing decode-only command profiling, dependency JSONL and per-token
observations, with Metal validation and hardware counters off. Require all
32 decode forwards, 101600 dispatches, 1536 layer passes and 15360 selected
expert records. Preserve every captured token, including disturbed intervals.

Bound execution to one inference process, 90 seconds; allow 150 seconds for
the stage including admission/preparation. The existing cancellation escalation
allows up to 45 seconds of additional draining. Offline analysis and sealing
also occur outside the work deadline. No automatic retry or packed variant.

Native allocation admission, disk/evidence limits and the single-GPU lease
remain active. Completed-run acceptance requires observed physical lifetime
peak within 12GiB, compressed lifetime peak at most 512MiB, no observed swap
growth, complete memory gauges and non-resetting cumulative counters. Swap
decreases and falling compressed bytes remain valid observations. The 512MiB
allowance is an operational diagnostic bound (one twenty-fourth of the engine
budget), not a measured threshold for safety or harmless performance impact.
The wrapper evaluates memory at saved boundaries and lifetime peaks after
inference: it does not cancel live at 512MiB or provide continuous host sampling.
Over-limit or incomplete runs remain blocked/incomplete, with raw files retained.

Report per-token compression/decompression changes and phase gaps separately,
plus observation-call cost without subtracting it from measured time. Rank
exclusive whole-token GPU/wait categories first. Within GPU work, preserve
complete stage-set and layer-set command classes; mixed commands stay mixed.
Read service, command sums, boundary gaps and CPU times overlap those categories
and must not be added to them or described as recoverable latency.

The output is a ranked hypothesis report and one proposed bounded experiment.
Choose the intervention only after reviewing the reference trace. Consider
packed Q4 only if that trace supports it. No paired speed claim, confidence
qualification, production promotion, long-context or coding-quality acceptance
follows from this reference diagnostic, even if memory happens to be clean.
