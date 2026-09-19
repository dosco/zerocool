# Usability implementation and qualification

The server now binds before model loading and runs inference on one owning
worker. The new terminal client manages an owned child or connects to an existing
server. Both use the existing native session, not the experimental MTP path.
See [usage and validation](../../qwen_usability.md).

A subsequent [review](review/README.md) found and fixed premature stream success,
server replacement handling, failed-startup resource retention, and cancellation
evidence matching. Its reports supersede the initial implementation checks below
for the revised build; the earlier raw reports are preserved.

Qualification is **incomplete**. Model-free transport/PTY checks and an Aider
protocol check establish interface behavior only. The first native startup
attempt was stopped before submitting inference because the model process was
already compressed. No latency or real coding result is claimed from that run.

## Evidence

Final verification results and provenance are recorded in
[`verification.json`](verification.json), with the referenced logs and reports
in `raw/`. Additional development logs remain under `.cache/usability-stage/`.

- 72 native tests / 49,201 assertions passed with Metal API and shader validation.
- 7 transport tests / 476 assertions and 529 Python tests passed.
- Four PTY cases passed, including production startup failure, interrupted
  output, safe paste, exports, terminal restoration and child lifecycle.
- Sampled TUI physical footprint peaked at 3.61MiB in the model-free check.
- Aider 0.86.2 protocol check passed against the test-only executor.
- The build with `FREELLM_BUILD_TUI=OFF` passed.

The old server executable was preserved before editing at
`.cache/usability-stage/baseline/freellm`, from commit
`f26f813574ae87baaee26265c9213ca372aafd36`, SHA-256
`05495d0baf993b1f7f1bc4b9365753e88603909ac66db3309a1380ea8c7c519f`.

The guarded 12GiB performance screen requires 13.5GiB available memory. The
initial host had about 8.1GiB available; normal admission would reduce the cache,
invalidating that fixed-budget comparison. A separate 6GiB functional startup
attempt admitted about 5.5GiB, then reported approximately 2.6GB compressed in
the engine process. It was shut down without a model request. These are resource
observations, not performance measurements.
The final paired-screen preflight observed AC power, Low Power Mode off and
nominal thermal state, but only 7.22GiB available. It stopped at memory admission
and contains no request measurements.

## Pending gates

- Real-model plain/streamed requests, reasoning/tools, history edits, context
  limits, cancellation during ingestion/generation and subsequent recovery.
- Aider 0.86.2 read/edit/test/recovery on the real model.
- Two alternating paired old/new request screens with equal admitted memory,
  identical artifacts, prompts, sampling and outputs; investigate regression
  above 3% or ambiguous measurements.
- Separate actual TUI overhead; under 128MiB physical footprint, status p95
  below 500ms during inference, acknowledgement below 500ms with drain separate.
- A 20-minute interactive coding session without progressive memory or sustained
  swap growth. Scripted soak alone does not complete this gate.

Missing and resource-blocked checks remain unqualified. No inference speed claim
or default experimental execution change accompanies this implementation.
