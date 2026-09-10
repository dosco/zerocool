# Cached replay phase progress

Phase reporting is implemented in native cached replay and connected to the
selector qualification runner. It records model load, pipeline setup, reference
history priming, snapshot, reference continuation, expert/ngram preload, candidate
setup, each warmup and timed arm, and report writing. History records identify
the completed token prefix and active chunk; timed arms identify their variant,
repetition and restore/forward/verification step.

`--cached-progress FILE` writes a new JSONL file and flushes every event. Existing
traces cannot be overwritten. Progress writes remain outside the measured
forward interval and add no GPU waits. Completed phase durations and unfinished
work remain distinct. The reader rejects inconsistent identities/sequences and
does not accept an incomplete final line as an event.

Use [the documented command](../../qwen_selector_qualification_stage.md#cached-phase-progress)
to read a live or stopped trace. The outer cached triage summary also includes
the last phase after a timeout, cancellation or error. Progress describes the
diagnostic; partial pairs cannot establish correctness or performance.

## Verification

- **52 native tests / 6,962 assertions passed**, with Metal API and shader
  validation and no skipped tests: [native output](native-tests.log).
- **131 Python tests passed**, including live native load failure, existing-file
  preservation, truncated traces, phase durations, changed identities, invalid
  completion and timeout-summary integration: [Python output](python-tests.log).
- A saved native failure fixture confirms that the file contains `model_load`
  followed by `failed` when a deliberately missing checkpoint stops construction.
  [The trace](load-failure/progress.jsonl) and [reader result](load-failure/summary.json)
  are real tool outputs. This is a failure-path check, not model execution.
- A real 4K diagnostic was attempted with a 120-second total cap. It stopped at
  admission after **7.55 seconds**, before launching inference. Available engine
  memory was about **7.44GiB**, below both the initial 8.70GiB minimum and the
  required fixed 12GiB budget. Three bounded metadata admission attempts are
  preserved in [the original summary](admission-attempt/summary.json) and its
  adjacent reports. No native progress trace was created for that attempt.

The final native fingerprint is
`1134107188e63a9bc47491ca79500e6b749a49f8a133b6f65ed6cef0e363e4cb`.
The blocked admission check used the earlier instrumentation build recorded in
its own identity; it was not relabeled after final consistency fixes. Tests and
the saved native failure fixture cover the final build.
[Verification hashes](verification.json) bind both sets of evidence.

Full-model progress through priming, warmup and paired timing remains unobserved.
The earlier ten-minute timeout still has no phase trace. No speedup, numerical
qualification of this rebuilt model, or production promotion is claimed.
The next live diagnostic should use the same fixed budget when admission permits.
