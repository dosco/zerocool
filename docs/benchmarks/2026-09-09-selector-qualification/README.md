# Selector qualification implementation

Disk space is now sufficient. [Run 02](run-02/README.md) passed recovery and
attention replay, then stopped at memory admission inside the state harness.
Investigation found and fixed [Metal buffer retention](submit-cleanup/README.md).
[Run 03](run-03/README.md) has passed recovery, all eight attention replays, and
the full-model boundary and retained-4K append comparisons with exact logits,
routes and persistent state. Model recreation now succeeds. The 7K reference
also passed all nine checks. At the user's request, the long-running 7K candidate
was interrupted to move timing screens earlier; no completed 7K comparison exists.
Passed evidence is retained. Normal performance and profiling remain outstanding.
There is no qualified speedup yet.

The first [capped cached screen](triage-01/README.md) stopped at memory admission
before inference: roughly 2.57GiB was available against an 8.70GiB minimum. The
comparison still requires its fixed 12GiB budget. Disk space is no longer the
blocker. No timings were produced. The reusable command was then [tried live](triage-02/notes.md)
and correctly reported `resource_blocked` in 7.86 seconds, again before inference.

After memory cleanup, the [September 10 capped attempt](../2026-09-10-selector-triage/README.md)
passed the full 12GiB admission, then reached its 600-second deadline without
emitting timings. It shut down cleanly and remains inconclusive. There is no
current memory-admission blocker from that attempt, and no performance result
that permits advancing to normal screening. The new short request-screen
helper is implemented; all 124 Python tests passed at that stage.

[Native phase progress](../2026-09-10-phase-progress/README.md) is now implemented.
The final build passes 52 native tests and 131 Python tests. Its real-model
progress check was blocked before inference by current memory admission
(about 7.44GiB available to the engine). That attempt cannot explain the earlier
timeout; it preserves an explicit blocked result and launches no inference.

The reusable [triage command](../../../scripts/qwen/triage_selector.py) now enforces
a 600-second deadline, five exact cached 4K pairs and explicit blocked/inconclusive
outcomes. It can recommend a normal-request screen but never launch one or full
validation. See [the revised workflow](../../qwen_selector_qualification_stage.md)
and [tooling verification](triage-tooling/verification.json).

## Original disk-blocked attempt

The [bounded qualification runner](../../qwen_selector_qualification_stage.md)
is implemented. The real-model experiment is **resource blocked and unfinished**.
There is no new speedup, fidelity or production-promotion claim.

The runner freezes source, binary, artifact and workload identities; validates
whole-pair resume against original raw reports; checks exact cancellation trace
windows; and enforces the prescribed screen, paired-comparison and profiling
sequence. Negative valid results finish the experiment. Missing or inadmissible
measurements do not pass it.

Verification on 2026-09-09:

- All **50 native tests / 6,888 assertions** passed with Metal API and shader
  validation, with no skipped tests and a 600-second external timeout.
- All **83 Python tests** passed, including source changes during subprocess
  execution, invalid screen evidence, complete-pair resume, corrupted/extra
  files, capture geometry/hashes and blocked-run reporting.
- The all-phase runner froze its identity, then refused disk admission before
  launching any model workload. Available space was **4,115,861,504 bytes
  (3.83GiB)**; the initial requirement is **7GiB**, leaving a 5GiB reserve and
  at most 2GiB of new evidence. No files were removed or limits changed.

The native source fingerprint remains
`857ee812709cba3817cfef46d431060744a529bd2d3fa802fa3ece6dd430cb5c`.
Executable and tooling hashes are separately frozen in [identity.json](identity.json).
[summary.json](summary.json) is the original blocked report copied without
modification; [verification.json](verification.json) records the checks and
source directory. Earlier benchmark reports retain their original identities.

These were the limitations of the original disk-blocked attempt. Subsequent
runs above supersede its progress status while preserving its raw evidence and
identity. Native memory admission must still independently approve the fixed
12GiB budget. Use the capped triage entry point for the next candidate screen;
do not automatically resume the old all-phase run.
