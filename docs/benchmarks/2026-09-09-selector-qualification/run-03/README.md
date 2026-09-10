# Fresh qualification after submission cleanup

This checkpoint records passed recovery, eight exact attention replays, and the
full-model boundary and retained-4K append comparisons. The 7K reference has
passed; its GPU-selector candidate was interrupted at the user's request after
roughly 72 minutes so short timing screens could run first. No completed 7K
candidate report exists. **No normal-request speedup or production promotion is
established.** The raw runner records `failed` with an empty error from
`KeyboardInterrupt`; this is an intentional interruption, not a numerical
mismatch. All completed evidence below remains valid for its original build.

Build:
`c4f983f03f28974dd2c4a935d6d5233f1b28a1f5e7cb4800fe7529a64c058c2b`.
The fixed 12GiB engine budget, checkpoint, workload and CPU/full versus GPU/full
comparison are unchanged. All evidence was generated afresh after the
[submission cleanup](../submit-cleanup/README.md); no earlier capture was
relabeled or imported.

- Native tests: 51 / 6,931 assertions passed under Metal API and shader validation.
- Python tests: 84 passed.
- Real-model recovery: both cancellations occurred in their required windows;
  settings were restored, GPU users drained, and the same model ran again.
- Attention captures: eight exact cases at 4K, retained append and 7K,
  totaling 181,934,336 original bytes.
- Append: 4,096 tokens of computation reused, 128 new tokens ingested.
- 7K capture: 10.403GiB peak Metal allocation, 10.389GiB reported process footprint.
- Full-model boundary harness: both paths passed all nine checks, including fresh
  replay, failure and cancellation. All 48 layers' logits, routes and persistent
  state matched exactly. The [comparison](qualify/boundary/summary.json) binds
  both original reports. The candidate completed the model-recreation recovery
  checks that previously stopped at memory admission.
- Retained-4K append: both paths passed all nine checks and matched logits,
  routes and persistent state across all 48 layers.
  The [comparison](qualify/append/summary.json) binds both original reports.
- 7K reference: all nine state and recovery checks passed under both Metal
  validators. It retained the fixed 12GiB budget, reported 10.5277GiB peak Metal
  allocation, and approximately 10.42GiB physical footprint after continuation
  and fresh replay. The [original report](qualify/7k/reference.json) has SHA256
  `381eb760494b8ae5b7425cbd49caa28a60f0cd60035c348e4c46dba64b6f0965`.
  The candidate comparison is still outstanding.

The candidate's completed 4K run also records reclaimed prompt workspaces:

| Allocation | After continuation | After fresh replay |
|---|---:|---:|
| Resident weights | 4.994GiB | 4.994GiB |
| Persistent state | 0.532GiB | 0.532GiB |
| Occupied expert cache | 4.766GiB / 1,848 slots | 3.765GiB / 1,460 slots |
| Live prompt scratch | 0 | 0 |
| Process physical footprint | 10.418GiB | 9.418GiB |

Both snapshots retain an admitted capacity of 1,848 expert slots and report zero
memory-pressure resizes. The different live allocations reflect cache occupancy;
the released prompt scratch is not accumulating between phases. Values come from
[the candidate report](qualify/append/candidate.json).

A [process sample](diagnostics/boundary-reference.sample.txt) during the untimed
reference harness showed active Metal panel work and 10.5GiB physical footprint
(10.7GiB peak). The smaller resident-memory samples alone are insufficient to
describe Metal memory use. No timed performance result uses this diagnostic run.
The [candidate sample](diagnostics/boundary-candidate.sample.txt) showed active
expert work and 9.9GiB physical footprint (10.4GiB peak).

During the untimed 7K reference check, a [process sample](diagnostics/7k-reference.sample.txt)
showed active expert execution in the fresh-replay call, after resetting the
expert cache and rebuilding state. Four [direct physical-memory readings](diagnostics/7k-reference-physical.jsonl)
from `proc_pid_rusage` between 17:30 and 18:37 Pacific recorded approximately
10.49–10.53GiB in use and a lifetime peak below 10.68GiB. The process continued
doing work throughout those samples. Its completed reference report subsequently
passed. The 7K candidate comparison and normal performance qualification remain
unfinished.

The [7K candidate sample](diagnostics/7k-candidate.sample.txt) at 19:33 Pacific
also reached the fresh-replay call after initial ingestion and continuation.
Its [direct memory readings](diagnostics/7k-candidate-physical.jsonl) recorded
10.33GiB during initial ingestion and 9.75GiB during fresh replay, with a lifetime
peak below 10.59GiB at the latter reading. Candidate/reference state equivalence
has not yet been established for this 7K case.

[identity.json](identity.json), [summary.json](summary.json) and
[checkpoint.json](checkpoint.json) preserve the checkpoint and its provenance.
The complete recovery directory is copied with its original sealed inventory.
Capture reports are copied; the [capture inventory](capture-evidence-files.json)
binds the original tensor payloads retained at
`/repo/.cache/benchmarks/selector-qualification/2026-09-09-run-03`.
Both completed phase seals were verified before copying.

Capture runs are instrumented correctness checks. Their timing is not evidence
for the selector performance decision. The 7K candidate comparison, alternating
normal requests, decision gate and final profiling remain outstanding at this
checkpoint.
