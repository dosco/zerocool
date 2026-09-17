# Target-state recovery implementation and optimization tools

This stage is implemented as an isolated C++/Metal developer candidate. Clean
real capture/replay now passes; the complete full-model correctness gate and
performance screening remain unfinished. It is not promoted.

## Latest continuation

- [Full-model numerical diagnostics](numerical-diagnostic-01/summary.json) now
  match all five cases: forced accepted prefixes 1/2/3/4 and immediate EOS.
  The comparison covers all logits, generated tokens, later proposals and final
  target/draft buffers. Candidate recovery performs zero target forward calls
  and zero expert reads. Compression peaked at 31.531MiB; these are numerical
  diagnostics, explicitly ineligible for performance or clean-stage qualification.
- [Trial 09](trial-09/summary.json) stopped on 23.141MiB of control-process
  compression. A [two-boundary VM diagnostic](memory-diagnostic-02/summary.json)
  reproduced compression during prompt processing, with about 24MiB in
  `MALLOC_MEDIUM` swapped/compressed pages and none in the sampled Metal graphics
  category. VM categories do not identify individual C++ allocation owners.
- [Startup heap release](startup-heap-01/summary.json) did not qualify a fix:
  both runs recorded compression, the release API reported zero bytes reclaimed,
  and the candidate encountered 0.3125MiB of system swap growth. It remains a
  source-copy diagnostic, with production unchanged.
- [Early timing screen](early-screen-01/summary.json) completed its 64-token
  control, which hit 7.953MiB peak compression. The strict gate stopped before
  running the candidate. No usable timing pair or speed comparison resulted.
  The new early-screen tool requires clean real fixtures and an exact diagnostic
  pair; it preserves all timing gates and cannot qualify advancement.
- Correctness can resume at the individual clean run. The new resume tool
  rechecks seals, inputs, producer, native resource observations and numerical
  comparisons, then collects missing runs. Original reports remain unchanged;
  incomplete or disturbed runs and timing measurements cannot be reused.
  [Trial 10](trial-10/summary.json) reassembled and exactly compared a clean
  prefix-one pair without rerunning it. Its fresh prefix-two control encountered
  compression, leaving the stage incomplete. This retained progress does not
  qualify the remaining cases or a performance gain.

- [Trial 06 fixtures](trial-06/fixtures/summary.json) pass clean capture and all
  eight real-state prefix checks, including cancellation, failure, corrupted
  input and delayed completion. Peak capture footprint was 9.794GiB; replay was
  0.721GiB. Both report zero compression/decompression and unchanged swap.
- Capture now uses exclusive, bounded, uncached payload writes instead of a
  buffered stream. Ten memory boundaries identify pressure during snapshot
  writing, reference replay and final hashing. Trial 05 reached 165.27MiB of
  peak compression during capture; the changed writer's Trial 06 was clean.
  These are validation observations, not a controlled latency comparison.
- [Trial 08](trial-08/summary.json) reached the first full-model pair. All logits,
  generated tokens, future proposals and final target/draft state match for an
  eight-token continuation with forced rejection after its first accepted
  input. Candidate recovery makes zero full-target calls and zero expert reads;
  the control replays one target row, which was already cached.
- Both processes in that pair recorded zero compression/decompression. Total
  system swap fell by **8MiB** during candidate startup; the existing strict
  unchanged-swap rule rejected the sample. It is numerical diagnostic evidence,
  not a passed stage or a timing result. Earlier retries genuinely recorded
  process compression. Failure messages now distinguish those causes without
  relaxing the gate.
- A [memory diagnostic](memory-diagnostic-01/summary.json) also completed with
  clean process readings. Its process-region snapshot showed roughly 9.5GiB in
  resident Metal allocations. It did not reproduce the earlier compression,
  and its observer can pause the process, so it establishes no latency result.
- [Allocation attribution and pipeline preparation](allocation-followup.md)
  identify checkpoint, ngram and pipeline metadata owners. Preparing 39 of 77
  pipelines did not establish a clean-memory fix; that intervention is set
  aside. Allocation size is not physical residency or compressed-page ownership.
- The [fixed-width candidate](../2026-09-17-mtp-widths/README.md) now passes all
  ten full-model numerical cases cleanly, in addition to twelve synthetic state
  checks. It explicitly retains full-replay; this does not qualify state-only
  recovery. Two fresh 128-token LRU pairs favor width one by 17.77% generation
  latency, with identical logits/state and clean memory. No policy is promoted.

The current Python suite passes **505 tests**. Native production sources and
the production executable retain the fingerprints recorded below. Trial source
snapshots under `.cache/experiment-sources` prevented concurrent edits from
invalidating later attempts. Raw failed and blocked reports remain unchanged.

## What is available

- `query_evidence.py cycles`, `opportunity`, MTP-aware `compare` and `next`.
  [The offline audit](direct-output-audit.json) reproduces the original direct-output
  comparisons and optimistic recovery-free bounds: **4.7572 / 3.8080 / 4.3419
  tokens/s**. No inference was launched for this calculation. The workloads are
  separate cases with one pair each, without a confidence bound.
- `build_target_recovery.py` derives an isolated binary from the existing
  direct-output producer, preserving historical code and production defaults.
  The explicit arms are `full-replay` and `state-only`.
- `mtp_target_recovery.hpp` preallocates the same journal/output capacity in both
  arms. During verification, the candidate copies original state-update inputs
  into reusable buffers. Recovery restores recurrence/convolution and rejected
  attention/index tails, applies the original kernels in groups of four layers
  with at most two outstanding groups, then commits history and positions.
  The bound includes allocator rounding, metadata, transient convolution owners
  and scratch under the 16MiB incremental ceiling. Resident constants are shared.
- New native reports measure checkpoint saving, drafting, verification, target
  restore/repair, draft restore/catch-up and residual overhead. Actual target
  forward-entry and expert-byte counters surround recovery. The candidate
  requires zero of each; no synchronization is added solely to time a phase.
- `probe_target_recovery.cpp` contains bounded capture/replay, manifest/tensor
  hash checks, independent full-target golden prefix generation and transaction
  validation. Bundles remain local under `.cache/target-recovery-fixtures` and
  are limited to 2GiB. Replay loads no complete model and enforces a separate
  2GiB process ceiling. The native binary hash binds a fixture to its producer.
- `trial_recovery.py` executes the registered recipe through existing resource
  guards and progress reporting, seals every stage, records decisions in the
  ledger and resumes only compatible correctness prerequisites. It does not
  reuse timing samples or automatically promote a candidate.

## Checks and outstanding gate

The initial Python suite passed **485 tests**, including malformed evidence, changed
hashes, duplicate pairs, EOS-shortened accounting, missing/overlapping phases,
legacy compatibility, numerical comparison checks, bounded recipes, early stops
and ledger recall. The isolated native build succeeds.

[Low-memory native checks](native-self-test-01/summary.json) pass with Metal API
and shader validation: accepted prefixes 1/2/3/4 at absolute offsets 3 and 8,188,
EOS-valued history/input tokens, and exact comparison of every persistent buffer,
including untouched attention tails. The independent reference runs the
existing operators one row at a time. Peak physical footprint is **1.2505GiB**;
all buffers are released. These inputs are synthetic, so this result does not
replace real capture or full-model validation.

[Trial 01](trial-01/summary.json) stopped before loading the target. Available
memory was **10.59GiB**, below the unchanged **13.5GiB** preflight requirement
for the 12GiB experiment. AC power, Low Power Mode off and nominal thermal state
passed. No real fixture or timing sample was produced. The attempt and blocked
ledger entry are retained. Later source changes mean the next attempt must use
a fresh build rather than resuming this old producer.
The earlier [read-only preflight](preflight-final.json) found **9.61GiB** available.
The review below subsequently passed admission; that older preflight no longer
describes the current blocker.

## Double-check results

The review fixed evidence validation gaps: replay now checks compression,
decompression, swap and power/thermal observations; resumed correctness must
revalidate the exact four prefix cases plus EOS and all ten raw samples; external
stage references, fixture source identities and kernel policies are checked.
Missing checkpoint timings are rejected instead of silently becoming zero.
Saved-fixture numeric fields are validated before narrowing. Synthetic fixtures
carry an explicit origin and cannot satisfy a real-fixture gate. External replay
requires the original sealed capture report and its resource observations; a
clean replay cannot qualify a resource-disturbed capture. Capture progress now
uses the same filename stem as the native phase log.

[Native review checks](native-self-test-02/summary.json) exercise both the eight
operator cases and eight serialized-fixture replay cases with Metal API and
shader validation. Corrupted tensor hashes, invalid geometry, missing coverage,
invalid prefixes, numeric overflow, delayed GPU completion, cancellation,
injected failure and buffer cleanup checks pass. Peak footprint is **1.2515GiB**,
with zero recorded compression/decompression and unchanged swap.

[Real capture 01](real-fixture-01/summary.json) passed admission and completed
the real cycle and four independent full-target reference prefixes. It then
stopped at the resource gate: peak compressed memory was **28.53MiB**, despite
unchanged swap and clean power/thermal observations. It does not qualify.

[Diagnostic replay](real-replay-diagnostic-01/summary.json) loads that captured
cycle without the model. All four prefix states match exactly on two repetitions,
including untouched attention tails; invalid-input, cancellation, delayed-
completion and cleanup checks pass. Replay peaks at **0.720GiB** with clean
resources. The journal's incremental bound is **14.672MiB**, below 16MiB.
This useful arithmetic evidence does not erase the capture's resource failure
or replace full-model continuation comparisons. The diagnostic report explicitly
sets `qualification_eligible: false`. Historical reports are unchanged.

Clean real capture/replay is complete. The remaining qualification needs all ten
clean full-model runs covering prefixes and EOS, followed by fresh predeclared
timing gates. Numerical diagnostics alone do not pass the resource gate.
Five-pair confirmation, 2K/4K/7K context targets and sustained coding qualification
remain beyond this screening stage.

## Run the stage

```sh
.cache/qwen-reference-venv/bin/python scripts/qwen/query_evidence.py trial \
  scripts/qwen/recipes/target_recovery.json \
  --output docs/benchmarks/2026-09-17-target-recovery/trial-next
```

The standalone developer operations are:

```sh
.cache/qwen-reference-venv/bin/python scripts/qwen/build_target_recovery.py \
  --output .cache/target-recovery-build-new

.cache/qwen-reference-venv/bin/python scripts/qwen/trial_recovery.py capture-recovery \
  --recipe scripts/qwen/recipes/target_recovery.json \
  --build .cache/target-recovery-build-new --output .cache/recovery-capture-new

.cache/qwen-reference-venv/bin/python scripts/qwen/trial_recovery.py replay-recovery \
  --recipe scripts/qwen/recipes/target_recovery.json \
  --build .cache/target-recovery-build-new --fixture FIXTURE_DIRECTORY \
  --output .cache/recovery-replay-new
```

Production source fingerprint remains
`51877f96f015386dd642fd4e63fbc39ff2e75c1156f5ae1bcba764f546963594`, and the
production executable SHA256 remains
`05495d0baf993b1f7f1bc4b9365753e88603909ac66db3309a1380ea8c7c519f`.
