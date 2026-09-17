# Longer MTP continuations and updated target profiling

The native developer stage now completes exact, memory-clean **128-token
comparisons on all three coding prompts**. Fixed four-token speculation helps
one case and slows two; it is not ready to become the default. These are single
screening pairs, not confidence-bounded performance or coding-quality results.

| Coding case | Prompt tokens | Serial tokens/s | Real MTP tokens/s | Proposal acceptance | Cycle latency change |
|---|---:|---:|---:|---:|---:|
| Merge intervals | 72 | 4.1173 | 4.4038 | 86.67% | -6.51% |
| Repair LRU cache | 198 | 4.0629 | 3.1987 | 66.67% | +27.02% |
| TypeScript retry/backoff | 93 | 3.9437 | 3.6897 | 82.41% | +6.89% |

The first two pairs come from [screen 01](screen-01/summary.json). That run
preserves `complete: false`, `resource_blocked`: thermals rose before its final
candidate started. The TypeScript row uses a new, complete
[selected-case pair](retry-backoff-01/summary.json); its earlier unpaired serial
measurement is not reused. See the reconstructed
[first-screen audit](audit-screen-01.json), [selected-case audit](audit-retry-backoff-01.json)
and [combined analysis](continuation-analysis-01.json). No timings are pooled
into a repeated-pair confidence claim.

Every compared committed token, full-vocabulary logit hash and final target
state matches a fresh serial process. The LRU prompt exercises ingestion across
the 128-row target panel boundary. All compared processes have zero recorded
compression and decompression, unchanged swap, nominal thermals and AC power
with Low Power Mode off. Peak physical footprint is approximately **9.77GiB**
under the unchanged **12GiB** admission, 1,460 target slots and 32 draft slots.

## Implemented

- A bounded native continuation harness supports up to 256 generated tokens and
  512 prompt tokens. It streams target hidden panels of at most 128 rows into
  draft microchunks of at most 16 rows, retaining the exact next-token alignment
  across panels. Target and draft arithmetic are shared with the qualified short
  prototype.
- Three natural coding prompts cover interval merging, LRU cache repair and
  TypeScript retry/backoff. The LRU prompt crosses a 128-token panel boundary.
  Each run computes its own greedy proposals. Normal comparisons check every
  committed token, full-vocabulary logit hash and final target state against a
  fresh serial process. EOS truncates a speculative block and restores its unused
  tail; an early stop is not a completed requested-length benchmark.
- An eight-token rejection check exercises future proposals after recovery.
  Measured cycles include drafting, checkpointing, target verification and replay.
  Priming is separate; the wall interval including evidence reporting is also
  recorded. Serial and candidate retain the same 12GiB budget, 1,460 target slots,
  32 draft slots and 8,192-token state capacity.
- A separate profiler captures the updated expanded-Q8 target while real MTP
  proposals execute. It admits 512MiB for trace data inside the same budget and
  reconciles each target call's dispatch counts separately from draft work.
  The existing audited timeline analysis joins all 48 layers per call. Truncation,
  missing operations and changed outputs fail closed. Instrumented timing cannot
  qualify normal throughput.

## Verification and limits

Both native executables compile with the project's C++23 flags. Native input-bound
and checkpoint self-tests pass without loading a model. **30 focused MTP tests
plus seven existing timeline tests pass.** Regression checks cover incomplete
logit coverage, EOS inside a speculative block, invalid positions/timings,
compression peaks, and missing target dispatches when draft counters are present.
Single-case selection is tested and its scope is explicit in each report.
Production sources, executable, fingerprint and defaults are unchanged by this
stage. The original harness before case selection is preserved in
`source-before-case-selection/`.

The independent [CPU/Metal fixture](fixture-02/summary.json) passes all 22 saved
boundaries. The [eight-token rejection check](validation-01/summary.json) passes
full-layer versus state-only catch-up, including future proposals after forced
recovery, with exact target and draft boundaries. Both pass the
[prerequisite audit](audit-prerequisites-01.json). An external memory snapshot
during correctness validation is recorded in `memory-observation-01/`; that
validation's timing is not used as performance evidence.

No 128-token run reaches EOS; actual native EOS truncation remains to be exercised
beyond its current unit coverage. No 256-token, 2K/4K/7K, sparse-draft-attention,
retained-session, non-greedy sampling or sustained tool-workflow qualification
follows from this stage. Fresh-process memory cleanliness does not prove stable
memory across repeated model destruction in one process. Output reports retain
both cycle time and the longer interval including evidence hashing/report writes.

Earlier resource stops remain separate: `fixture-01` stopped before GPU work;
initial available-memory observations ranged from 6.84 to 12.91GiB. After the
user closed Brave, availability rose to 15.72GiB with nominal thermals and later
18.63GiB. No unrelated process was stopped and no OS limit was raised.

The original catch-up comparison is now also complete:
[recovery timing 03](../2026-09-16-mtp-forward/recovery-timing-03/summary.json)
records **4.9713 tokens/s** versus 4.8719 for full catch-up, with exact state and
clean memory. The 5-token/s floor fails, so it correctly stops after one pair.
That short result and the earlier 4.7904 result retain their distinct scope.

## What the longer cases change

| Case | Draft ms/token | Target verification ms/token | Recovery ms/token | Entire cycle ms/token |
|---|---:|---:|---:|---:|
| Merge intervals | 9.47 | 199.31 | 16.68 | 227.07 |
| LRU repair | 11.50 | 245.69 | 53.56 | 312.63 |
| Retry/backoff | 9.83 | 220.54 | 38.96 | 271.03 |

Removing every measured recovery cost, while holding the other costs fixed,
would yield only 4.753, 3.860 and 4.309 tokens/s respectively. These are hypothetical
upper bounds, not attainable speedups. Exact prefix-state restoration could
reduce replay waste, but it cannot by itself satisfy 5 tokens/s on these cases.
Faster verification and/or better use of draft acceptance is also necessary.

Do not spend the longer 256-token or sustained qualification budget on the
unchanged fixed-width policy. Next collect the updated target dependency profile,
then choose a small controlled experiment that addresses the observed cost.
Keep exact target verification authoritative; a future acceptance-aware block
policy may change proposal length or read timing, never committed target tokens.
Select any policy on separate development data and measure it on held-out prompts.

## Current profiling step

The [short reference](short-reference-01/summary.json) is complete. It uses
`--max-tokens 16 --case merge_intervals` to avoid rerunning unrelated prompts.
The first [profile attempt](profile-01/summary.json) stopped at elevated thermals
before model loading. It remains resource-blocked and contains no GPU timeline.
The second attempt completed native inference but its analysis correctly rejected
three hidden-state copies captured outside their target-call windows. The
[diagnosis](profile-boundary-diagnosis-01.json) and original sources remain
preserved; no intervals were silently dropped. The isolated profiler now disables
capture after each target call and asserts empty capture buffers outside every
window, including the final one. A regression test retains strict rejection of
commands left over from an earlier block. Normal inference code is unchanged.

The repaired [profile 03](profile-03/summary.json), built in
`.cache/mtp-target-profile-build-02`, passes exact full logits, target/draft state,
initial target cache, clean memory and complete coverage. Its
[independent audit](audit-profile-03.json) reconstructs **192 layer passes,
21,751 dispatches and 4,107 submissions**. Mean instrumented intervals per target
input are 106.95ms GPU-active, 26.47ms idle after submission, 18.88ms idle with an
expert ready, 29.66ms idle with a required read pending, 2.78ms callback delay and
18.60ms other idle time. These exclusive overlap categories describe concurrency,
not the cause of each gap or attainable savings. The capture covers four target
calls on one short coding prompt, not the whole longer suite; its timing cannot
qualify normal throughput.

## Selected next experiment

**Follow-up completed:** the [bounded scratch trial](../2026-09-16-mtp-ngram-init/README.md)
passes exactness and clean memory, but its 1.2465% short-pair latency reduction
fails the 2% gate despite 70.37% fewer allocations. The rationale below records
the original experiment. The [next isolated trial](../2026-09-16-mtp-ngram-init/next-protocol.md)
targets redundant single-row expert scatter copies; no new Q4 arithmetic is proposed.

The profile counts **26,932 new target buffers and zero scratch reuse** over
16 inputs. The normal short reference independently records 28,000 allocations
including draft work and zero reuse, versus 3,825 allocations and 48,015 reuse
events for serial decoding. Source review confirms that `Model::forward` enables
the existing temporary pool only for one token. Counts alone do not measure the
time spent allocating, but they identify a concrete difference in the verifier.

Prototype completion-bound reuse of routed-expert temporaries for four-token
verification in a separate developer build. Retain two live GPU groups, the
32-lease window, fixed cache slots, identical kernels and the 12GiB total budget.
Each temporary pool may be recycled only after its last GPU user completes;
inputs, expert contributions, persistent state and captured target hidden values
must remain outside those short lifetimes. Account for both pools and serial
recovery transitions explicitly. Do not simply retain every buffer for an entire
four-token pass: that could exchange allocation cost for memory pressure.

First screen representative one-/two-/three-/four-row expert work, including
delayed completion, cancellation and forced misses. Then require exact full logits
and state under rejection/replay, and a fresh normal comparison before extending
to the same 128-token cases. Record allocation time/count, complete cycle latency,
read bytes and physical memory. Stop if allocation reductions do not improve
normal requests. This is a new four-token lifetime experiment; it does not revive
the rejected packed-Q4 kernel or cache-capacity changes. Acceptance-aware proposal
length remains a separate follow-up for the two measured regressions.

All tools require explicit build/output paths. Continuation screening requires
the completed fixture and validation source. `--case` permits a fresh selected
comparison after a stopped suite; it never borrows the previous arm's timing or
relabels the original run. No benchmark automatically starts later.

See the [continuation protocol](protocol.md) and [profile protocol](profile-protocol.md).
