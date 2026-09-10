# Five-pair residency confirmation: benefit not established

The completed five-pair run **failed its predeclared timing gate**. Core residency
won three pairs, lost one narrowly, and lost the final pair substantially. The
median conversation ratio favored core by 1.60%, but the primary geometric-mean
ratio was **1.0228**: 2.28% more elapsed time. Its model-based 95% interval was
**0.7707–1.3573**, covering both benefit and regression.

Keep the result inconclusive and the defaults unchanged. No longer qualification
was started. The earlier short-screen result remains valid evidence for those
runs; it is not a reliable general speedup claim.

## Method and observed results

Five new off/core pairs ran on the actual 32GiB M1 Pro, alternating which arm ran
first. Each arm was a fresh process with a 72-token prompt, 33 outputs, a
128-token retained-history append and 33 more outputs. Each follow-up reused
104 computed tokens and ingested the pending output plus new input. All twenty
requests matched the previous control's output tokens and reuse.

The mixed artifact, native build, reference arithmetic, CLOCK, 12GiB allocation,
1,848 expert slots, panel 512, microchunk 128, four ready experts and eight I/O
workers were fixed. Diagnostics and Metal validation were disabled. Earlier
screening pairs and the incomplete first attempt were not pooled. The completed
capture took **717.54 seconds**, within its 900-second deadline.

| Pair | Order | Off conversation | Core conversation | Core/off ratio |
|---|---|---:|---:|---:|
| 0 | off, core | 65.28 s | 64.23 s | 0.9840 |
| 1 | core, off | 64.92 s | 63.74 s | 0.9818 |
| 2 | off, core | 79.74 s | 62.45 s | 0.7833 |
| 3 | core, off | 63.72 s | 63.98 s | 1.0042 |
| 4 | off, core | 63.94 s | 94.19 s | 1.4730 |

The gate required a conversation geometric mean at most 0.99, an interval upper
bound below 1.00, and each initial/follow-up request, first-token and decode
metric's upper bound at most 1.03. The primary test and several secondary guards
failed. All measured runs, including both large slowdowns, remain included.

Intervals use paired log-ratios with Student-t and four degrees of freedom. The
pair, rather than a token, is the unit of analysis. Five pairs provide limited
precision, and temporal host effects and large stalls challenge the independence
and approximate-normality assumptions. These are marginal intervals; they are
not simultaneous 95% coverage for all metrics or evidence of a causal effect.

## Two different slowdown patterns

Pair 2 off reproduced the startup-memory pattern. Its initial generation took
25.28 seconds, including **14.40 seconds in the first four forwards**, and recorded
1,993,688 decompressions across the generation phase. Its GPU command durations
totaled 8.18 seconds, while CPU waits for GPU completion totaled 17.28 seconds.
The corresponding core generation took 12.08 seconds with zero decompressions.
Phase counters cannot attribute every decompression to a particular token.

Pair 4 core showed a different pattern:

| Generation phase | Off wall time | Core wall time | Off GPU command duration | Core GPU command duration | Core decompressions |
|---|---:|---:|---:|---:|---:|
| Initial request | 12.53 s | 25.44 s | 7.21 s | 12.26 s | 963 |
| Follow-up | 13.37 s | 28.03 s | 7.69 s | 12.86 s | 96 |

Core's first four initial forwards took only 1.91 seconds; the delay extended
through later generation. Coordinator waits increased to 7.64 / 9.20 seconds.
Completed-read service sums were **lower**, 13.24 / 15.38 seconds versus the
control's 15.13 / 18.00 seconds. These observations do not establish an SSD
slowdown. GPU durations, CPU waits, concurrent I/O and coordinator waiting
overlap; they must not be added into a critical-path breakdown.

Core buffer enrollment remained correct. The counters do not establish whether
GPU contention, host scheduling, power/thermal changes or another mechanism
caused the final slowdown. No live host-condition trace or independent fixed GPU
timing reference was recorded. Do not attribute this result solely to compression,
nor remove the final run as an assumed environmental outlier.

## Memory admission and system conditions

An [earlier attempt](blocked-01/README.md) stopped after one control because the
next process could not admit 12GiB. A later metadata check admitted the same
budget. The failed capture and its original tooling are preserved; its unpaired
control is excluded because the pair never completed.

The runner now uses the existing three-check metadata retry, with 2- and
5-second waits and a remaining-deadline bound. Rejected checks remain in the raw
evidence. One retry was needed for pair 2 off in the completed run. The helper
does not retry inference or reduce the requested allocation.

The largest sampled process footprint was **10.47GiB**. Every request retained
the same 12GiB plan. Core enrolled 5,933,858,816 resident/state/control bytes,
with no expert enrollment or pending retirements at observed boundaries.
Registration requests residency; it does not guarantee it.

Observed system swap rose from 1,891,368,960 to 2,706,833,408 bytes during the
capture. This is a system-wide counter that includes other processes; it is not
attributed to FreeLLM alone. Boundary snapshots do not establish continuous peaks
or sustained-session behavior. The normal host conditions changed, and that
limitation remains part of the recorded result.

## Verification and next step

**178 Python tests passed**, including five-pair ordering, interval arithmetic,
uncertainty/regression gates, prerequisite checks and deadline-bounded metadata
retries. Two existing SQLite resource warnings were emitted without test failures.
Native sources and the build are unchanged from the earlier 56-test,
45,682-assertion Metal validation run; native tests were not rerun for these
Python-only changes.

Frozen source/artifact identity, the prerequisite short screen, all ten raw
conversations, memory/residency/output/reuse checks, original/copied seals and
the fixed decision were revalidated offline. The failed attempt retains its
pre-change tooling separately.

Native build:
`968132061556ac0077ff41b7c922b5b7c632bb12e21b8ea8e2c5f82f10bb3c73`.
Mixed artifact: `b2c422f3c643e36f04227a64d61796b44a4b1029`; prepared Q4 bytes are
unchanged. [Raw summary](raw/summary.json), [derived observations](observations.json),
[Python tests](python-tests.log), [ledger result](ledger-result.json) and
[verification hashes](verification.json) preserve the evidence.

The next useful experiment is a small measurement of the later GPU slowdown:
add a fixed, resident GPU operation before and after each measured conversation,
outside request timing, and record available host power/thermal information with
explicit missing values. Compare that timing reference with the existing GPU,
read and memory counters. Keep the residency setting fixed during that diagnostic.
This can distinguish changing device execution speed from a request-specific
dependency before another long comparison or kernel change.

The normal fast runs still generate about 2.5 tokens/s, and follow-up first-token
latency is about 24 seconds. Neither the 5 tokens/s target nor 2K/4K/7K and
sustained coding acceptance has been established.
