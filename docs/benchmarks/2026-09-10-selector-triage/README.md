# Cached selector triage after memory cleanup

The fixed **12GiB memory admission passed**. The cached comparison then reached
its **600-second total deadline** without emitting a timing report. This attempt
is **inconclusive and unfinished**; it establishes no selector speedup or
regression. The short normal-request screen and full validation were not run.

The original [outer summary](raw/summary.json) records `time_budget_exhausted`,
`complete: false`, and 600.521 seconds elapsed including shutdown. The native
[log](raw/cached-run/cached-4096/cached.log) records cancellation. A process check
after exit found neither the benchmark runner nor its native child remaining.
The inner runner's original `failed` status and empty interruption error are
preserved in [its summary](raw/cached-run/summary.json); they do not establish a
numerical failure.

## Experiment identity and memory

- Native source fingerprint:
  `c4f983f03f28974dd2c4a935d6d5233f1b28a1f5e7cb4800fe7529a64c058c2b`.
- Mixed 4/8-bit artifact revision:
  `b2c422f3c643e36f04227a64d61796b44a4b1029`.
- Prepared Q4 manifest:
  `c4bb4db3220a0de2218128738b2a6ac650a3d65086690c8dc2e8c23d097160da`.
- Actual Apple M1 Pro, 32GiB physical memory; 12GiB engine budget.
- Five alternating CPU/full and GPU/full cached pairs at 4K; API and shader
  validation disabled during timing. Only sparse selection differs.

The [admission report](raw/cached-run/cached-4096/admission-000.admission.json)
accepted 12,884,901,888 bytes with 480 expert slots. Its prompt plan was
10,741,235,712 bytes and its generation plan was 9,667,297,280 bytes. These are
planned allocations, not observed peak inference memory. No final footprint or
completed timing measurements are available from this attempt.

## What this reveals about iteration

The cached diagnostic first rebuilds the full 4K history using the original
reference kernels and serial prefill. It then snapshots state, obtains the
reference continuation, preloads its experts, warms both arms and measures the
pairs. It currently emits the native report only after all those steps finish.
The timeout therefore cannot tell us whether setup, warmup or timed replay
consumed the deadline.

The next useful harness change is bounded phase progress for model load,
reference priming, expert preload, warmup and each completed pair. Saved real
attention inputs are also a candidate for a cheaper first timing filter. Any
such operator timing remains separate from full-token and normal-request
latency. Do not extend this attempt or claim a performance rejection from its
missing timings.

## Short request-screen helper

[`triage_requests.py`](../../../scripts/qwen/triage_requests.py) is implemented
and tested. It independently verifies a promising cached result and its frozen
dependencies before starting one CPU/GPU normal-request pair. Each arm primes
4K history in a fresh process, appends 128 tokens to retained state and produces
eight outputs. Both history primes are inside one 900-second deadline.

The helper requires identical output tokens, actual 4K computation reuse, exact
workload/configuration identities, raw report hashes and disabled timing
instrumentation. A promising result requires at least 1% request improvement
with no first-token or decode regression above 3%. One pair cannot establish
confidence bounds or qualify production. The helper never launches full
validation. Its native positive path remains unexecuted because this cached
attempt did not pass the prerequisite.

All **124 Python tests** passed, including eight new short-screen cases for
prerequisite provenance, result reconstruction, total deadlines, interruption,
changed outputs/workloads, missing pairs and instrumentation mismatch. The
native runtime was not changed in this step. [Test output](python-tests.log) and
[verification hashes](verification.json) bind the helper, tests and unmodified
raw evidence copies.

The [experiment ledger entry](../../experiments/b8946f855ecb3a40b08513211afb9efcf455887a76a986341f4f7836d824f0b1.json)
records the hypothesis, failed measurement attempt and next smallest experiment
without changing the original run status.
