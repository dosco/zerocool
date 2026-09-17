# Perfect-draft verifier: implemented, first screen blocked by compression

Follow-up: the [September 16 screen](../2026-09-16-perfect-draft/README.md) now
passes two-/four-token correctness and recovery. Its partial clean timings miss
the 5-token/s gate; the next experiment tests the block working set against cache
capacity. This report preserves the original attempt below.

The standalone C++/Metal verifier, bounded rollback checkpoint, comparison runner
and source-provenance checks are implemented. The first attempt stopped after
serial validation because memory observations failed the predeclared gate.
**There is no two-/four-token correctness result or performance comparison yet.**
This remains an open experiment, not a rejected speculative-decoding idea.

## Delivered

- Existing native multi-token kernels verify 2/4 consumed inputs and return every
  vocabulary-logit row. Production model source and binary are unchanged.
- A 113.099MiB detached checkpoint preserves recurrent/convolution state and the
  appended attention/index rows, with no ownership of replaced GPU buffers.
  All arms reserve and touch identical capacity before priming; candidate saves
  are inside the measured interval. Total planned allocation including checkpoint
  and logit allowance is 10.124GiB within the fixed 12GiB experiment limit.
- Full logits, router selections, persistent state, causality and rejected-prefix
  recovery have explicit comparisons against separate serial execution. Timing
  collects full state at matching 4-token boundaries to avoid asymmetric scans.
- The runner requires three validation processes, then six fresh timing processes
  in order 1/2/4/4/2/1. It checks exact starting cache state, native identities,
  execution counts, physical/compressed memory, swap, thermal and power state.
  A candidate must win both pairs and reach 5 verified tokens/s in both runs.

The five build/provenance tests and eight comparison/gating tests pass. The
[CPU checkpoint self-test](checkpoint-self-test.json) proves restoration into a
replacement buffer, release of the old owner, full speculative-tail restoration
before partial acceptance, and rejection of bad geometry before any writes.
It uses small synthetic state and does not replace full-model verification.

## First attempt

[screen-01](screen-01/summary.json) lasted 32.09 seconds. Its serial process completed
all four requested input/next-token checks after the 72-token prime under Metal
API and shader validation. All 48 layers' state/route evidence was recorded.

| Observation | Value |
|---|---:|
| Peak physical footprint | 8.604GiB |
| Compression peak at prime | 107.172MiB |
| Final lifetime compression peak | 143.375MiB |
| Decompressions, prime → final decode → destruction | 23 → 37 → 68 |
| System swap | 1.748GiB, unchanged |
| Post-destruction physical footprint | 63.113MiB |
| Checkpoint allocation | 113.099MiB |
| Host conditions | Nominal, AC power, low-power mode off |

Compression was already present at the prime boundary. Aggregate counters do
not identify which allocation was compressed; similarity to checkpoint size
is not attribution. Full model destruction released the allocations. This
attempt does not establish a retention leak or prove that closing an application
would eliminate compression.

The guard stopped before width 2, width 4, and every timing process. The raw
serial validation contains instrumented durations, but those are neither normal
inference throughput nor an estimate of verifier speed. Keep the memory gate
and the result's `resource_blocked` status. Do not relax either to obtain a gain.

The [independently reconstructed audit](audit-01.json) passes. A second read-only
review checked the seal, producer/object hashes, full state geometry, every
shifted next-token check, and the timing denominator. An audit pass means the
blocked result was represented correctly; it does not mean the screen passed.

## Identity and next step

Base native fingerprint:
`51877f96f015386dd642fd4e63fbc39ff2e75c1156f5ae1bcba764f546963594`.
Actual developer executable SHA256:
`26bc0c551123bfceff1fbd7854d1a825bced301c2fb3cf439ec39853be7cd85e`.
The [producer receipt](producer.json) binds both source copies, compiler/link
commands, original native inputs, harness and executable. Archived sources are
in `initial-sources`. The unchanged mixed artifact and prepared manifest are
recorded in the sealed identity. The older source report supplies tokens only;
its disturbed timings remain excluded.

Resume this same bounded screen in a new directory when memory observations
permit it. If compression recurs, first attribute it with a separate lifecycle
measurement; do not repeatedly rerun the full comparison or start a draft model.
A passing perfect verifier would still omit real drafting cost and rejection,
long context, production session/RNG rollback, and sustained coding quality.
No production promotion or 5-token/s claim follows from this work.

```sh
.cache/qwen-reference-venv/bin/python scripts/qwen/perfect_draft.py run \
  --output docs/benchmarks/2026-09-15-perfect-draft/screen-02 \
  --binary .cache/perfect-draft-build-02/probe-perfect-draft
```
