# Measure temporary-buffer cost before reuse

Current stage: counters measured; isolated reuse opportunity confirmed. Keep the
failed tail-overlap candidate disabled (`expert_tail=wait`). The question is
whether the 3238 buffer allocations per generated token consume enough exposed
CPU time to justify a bounded reuse experiment toward 200ms/token.

`--decode-diagnostics` now enables per-class physical allocation counts, charged
bytes and CPU call duration. It also measures completed command-group destruction
(including Metal command-buffer references) and last-engine-owner callbacks
outside that destruction. Same-executor owner callbacks nested inside group
destruction are excluded from the outside total. Instrumentation adds no GPU
commands, waits, kernel changes or new GPU allocations. It is disabled by default;
old or disabled reports carry unavailable measurements, never a zero-cost claim.

The timers distinguish temporary, resident, state, expert, snapshot and workspace
allocations. Views obtained from an existing scratch pool count only the underlying
physical workspace allocation. Charged/released byte counters describe allocation
traffic and engine ownership, not physical footprint or immediate driver release.
These CPU interval sums may overlap GPU/I/O work. Even the nonduplicated sum cannot
be subtracted from token wall time as a predicted speedup.

Run `scripts/qwen/measure_buffer_costs.py --output FRESH_DIRECTORY`. It executes
one ordinary and one measured conversation on the same build: 72 prompt tokens,
33 output tokens, a 128-token retained append, and another 33 output tokens.
Each decode phase has all 32 per-step snapshots; allocation counts must match
the executor, retired groups must match submissions, and phase deltas must equal
the complete per-step sums. Outputs must match the previously confirmed control.
Both arms use mixed 4/8-bit weights, prepared Q4 records, packed Q8 rows 2, SIMD
routing, 12GiB, 1848 CLOCK slots, panel 512, chunk 128, eight readers and ready
groups of four. Metal validation and command/per-kernel profiling are off.

The collector freezes sources, binaries and artifact receipts, takes the shared
GPU lease, and enforces 360 seconds total / 150 seconds per inference process.
Keep admission failures and interrupted runs incomplete. Report measured/ordinary
latency ratios without using this one pair as an overhead correction. Report
process compression/decompressions alongside footprint and net system swap.

Consider a bounded reuse probe when temporary allocation plus all measured
retirement intervals reach at least 20ms/token in both decode phases without an
observed decode compression disturbance. This is a generous CPU opportunity bound:
state, command and residency work may remain even if temporary allocations vanish.
It authorizes a small lifetime/operator experiment, not runtime promotion. If the
cost is small or the evidence is disturbed, do not build a pool merely to reduce
allocation counts. No 2K/4K, sustained-use or 5 tokens/s claim comes from this capture.

## Captured buffer costs

Native build `870a3e3a` completed both conversations and reproduced every prior
control output token. All 64 step snapshots reconcile exactly with phase counters.

| Measurement, ms per generated token | Initial | After append |
|---|---:|---:|
| Ordinary decode wall time | 325.66 | 322.60 |
| Instrumented decode wall time | 283.40 | 290.68 |
| Temporary buffer creation | 25.87 | 26.88 |
| All buffer creation | 26.13 | 27.16 |
| Completed command-group retirement | 23.06 | 23.65 |
| Owner callbacks outside retirement | 4.83 | 7.81 |
| Observed CPU interval sum | 54.01 | 58.62 |

Each token creates 3201 temporary buffers and 37 state buffers. Temporary
allocation traffic is 74.39–74.58MiB per token; this is not live memory.
Instrumented/ordinary decode ratios were 0.8702 and 0.9010. These are separate
sequential runs with host variation, not evidence of negative profiling overhead.

The full-model capture **does not pass the clean-memory opportunity gate**.
Measured process compression peaked at 7.43GiB over the process lifetime, with
about 4.64GiB still compressed during generation. Decode decompressions were
94811 initially and 4 after append. The ordinary arm also experienced compression
(8.21GiB peak; 276019/27082 decode decompressions). Net system swap did not grow
within either conversation. These observations do not establish the cause of
compression, and a positive CPU interval cannot be subtracted from token latency.

[Sealed capture](raw/summary.json), [per-step analysis](raw/analysis.json),
[ordinary request](raw/normal.json), [instrumented request](raw/measured.json).

## Isolated lifetime and timing probe

To separate allocation overhead from compressed model pages, the follow-up uses
the **existing** scratch-pool implementation. It changes no inference schedule.
Run `scripts/qwen/probe_buffer_reuse.py --output FRESH_DIRECTORY`; the runner
compiles the developer executable, freezes native/library/probe identities,
holds the shared GPU lock, and caps execution at 120 seconds. Compiler arguments
are saved. The probe holds at most 512MiB and loads no model assets.

The synthetic iteration has 48 groups of 64 temporary buffers, with widths 2560,
6144, 10240 and 768 floats. Each group executes identical binary GPU operations,
waits for completion and checks every result against a CPU calculation. It is
allocation pressure, **not the model's allocation sequence or dependency graph**.
Five alternating pairs each contain a cold iteration, another warmup, then eight
timed warm iterations. Output checking is included in both wall times.

| Pair | Fresh allocation, ms/iteration | Existing pool, ms/iteration | Saved |
|---|---:|---:|---:|
| 1 | 76.86 | 25.08 | 51.78 |
| 2 | 72.91 | 24.25 | 48.66 |
| 3 | 77.24 | 25.78 | 51.46 |
| 4 | 89.01 | 25.73 | 63.27 |
| 5 | 78.94 | 24.84 | 54.11 |

Median paired saving is **51.78ms per synthetic iteration**, with zero observed
process compression or decompression. The pool retains 84MiB plus a 48KiB input;
warm iterations perform zero physical allocations and 3072 pool reuses. The
control allocates and releases 3072 buffers per iteration. All GPU users finish
before reuse, and final release returns engine accounting to zero. Cold pool
time has median 55.38ms; final pool release costs median 43.25ms and remains part
of any future complete-request comparison. Warm timing does not hide this cost.

[Probe analysis](probe/analysis.json), [all iterations](probe/probe.json).
The [first probe attempt](probe-failed/summary.json) failed while assembling JSON
after invalidating an ordered-JSON iterator. It has no complete timing result;
the corrected probe iterates a separate immutable order list. All five new pairs
are retained. This synthetic result does not override the disturbed full-model
capture or qualify an inference speedup. It independently supports a small
runtime reuse experiment within the existing scratch allowance.

The native suite passed **60 tests / 48774 assertions** with Metal API and shader
validation ([log](native-tests.log)); tooling tests cover absent measurements,
unretired GPU users, counter resets, incomplete phases/pairs and memory disturbance.
The empty-phase guard and isolated probe were added after the full-model capture;
the final analyzer reproduces its saved analysis exactly. Native sources remain
the measured build. Production defaults and weights are unchanged.
