# Confirmation blocked before inference; narrow projection probe is promising

The five-pair router confirmation stopped at memory admission after three
bounded metadata-only attempts. **No timing samples were collected.** Its
status is `resource_blocked`, `complete: false`. Preserve this attempt and use
a new output directory after memory becomes available; do not lower the
12GiB budget, pool earlier pairs, or count missing results as a pass.

The requested allocation remains 1848 CLOCK slots and a 512-token panel.
Reclaimable memory was 11.28–11.31GiB across admission attempts; the 12GiB
engine allocation needs 13.5GiB including the system safety margin. The
engine did not load the model or generate tokens. Disk space was sufficient.

The new runner reuses the prior sealed operator and full-state evidence only
on the identical native build. It requests five fresh alternating pairs,
allows 900 seconds total and 150 seconds per process, and retains the
existing confidence guards. Offline queries reconstruct results from the
original sources and refuse incomplete confirmations or undeclared changes.

[Unfinished confirmation](raw/summary.json), [admission](raw/pair-0-control.admission.json),
[protocol](../../qwen_route_selection_stage.md), [Python checks](python-tests.log).

## Independent small experiment

While waiting for memory, a standalone probe tested loading four successive
input/weight pairs ahead of accumulation in `plain_mm`. It preserves the
original addition order, SIMD reduction, and final rounding. It extracts the
reference shader directly from the native source and adds a separate
experimental function. The engine and its selectors are unchanged.

The motivation is the two per-layer narrow hyper-connection injection
projections that precede expert routing. The previous
[instrumented diagnostic](../2026-09-11-route-selection/diagnostic-before-change/dispatch.json)
identified these as material GPU operations. That first-token trace belongs
to the older build and cannot establish the current request critical path.

Ten alternating pairs of 32 dispatches, using the native 32-thread group size:

| BF16 matrix / token rows | Reference median µs | Candidate median µs | Median paired ratio |
|---|---:|---:|---:|
| K=10240, N=4, T=1 | 157.86 | 111.59 | 0.7068 |
| K=10240, N=4, T=72 | 155.87 | 108.58 | 0.6966 |
| K=10240, N=4, T=128 | 158.62 | 128.98 | 0.8131 |
| Irregular K=10239, N=5, T=3 | 155.98 | 108.93 | 0.6983 |
| Tiny K=31, N=1, T=1 | 2.29 | 2.45 | 1.0727 |

All 80 comparisons were byte exact against the unchanged shader: five
shapes, eight fixed seeds, and both BF16-rounded and FP32 outputs. A separate
run passed the same comparisons with Metal API and shader validation. The
timings above exclude validation. Maximum shared-buffer allocation was
5,328,896 bytes, about 5.08MiB; this excludes compiler and driver memory.

**These are synthetic weights and inputs, not a real-model qualification.**
GPU times include command overhead amortized over 32 dispatches. The tiny
case regressed; a later runtime experiment should target the measured narrow,
long-reduction shape only. No whole-request benefit is established and no
production default changed. An exploratory first run used larger thread
groups; its timings are excluded from this table.

[Timing originals](projection-probe/timing.json),
[validation originals](projection-probe/validation.json),
[source and compiler identity](projection-probe/identity.json).

Build the standalone experiment using the command recorded in its identity,
then run the resulting binary with `kernels/metal/qwen.metal` and a fresh
JSON output path. Set `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` for the
separate validation run. It requires only Foundation and Metal, and neither
links nor rebuilds the native engine.

Next: complete the five router pairs once the fixed allocation is admitted;
capture the selected projection's real weights and activations; replay them
byte exactly and time them; then screen a narrowly enabled candidate on two
normal conversation pairs before spending time on long qualification.

Native build remains
`3a0ea9334c448434ca7a534a623888e94c20350c29d29f7354bb4be2cc334a53`.
The 5 tokens/s and long-context/session acceptance targets remain open.
