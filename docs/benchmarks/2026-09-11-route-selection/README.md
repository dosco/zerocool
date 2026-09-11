# Parallel expert selection passes the short request screen

Two fresh alternating pairs reduced complete-conversation time by **1.88%**
and **1.82%**, median **1.85%**, at the same 12GiB budget and 1848 CLOCK slots.
Initial generation improved from median **3.53 to 3.71 tokens/s**; generation
after the retained append improved from **3.27 to 3.43 tokens/s**. All secondary
median guards passed. This is a two-pair screen, not confidence-qualified
promotion or a 2K/4K performance result.

The new `--route-selection simd` kernel selects exactly ten of the 512 routed
experts using all 32 SIMD lanes. Ties retain original expert order; selected
scores are reloaded before the unchanged softmax. Expert execution and final
reduction remain unchanged. `serial` is the default and automatic path. The
candidate is allowed only in experimental kernel mode for bench/inspect.

Both timed arms explicitly used the same experimental packed Q8 two-row path,
original GDN and token tile, fixed phase memory, serial prefill, reference expert
execution, eight readers and ready groups of four. This experiment changes
only selection. It does not retroactively pass the earlier Q8 TTFT guard.

The 72-token prompt generated 33 tokens; the retained 128-token append generated
another 33. All eight requests matched tokens, actual 104-token computation
reuse, memory allocation, expert hit/miss counts, and non-routing dispatches.
Timing excluded capture, profiling, validation and boundary probes.

Before normal timing, all **50 captured router inputs** passed independent
stable CPU ranking and byte-identical GPU IDs/weights. The capture includes
one decode input from each of 48 layers and 72-row prefill inputs from layers
0 and 47. Ten alternating operator pairs of eight dispatches each had median
candidate/control GPU ratio **0.01682**; the slowest input's median ratio was
**0.02216**. Median per-dispatch decode GPU time was **353.49 to 5.96 microseconds**.
These isolated costs are not end-to-end speedups.

The pre-change diagnostic attributed 43.88ms to the first token's 48 serial
selection passes. Its per-pass instrumentation and first-token execution differ
from warmed operator and ordinary request timing. It identifies a dependency
worth testing; do not subtract it from normal token latency or claim it all
became a request-level saving.

Only after the short timing gate passed did the fresh all-layer state checks
run. Original arithmetic and the combined packed-Q8/parallel-router candidate
matched all logits, route identities and retained state at each tested stage.
Continued sessions matched fresh replay with 32 expert slots forcing eviction.
Both arms passed partial-state invalidation, refusal of invalid reuse, GPU drain,
cancellation and failure tests under Metal validation. These short checks do
not qualify long contexts or independently establish coding quality.

**58 native tests / 48,454 assertions** passed under Metal validation and
**195 Python tests** passed. The capture, operator replay, four conversations
and two full-state/failure arms completed in **288.01 seconds**.

[Protocol](../../qwen_route_selection_stage.md), [raw summary](raw/summary.json),
[source-revalidated comparison](comparison.json), [operator replay](raw/operators.json),
[native tests](native-tests.log), [Python tests](python-tests.log).
The small captured-logit payloads are included; they are model activations,
not model weights. Unrelated temporary diagnostic tensors were discarded.

Native build: `3a0ea9334c448434ca7a534a623888e94c20350c29d29f7354bb4be2cc334a53`.
Artifact: `b2c422f3c643e36f04227a64d61796b44a4b1029`.
The pre-change diagnostic is separately bound to build
`1ddec378e0a2550eda51c351e5e9315863c15b61773617c457b4d8e2f8df9189`.

Next run five fresh paired confirmations with the same settings, preserving
all samples and the original confidence guards. Then address prompt/append
latency and the remaining generation dependencies at 12GiB. The 5 tokens/s,
2K/4K first-token, 7K-context, and sustained coding-session targets remain open;
production defaults are unchanged.
