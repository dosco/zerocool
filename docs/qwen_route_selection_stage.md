# Exact parallel expert selection

Keep 12GiB and 1848 CLOCK slots after the 18GiB experiment increased memory
compression and complete-conversation time. Optimize only the serial top-ten
ranking step, which must finish before a layer's expert reads can begin.

The candidate assigns sixteen of the 512 scores to each of 32 SIMD lanes.
Ten rounds select the maximum score and lowest original expert index on ties.
Scores are reloaded from their winning indices to preserve signed zero; the
existing softmax and contribution reduction remain unchanged. NaN and negative
infinity do not outrank the reference insertion kernel's sentinel. No router
prediction, expert skipping, precision change or extra cache is involved.

`--kernel-policy candidate --route-selection simd` is restricted to experiments.
The default and automatic path keep `serial`. Both timing arms explicitly use
packed Q8 two-row kernels; this does not resolve the earlier Q8 TTFT guard.

Run `scripts/qwen/screen_route_selection.py --output DIRECTORY` after native
and Python tests. The bounded sequence is:

1. Capture all 48 layers' router scores for one generated token after the
   established 72-token prompt, plus 72-row prefill scores at layers 0 and 47.
   Check scores against independent stable CPU ranking; compare GPU IDs and
   weights byte for byte. Keep exceptional-score and irregular-batch tests in
   the native suite. Temporary unrelated tensors are discarded after capture.
2. Ten alternating operator pairs of eight dispatches per input. Require every
   input's median GPU ratio below 0.9. Allow 180 seconds for capture and replay.
3. Two fresh alternating normal conversation pairs at the same 12GiB budget,
   without validation, profiling or capture. Require both conversation ratios
   below one, median at most 0.99 and every secondary median at most 1.03.
   Allow 600 seconds total and 150 seconds per conversation.
4. Only a timing survivor runs both original and candidate all-48-layer exact
   state, fresh replay, forced eviction, cancellation and failure checks, under
   Metal validation. This compares the combined Q8/router candidate against
   original arithmetic; request timing isolates only router selection. Allow
   360 seconds total, 180 seconds per correctness arm.

All timing decisions are reconstructed from hashed originals. Changed route
selection must be declared by offline comparisons, and other kernel dispatch
counts, memory plans and computation reuse must match. Incomplete runs remain
incomplete. A survivor still needs five fresh timing pairs, 2K/4K acceptance,
7K reporting and a sustained coding session before promotion.
