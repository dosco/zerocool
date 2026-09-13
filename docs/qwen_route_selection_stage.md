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

Run `scripts/qwen/confirm_route_selection.py --output DIRECTORY` for the next
five fresh pairs, with a 900-second total deadline and 150 seconds per process.
It reopens the sealed short screen and reuses its correctness evidence only on
the same native build, artifact, allocation and workload. All prior timings are
excluded from the new samples; no early success stop is allowed. Require the
conversation geometric-mean ratio at most 0.99, the paired log-ratio Student-t
95% upper bound below one, and each secondary upper bound at most 1.03.
Inconclusive results do not pass. Offline comparison uses the same
`--change route_selection` axis for `route_selection_confirmation_v1`.

The [five-pair confirmation](benchmarks/2026-09-12-route-five-pairs/README.md)
passed on the unchanged build after host memory became available. Complete
conversation time fell 2.00% geometrically; the paired 95% ratio interval was
0.96644–0.99380. All secondary upper bounds stayed below 1.03. This advances
the selector to later qualification only; the prior packed-Q8 guard and the
2K/4K, 7K, and sustained-session targets remain open. The current priority is
the [200ms/token generation stage](qwen_decode_target_stage.md). The narrow
BF16 projection is deferred because its estimated total saving is too small
to lead the remaining 79–96ms/token reduction.
