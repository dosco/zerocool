# Prompt-memory reclamation and exact matrix experiments

Native source fingerprint:
`6d60860fdeb4945ce8d6563add992dce83a89eb002d5da6a23e8d6bb2476ba34`.
Measurements below use the actual 32GiB M1 Pro, a 12GiB engine budget,
8192 context, pinned mixed 4/8-bit artifact, unchanged prepared Q4 records,
panel 512, microchunk 128, eight I/O workers and ready-group two.
All execution options remain experimental; production defaults are unchanged.

## Normal memory screen

Each arm is a fresh process with the same 2K prompt and 64 greedy output tokens.
Profiling and GPU validation are disabled. All generated token IDs match.
This is one screen, not five alternating pairs or promotion evidence.

| Workspace policy | First token | Generation | Decode slots | Decode hit rate | Application decode reads |
|---|---:|---:|---:|---:|---:|
| Serial, fixed | 221.893s | 2.700 tokens/s | 1848 | 54.51% | 38.03GB |
| Double, fixed | 189.330s | 2.608 tokens/s | 1460 | 48.86% | 42.76GB |
| Double, reclaim | 189.280s | 2.699 tokens/s | 1848 | 54.45% | 38.08GB |

Reclamation recovered generation cache capacity and throughput while retaining
the double-workspace prompt latency in this screen. The prompt-to-generation
transition took 5.64ms, included in first-token timing. Ending footprint was
10.42GiB with zero emergency pressure resizes. This is not a sustained-memory
qualification. Read bytes above are application traffic, not device traffic.

See [measurements](memory-screen.json) and [configurations](memory-configs.json).
Raw native reports remain in `.cache/benchmarks/phase-memory/memory-screen`;
the copied summary records their names and SHA-256 values.

## Normal kernel screen

Both arms use double-workspace reclamation. Only affine output blocking changes;
shared gate/up loading remains disabled. Same prompt, output length and tokens.

| Kernels | First token | Generation |
|---|---:|---:|
| Existing tile-eight/precomputed GDN | 189.52s | 2.699 tokens/s |
| Four output rows; two rows for T1 | 146.69s | 2.846 tokens/s |

This is approximately 22.6% lower prompt latency and 5.5% higher generation
throughput in one screen. It does not meet the 60s/5-token-per-second product
targets and does not qualify a production default.
See [measurements](kernel-screen.json) and [configurations](kernel-configs.json).

## Correctness and tuning evidence

The native Metal API/shader validation suite passes 43 cases and 6408 assertions,
including Q4/Q8 output-row tails, exact FP32/BF16 outputs, grouped expert
positions, buffer lifetime, and cache resizing. The focused Python suite passes
34 tests, including phase-transition rejection and held-out policy validation.
See [native tests](native-tests.txt) and [tool tests](tool-tests.txt).

Both the combined two-row/shared-gate candidate and the screened four-row
configuration pass the 48-layer mixed
257+129 prompt/append, continuation, fresh replay, injected failure and
cancellation checks against original arithmetic. See [state summary](mixed-summary.json) and the [four-row comparison](mixed-rows4-state.json).
Raw state hashes and reference/candidate reports remain in
`.cache/benchmarks/phase-memory/qualify-mixed` and are hashed by that summary.
This is native parity, not a newly rerun independent MLX oracle.

Nine tuning captures cover early/later GDN, sparse attention, prefill, append,
and generation. Their 41 real-input cases produce 1250 exact operator
comparisons. T128 Q8 projection times improve substantially with four output
rows, including layer 30 and appended input. T1 expert variants are initially
inconclusive and must not be selected from their median alone. Tuning data and
policies are under `.cache/benchmarks/phase-memory/tuning`.

## Supplemental request screens

Shared gate/up measures 146.57s / 2.677 tokens/s with four-row blocking.
The frozen shape policy measures 196.49s / 2.796 tokens/s. Neither becomes the
preferred configuration: shared gate/up failed to improve the full-request
screen, and shape fallback lost prompt performance despite operator-level gains.
These are supplemental single screens, not paired significance tests against
the earlier controls. All seven screens have identical 64-token output.
See [supplemental measurements](supplemental-screen.json).

## Qualification limits

Held-out captures also pass all 1250 arithmetic comparisons. The initial frozen
shape-policy check was inconclusive: one T1 attention shape had an upper
latency ratio of 1.0183. The single fresh ten-pair confirmation passes without
changing the policy; the original five-pair reports remain intact. No shapes were missing. See [confirmation](operator-confirmation.json),
[held-out validation](heldout-policy-validation.json),
[tuning captures](tuning-captures.json), and [held-out captures](heldout-captures.json).

The optimized 2K cached-token diagnostic takes 270.85–276.99ms (median 274.90ms),
with exactly 480 ready hits, zero model reads, and bitwise logits/routes/state
agreement in each of five restored repetitions. It uses the same four-row,
precomputed-GDN, grouped configuration as the normal candidate. This exceeds
the 200ms budget for 5 tokens/s even with the required weights cached. It is
not a universal hardware lower bound or a normal-request performance result.
See [cached-token evidence](cached-token-2048.json).

The Q4 control passes all-layer short replay, cancellation/failure, sample-free
priming and continued-versus-fresh checks with reclamation and four-row
selection. See [Q4 state summary](q4-summary.json).

The initial normal-request targets remain unmet, so production promotion stays
blocked. Full paired-workload qualification, 7K reporting, and the 20-minute
coding/recovery workflow remain unfinished and are not claimed as passing. Missing or
unfinished evidence is not a pass. See the [stage contract](../../qwen_memory_compute_stage.md).

## Follow-up review

The native build remains unchanged. A fresh build and Metal API/shader validation
rerun pass all 43 cases and 6408 assertions. Review found that held-out validation
could silently discard unmatched candidate samples and accept gaps in repetition
IDs. The validator now rejects both; regression tests reproduce the previous
failure. The focused tool suite passes 36 tests after this fix.

Saved normal reports, output-token equality, native state parity, and cached-token
evidence were rechecked. The stricter held-out validator preserves the original
inconclusive result and the later confirmation's bounds. This review does not add
new full-model timing runs or complete the outstanding release qualification.
