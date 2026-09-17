# Bounded real MTP forward and first cost screen

Use the pinned prepared trained head (`mtp-affine-q4-experts-q8-dense-64-v1`)
with the existing mixed target and qualified expanded Q8 verifier. Production
remains unchanged. Draft normalization uses BF16(1 + original weight) once;
the pre-FC hidden norm covers all 10,240 values. Token IDs shift one position
ahead of hidden inputs; attention positions remain those of the hidden inputs,
following the reference MTP proposer. The head has its own mixer and state.

First validate a four-row real-weight draft fixture, serial versus block
execution, forced eviction with ten expert slots, causal independence, rollback,
every proper prefix and pre-update cancellation. An independent CPU NumPy
implementation decodes prepared bytes and evaluates the entire draft. It must
agree on all selected experts and final greedy IDs, with relative L2 error at
most 0.02 at each saved boundary. This accommodates independent FP32 reductions
and BF16 rounding; it does not qualify draft quantization quality. The native
serial/block and rollback checks require exact bytes. Metal API and shader
validation are mandatory for this stage.

Only then run a small normal greedy continuation, including actual drafting,
target verification, rejected work, checkpoint copies, rollback and draft state
catch-up. Prime from bounded target hidden panels, at most 128 target rows and
16 draft rows at once. Never save context-wide hidden history. Both timing arms
reserve the same resources: 1,460 target expert slots and an explicit 32 or 128 draft slots under
a combined 12GiB plan, including host checkpoints and driver allowances. The
initial 128-slot runs were compression-disturbed; the separately reported
32-slot screen reduces draft storage by 253.5MiB without changing target capacity
or either model's weights. Host rollback storage is admitted before model load
and committed after priming, when generation first needs it. Do not pool samples
across these configurations or weaken the compression gate.

The initial joint screen is one validation process, one serial timing process,
and one MTP timing process, using the existing 72-token prompt and 16-token
greedy continuation. Numerical outputs must match that continuation, and final
target state must match serial execution. Candidate proposals do not affect
acceptance: only the target argmax decides. The candidate uses three actual
proposals per four-input verification block. Rejection replays only the accepted
target prefix and replaces tentative draft state with target-aligned inputs.
Every recovery cost is included. No timing from older free-draft probes is pooled.

Require AC power, Low Power Mode off, nominal thermals, no observed process
compression/decompression or swap growth. Resource-disturbed results stay
separate. A result slower than serial stops expensive performance qualification.
One short timing pair cannot establish a confidence interval or production
speed. Long-context sparse behavior, cancellation during execution, tool-use,
sustained sessions and non-greedy exact sampling need further validation before
promotion, regardless of this screen's result.

Architecture sources: [MTP head](https://docs.vllm.ai/en/latest/api/vllm/models/qwen4_exp/nvidia/mtp/)
and [proposer token alignment](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/llm_base_proposer.py).
