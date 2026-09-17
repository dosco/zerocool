# Bounded temporary reuse for four-token verification

The current verifier allocates thousands of short-lived Metal buffers per block.
Test whether two completion-owned pools reduce complete request latency without
changing weights, kernels, proposal decisions or outputs. Counts alone do not
establish time savings.

Use an isolated native source-copy build, with an explicit off/on control in the
same binary. Each pool is capped at 1MiB. Both arms reserve the additional 2MiB
within the same 12GiB joint admission, 1460 target slots, 32 draft slots and 8192
context capacity. At most two command groups and 32 expert leases remain live.
Only four-token target decode uses group pools. Persistent activations and state
bypass them. Changing between group pools and ordinary single-token replay drains
and releases the old pools. Submit before ending a scratch scope so the scheduler
keeps the command completion and expert leases. Cancellation and failure drain
both GPU and I/O before reuse.

1. Native Metal validation: unchanged real expert records at 1/2/3/4 rows,
   reversed read completion, forced eviction, all hits, cancellation, read and
   encode failures, delayed GPU completion, and release after drain.
2. Full model eight-token forced rejection and replay, comparing all logits,
   proposals, intermediate/final target and draft state. This is correctness
   evidence only, with Metal validation enabled.
3. A fresh 16-token ordinary MTP pair, off then on, with profiling and validation
   disabled. Stop unless latency falls at least 2%. If it survives, repeat with
   reversed order. Advance only if both pairs improve and their geometric mean
   latency ratio is <=0.98. These are exploratory gates, not confidence bounds.
4. Surviving candidates get fresh 128-token pairs on all three established coding
   prompts, alternating arm order. Promotion still needs five paired repetitions,
   long context, sustained use and the existing recovery/API qualification.

Require AC, Low Power Mode off, nominal thermal state, at least 13.5GiB available
before full-model load, no process compression/decompression or swap growth and
physical footprint <=12GiB. Preserve resource-blocked reports as unfinished.
Never combine unpaired runs. Raw reports are sealed and bound to the exact native
producer, artifact, workload and controls. Report allocation/reuse counts along
with complete request times. Callback-delay instrumentation is disabled in both
timing arms. Production remains unchanged throughout this experiment.
