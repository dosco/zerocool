# Native Q4 replay protocol

This diagnostic investigates why the earlier packed-Q4 probe improved isolated
GPU execution while the normal four-arm request screen regressed. No default,
weight bytes, native arithmetic or native library source changes in this stage.

Primary condition: `scatter`, using `encode_expert_rows` with the same temporary
activation, temporary down projection, position buffer and scatter as the
screened `decode_path=reference`. The `direct` condition is a separately reported
bridge to the original isolated probe. Direct output plus scratch reuse is a
synthetic combination here, not an admitted production setting.

Each condition runs in one process, with five alternating reference/packed pairs
at group sizes one and four. Each arm executes four repetitions of 64 saved
expert/input cases, with identical one-repetition warmup. At most two command
groups remain live. Scratch resets once per eight-expert batch; it does not model
the entire 480-expert token lifetime. Saved records are verified against the
original prepared Q4 bytes, covering two experts at each of layers 0/16/32/47
and eight decode inputs per layer. Full-state correctness uses the separately
sealed prior proof for native build `51877f96f015386dd642fd4e63fbc39ff2e75c1156f5ae1bcba764f546963594`.

Run native operator validation separately with Metal validation enabled. Timing
has validation and profiling disabled. Compare output bytes outside timing in
both command geometries, check the actual final timed batch before reexecution,
and protect untouched output destinations. Reject missing work, changed kernel
selection, outstanding users or changed identities. Timed arms must dispatch
256 gate/up, 256 down and, except for direct output, 256 scatter operations.

Report CPU encoding, CPU completion wait, GPU command time and wall time
separately; they overlap. A diagnostic gain requires both group sizes to have a
paired GPU-time ratio upper 95% bound below one and a wall-time upper bound at
most 1.03, with clean observed memory and host conditions. Missing observations
remain unknown. Five correlated replay pairs do not qualify request latency.

Each explicit condition has a 120-second stage deadline and 60-second process
deadline, the shared GPU lease and the existing resource admission checks.
Use the fixed 12GiB allocation ceiling; do not expand it on admission failure.

If gain survives the native path, add `resident` separately: load and register
the actual mixed artifact's resident weights, with unchanged replay computation.
It tests resident allocations, not preceding resident computation. If that also
survives, the next interventions are preceding resident work and bounded expert
read contention, declared separately before timing. Do not pool conditions or
project isolated timings into tokens/s. Do not rerun the unchanged rejected
eight-conversation comparison. Larger expert-cache work remains a separate lead.
