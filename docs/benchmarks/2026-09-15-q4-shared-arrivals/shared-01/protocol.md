# Queued shared-expert work before routed experts

Declared before native timing. The preceding verified-storage replay preserved
packed Q4's GPU advantage with six misses per eight-expert batch, but did not
reproduce its full-request regression. This stage adds the actual three shared
operations queued by Model::moe before execute_experts. Production inference,
model weights, arithmetic and defaults remain unchanged.

Run two independently sealed conditions with fresh samples: control without
shared work, and one shared chain queued before each eight-expert batch. Each
condition has five alternating reference/packed pairs at group caps one/four.
Do not reuse earlier timings, pool conditions or interpret their difference as
a paired causal estimate. Fixed resident weights, eight expert buffers, two
ready hits/six misses, eight readers, two live command groups and the 12GiB
ceiling remain unchanged. Each timing arm runs four cycles of eight batches
(256 routed expert chains). The full stage is limited to 180 seconds per
condition and 60 seconds per native process under the existing exclusive lease
and memory/disk admission. Stop on a failed correctness, identity or resource
gate; do not spend time on a full normal-request qualification here.

One shared chain per batch keeps the work ratio close to production's one chain
per ten selected experts. These eight fixture experts span four layers; this
is a declared diagnostic, not a replay of one real layer's routing. The shared
layer rotates [0,16,32,47,0,16,32,47] and consumes that layer's saved input row
at offsets 72–79. Queue Q8 gate/up (2560→640), Q8 down (640→2560), and the BF16
shared gate (2560→1) in that order. Add no submit or wait before execute_experts.
The initial ready experts share the first command with that queued work.

The scalar gate is the shared_expert_gate linear output; final MoE weighting,
reduction, attention, routing and recurrent updates remain outside this replay.
Changing those operations is not part of this stage.

Use an independent NumPy CPU reference for the pinned mixed checkpoint's
actual affine-Q8/BF16 shared tensors and original saved input rows. Bind its
manifest to selected tensor hashes, checkpoint identity and input hashes. The
native checker hashes the actual resident buffers against those tensor hashes.
Validate 32 layer/row cases before measurement: activation and down-output
relative L2 <=0.01 and cosine >=0.99995; scalar gate absolute error
<=1e-6+abs(reference)/128. These are CPU/operator tolerances, not permission for
the candidate to change outputs. Establish native reference bytes separately
and require byte-identical shared activation/down/gate and routed outputs under
both Q4 variants, including the final timed batch. Keep destination canaries.
Validate every batch in check/trace and warm/post-timing passes. CPU preparation
and hashing occur outside measurement. Keep the same native shared-output hash
across all processes. Production release still requires full-model checks.

Memory includes the existing scratch arena plus three shared buffers (48KiB
allocated capacity) per batch and 819,456 bytes of retained CPU/native reference
payload. Shared output views survive until GPU completion and are cleared before
scratch reuse/release. Timing performs no new Metal allocations after identical
warmup. Shared arms require 32 of each shared dispatch and 864 scratch reuses;
control arms require no shared dispatches and 768 scratch reuses. Both execute
256 of each routed gate/up, down and scatter operation.

Keep explicit selected-file-range invalidation after GPU/I/O drain and before
hit priming. Report 256 invalidations per timing arm inside preparation, excluded
from measured work. Application bytes are 530,841,600 during demand and
176,947,200 during preparation. Every timed arm must observe device read bytes
within 90–110% of their sum. Systemwide counters include unrelated processes;
this establishes neither per-process attribution nor NAND/controller-cold state.

Each condition has separate check, timing and trace processes. Check enables
Metal validation; timing enables neither validation nor profiling. Shared trace
uses existing command-group profiling plus dependency events, without per-pass
GPU counters. Control trace retains the preceding dependency-only mode. Never
compare trace time to normal timing. Join command submission times to dependency
events; require the three shared operations at the beginning of the first
expert-bearing command, exact remaining routed dispatches, no extra submission,
correct layer/offsets, and safe final-user release. No trace truncation may pass.

Call the timing GPU sum combined shared-and-routed command duration when shared
work is present. That mixed command cannot be apportioned into independent
expert GPU costs. The control measures routed-only commands separately; do not
subtract it as if it were a matched GPU attribution. Report this measurement
limit explicitly rather than adding expensive per-dispatch instrumentation.

Preserve the diagnostic criterion: paired GPU-duration upper 95% bound below one
and combined-wall upper bound at most 1.03 at both group caps, with exact outputs,
clean observed memory/host state and verified device traffic. Missing/disturbed
or inconclusive results cannot pass. If the gain survives shared work, test
full-token scratch lifetime as a separately declared change next. If it does
not, inspect the mixed first command before selecting a narrow scheduling fix.
Neither result overturns the existing normal-request rejection or qualifies
5 tokens/s, 2K/4K, 7K, sustained coding or a production default change.
