# Locate the validation-associated memory difference

Declared before native execution. The preceding scratch-lifetime stage retains
two validation-process resource blocks, one inconclusive fresh batch timing,
and a clean separately instrumented forward trace. Preserve every original
decision. This stage changes only Metal API/shader validation between otherwise
identical memory-attribution processes; no production source or arithmetic change.

Use four fresh processes in order off, on, on, off. Both validation flags are
absent in off processes and present in on processes. Disable command profiling
and hardware counter profiling in all four. Existing aggregate counters remain
available. Each process uses reference Q4, group cap one, one complete 48-pass
warmup, then one complete 48-pass checked execution. The same 48 shared Q8/BF16
chains, 384 routed Q4 chains, eight fixed expert slots, two ready hits and six
SSD-backed misses per pass, eight readers, 128MiB arena ceiling and 20.25MiB
retained scratch remain unchanged. Inputs, bytes, shared CPU oracle, destinations
and reduction arithmetic remain fixed. This does not execute a full model token.

Capture 57 lifecycle observations: startup before Metal construction, pipelines
ready, resident ready, fixtures ready, shared reference ready, routed reference
ready, expert pool ready, post warmup, all 48 observed pass completions, and final
scratch release. Each records process current and peak footprint, current and
peak compression, cumulative decompressions, swap, host state, charged native
allocation, device allocation size, scratch pools and outstanding users. Flush
each observation immediately to a progress file so setup failures retain their
last completed phase. Observations add no GPU submission or wait; pass snapshots
follow drain and exact output checks and sit outside coordinator windows.

Require identical native work/allocation signatures and output hashes across
all four processes. Require complete phase order, growing used scratch prefix,
zero new allocations during the observed execution, 1,296 scratch reuses,
384 submissions and zero live users at each boundary. Final scratch release
must free exactly 21,233,664 charged bytes while resident/expert allocations
remain held. Do not describe that boundary as complete model destruction.

Compression is the outcome to observe in this attribution-only stage. Preserve
strict resource_clean=false whenever current/peak compression is nonzero or
decompression counters change; do not relabel it a clean performance result.
Hard guards remain in force at every native observation: known positive process
current/peak footprint <=12GiB, native charged allocation <=12GiB, stable known
system swap usage and power source, nominal thermal state and low-power off.
Persist the offending phase before a hard guard stops the process. Missing
gauges fail closed. Keep the existing exclusive GPU lease, native admission,
source/artifact checks, disk admission, evidence cap, 60-second process limit
and 180-second stage limit. No system-limit changes or global cache purge.

Prepared ranges are explicitly invalidated before hit priming as in the earlier
diagnostic. Each observed execution reads 796,262,400 demand bytes plus
265,420,800 preparation bytes, with 384 invalidations. Require observed device
read bytes within 90–110% of this total, retaining the limitation that systemwide
counters include preparation and unrelated processes.

Two clean off processes and two dirty on processes with the same first dirty
phase support a reproducible phase-localized association with validation. They
do not identify compressed pages or provide a statistical causal estimate.
Otherwise report the captured observations without inventing an attribution.
Boundary snapshots and cumulative peaks cannot locate events within phases;
device-allocation differences are not direct driver physical-memory measurements.

If both off processes complete cleanly and both on processes complete exact
work without a hard guard failure, a separate timing-only follow-up is admitted.
That follow-up runs one fresh forward and one fresh batch condition, each with
five alternating reference/packed pairs at group caps one/four, using the
existing uninstrumented timing mode, fixed work, source/byte guards and a
60-second process /180-second condition cap. No setup phase observer is active
there. Keep every previous timing observation, including the slow batch sample;
do not pool it with fresh pairs or replace the earlier decision.

If all four attribution processes have clean memory, prefer a fresh full
check/timing/trace sequence for each scope using the existing lifetime runner
and its unchanged original gates. This adds the ordinary validation and trace
checks rather than carrying forward the earlier blocked validation result.
The new complete sequences remain separate from prior attempts and qualify
only the bounded diagnostic if they pass, never a production or full-request
change. Do not run both follow-up variants or retry a failed fresh sequence.

Such timing-only results may describe this partial replay's performance but
cannot qualify the original check/timing/trace stage, a production change or a
normal request. Preserve the original_lifetime_stage_qualified,
normal_request_latency_qualified and production_promoted flags as false.
Comparison thresholds remain GPU upper paired 95% bound <1 and coordinator
wall upper bound <=1.03 at both caps, with unchanged exactness, memory, host,
storage and identity requirements. Missing or disturbed measurements do not
pass. No further repeat is prescribed after the fresh two-condition follow-up.
