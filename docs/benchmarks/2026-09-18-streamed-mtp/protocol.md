# Exact embedding storage in real MTP

Registered before model execution. Compare resident embeddings with the exact
256-row packed embedding provider in one native producer. Width one and width
four are separate comparisons. Keep the mixed artifact, 1460 target/32 draft
slots, 12GiB total admission, 8192 context, full target replay on rejection,
reference Q4, packed Q8, direct expert output and lazy ngrams unchanged.

The candidate removes 675,446,784 resident bytes and admits a 2MiB host cache:
673,349,632 bytes (642.15625MiB) less planned allocation. Keep the saved space
unused in this experiment. Arithmetic, expert replacement and precision do not
change. Target and draft share the provider; each GPU use owns copied row bytes.

1. Run the ten real embedding fixtures with Metal validation, including cache
   eviction during outstanding GPU use, failed reads and owner destruction.
2. Compare six fresh resident/rows numerical pairs: width one with seven output
   tokens; width four with first accepted prefixes 1/2/3/4 and a seven-token tail;
   immediate EOS. Compare every actual proposal, every draft and verification
   logit hash (including rejected rows), primed state, committed state boundaries
   and final target/draft buffers. Compare target outputs with the existing
   independent oracle. Do not reuse old timings or claim oracle resource fitness.
3. Only after clean numerical validation, run up to two alternating 64-token
   interval-merging pairs at each width (resident/rows, then rows/resident).
   Stop a width early if an observed pair has over 10% latency regression.
   Report widths separately, actual peak physical memory, planned bytes, decode
   time, real generated tokens/s and embedding read counts. This is a storage
   cost screen; no speedup or larger cache is assumed. Five-pair confidence,
   other coding cases and full session acceptance remain outstanding.

Keep the existing host/memory rules: 13.5GiB current available memory for a full
model process, AC power, Low Power Mode off, nominal thermal state, no process
compression/decompression or changed swap. Missing/resource-disturbed evidence
stays unqualified. Validation has an 1800-second stage bound; normal timing has
1200 seconds; each model process is limited to 180 seconds. Acquire the existing
GPU lease, preserve all raw reports and source identities, and drain native users
on cancellation. Production defaults remain unchanged.

Next: only a numerically exact and acceptably cheap provider can become the
control for a separate cache admission/phase-workspace experiment. Trace actual
lease lifetimes and reproduce CLOCK before extrapolating read savings.
