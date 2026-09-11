# Extra memory headroom did not improve this short conversation

The bounded stage completed in **515.55 seconds**, including real-model
correctness, three diagnostic conversations and two alternating capacity pairs.
The smaller cache failed the predeclared speed screen, so no five-pair
confirmation or long qualification ran. Defaults remain unchanged.

## Measured capacity tradeoff

Both arms used the actual 32GiB M1 Pro, the same 12GiB ceiling, pinned mixed
4/8-bit artifact, unchanged prepared Q4 records, CLOCK, residency off, reference
kernels, panel 512, microchunk 128, four ready experts and eight readers. Each fresh
process executed 72 prompt tokens + 33 outputs, then a retained 128-token append
+ 33 outputs. Every follow-up reused 104 computed tokens and ingested 129 tokens
including the pending previous output. Metal validation and detailed profiling
were disabled during timing; boundary GPU probes were enabled in both arms.

| Pair | Order | 1,848 slots | 1,460 slots | Smaller/current ratio |
|---|---|---:|---:|---:|
|0|Current, smaller|63.378s|63.983s|1.00954|
|1|Smaller, current|63.569s|63.950s|1.00598|

The median conversation ratio was **1.00776**: the smaller cache was 0.78% slower
in this two-pair screen. Both ratios were above 1, failing the requirement that
both improve and the median be at most 0.99. Two pairs do not establish a precise
regression estimate or a confidence bound; they establish no qualifying benefit.
The [offline comparison](comparison.json) reconstructs the same decision from
all four original reports rather than trusting copied summary metrics.

The capacity change left **1,074,331,648 bytes** more room in the admitted plan.
Request-end process footprints fell from approximately **10.41GiB to 9.42GiB**.
This reduction did not bring a measured latency benefit under these conditions.
There were no process decompressions during any of the screen's ingest/decode
phases. Expert application reads increased from **77.927GiB to 81.434GiB per
conversation**, about 4.50%. This includes initial ingestion, generation, append
and subsequent generation; application bytes are not physical SSD traffic.

## GPU reference observations

The preceding three baseline conversations took 66.10, 63.83 and 63.46 seconds.
Their warm reference operation, measured at the start and end of each complete
conversation, took:

| Conversation | Before | After |
|---|---:|---:|
|0|532.02µs|500.81µs|
|1|633.95µs|500.00µs|
|2|523.59µs|501.92µs|

The earlier large late-generation GPU slowdown was not reproduced in this
small sample. The first diagnostic generation recorded 94,877 decompressions;
subsequent diagnostic generation phases recorded zero except for two in its
first append. These are phase observations, not token-level attribution.

Each probe used the same existing resident Q8 matrix and exactly 48KiB of
transient buffers. Every checksum matched; GPU users drained and live allocation
returned to its previous value. Before/after probes also confirmed unchanged
expert/ngram counters and memory plans. All host snapshots reported AC power,
nominal thermal state and low-power mode off, with the SDK's unknown/unsupported
limitations retained. No temperature, GPU frequency or wattage was measured.

Probe-enabled timings are diagnostic. They can warm hardware and touch memory,
even outside request timers. Endpoint measurements do not prove conditions
throughout a conversation or explain the older slowdown. GPU durations and
CPU/I/O waits overlap and must not be summed into a critical-path estimate.
Sampled footprints do not establish continuous physical peaks or sustained swap
behavior. Ordinary probe-off performance remains unqualified by this stage.

## Correctness, evidence and implementation

**57 native tests / 45,704 assertions passed** with Metal API and shader validation.
**185 Python tests passed**, including timing/protocol validation, strict capacity
accounting, revalidation from raw sources, duplicate/incomplete-pair rejection,
and nested report checksums. The original capture ran after 184 Python tests;
the additional test covers the packaging correction described below.

The bounded real-model checks passed with probes both disabled and enabled,
using all 48 layers and 32 slots to force eviction. Logits, routes and every
persistent-state snapshot were identical. Continued state matched fresh replay;
cancellation and injected failure invalidated partial state and drained users.
These parity checks are not an independent full-model oracle or coding-quality
assessment. The new probe's synthetic operator test uses an independent CPU
result and verifies fixed reference dispatch even with candidate kernels selected.

The new `--gpu-reference` and `--bench-progress` controls, bounded runner,
comparison query, tests and usage contract are described in the
[stage documentation](../../qwen_memory_balance_stage.md). The inference
arithmetic, quantization and production defaults are unchanged.

The final archive check found that the generic outer checksum writer omitted
nested `evidence-files.json` entries. All original payload hashes and both
individual stage manifests verified. The writer now excludes only its own
manifest, with a regression test. The original root index and exact pre-fix
helper/test sources are preserved; the corrected index covers those records
too. No measurement or decision was changed. See the
[packaging correction](raw/seal-correction.json).

Native build: `1ddec378e0a2550eda51c351e5e9315863c15b61773617c457b4d8e2f8df9189`.
Mixed artifact: `b2c422f3c643e36f04227a64d61796b44a4b1029`.
[Raw stage result](raw/summary.json), [derived observations](observations.json),
[native tests](native-tests.log), [Python tests](python-tests.log) and
[phase progress](runner.log) retain the evidence.

The bounded stage is complete. The cache-capacity hypothesis is not supported
for speed on this workload; further capacity sweeps, new cache policies,
compression changes and long-context qualification were not started. The
approximately 2.5 tokens/s generation and 24-second follow-up first-token latency
remain below the product targets.
