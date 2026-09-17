# Memory, cache capacity and verification must be planned together

This stage tests a larger change than the previously proposed gate/up kernel
tweak. It adds exact streamed token embeddings and asks whether more tokens can
share a target-model pass. Native inference remains C++/Metal. Every path is an
isolated developer producer; no production default or artifact is changed.

## Implemented and checked

- A 256-row shared target/draft cache preserves packed Q8 embedding codes and
  BF16 scales/biases. It replaces 675,446,784 aligned resident bytes with a 2MiB
  host allowance, reducing planned memory by **642.15625MiB**. Each miss reads
  2720 bytes through the existing uncached checkpoint files.
- Per-call Metal buffers own gathered rows until GPU completion. Ten real-table
  checks match the full resident table, including repeated IDs, both vocabulary
  endpoints, 128-row batches, FIFO eviction, fourfold replication, eviction before
  GPU submission, invalid IDs, a failed read after a successful first range, and
  owner destruction. Cache capacity and failed-read publication are bounded.
- The verifier accepts eight token inputs with exact full-vocabulary outputs.
  The initial packed Q8 extension increases its independent token dimension.
  A second implementation groups eight tokens for storage while using four-token
  GPU tiles. Both pass low-memory Metal checks and nine real-model input checks,
  including an eight-row block, one-row tail and full serial state boundaries.
- All arms retain 12GiB joint admission, 1460/32 expert slots, reference Q4,
  packed Q8, direct output, lazy ngrams, no expert scratch reuse, two GPU groups,
  and 8192 context. The real MTP object is allocated and primed, then idle during
  the verifier ceiling. Wider MTP proposals/rejection are not implemented here.

## Measurements and limits

All rates below use known-correct continuation inputs, excluding proposal
generation, rejection recovery and draft catch-up. They cannot establish
5 generated tokens/s. Each row is its own fresh paired comparison over the
same 64-token interval-merging continuation; no timings are pooled across rows.

| Screen | Four-token control | Eight-token candidate | Decision |
|---|---:|---:|---|
| Eight-token GPU tile | 5.6127 verified tokens/s | 4.3174 | Stop: insufficient headroom |
| Four-token GPU tile, eight-token storage window | 5.5641 | 5.4303 | Stop: insufficient headroom |

Normal timing in both pairs has zero compression/decompression, unchanged swap,
AC power and nominal thermal state. The second pair peaks at 9.0476GiB physical
footprint. One pair per candidate does not provide a confidence interval; the
predeclared 15% reduction and 6.5 verified-token/s thresholds are early rejection
rules, not promotion criteria.

The first pair's eight-token path increases measured GPU command time per token
from 98.23 to 151.57ms. The separate tiled pair measures 100.73/96.80ms, removing
that observed penalty within its fresh comparison. This does not isolate the
hardware mechanism, and no cross-producer speedup is claimed.

The remaining cache problem is concrete. The first pair visits about 1237
distinct layer/expert pairs per four-token pass and about 1990 per eight-token
pass. At 1460 slots, the eight-token path records **zero cache hits**; four tokens
retain 31.23%. Application expert traffic rises from 560.71 to 655.72MiB per
verified token. Those are application reads, not isolated device traffic. The
tiled eight-token path also records zero hits. More work per pass cannot help
when the resulting access pattern defeats the cache.

Both original full-qualification attempts remain resource-blocked during serial
validation setup: resident 28.8125MiB compression, streamed 33.59375MiB. The
streamed serial physical peak is about 641.6MiB lower, consistent with the
allocation change; this is a diagnostic comparison, not a clean paired memory
or latency qualification. The eight-token numerical diagnostics later complete
cleanly. Reusing the compressed serial run checks numerical equality only.

The suite passes **525 Python tests**. Real MTP generation with streamed rows,
eight-row rejection, longer prompts, 7K context, a sustained coding session and
production integration remain unqualified.

## Next experiment

1. Carry the exact embedding provider into the existing real-MTP width 1/4
   experiment. Compare every proposal, logit and persistent state with resident
   embeddings, keeping 1460/32 slots. Establish its actual request memory and
   latency cost before adopting it as the control. Do not spend the saved bytes
   on a larger cache in that same comparison.
2. Capture a bounded complete cache access sequence for four- and eight-token
   storage windows with compute tiles capped at four. Include priming, actual
   leases and per-expert token-row counts. Reproduce native CLOCK exactly before
   any simulation. Use several complete eight-token passes, and state the
   coverage limit explicitly.
3. Screen two distinct ways to avoid the observed cache thrashing: return unused
   prefill workspace to the generation cache, or bypass one-use experts through
   a bounded transient pool while retaining recurrently useful experts. The
   latter changes admission, unlike the previously rejected eviction-only SLRU
   experiment. Simulated read savings are not predicted latency.
4. For phase memory, first prove the byte budget and lifetimes. Current streamed
   joint admission permits at most 1909 slots with the unchanged workspace
   reserves. A separately admitted generation workspace of 128MiB instead of
   512MiB would make about 2048 slots possible inside 12GiB, retaining the driver
   reserve. This is allocation arithmetic, not a validated runtime plan. Drain
   all prefill users before changing plans; restore prompt capacity before an
   append that needs it. Keep panel/state/context allowances and rollback explicit.
5. Only a cache candidate with a substantial trace benefit gets a fresh normal
   request comparison. Revisit wider MTP only if verifier cost plus measured
   acceptance, proposal and recovery costs support 5 generated tokens/s. Keep
   original-source Q3 calibration as the separate quality-controlled alternative
   if exact storage/scheduling cannot provide the required working set.

The older CLOCK 1536 and SLRU experiments are not reopened unchanged. This stage
identified a different approximately 1990-expert pass and a phase-memory/admission
question. The first deliverable of the next step is a measured memory reduction
on real MTP and a cache trace, not another larger draft model.

Protocols: [original horizon](protocol.md), [streamed rows](streamed-protocol.md),
[early numerical/timing split](early-protocol.md), [compute/storage tiling](tiled-protocol.md).
Raw attempts: [resident](screen-01/summary.json), [streamed](streamed-screen-01/summary.json),
[first early pair](early-01/summary.json), [tiled pair](tiled-early-01/summary.json).
