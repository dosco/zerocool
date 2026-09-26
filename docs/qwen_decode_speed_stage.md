# Decode speed: GPU clock, row kernels and next-layer prefetch

Single-token generation on the Q4 control reached **5.3–5.5 tokens/s** after
the established 72-token coding prompt and **4.9 tokens/s** after the retained
128-token append, with the new defaults and 850 expert slots on the 32GiB
M1 Pro. The same build with the previous defaults measured 2.40 and 2.27
tokens/s. Every generated token, all 248,320 final logits and all 48 layers'
recurrent, convolution and attention state are bit-identical between the two.

On the 2K/256 acceptance workload, one clean run measured **5.86 tokens/s** with
the 1,977 expert slots that free memory allowed. Pinned to 900 slots, it
measured 4.89.

These are single screens on a busy laptop, not the repeated 2K/4K/7K
qualification. [The measurements section](#measurements) states each run's
conditions.

Four changes produced the gain. None alters arithmetic, routes, experts,
precision or artifact bytes.

## 1. The GPU was running at a low clock

The per-dispatch profile put single-token GPU work at roughly 300ms per token,
about ten times the memory-bandwidth bound for the ~3.9GB of resident and
expert weights a token reads. Kernel for kernel, the engine ran about three
times slower than the same kernels in an isolated loop: 7.8ms versus 2.5ms for
`lm_head`, 316µs versus 109µs for a GDN input projection.

The cause is the GPU's performance controller. Decode is a long sequence of
short command groups separated by CPU work: 48 router boundaries per token,
reads and completion waits. A bounded probe showed the effect directly: one
14.7MB Q4 projection per command buffer took 363µs (41GB/s); the identical
dispatch took 108–119µs once the GPU had sustained work, including when that
work was a single 32-thread spinning SIMD group on a second queue. Adding idle
gaps between dispatches made no further difference, and a residency set did not
remove the effect.

`Metal::keep_warm` implements that remedy. A dedicated command queue and thread
run one SIMD group of dependent integer arithmetic (`gpu_keepwarm`, no model
data, a 16KiB accounted sink) whenever a forward has started within the last
250ms. `Model::forward` renews the window; the thread sleeps otherwise and is
joined before the Metal device is released. With it, decode GPU time fell from
about 200 to about 93ms per token and generation went from 3.37 to 4.88
tokens/s in otherwise identical runs. `--gpu-warm off` disables it.

## 2. Single-token row kernels

The reference Q4 matrix-vector kernel gives each output row its own SIMD group,
so every row reloads the whole input vector and recomputes its BF16 bias sums.
`q4_mv_r{4,8}_w{8,16}` and `q4_gate_mv_r2` let each lane load its input chunk
and bias sum once per block and apply it to four or eight rows (two for fused
gate/up). The lane partition follows `affine_dot`'s width rule
(`K%512==0 && N%8==0 ? 16 : 8`) and each row keeps the reference expression
order, BF16 boundaries and final `simd_sum`, so the outputs are bit-identical.
Only load reuse changes.

Two latency-bound kernels got the same treatment:

- `plain_mv` (single token, N ≤ 64) stages 2048-element chunks of the input and
  weight row in threadgroup memory, then SIMD group 0 runs `plain_mm`'s exact
  lane chain from on-chip memory. It replaces four-SIMD-group dispatches whose
  serial chain paid one memory round trip per step (hyper-connection mixes,
  shared-expert gate, GDN gate projections).
- `norm_wide` computes the grouped RMS norm with one SIMD group per 128-element
  block, combines the block sums in the same `simd_sum` lanes as
  `rms_square_sum`, and writes outputs with the whole threadgroup. It applies to
  every token count.

Single-row expert contributions are written in place instead of through
`scatter_experts`; the bytes are those the copy would have written.

Isolated screens on real shapes, with random codes, compared raw FP32 sums
(before output rounding) as well as BF16 outputs: every admitted shape matched
exactly, and deliberately wrong lane widths were detected. The large resident
projections ran 2.6–3.1× faster in isolation, experts about 2×, `plain_mv`
about 3× and `norm_wide` 8×. With the clock held up, whole-token GPU time fell
from about 93 to about 66ms. `--q4-rows off` restores the reference kernels.

## 3. Next-layer expert prefetch

Each layer's experts were read only after that layer's router, so the SSD saw
four or five reads at a time and the GPU waited on every miss. At each router
boundary the engine now also evaluates the next layer's router on that layer's
own MLP hyper-connection of the current post-attention stream
(`--expert-prefetch next-hyper`) and starts future-priority reads for the
predicted experts that are not cached. Details:

- Speculative entries enter CLOCK unreferenced and are counted separately.
  `ExpertCache::prefetch` uses the same victim rules as demand, so it never
  evicts a leased or loading slot.
- `execute_experts` issues them only after every selected expert of the current
  layer holds a lease, so a guess can never evict a selected hit.
- A demand acquire that finds its expert still queued as a guess promotes that
  read to the demand queue (`ReadPool::promote`).
- Routes and computed experts are unchanged; only cache timing differs.

In the 96-token screen the predictor's top-ten guesses matched 77% of the next
layer's actual experts, against 63% for the plain `next` variant (the next
router on this layer's MLP input). Precision falls steadily with rank, from 99%
for the first guess to 44% for the tenth. Demand misses fell from 263 to 86
per token at 900 slots. Depths of 6, 8 and 10 were screened; 10 is the default
(`--prefetch-depth`).

## 4. Defaults

`q4_rows`, the keep-warm, SIMD route selection (qualified exact in
[its stage](qwen_route_selection_stage.md)), next-hyper prefetch and bounded
single-token scratch reuse are now the defaults for every command, including
`serve` and `chat`. `Options::resolve()` turns `auto` into `reuse` and
`next-hyper` only when the schedule supports them: completion pipeline, wait
tail, reference decode path, serial fixed memory, original GDN, CLOCK and a
resident trunk. Otherwise it selects `none` and `off`. Statistics report the
resolved values.

Residency also defaults to `auto`, which selects core-plus-expert residency
(`core-cache`) wherever Metal residency sets exist and `off` elsewhere, such as
on virtualized CI GPUs. On short prompts it changed speed by under 1%. On the
2K prompt it removed the memory compression this project has recorded during
long ingestion since September 13. Without it, both 2K runs compressed exactly
3.34GiB of the engine's pages while ingesting and then paid 50–65 thousand
decompressions during decode. That happened even at 900 slots with 5.9GiB
reported reclaimable. With it, the same run peaked at 0.09GiB compressed with
132 decompressions.

The keep-warm sink is counted as a second runtime control page
(`runtime_control_bytes` is 32KiB when it is on). A demand acquire that finds
only leased or loading slots now waits for an unclaimed guess to finish instead
of failing admission. That case is only reachable with a few dozen slots.

## Measurements

[The benchmark record](benchmarks/2026-09-25-decode-speed/README.md) and its
`summary.json` bind the headline runs to build fingerprints, report hashes,
resolved settings and output-token hashes.

All runs used the Q4 control, prepared records, the 72-token coding prompt from
the earlier screens (96 generated tokens), then where stated a retained
128-token append (48 generated tokens). Settings: temperature zero, fresh
processes, and a 12GiB request admitted at 6.5–7.9GiB because other
applications held memory. Expert slots were pinned at 850 or 900 so arms were
comparable. The host was not otherwise idle; one pair ran during an unrelated
six-core compile and is excluded. Tokens/s is the decode rate reported by
`zerocool bench`.

| Configuration (slots) | Initial | After append | GPU ms/token |
|---|---:|---:|---:|
| Previous defaults (850) | 2.40 | 2.27 | 279–291 |
| Previous best experimental flags (900) | 3.04 | — | 222 |
| + row kernels (900) | 3.17 | — | 193 |
| + next-hyper prefetch (900) | 3.37 | — | 200 |
| + external clock probe (900) | 4.96 | — | 94 |
| + in-engine keep-warm (900) | 4.88 | — | 93 |
| + `plain_mv`, `norm_wide` (900), three runs | 5.02–5.49 | — | 64–66 |
| New defaults (850), four runs | 5.29–5.41 | 4.89 | 64–66 |
| Final defaults incl. residency (850) | 5.29 | 4.89 | 65–68 |

The 2K acceptance workload is `docs/benchmarks/2026-09-08-validation/workload-2k.json`:
a 2048-token prompt, 256 generated tokens, panel 512.

| 2K/256 run | Slots | Decode tokens/s | First token | Compression |
|---|---:|---:|---:|---|
| Defaults before residency, unpinned | 1492 | 5.24 | 200s | 3.34GiB during ingest |
| Same, pinned | 900 | 4.86 | 199s | 3.34GiB during ingest |
| With core-cache residency, pinned | 900 | 4.89 | 194s | none (0.09GiB) |
| Final defaults, unpinned | 1977 | 5.86 | 194s | none (0.10GiB) |

With the GPU fast, the SSD is the bound again. Demand misses plus prefetches
come to about 350 expert records (~970MB) per token, or about 5.2GB/s at
185ms per token, at the SSD's measured random-read ceiling. Sixteen readers
instead of eight changed nothing.

## Negative results kept

- **Splitting an expert read** across workers: a lone 2.76MB uncached read
  already takes about 0.9ms because the controller parallelizes it, and five
  concurrent experts saturate at 4.9GB/s whether split or not.
- **Per-layer LFU with equal soft quotas.** An offline simulation of the
  captured routes predicted about 10% fewer demand misses than CLOCK without
  prefetch, but with prefetch it raised demand misses from 88 to 113 per token
  and lost both pairs (5.03/5.25 versus 5.39/5.43 tokens/s). A per-layer quota
  near 18 slots lets each layer's own guesses evict its hot experts, whereas
  global CLOCK takes prefetch victims from the globally coldest entries. The
  policy was removed.
- **Prefetch depth 6 or 8** did not beat 10 on the same footing.

## What remains

- The 5 tokens/s gate is defined over 256 generated tokens after 2K and 4K
  prompts, with repeated comparable runs.
  - Single 2K runs so far come out at 4.9 tokens/s at 900 slots and 5.9 with
    the final defaults at 1977 slots, so on this host the result depends on how
    much memory other applications leave for the expert cache.
  - The 4K run, repeated pairs, the 7K report and the sustained coding workflow
    are still to be done.
- Generation is now SSD-bound. Further gains need fewer bytes per token:
  - more expert slots (freed trunk or state memory, or a less loaded host);
  - fewer wasted guesses (about 74 unclaimed per token at depth 10);
  - a replacement policy that works with prefetch.
- First-token latency is unchanged in kind.
