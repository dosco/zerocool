# Building an inference engine for a 104GB model on a 32GB laptop

This is a walkthrough of how ZeroCool runs Qwen3.8-Flash-Next on an Apple M1 Pro
with 32GiB of unified memory, written for someone who wants to build something
like it. It covers the machine, the model architecture, the design that follows
from putting those two together, the optimizations that were tried, and what the
measurements actually showed.

**On the numbers.** Generation has gone from 1.75 tokens/s on the first
working build to 5.3–5.5 tokens/s after a short prompt, and 4.9 after an
append, with the current defaults. One clean 2K/256 run measured 5.86 tokens/s
(section 10). The 5 tokens/s gate is defined over repeated 2K and 4K runs and
remains open until those are qualified. Where this document gives a number, it is a
measured one with its conditions; where it gives a target, it says so. That
distinction is the most transferable habit in the project, so it is kept here.

## 1. The problem

The checkpoint is 103,769,745,912 bytes of tensor payload across 11 safetensors
shards and 3,215 tensors. The machine has 32GiB of memory shared between CPU and
GPU, of which the engine may use at most 22GiB, and in practice is admitted
around 12GiB because other applications are running.

The model does not fit. It is not close to fitting. So the engine's real subject
is not matrix multiplication, it is **deciding what to keep in memory and
scheduling everything else off the SSD without stalling the GPU.**

What makes this tractable is the shape of the payload:

| Payload | Bytes | Fraction | Where it lives |
|---|---:|---:|---|
| Routed experts | 67,947,724,800 | 65.5% | SSD, read on demand |
| Ngram tables | 32,000,153,600 | 30.8% | SSD, indexed row reads |
| Resident forward allocations | 2,937,683,968 | 2.8% | Memory, always |
| Other tensors | 3,821,867,512 | 3.7% | Mostly outside the forward path |

Only 2.9GB has to be resident. The other 97% is touched sparsely: a
mixture-of-experts layer consults 10 of 512 experts per token, and the ngram
table is a lookup. That is the entire reason this is possible.

## 2. The roofline that decides everything

Work out the traffic per token before writing any code. This model has 48
layers, each selecting 10 of 512 experts, so a single token touches
48 × 10 = **480 expert records**. Each record is 2,764,800 bytes of payload in a
2,768,896-byte aligned slot. So one token pulls

```
480 × 2,764,800 = 1,327,104,000 bytes ≈ 1.33 GB per token
```

if nothing is cached. Now measure the SSD. These are medians of three
repetitions of real expert-sized reads with `F_NOCACHE`, on this laptop:

| I/O workers | Expert read GB/s | Minimum cache hit rate for 5 tok/s | For 8 tok/s |
|---:|---:|---:|---:|
| 1 | 1.28 | 80.7% | 88.0% |
| 4 | 3.75 | 43.6% | 64.7% |
| 8 | 4.88 | 26.4% | 54.0% |
| 16 | 5.62 | 15.3% | 47.1% |
| 32 | 5.82 | 12.3% | 45.2% |

The hit-rate column is `max(0, 1 - measured_Bps / (bytes_per_token × target_tps))`.
It assumes compute, routing, ngram reads and synchronization take *zero* time,
so it is a lower bound that no real system reaches.

Three conclusions fall out immediately, and they shaped the whole engine:

1. **Reading is the budget.** At 8 workers you need better than a 26% expert
   cache hit rate before 5 tokens/s is even arithmetically possible.
2. **Parallel reads matter enormously** — 1 worker to 8 nearly quadruples
   bandwidth — but the returns flatten after 8, so 8 persistent workers is the
   default rather than 32.
3. **Cache capacity is a first-class design parameter**, not a tuning detail.
   Every byte spent on something else is a byte not spent on expert slots.

If you build one of these, do this calculation first. It tells you whether your
target is reachable at all, and it tells you which optimization is worth
attempting.

## 3. The machine: what Apple Silicon gives you and takes away

**Unified memory** is the enabling feature. CPU and GPU address the same
physical pages, so a buffer read from the SSD into memory is usable by a kernel
without a copy across a PCIe bus. On a discrete GPU, streaming 1.33GB per token
over the bus would end the discussion. Here the read lands where the GPU can use
it. A buffer visible to both is counted once in the memory budget.

**The memory budget is enforced, not advisory.** Metal reports a recommended
working-set size; exceeding it causes the system to compress or swap, and
compression on this path is fatal to a measurement because it silently turns a
memory-bound workload into a CPU-bound one. So the engine computes an admission
budget before allocating anything:

```
min(requested budget,
    physical memory - 8GiB,
    Metal's recommended working set,
    currently reclaimable memory - 1.5GiB)
```

and then charges every allocation against it: resident weights, recurrent and
attention state, expert cache slots, scratch, a separate 64MiB ngram cache, and
a 1GiB CPU/driver reserve. If admission fails, the engine refuses to start and
explains why, rather than running into swap. No code raises macOS wired-memory
limits, because a program that quietly reconfigures the machine it runs on
cannot produce an honest measurement.

**One device, one work queue.** The engine wraps a single `MTLDevice`, and all
model work goes through a single `MTLCommandQueue`; the expert executor keeps at
most two GPU groups in flight against it. Dispatching is cheap; the expensive act is waiting. So the rule is: *encode
freely, wait only at real dependencies.* Those dependencies are routing (the CPU
must read which experts were selected), sparse selection, and any point where a
buffer is about to be reused. Not after every kernel.

**The GPU clock follows the shape of the work.** Single-token decode cannot
avoid its waits: every layer's router result has to reach the CPU before that
layer's experts can be chosen and read. That is 48 CPU dependencies per token,
plus read and completion waits. The GPU therefore sees a long stream of short
command groups separated by idle gaps, and Apple's performance controller treats
that as a light load and holds a low clock.

The effect is large and easy to miss. The same kernels ran about three times
slower inside the engine than in a benchmark loop. One 14.7MB projection took
363µs dispatched alone per command buffer and 108–119µs once the GPU had
sustained work, and adding gaps between dispatches changed nothing further. The
engine therefore runs a second queue that exists only to hold the clock: while a
forward has started within the last 250ms, it keeps one 32-thread SIMD group
spinning on dependent integer arithmetic. It reads no model data. That halved
whole-token GPU time. Any kernel timing taken in a tight loop describes a faster
GPU than the one decode actually gets.

**Completion, not submission, is what releases a resource.** A GPU buffer
remains in use until its command group completes; a cache slot cannot be
overwritten while an encoded kernel still references it. This single fact
generates most of the engine's ownership machinery, and is where a naive
implementation corrupts its own weights under memory pressure.

**Metal source is compiled at runtime** from a string embedded in the binary, so
there is no separate metallib to ship or version-match. The cost is a per-process
compile of roughly 49 kernels, which is why the server now builds every pipeline
before reporting ready rather than paying it on the first request.

Allocations are tagged by class — `Temporary`, `Resident`, `State`, `Expert`,
`Snapshot`, `Workspace` — which makes memory accounting explainable rather than
a single opaque total. That sounds like bookkeeping until the first time a
measurement is invalidated by an unexplained 51MiB, at which point it is the
only thing that lets you find it.

## 4. The model: what Qwen3.8-Flash-Next actually does

The architecture is fixed and validated at load time. Per layer, in order:

1. **A four-stream gated residual** ("hyper-connection"). The residual stream is
   four copies of the 2560-wide hidden state, 10,240 floats wide. Each block
   reads a normalized mix of the four streams and writes back an injection. This
   is why the engine's `Hyper` constant is 10240 and why a per-layer normalize
   is a grouped operation rather than a plain one.
2. **A token mixer, alternating by position.** Three layers out of every four
   use **Gated DeltaNet**, a linear-attention recurrence; every fourth layer
   uses **full sparse attention**. With 48 layers that is 36 recurrent layers
   and 12 attention layers.
3. **A mixture-of-experts feed-forward**: a router over 512 experts selecting
   exactly 10, plus a shared expert that every token uses. Expert intermediate
   width is 640.

Two extra pieces sit outside that loop. At one layer the hidden state is mixed
with a **per-layer-embedding (PLE) lookup** driven by an **ngram table**: a
3-gram hash of the current and two previous tokens indexes 16 rows per token out
of the 32GB table. And the vocabulary is 248,320 entries wide, which makes the
final logits projection one of the largest single matrices in the forward pass.

### Why the layer mix matters for memory

The two mixer types have completely different state profiles, and the engine
allocates accordingly:

| Layer type | Per-layer state | Grows with context? |
|---|---|---|
| Gated DeltaNet (36 layers) | Convolution history, plus a fixed recurrent matrix | No |
| Sparse attention (12 layers) | Keys, values, and an index stream | Yes |

A recurrent layer's state is *constant size* no matter how long the
conversation. Only the 12 attention layers pay for context. That is what makes
an 8192-token context affordable inside a 12GiB budget alongside a 2.9GB
resident trunk and an expert cache. If all 48 layers had been full attention,
the KV cache alone would have eaten the expert cache and the roofline above
would have been unreachable.

The attention layers use 24 query heads of 256 dimensions with grouped-query
attention at a ratio of 12, so 2 key/value heads. They are also *sparse*: a
small indexer (4 heads, 128 dimensions, compression ratio 4) scores blocks of
keys and selects a budget of 2048, so a long context does not turn into a
quadratic score matrix.

### The recurrence has a consequence you must design around

A recurrent state cannot be rolled back. If a conversation's history is edited,
truncated, or compacted, there is no way to "un-apply" tokens from the
DeltaNet state — it must be rebuilt from the beginning. This is not an
implementation shortcut, it is a property of the architecture.

So the session contract is: **an exact continuation of the retained token
sequence reuses state; anything else rebuilds it.** The engine reports how many
tokens were genuinely reused rather than how much text happened to match, and a
failed forward marks the state invalid so it can never be silently continued
from a half-applied update. A client that rewrites history pays a full re-ingest,
and it is better to tell the user that honestly than to hide it.

## 5. Storage: the layout is an optimization

The source checkpoint stores each expert as three projections, each with
separate code, scale and bias tensors. Reading one expert from the source layout
therefore means **nine reads**, scattered across a shard. An ngram row means
three.

The engine ships a preparation step that writes a **lossless sidecar**: the same
bytes, rearranged so that one expert is one contiguous read and one ngram row is
one 100-byte read. Nothing is requantized, reordered within a tensor, or
rounded — a byte comparison proves payload equivalence, and the manifest pins
both the source hashes and the prepared hashes. The sidecar costs
100,048,541,696 bytes of additional disk while preserving the original.

This is the highest-leverage change in the whole storage layer, and it is worth
understanding why: an SSD delivers its rated bandwidth on large sequential
reads. Nine scattered reads per expert do not just cost nine system calls, they
forfeit the read-ahead and queue depth that produce the 4.88 GB/s in the table
above. Rearranging bytes on disk bought more than any kernel change did.

Around that sit three smaller decisions that each earned their place:

- **`F_NOCACHE`.** The engine bypasses the OS file cache for expert reads. This
  looks wrong — why refuse free caching? — but the OS cache competes for the
  same physical memory the expert cache is using, and it is not under the
  engine's accounting. Two caches fighting over one budget is worse than one
  cache that knows its size. It also makes measurements reproducible, which
  matters more than it sounds: an early storage experiment produced
  inconclusive results precisely because the first reads were served from the
  OS cache.
- **Eight persistent I/O workers**, with **half the queue reserved for demand**.
  Speculative ngram and expert prefetch must never delay a read that the GPU is
  currently waiting on. Queued demand is always serviced first, and a guess that
  demand turns out to need is moved to the demand queue. Sixteen workers
  measured the same as eight once the SSD was saturated. Splitting one expert
  read across workers did not help either: the controller already parallelizes
  a lone 2.76MB read, which takes about 0.9ms.
- **Page-aligned coalescing.** Requests that begin on the same 4KiB page merge
  into one bounded range, while a lone row still reads only its 100 bytes.

## 6. The expert cache: ownership is the hard part

The expert cache is the engine's central data structure. Default replacement is
global CLOCK; an SLRU variant exists as an explicit experiment. But replacement
policy is the easy half. The hard half is that a cache slot has three states and
only one of them is safe to reuse:

- **Loading** — an I/O job is writing into it.
- **Ready** — it holds valid bytes and nothing is using it.
- **Leased** — an encoded GPU command references it, and *the command may not
  have executed yet*.

A slot may only be evicted from the ready state. Evicting a leased slot does not
crash; it silently computes with the wrong weights, which is far worse, because
the output is plausible text. Every failure path — cancellation, an I/O error, a
memory-pressure shrink, a cache resize — must drain outstanding GPU users before
releasing anything. The engine's test suite spends much of its effort here:
forced eviction under load, cancellation mid-group, resize preserving survivors,
and leases that refuse to release early.

If you build one of these, write the ownership model before the replacement
policy, and test it adversarially. A hit-rate bug costs you speed; an ownership
bug costs you correctness in a way that looks like the model being bad.

Prefetch (section 7) adds a fourth kind of entry: a **speculative** record that
is loading but has no lease. Its rules follow from the ones above:

- A guess is issued only after every expert the current layer selected already
  holds a lease, so it can never evict a hit that is about to be used.
- It takes a victim by the same rules as demand, so it never evicts a leased or
  loading slot.
- It enters CLOCK unreferenced, so a guess nobody claims is the next thing
  evicted.
- A demand acquire that finds its expert still queued as a guess promotes that
  read to demand priority.
- An admission that finds only leased or loading slots waits for an unclaimed
  guess to finish instead of failing.

Claims, unclaimed evictions and promotions are counted separately from demand
hits and misses, so the cost of guessing stays visible.

## 7. Scheduling: three different problems wearing one interface

A request is not one workload. The engine recognizes three shapes and schedules
each differently:

| Shape | Situation | Strategy |
|---|---|---|
| **Single-token generation** | Ordinary decoding | Compact router readback, feed expert input directly, overlap missing reads with available and shared GPU work, and start reads for the next layer's predicted experts |
| **Short append** (≤32 tokens) | Continuing a conversation | Group the token rows by expert; consume live cache hits *before* admitting misses, so a miss cannot evict a hit you are about to use |
| **Large prefill** | First message | Chunk-major by default, or explicitly requested layer-major "panels" |

The scheduler that makes decode work is **completion-driven**, not batched. The
naive design acquires a batch of experts, waits for all of them, then encodes.
That leaves the GPU idle for the slowest read in every batch. Instead:

- The coordinator admits up to 32 leases and keeps two GPU groups in flight.
- Readers and GPU completion callbacks publish events.
- The coordinator encodes experts *as they become ready*, reaps finished
  buffers, and admits replacements without waiting for a batch boundary.

**Cross-layer prefetch** attacks the dependency that completion-driven
scheduling cannot remove. A layer's reads can start only after its own router
runs. With a few thousand cache slots, that meant only four or five misses in
flight per layer, an SSD far from saturation, and a GPU idle on every miss.

So at each router boundary the engine also runs the *next* layer's router, on
that layer's own MLP hyper-connection of the current post-attention stream. That
skips only the current layer's expert output and the next layer's attention. It
then starts future-priority reads for the predicted experts that are not cached.
This input matched 77% of the next layer's actual top ten. Feeding the next
router this layer's MLP input instead matched 63%. Precision falls with rank,
from 99% for the first guess to 44% for the tenth. Demand misses fell from 263
to 86 per token.

A guess only warms the cache. The experts that are computed are still exactly
the ones the real router picks, so a wrong guess costs bandwidth and one slot,
never correctness. Once the SSD is saturated, though, every unclaimed guess is
bytes a demand read could have used, so depth and precision matter. Depths of
6, 8 and 10 were screened, and 10 stays the default.

The same pressure explains a failed idea. An offline replay of captured routes
predicted about 10% fewer demand misses from per-layer LFU with equal quotas.
With prefetch running, it did worse: each layer's guesses evicted that layer's
own hot experts from a quota of about 18 slots. Global CLOCK takes prefetch
victims from the globally coldest entries.

The **panel** idea is worth explaining because it targets a different cost.
Normally the forward pass walks chunk by chunk through all 48 layers, so a long
prefill visits layer 12's experts once per chunk. In layer-major panel mode, a
panel of up to 1024 tokens is processed one layer at a time, so a layer's
selected experts are loaded once and used for the whole panel. On a 48-layer
check with a 257-token prompt and a 129-token append at only 32 cache slots,
panel 512 reduced expert read bytes by **59.4%**. Attention, DeltaNet and PLE
still run in bounded chunk units inside the panel, and global position commits
only when the whole panel finishes, so a failure invalidates cleanly.

## 8. Kernel work, and an honest account of what it bought

The GPU side has about 49 kernels. The ones that reflect genuine design
decisions rather than plumbing:

- **Fused gate/up projection.** The expert feed-forward computes a gate and an
  up projection from the same input. Fusing them avoids materializing a
  dequantized weight matrix and both intermediate buffers. This is a clean win
  and is on by default.
- **Packed affine Q4 and Q8 matrix kernels**, operating directly on the stored
  codes, scales and biases. Weights are never expanded to FP32 in memory. There
  is no GGUF-style dequantize-then-multiply step, because at 1.33GB of weights
  per token, materializing them would dominate everything.
- **Sparse attention** with exact block selection and skipping of wholly masked
  score tiles.
- **A SIMD router** preserving top-ten identity, tie order and softmax
  arithmetic exactly.
- **Single-token row kernels.** The reference Q4 matrix-vector kernel gives each
  output row its own SIMD group, so every row reloads the whole input vector and
  recomputes its BF16 input sums. Cache traffic for the input was about eight
  times the weight traffic. The row kernels load each lane's input chunk and bias
  sum once per block and apply them to four or eight rows (two for fused
  gate/up). The per-row arithmetic is untouched. Two latency-bound shapes get
  the same treatment:
  - `plain_mv`, for BF16 projections with four or fewer rows, stages its operands
    in threadgroup memory. Previously a 10,240-long serial chain paid one memory
    round trip per step.
  - `norm_wide` spreads the grouped RMS norm over twenty SIMD groups instead of
    four.

Now the part that is usually left out of write-ups: the outcome of each measured
change, including those **rejected**:

| Change | Isolated result | End-to-end result | Outcome |
|---|---|---|---|
| Packed Q4 decode kernels | 43–47% less expert GPU time, byte-exact on 64 real cases | 1.98% *slower* for whole conversations | Rejected at the time |
| Coalescing expert reads | — | Failed its speed gate | Rejected |
| Q8 resident load-ahead | — | Failed | Rejected |
| Larger expert cache (1460 vs 1072 slots) | 1.77% lower conversation latency | Directional only, missed the stage gate | Not promoted |
| SIMD router selection | — | 1.85–2.00% lower conversation time | Default |
| Core-plus-expert residency | — | 4.35% lower latency (95% CI 3.21–5.48%); removes ingestion compression on 2K prompts | Default where Metal supports residency sets |
| GPU clock keep-warm | Same projection 363µs → 108–119µs | 3.37 → 4.88 tokens/s; GPU time 200 → 93ms/token | Default |
| Single-token row kernels | 2.6–3.1× on resident projections, ~2× experts; bit-identical in raw FP32 | GPU time 93 → 66ms/token with the clock held | Default |
| Next-layer expert prefetch | 77% of next-layer experts predicted | Demand misses 263 → 86 per token | Default |
| Splitting one expert read across workers | A lone read already takes ~0.9ms | No change | Rejected |
| Sixteen I/O workers | — | No change once the SSD is saturated | Not adopted |
| Per-layer LFU replacement | ~10% fewer misses in replay, without prefetch | 88 → 113 demand misses per token with prefetch; slower | Rejected |

The packed Q4 line is the most instructive result in the project. A kernel that
was nearly twice as fast in isolation made the whole system slower. The reason
is visible in a command trace: **72–77% of expert-bearing GPU groups contain
exactly one expert.** Decode is a long sequence of tiny GPU groups, each waiting
on a read. Making the arithmetic inside those groups faster optimizes a part of
the token that was never the constraint, while the change's effect on
occupancy, register pressure and scheduling costs more than it saves. The clock
finding in section 3 may be part of the story: isolated benchmarks ran at a
clock that decode never reached. The packed kernels were not re-measured with
the clock held; the row kernels that replaced them were.

A per-token breakdown from a clean diagnostic run puts it plainly: GPU execution
occupies about 126–145ms per token, and *pending-read idle* accounts for another
58–62ms. The largest mixed GPU class contains resident work before routing —
that is, work the engine does while waiting for expert reads to arrive.

**The GPU clock is part of the whole request.** Much of that 126–145ms turned out
not to be arithmetic at all, but the low clock described in section 3. A
per-dispatch profile put single-token GPU work at about ten times the
memory-bandwidth bound for the ~3.9GB a token reads. Kernels timed in place were
about three times slower than the same kernels timed alone. Only after the clock
was held did kernel structure pay off end to end: the row kernels then took
whole-token GPU time from about 93 to 66ms.

**The lesson: measure the whole request.** An isolated kernel benchmark, a
cached replay with no SSD traffic, or a profiled run with instrumentation
boundaries can each show a large gain that does not survive contact with a real
request. This engine treats those as screens that justify a full measurement,
never as evidence on their own.

## 9. Bit-exactness, and why it is a performance topic

The engine reproduces a reference implementation's arithmetic exactly: all 48
layer outputs and all 248,320 final logits match the original MLX reference bit
for bit on the saved fixture, for both the Q4 and mixed 4/8-bit artifacts.

That is a strong constraint, and it costs real performance. It means:

- Explicit BF16 rounding boundaries at specific operations — sigmoid, softplus,
  normalization, attention scores and probabilities, and a four-way reduction.
- The expert sum follows a specific eight-part reduction order: combine
  positions 0/8 and 1/9 before 2–7.
- The router's FP32 matrix multiply uses a fixed geometry (M1 SIMD 8×8
  fragments, sixteen fixed K partitions) so routing arithmetic cannot vary with
  chunk size.
- The recurrent memory dot uses an ordinary four-value sum. Compensated
  summation — numerically "better" — changed a BF16 midpoint decision on a real
  fixture and was therefore wrong for this purpose.

Why accept that cost? Because without it you cannot tell a scheduling bug from a
quality regression. When the output changes, you need to know whether you broke
the engine or merely reordered a sum. Two rounding discrepancies, in the PLE
gate and GDN normalization, were found *only* because full-model comparison was
exact rather than approximate. With a tolerance-based check they would have
passed and shipped as a slight, permanent quality loss.

The rule the project settled on: **every optimization must preserve logits,
routes and persistent state bit for bit within an artifact.** An optimization
that changes numerics is a different model, and must be qualified as one.

That rule still leaves room for kernel work. What must not change is the
arithmetic each output sees, not which thread computes it. A row kernel keeps
the reference kernel's lane partition, including its rule for the lane width
(sixteen when K is a multiple of 512, else eight). It also keeps each row's
expression order, BF16 boundaries and final SIMD reduction. Only the loads are
shared.

Proving it takes two levels, and the first has a trap. Rounding outputs to BF16
hides most arithmetic differences. A deliberately wrong lane width changed only
one to three of 10,240 rounded outputs, yet it changed nearly every raw sum. So
operator checks compare raw FP32 sums, before output rounding, on every shape
the dispatch admits. Then the whole model runs one token at a time (`--chunk 1`,
so every token takes the single-token path) and compares all 248,320 logits and
every layer's recurrent, convolution and attention state against the reference
configuration.

## 10. What actually moved the number

The measured arc, each figure with its conditions:

| Build | Initial generation | After append | Conditions |
|---|---:|---:|---|
| First full-model run (Sept 8 audit) | 1.753 tok/s | — | 2K prompt, 256 output |
| Mixed artifact, panel 512 | 1.985 tok/s | — | 2K/256, 12GiB budget, 470.6s to first token |
| Route-selection screen | 3.53 → 3.71 tok/s | — | Two alternating pairs |
| Submission/residency follow-up | 3.80 tok/s | 3.31 tok/s | Five fresh alternating pairs |
| Cache capacity 1072 slots | 4.11 tok/s | 3.69 tok/s | Two clean pairs, 12GiB |
| Cache capacity 1460 slots | 4.37 tok/s | 3.87 tok/s | Same, directional screen only |
| Previous defaults, Q4, 850 slots | 2.40 tok/s | 2.27 tok/s | 96/48 outputs, busy host, ~7GiB admitted |
| GPU keep-warm, row kernels, next-layer prefetch (new defaults) | 5.29–5.41 tok/s | 4.89 tok/s | Same conditions, identical tokens |
| New defaults, 2K prompt | 5.86 tok/s | — | 2K/256, 1,977 slots, one clean run; 4.89 at 900 slots |

Target: 5 tokens/s. Measured in short screens and in a clean 2K run. It is not
yet qualified by repeated 2K/4K runs, and on this laptop it depends on how much
memory other applications leave for the expert cache. First-token latency on a
2K prompt is still measured in minutes (about 194 seconds). That is a separate
open problem from generation speed.

Ranked by what produced those gains:

1. **The prepared contiguous layout** — nine reads per expert to one. Nothing
   else changed the bandwidth ceiling.
2. **Completion-driven scheduling** instead of batch-and-wait, which is what
   lets resident computation cover read latency at all.
3. **Expert cache capacity**, directly, because the roofline is a hit-rate
   problem. Every memory decision is ultimately a decision about this number.
4. **Hit-ordered short appends and panels**, which cut how many reads are needed
   rather than how fast they are.
5. **Kernel-level arithmetic**, which produced large isolated improvements and
   small or negative end-to-end ones, until the GPU clock was held up. Then
   exact row kernels cut whole-token GPU time by about a third.
6. **Holding the GPU clock and prefetching the next layer's experts**, which
   together took generation past 5 tokens/s in short screens. Both left every
   byte of the computation unchanged.

That ordering is the opposite of where intuition sends most people, and it is
the single most useful thing to take from this project.

## 11. If you are building one of these

In order:

1. **Do the roofline arithmetic first.** Bytes per token, divided by your
   measured storage bandwidth, gives you the hit rate you need. If that number
   is above about 90%, change the plan rather than the code.
2. **Measure your storage with the real access pattern**, not a sequential
   benchmark. Record the parallelism curve; it tells you your worker count.
3. **Design the residency split from the architecture.** Find the part that is
   sparsely accessed — for a mixture-of-experts model, the experts — and build
   everything around streaming exactly that.
4. **Write the ownership model before the replacement policy.** Loading, ready,
   leased. Test eviction under load, cancellation mid-flight, and resize.
5. **Enforce a memory budget at every allocation**, and refuse to start rather
   than swap. Account for buffers visible to both processors once.
6. **Build a bit-exact reference check before optimizing anything.** You cannot
   evaluate an optimization you cannot prove is neutral.
7. **Never promote on an isolated benchmark.** Screen cheaply, qualify with
   paired, fresh-process, whole-request measurements in a known memory state.
8. **Check the GPU clock before optimizing kernels.** If kernels timed inside
   your decode loop run several times slower than the same kernels timed alone,
   the GPU is idling down between command groups. Keep it busy while generating.
9. **Start the next layer's reads before its router runs.** Predict with the
   next router on a stream you already have. A guess only warms the cache, so it
   can be wrong without being incorrect.
10. **Once the GPU is fast, count bytes per token again.** SSD bandwidth becomes
    the ceiling, and cache capacity and wasted guesses are the levers.
11. **Treat memory compression as measurement failure.** On macOS, a run with
    compression is not a slower run, it is a different experiment. On Metal,
    residency sets kept the trunk and expert cache from being compressed during
    long ingestion.

## 12. Reading the code

| Where | What |
|---|---|
| `include/engine/storage.hpp`, `src/engine/storage.cpp` | Checkpoint parsing, prepared sidecar, expert and ngram stores, read pool with demand promotion, expert cache with speculative entries, memory plan |
| `include/engine/metal.hpp`, `src/engine/metal.mm` | The Metal wrapper: allocation classes, budget, residency, scratch arenas, kernel selection, dispatch and completion, and the GPU clock keep-warm |
| `include/engine/model.hpp`, `src/engine/model.cpp` | The forward pass, layer by layer, the next-layer predictor, and `Options::resolve()` for the defaults |
| `src/engine/pipeline.cpp` | Completion-driven expert execution and its ownership rules |
| `src/engine/prefill.cpp` | Layer-major panel prefill |
| `src/engine/session.cpp` | State reuse, sampling, streaming output parsing |
| `kernels/metal/qwen.metal` | Every compute kernel |
| `tests/test_qwen.cpp` | The exactness and ownership checks, which are the real specification |

Implementation detail and current qualification status are in
[qwen_engine.md](qwen_engine.md); the plan of record is
[qwen_plan.md](qwen_plan.md); measurement evidence is indexed from
[README.md](README.md).
