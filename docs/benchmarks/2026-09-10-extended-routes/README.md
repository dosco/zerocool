# Extended normal-generation and retained-history route capture

The real M1 Pro capture completed in **166.94 seconds** within its five-minute
deadline. It recorded all **256 generation steps**, then a **128-token follow-up
and 32 more generation steps**, at a fixed 12GiB engine budget. All 13,920 layer
passes committed; the trace contains no aborted, missing or incomplete forwards.

SLRU's simulated read benefit persists at the actual 1,848-slot allocation, but
is modest and varies by phase. Keep native CLOCK. The evidence supports one
experimental SLRU implementation and a short paired timing screen; it does not
justify promotion or explain away the slow follow-up.

## Observed native behavior

| Measurement | Initial coding request | Retained-history follow-up |
|---|---:|---:|
| New input tokens | 72 | 128 |
| Actually reused model-state tokens | 0 | 328 |
| Tokens computed during ingestion | 72 | 129 |
| Output tokens / subsequent decode steps | 257 / 256 | 33 / 32 |
| Instrumented time to first token | 15.73 s | 24.13 s |
| Instrumented generation rate | 2.469 tokens/s | 2.367 tokens/s |
| Complete request time | 119.54 s | 37.67 s |
| Decode token latency, median / p95 | 384 / 475 ms | 406 / 544 ms |
| Native decode expert hits / misses | 71,195 / 51,685 | 7,591 / 7,769 |

The prior history contains 72 prompt and 257 output tokens. Its last sampled
output has not yet passed through the model, so the follow-up reuses 328 tokens
and computes that pending output plus 128 new input tokens. The session identity
stays unchanged. The validator checks this against every committed input and
the native reuse counters, rather than inferring reuse from a textual prefix.

The append's ingestion phase had **123 expert hits and 8,835 misses**, just a
1.37% hit fraction after within-pass grouping. Retaining the session avoids
recomputing history but does not guarantee that its new expert demands hit cache.
The 24.13-second first-token wait warrants attention; no measurement here assigns
that entire wait to storage or predicts the latency saved by a different policy.

Generation remains below the 5 tokens/s objective. Timings include route tracing
and are not paired acceptance measurements. The prompt is short, and the append
extends 329 tokens of total history, not the required 4K acceptance history.
Both answers are deliberately length limited; this is not coding-quality proof.

## Equal-byte simulation with the cache retained across phases

The [simulation](cache-actual-capacity.json) keeps the same cache throughout the
entire conversation, starting cold only once. The 4,880MiB input yields exactly
the native **1,848 slots / 5,116,919,808 allocated bytes**, leaving 131,072 input
budget bytes unused. New per-request/phase counters partition this continuous
simulation; filtering to decode alone would discard the warm start.

| Phase | CLOCK misses | SLRU misses | Change in expert reads |
|---|---:|---:|---:|
| Initial prefill | 7,688 | 7,688 | unchanged |
| Initial 256-step generation | 51,749 | 48,788 | 5.72% fewer |
| 128-token append plus pending output | 8,835 | 7,999 | 9.46% fewer |
| Generation after append | 7,771 | 7,804 | **0.42% more** |
| Entire conversation | 76,043 | 72,279 | 4.95% fewer |

The short capture's 10.71% decode read reduction did not remain that large over
256 steps. The small post-append regression is visible instead of being hidden
in the conversation average. This one workload cannot establish its statistical
reliability or behavior on other coding tasks.

Native CLOCK recorded 75,977 total misses: **66 fewer** than simulated CLOCK.
The model uses fixed ascending expert demand order and immediate release; it
omits native hit-first admission, outstanding GPU users, OS cache and read overlap.
Its agreement with CLOCK is a useful check, not proof of SLRU's native behavior.
Future-aware MIN recorded 46,582 misses under those same assumptions. It is not
implementable online, and its per-phase counts are not independent minima.

## Memory and provenance

Memory admission and all observed allocation plans remained unchanged at 12GiB.
The largest physical footprint at recorded phase boundaries was 11,211,594,304
bytes (10.44GiB); this is not a continuous peak measurement. The final footprint
was 11,191,081,536 bytes, with 6,020,988,928 reported compressed bytes. System swap
usage was 1,891,368,960 bytes before and after, with **zero observed growth**.
System swap and storage counters cover other applications too. This short run
does not qualify the 20-minute sustained-memory requirement.

The artifact remains mixed 4/8-bit revision
`b2c422f3c643e36f04227a64d61796b44a4b1029`, with unchanged prepared Q4 experts and
reference kernels. Native build:
`fc0207993bb8736f67f7b9c233f5b66cded6ca2c654e76c73db21763d50b12cb`.
Panel size is 512, recurrent/attention microchunks 128, ready groups four and I/O
workers eight. No precision, native scheduling or production default changed.

The initial 33 output tokens and all first 1,584 layer routes exactly reproduce
the [previous short capture](../2026-09-10-normal-routes/README.md). That is a
reproducibility check, not independent full-model logit validation.

**159 Python tests passed**, including phase counter conservation, preserved warm
cache state, exact pending-token reuse, changed-session/budget rejection, early
EOS, and append framing. [Test output](python-tests.log) and [verification
hashes](verification.json) identify the tested code. The native build is unchanged
from the earlier 53-test, 7,084-assertion Metal validation run; those native tests
were not rerun for these Python-only changes. The existing Python suite emitted
two SQLite resource warnings, with no failing tests.

The original capture and [copied raw evidence](raw/summary.json) pass the same
immutable seal. [Native measurements](raw/native.json), [routes](raw/routes.jsonl),
[derived observations](observations.json), and [ledger entry](ledger-result.json)
retain their source hashes. The preparation recipe records token truncation and
chat framing; the helper never forced generation tokens or silently retried.

## Next bounded experiment

Implement only the tested 75%-protected SLRU policy behind an explicit experimental
option, retaining CLOCK as the default. Verify eviction and delayed buffer release,
then screen normal request latency at equal admitted memory and identical inputs.
Require unchanged arithmetic, routes, outputs and session state. Include the
follow-up and its subsequent generation so the observed phase tradeoff is tested.

The available evidence suggests a small read reduction, not enough by itself to
claim the laptop targets. Reserve full 7K/session/recovery qualification for a
candidate whose complete-request latency improves in the short paired screen.
