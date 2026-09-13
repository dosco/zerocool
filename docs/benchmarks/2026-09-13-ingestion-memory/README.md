# Ingestion compression and delayed model cleanup

The bounded trace reproduced compression during ingestion while the measured
process peak remained below the fixed **12GiB** budget. It also demonstrated
that an immediate post-destruction footprint can overstate retained memory:
one fresh control fell from **1,543MiB to 31.7MiB within 250ms**.

These are memory diagnostics. No request-latency qualification or default,
precision, cache-capacity or residency change is included.

## What was measured

`scripts/qwen/measure_ingestion_memory.py --output FRESH_DIRECTORY` runs one
control/reuse pair through the native `qwen_memory_check` harness. Both use the
same 72-token initial prompt, 33 outputs, 128-token retained-history append and
33 outputs, mixed artifact, prepared Q4 payload, 1848 CLOCK slots, panel 512,
microchunks 128, eight I/O workers, ready groups of four, packed Q8 rows 2,
SIMD routing and reference expert arithmetic. Only decode scratch reuse differs.

The harness calls normal `Session::generate`. A developer-only observer samples
ingestion forward/layer encoding boundaries and panel microchunk boundaries.
It adds no GPU command, submission, reap or wait. Existing per-token decode
observations cover all 32 committed forwards per request. Trace serialization
and sampling are instrumentation and remain part of these diagnostic requests.

Every sample records process footprint, process-lifetime peaks, compression,
system VM categories, tracked Metal bytes, device-reported resource allocation,
queued users and per-class owner counters. Owner-held bytes are allocated bytes
minus final-owner release bytes; they do not identify physical residency or
which pages compressed. System categories overlap and include other processes.

Detail is capped at 512 records, with 32 separately admitted lifecycle records
and a 64MiB file limit. Flushed JSONL survives partial execution. Completed
analysis requires the exact layer/microchunk order, zero omissions, all cleanup
boundaries and the original output/control identity. Missing gauges stay missing.
Normal production options leave the callback empty.

## Initial capture

The [initial capture](initial/summary.json) completed in **122.79 seconds**.
Each arm recorded all 388 detailed ingestion boundaries, eight lifecycle records
and 64 decode intervals across its two requests.

In the control, the first nonzero compression observation was at **layer 9
(index 8)** after expert work, between `attention_encoded` and `layer_encoded`.
At that boundary, owner-held resident weights were about **4.99GiB**, recurrent/
attention state **0.53GiB**, routed experts **3.90GiB**, and temporaries **0.017GiB**.
The observed process footprint was 9.57GiB and compressed bytes were 674MiB.
System free pages had fallen to about 35MiB while about 8.68GiB of file-backed
pages remained. This brackets the onset during expert-cache growth; it does not
establish which allocations compressed or why the OS selected them.

At the end of ingestion, expert ownership was **4.77GiB**, the full admitted
cache. Model memory did not show an unaccounted Metal resource pileup: device
resource bytes exceeded tracked live buffer bytes by about 0.625MiB at these
boundaries. The process-lifetime footprint peak was **10.560GiB** for control
and **10.559GiB** for reuse. Neither exceeded the 12GiB total budget.

Control retained about 5.60GiB of reported compression through the final user
drain. Reuse recorded zero compression in this capture despite the same
resident weights, expert capacity and roughly 74MiB of reusable workspace.
That contrast is not a causal attribution: the pair has a fixed order and
uncontrolled host memory conditions.

Immediately after model destruction, control still reported 3,278.7MiB of
footprint, including 3,252.7MiB compressed; reuse reported 33.4MiB. The initial
protocol ended there. It cannot establish how long control's remaining charge
persisted.

## Delayed cleanup follow-up

That unresolved lifecycle observation motivated a versioned follow-up with
additional samples at least 250ms and one second after model destruction.
The [follow-up](delayed-cleanup/summary.json) completed in **119.53 seconds**.
Both arms captured all 388 detailed records, ten lifecycle records and 64 decode
intervals. Neither process recorded compression in this follow-up.

| Configuration | Peak footprint | Immediately destroyed | After 250ms | After 1s |
|---|---:|---:|---:|---:|
| Control | 10.561GiB | 1,543.0MiB | 31.7MiB | 31.7MiB |
| Reuse | 10.560GiB | 33.6MiB | 33.5MiB | 33.5MiB |

GPU and I/O users had already been drained before destruction. Session cleanup
released state ownership; model destruction released the remaining model owners.
The falling footprint demonstrates delayed reclamation or accounting in this
fresh control. It does **not** retrospectively prove cleanup of the earlier
compressed control. These snapshots are also not a sustained-session leak test.

## Verification and agent queries

- **63 native tests / 48,814 assertions passed** with Metal API and shader
  validation, including non-mutating observation, bounded trace admission,
  separate cleanup coverage and no retained resource owners.
- **232 tooling tests passed**, including irregular 129-token panel coverage,
  missing/reordered events, omitted cleanup, pending GPU users, unavailable
  memory values, delayed-sample timing and source-verified JSONL queries.
- All eight real-model requests matched the prior control's output IDs and
  arithmetic dispatch counts. This is output/execution parity, not independent
  full-logit/state or coding-quality qualification.
- Both raw seals and all derived analyses were revalidated. The original
  immediate-cleanup protocol remains readable after adding delayed samples.
  [Initial tooling snapshots](initial-tooling/measure_ingestion_memory.py) and
  [the earlier query implementation](delayed-cleanup-tooling/evidence_queries.py)
  preserve subsequently edited source files; every snapshot matches the hash
  frozen in its capture identity.

The offline agent query now reads memory JSONL directly, with source line
pointers, pagination and explicit null Metal gauges after destruction:

```sh
.cache/qwen-reference-venv/bin/python scripts/qwen/query_evidence.py import \
  docs/benchmarks/2026-09-13-ingestion-memory
.cache/qwen-reference-venv/bin/python scripts/qwen/query_evidence.py memory \
  docs/benchmarks/2026-09-13-ingestion-memory/delayed-cleanup/control.jsonl \
  --offset 392 --limit 6
```

[Derived observations](observations.json), [cleanup query](cleanup-query.json),
[verification](verification.json), [native tests](native-tests.log),
[tooling tests](tooling-tests.log).

Native library build: `bc161bf1f6869264d4784b8c62519ebd162f363bab03e076f2a723b3d507ff03`.
Mixed revision: `b2c422f3c643e36f04227a64d61796b44a4b1029`. The harness versions
have separately frozen source/binary hashes; inference arithmetic is unchanged.

## Next decision

Use delayed, stable cleanup readings before diagnosing retention. Do not fix
this as an engine-budget overrun: the process peaks in these runs were within
the allocation plan. Ingestion compression remains intermittent and distinct
from delayed cleanup. Preserve the prior failed residency confirmation and the
memory-disturbed scratch timing screens.

The next bounded intervention to consider is the existing core-residency option
with scratch reuse held on, using the same cache capacity and a fixed GPU timing
reference to distinguish device-speed variation from compression. Treat it as
a new combined-configuration experiment; prior core residency did not establish
a reliable speedup. An apparent reduction in compression alone cannot promote
it. The 5 tokens/s goal still requires clean normal 2K/4K request evidence.
