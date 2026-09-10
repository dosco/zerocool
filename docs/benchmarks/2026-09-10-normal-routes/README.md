# First complete normal-generation route capture

The bounded capture completed in **33.97 seconds** on the actual 32GiB M1 Pro.
A native chat template produced a 72-token coding prompt. The engine generated
33 tokens, giving **32 committed single-token forward steps** after prefill.
All 1,584 layer passes are present: 48 for prefill and 32 × 48 for generation.
There are no missing passes, aborted forwards or incomplete trace records.

This used the pinned mixed 4/8-bit artifact, prepared Q4 expert records and the
unchanged reference kernels at **12GiB**, with 1,848 expert slots. It did not use
cached replay, a truncated model, forced output tokens or a reduced memory budget.
The native build is
`fc0207993bb8736f67f7b9c233f5b66cded6ca2c654e76c73db21763d50b12cb`.

## Observations

| Measurement | Result |
|---|---|
| Instrumented time to first token | 16.29 seconds |
| Instrumented generation rate | 2.42 tokens/s over 32 steps |
| Generation expert-cache hit fraction | 57.47%: 8,827 hits, 6,533 misses |
| Generation expert application reads | 18,062,438,400 bytes |
| Expert allocation | 5,116,919,808 bytes in 1,848 aligned slots |
| Process footprint after the request | 11,189,590,720 bytes, within 12GiB |
| Compressed process memory after the request | 5,991,153,664 bytes, approximately 5.58GiB |
| Observed system swap growth during this request | 0 bytes |

The first decode step took 1.31 seconds; the final five took about 0.35–0.38 seconds
each. The run performed 123,534 additional decompressions during generation.
These observations do not identify which memory or computation dependency was
on the critical path. System swap counters cover other applications as well.

All timings include route instrumentation. This is a single short request, not a
paired performance comparison, a 2K/4K acceptance workload, a retained-history
append, coding-quality proof or a sustained-session check. The generated code is
an intentionally length-limited fragment. No production default is promoted.

## Cache-policy experiment at the actual allocation

The [fixed-order simulation](cache-actual-capacity.json) uses exactly 1,848 slots;
the MiB input rounds up to 4,880MiB and leaves 131,072 bytes unused. Its CLOCK model
predicts 14,236 total misses versus **14,221 observed native misses**. The 15-miss
difference is consistent with the model's omitted hit-first admission and live
leases; this agreement does not prove other policies' native behavior.

| Policy in the simulation | Total misses | Decode misses |
|---|---:|---:|
| CLOCK | 14,236 | 6,548 |
| Probation/protected SLRU | 13,535 | 5,847 |
| Future-aware MIN | 10,416 | 2,728 |

Each policy must fetch all 7,688 distinct experts in the single prefill window.
Subtracting those compulsory prefill misses gives the decode counts above.
SLRU therefore saves **4.92% of total expert reads, or 10.71% of decode expert
reads**, in this short simulation. That is a testable hypothesis, not a measured
latency improvement. MIN has unavailable future knowledge and is only a bound
for the same fixed-order, equal-size, immediate-release model.

The [wider sweep](raw/cache-curve.json) also shows SLRU performing worse than CLOCK
at 2GiB. Cache policy cannot be chosen independently of its byte budget. Keep
native CLOCK until the hypothesis survives a longer normal capture and a timing
screen at equal admitted memory.

## Implementation and verification

The new native `--route-trace` diagnostic records request inputs/results, explicit
session and forward identities, phase/position, selected experts, commit/abort
markers and termination. It writes routes at the existing router readback and
adds no GPU wait. A forward becomes usable evidence only after its full computation
and state update complete. Capture requires a new file, refuses reduced admission,
and marks benchmark timings as instrumented. Existing detailed traces and cached
phase progress retain their formats.

The offline reader verifies sequence, layer coverage, positions, token history
and request results. It discards uncommitted routes, carries simulated prefill
warmth across an explicit decode transition, and resets on session/cache changes
or omitted work. A filtered decode-only curve starts cold.

- **53 native tests / 7,084 assertions passed**, with Metal API and shader validation,
  no skipped tests: [native output](native-tests.log).
- **155 Python tests passed**, including committed/aborted traces, corrupted
  identities and token history, explicit resets, phase filtering and deadline
  termination: [Python output](python-tests.log).
- The actual capture passed artifact/build/machine/budget checks and verified
  its generated tokens against every committed forward. The copied raw evidence
  matches its original seal: [capture summary](raw/summary.json),
  [native report](raw/native.json), [trace](raw/routes.jsonl),
  [raw evidence hashes](raw/evidence-files.json).

[Derived observations](observations.json) and [verification hashes](verification.json)
bind the report to its sources. The full-model run exercises the trace writer and
independent reader together; it is not a new independent model-logit comparison.

## Next bounded step

The 32-step evidence is sufficient to justify extending the locality capture to
256 generation steps and a retained-history append, while retaining a deadline
and fixed admission. Use that to decide whether one SLRU implementation deserves
a short normal-request timing screen. Include compression and actual cache capacity
in comparisons; the trace does not establish that all waiting comes from SSD reads.
Full 7K/session/recovery qualification remains reserved for surviving candidates.
