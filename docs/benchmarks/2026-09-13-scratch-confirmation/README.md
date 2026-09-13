# Scratch reuse: validation fixes and repeated memory disturbance

The native build remains `62d1bd6f` from commit `28bfc3b`. This stage changes
offline qualification tools, not inference arithmetic, defaults or memory policy.

The first readiness check was blocked at metadata admission: **7.66GiB
reclaimable**, versus 13.5GiB required for the unchanged 12GiB test plus its
1.5GiB margin. No full-model inference started and no new throughput sample was
collected in that attempt. The [readiness record](readiness/summary.json) stays
incomplete. After the user freed memory, the unchanged full-model screen ran.

## Comparison while memory settled

The [fresh four-run screen](memory-settling-screen/summary.json) completed in
234.13 seconds. All outputs matched the prior control; arithmetic dispatch
counts, 1848 expert slots, the 12GiB budget and bounded workspace were unchanged.

| Pair | Phase | Control tokens/s | Reuse tokens/s | Decode memory |
|---|---|---:|---:|---|
| 0 | Initial | 3.049 | 4.339 | Control compressed; reuse clean |
| 0 | Append | 3.308 | 4.160 | Control compressed; reuse clean |
| 1 | Initial | 3.586 | 4.346 | Both clean |
| 1 | Append | 3.307 | 3.782 | Both clean |

Conversation ratios were 0.886346 and 0.952402. Reuse again won every decode
comparison, but the first control retained about 5.6GiB of compressed process
pages. The declared memory gate therefore leaves the entire screen
**inconclusive**, including its otherwise clean second pair. No timings were
discarded or adjusted, and no full-state qualification ran for this batch.
The [offline comparison](memory-settling-comparison.json) reconstructs that
decision from sealed original reports.

The last three processes having zero observed decode compression/decompression
justified one new four-request batch with the same controls and no pooled samples.

## Fresh attempt after clean process observations

The [second screen](settled-screen/summary.json) completed in **233.71 seconds**.
All four decode comparisons again favored reuse, and both conversation timing
gates passed. Memory disturbance recurred in both first-pair processes; both
second-pair processes were clean. The entire screen remains **inconclusive**.

| Pair | Phase | Control tokens/s | Reuse tokens/s | Saving per token |
|---|---|---:|---:|---:|
| 0 | Initial | 3.592 | 4.289 | 45.24ms |
| 0 | Append | 3.376 | 3.944 | 42.64ms |
| 1 | Initial | 3.310 | 4.389 | 74.34ms |
| 1 | Append | 3.215 | 4.082 | 66.06ms |

Conversation ratios were **0.952172 and 0.918661**. Output IDs, arithmetic
dispatch counts, computation reuse, capacity and workspace checks passed. The
[offline comparison](settled-comparison.json) preserves the failed memory gate.
Neither new screen launched full-model state qualification or five-pair
confirmation. The prepared runner remains unused for real confirmation.

## Memory observation and next measurement

The [phase-boundary review](memory-boundaries.json) preserves both screens and
all eight original conversations. In the first control of each screen, process
compression was zero before initial ingestion and nonzero afterward. The second
screen began ingestion with **18.64GiB reported reclaimable**, yet already had
about 514MiB compressed by its end. Successful admission alone did not prevent
compression. The highest tracked Metal allocation across both screens was
**10.433GiB**, within the unchanged 12GiB total plan. This is not a continuous
process-footprint peak and cannot rule out transient driver or other allocations.

A separate [system snapshot](system-memory-observation.json) during the second
screen showed about 10.43GiB FreeLLM RSS, 6.50GiB system wired pages and 9.41GiB
file-backed pages. These categories overlap and must not be added as independent
allocations. The snapshot includes other processes, is not synchronized to the
onset of compression and cannot identify which model buffers were compressed.

Stop repeating unchanged timing screens. The smallest next diagnostic is one
bounded control/reuse capture with memory observations at ingestion layer or
microchunk boundaries and existing per-token decode counters. Capture physical
footprint, compression/decompression, live/peak Metal bytes and buffer categories,
plus a final GPU/I/O drain and model-destruction boundary. Keep the same 12GiB
budget and 1848 slots. Label these instrumented requests as diagnostic and do not
use them to qualify latency. Their purpose is to locate the transient or page
retention mechanism before changing residency, admission or buffer ownership.

The earlier [five-pair core-residency experiment](../2026-09-10-residency-paired/README.md)
did not establish a request-latency benefit; it also experienced a different
slowdown with few decompressions. Do not assume that changing residency or
clearing other applications resolves this issue. The remaining short-workload
gap is approximately 28–54ms/token in this second screen, but 2K/4K acceptance
still needs its own measurements.

## Correct a false state-validation failure

The state fixture feeds a five-token prefix, a two-token append and two
one-token continuations. Fresh replay of those nine tokens with chunk size two
ends with a single-token forward. That final forward correctly retains its
reusable scratch pool. The old checker incorrectly required an empty pool after
every fresh replay, and would have rejected valid behavior after a timing win.

The corrected checker derives the last forward size from the actual admitted
panel/chunk limit. It requires retained scratch after a single-token remainder,
and requires no retained decode scratch after a multi-token final chunk. Both
cases still require inactive scopes, the same 128MiB bound, zero second-pool
allocation, and completed GPU users. Unit checks cover odd/even history lengths,
panel limits, missing/wrong retention and live/unbounded storage. Earlier raw
measurements are unchanged; their memory-disturbance decision remains unchanged.

## Real-weight diagnostic that fits current memory

`scripts/qwen/check_scratch_slice.py --output FRESH_DIRECTORY` uses the first
four layers of the pinned mixed checkpoint with prepared Q4 experts/ngrams,
32 expert slots and a separate **4GiB diagnostic budget**. These layers cover
Gated DeltaNet, attention and PLE/ngram state. It uses the existing native
state/failure harness with Metal API and shader validation, the shared GPU lease,
frozen source/artifact identities and a 180-second total deadline.

Both arms completed in **7.00 seconds total**. All eight control and twelve
candidate checks passed. Continued and fresh replay had identical captured
state and route hashes between control and candidate, with 295 evictions after
continuation and 581 after fresh replay. Candidate scratch retained after the
single-token remainder was **6,848,512 bytes (6.53MiB)**; the control retained
none. Sampled footprint was at most about 1.16GiB and process compression was
zero at captured boundaries. Cancellation and deliberately failed single-token
execution drained scratch/GPU ownership and invalidated partial state.

[Sealed diagnostic](state-slice/summary.json), [control checks](state-slice/control.json),
[candidate checks](state-slice/candidate.json). This diagnostic computes **no
model logits** and is not an independent implementation oracle. Four-layer
state parity cannot replace the required 48-layer check or qualify throughput.
The full-state validator explicitly rejects these reports as prerequisites.

## Five-pair confirmation runner

Once a fresh unchanged short screen passes both timing and memory gates and
completes its 48-layer state/failure checks, run:

```sh
.cache/qwen-reference-venv/bin/python scripts/qwen/confirm_decode_scratch.py \
  --screen PATH_TO_SEALED_PASSING_SCREEN --output FRESH_DIRECTORY
```

The runner revalidates the prerequisite against original reports before loading
model assets or launching inference. It copies the sealed prerequisite into its
evidence, requires matching native build/artifact/workload/budget, then runs five
fresh alternating pairs under the existing 900-second total / 150-second process
limits. No prior timing pairs are pooled, and there is no early success stopping.
All outputs, dispatch counts, actual pool reuse, ownership and admission remain
checked against the screen's control.

The existing paired log-ratio Student-t confidence rule applies: conversation
geometric mean ratio at most 0.99, its upper 95% bound below 1, and all request,
first-token and decode upper bounds at most 1.03. Both decode phases must also
have upper bounds below 1, every paired decode ratio below 1, and no observed
decode compression/decompression. Missing memory observations cannot pass.
The offline query tool reconstructs this decision from original report hashes
and requires `--change decode_scratch`.

Even a successful short confirmation only selects a candidate for later
qualification. The 2K/4K 256-output target, 7K report and sustained coding session
remain separate requirements. Current short-workload measurements remain
provisional. All **227 offline tooling tests passed**; see the
[test log](tooling-tests.log). Native runtime sources and binaries are unchanged.
