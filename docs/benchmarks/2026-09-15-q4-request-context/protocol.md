# Actual Q4 request context — declared before execution

Run the current native build on the actual 32GiB M1 Pro using the mixed reference
artifact and byte-preserved prepared Q4 experts. Reuse the sealed compatible
full-state/operator proof. Change only reference versus packed-r2 Q4 decode.
Keep 1072 expert slots, core-cache residency, immediate submission, scratch reuse,
12GiB engine admission, and the existing candidate configuration. No fallback
precision, reduced cache, OS limit changes, native changes or kernel counters.

Use the saved 72-token initial prompt followed by its 128-token retained append,
with 17 generated outputs per phase. Each phase has sixteen decode forwards.
The append reuses 88 computed tokens and ingests 129 including the pending output.

1. Run normal reference A, then normal packed B, validation and profiling off.
2. If either phase has packed/reference decode-wall ratio strictly greater than
   one, run traced B then traced A. Otherwise stop as slowdown_not_reproduced.
3. Each trace enables only existing decode diagnostics, command profiling and
   explicit dependency JSONL. Require untruncated coverage of 32 forwards,
   101600 dispatches, 1536 layer passes and 15360 selected-expert records.
4. Stop after this attempt: at most four inference processes, 90 seconds each,
   a 240-second work deadline including preparation and admission. On timeout,
   the existing guard permits up to 45 seconds to cancel and drain outstanding
   users; offline analysis and evidence sealing also occur outside that deadline.
   Lightweight inspect processes are additional. Existing disk, source and single-GPU lease guards
   apply. Preserve partial evidence on cancellation, timeout, mismatch or block.

Require identical generated tokens and normalized dispatch work, actual prefix
reuse, drained GPU users, resident cache enrollment, physical peaks within 12GiB,
zero observed compression including cumulative peak, unchanged decompressions,
and unchanged observed swap within each process. These are boundary/peak
observations, not continuous host monitoring. No missing gauge qualifies.

One normal pair is a directional trace trigger with no confidence interval; it
cannot establish a gain or regression. Preserve older normal-request rejection.
Report normal target distance from decode-wall/16 to 200ms (5 tokens/s).
Report traced/normal ratios without corrections. Rank exclusive whole-token
timeline buckets, then whole command stage sets and layer sets. Mixed commands
remain mixed; read service, CPU work, boundary gaps and command sums overlap and
must not be added to the exclusive timeline or presented as attainable savings.

No production promotion, 2K/4K/7K claim, coding-quality claim, or sustained-session
qualification follows from this diagnostic. Archive source identity and retain
raw reports as the source of truth. Reconstruct decisions in an offline audit.
