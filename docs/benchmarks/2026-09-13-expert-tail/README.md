# Expert-tail overlap toward 200ms/token

Status: implemented and screened; repeatable request improvement was not demonstrated.
No production default or precision changes.

The completed [dependency capture](../2026-09-13-decode-target/README.md)
observes 71.09ms/token between final expert GPU completion and submission of
its reduction group, with 22.4% greater traced request latency. That interval
includes trace writing and preparation of the next layer. It is an upper bound
on an exposed opportunity in this trace, not a predicted saving. A subsequent
unchanged request with GPU boundary probes returned to 283.73ms/token; the
473.64ms/token normal arm in the capture must remain part of the evidence.

The experiment starts encoding the reduction, residual and next attention work
as soon as every selected expert has been submitted. The final one or two GPU
groups keep their leases until the next router's required CPU-visibility wait,
or the end-of-token wait. No new expert admission starts before that tail is
drained. The queue still has at most two live command groups and the existing
32-lease limit. Cancellation and exceptional exits drain GPU/I/O users before
releasing slots; incomplete state remains invalid. No arithmetic or selected
expert changes. The same option applies to single-token appends; multi-token
panels retain the existing path.

`--expert-tail wait|overlap` is accepted only for benchmarks and inspection.
`wait` remains the default. Deferred expert records are finalized at the next
CPU dependency. Their lifetime includes intervening encode work, and the
pipeline's coordinator-wait counter excludes the caller's router wait, which
is recorded in Metal's CPU wait counter. Neither metric is a causal latency
breakdown. Exact output/state and actual request wall time govern the decision.

Before normal screening, run the native ownership/arithmetic tests with Metal
API/shader validation. Then `scripts/qwen/screen_expert_tail.py` runs two fresh
alternating conversations: 72 prompt tokens, 33 outputs, 128 appended tokens,
and 33 more outputs. Both arms use mixed 4/8-bit weights, the same prepared Q4
records, 12GiB, 1848 CLOCK slots, packed Q8 rows 2 and the SIMD router. Only the
tail option changes. Profiling, Metal validation and GPU boundary probes are
off during timing. All four runs are retained, with a 600-second total request
cap and 150 seconds per process.

Advance only when both conversations improve, median conversation ratio is at
most 0.99, all existing request/TTFT/decode median ratios are at most 1.03, and
every initial/append decode comparison improves. Report actual milliseconds
saved against the 200ms goal. Two pairs have no qualification confidence bound.
A survivor then receives all-48-layer exact logits/routes/state replay under
forced eviction, including cancellation and deliberately failed single-token
tails, before five fresh confirmation pairs are proposed. Slow or unfinished
screens do not qualify an improvement. The 2K/4K 256-output target remains open.

## Result: improvement not demonstrated

The complete screen finished in 244.19 seconds on native build `316e520d`.
Every output token in all four conversations matched both the paired arm and
the previous confirmed router baseline. Each candidate decode phase executed
1536 deferred tails (48 layers × 32 steps), kept at most two live GPU command
groups, and drained all outstanding tails by each phase boundary.

| Pair | Phase | Wait ms/token | Overlap ms/token | Change |
|---|---|---:|---:|---:|
| 1 | Initial generation | 296.09 | 259.19 | 36.90ms faster |
| 1 | Generation after append | 278.00 | 288.88 | 10.88ms slower |
| 2 | Initial generation | 274.57 | 341.88 | 67.30ms slower |
| 2 | Generation after append | 403.10 | 406.38 | 3.28ms slower |

Conversation ratios were 0.98497 and 1.06039, for a median ratio of 1.02268.
The candidate fails the declared screen. The first initial request reached
3.86 tokens/s, but that isolated result is not a repeatable improvement and
is not adopted. Full-model state qualification and five-pair confirmation
were not run, as required for a failed early screen. No default changes.

GPU command durations varied from 111.15 to 254.99ms/token across the measured
phases. The second candidate's append also experienced **7.56GiB peak process
compression and 497,771 decompressions**, despite no emergency cache resizing.
Sampled footprint stayed at most 10.44GiB across all runs; process compression
is part of that footprint, not additional allocated bytes. No request had a
positive net system-swap change. Zero swap growth alone therefore would have
missed a substantial memory disturbance. Other measured phases had zero
process compression, so compression does not explain all timing variation.
No samples are removed or adjusted using GPU time or memory observations.

[Raw summary](raw/summary.json), [source-bound analysis](analysis.json),
[query reconstruction](comparison.json). The full native suite passed 59 tests
and 48,747 assertions with Metal API/shader validation. A subsequent change
only corrected test-object destruction order; the focused final-source run
passed all three tail/pipeline tests and 559 assertions. The other 56 tests
were intentionally excluded from that focused invocation. The tooling suite
passes 209 tests, including strict tail execution, request guards, output/state
comparisons, missing failure evidence, and query-axis checks. The query adapter/test and stricter checks for complete future state snapshots
and failed-run flags were added after timing. Native sources, request collection
and this negative screen decision are unchanged. See [native suite](native-tests.log),
[final ownership checks](native-tail-tests.log), and [tooling](python-tests.log).

## Next experiment

Keep `expert_tail=wait`. Do not spend a five-pair confirmation or long-context
run on this failed screen. The next hypothesis is reducing temporary Metal
allocation/release work: both schedules still create **3238 allocations per
generated token**, and the trace leaves sizeable CPU/submission gaps. Measure
allocation and retirement time with bounded scalar counters before selecting
reuse; allocation count alone does not prove a bottleneck.

If that cost is material, screen bounded reuse of single-token temporaries
within the already admitted 512MiB scratch allowance, keeping the 1848 expert
slots and unchanged arithmetic. Reuse requires completed GPU users, and a
transition into prefill must release or account for retained decode buffers.
Begin with a small operator/lifetime check, then the same normal paired screen.
Report compression/decompressions as well as footprint and net swap so a repeat
of this host disturbance remains visible. This is a hypothesis, not an
implemented or measured buffer-reuse gain.
