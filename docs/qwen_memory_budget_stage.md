# Use the safely available laptop memory

After recording the packed Q8 confirmation, evaluate one 12GiB-versus-18GiB engine
budget experiment on the same 32GiB M1 Pro. This is a memory/performance tradeoff:
the candidate spends six additional GiB on the existing expert cache. It does
not introduce a new quantization, cache policy, kernel, model, or backend.

```sh
python3 scripts/qwen/screen_memory_budget.py --output .cache/benchmarks/memory-budget-NEW
```

The Q8 confirmation's primary conversation interval favored packed Q8, while
its initial first-token guard remained inconclusive. This stage keeps that
result unchanged and uses packed Q8 explicitly in both experimental arms.

First simulate the existing complete 256-step route trace at the corresponding
expert capacities, retaining cache state across ingestion, generation and the
follow-up. Report predicted read counts only. This simulation cannot admit
memory or predict tokens/s. Then inspect actual system admission after draining
the preceding model. Run only if the complete 18GiB budget and requested cache
fit the existing OS/Metal limits; never raise those limits or relabel a smaller
admission as 18GiB.

Both arms use the experimental packed Q8 two-row configuration, CLOCK, residency
off, serial prefill, fixed phase memory, reference expert execution, panel 512,
chunk 128, ready group four, eight readers and context capacity 8192. Fixed
allocation categories must match; only the engine ceiling, expert capacity,
expert bytes and total planned bytes may differ. Under the current layout this
is 1848 versus 4175 slots. No mid-request pressure resize is allowed.

Run two fresh alternating pairs of the established 72+33, retained-128+33
conversation, with no profiling, validation or GPU probes during timing. Retain
actual 104-token computation reuse and the pending-output ingestion check.
Use a 600-second stage deadline and 150 seconds per process, with bounded
metadata-only admission retries and no inference retry. Preserve every result.

Require matching generated tokens, normalized dispatch counts, exact requested
allocation at every phase snapshot, and observed process/Metal memory within
each arm's admitted ceiling. Existing forced-eviction state checks cover the
unchanged cache implementation; this experiment changes capacity, not arithmetic.

The two-pair gate requires both complete-conversation ratios below one, median
at most 0.99 and each request/first-token/decode secondary median at most 1.03.
Report expert reads, hits, coordinator waits, physical footprints, compression
and system swap observations. Six additional GiB must be counted explicitly in
the result; it cannot qualify as an equal-memory kernel improvement.

A survivor still needs five fresh pairs and the original 2K/4K output-length,
7K, sustained-memory and coding-workflow acceptance. A failed admission or
inconclusive timing result stops this one budget experiment. The next computation
candidate is the serial expert-ranking step, whose completion precedes each
layer's actual expert reads; measure it on current inputs before implementation.

The offline comparison accepts `q8_memory_budget_screen_v1` with
`--control control --candidate candidate --change memory_gb --change expert_slots`.
It reopens the prior confirmation and every new request by hash, enforces each
arm's explicit budget, and preserves the same fixed allocations across arms.
Legacy comparisons still default to exactly 12GiB and reject an 18GiB report
unless the caller explicitly selects that budget.
