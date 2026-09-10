# Offline cache curves: first evidence

The evidence tool now implements `cache SOURCE`: equal-byte CLOCK,
probation/protected SLRU and future-aware MIN simulations over saved routes.
This advances the plan's cache-locality investigation without changing native
inference, weights, cache defaults, memory admission or the selector experiment.
No model run was launched for this work.

The important outcome is an evidence gap: **these sources cannot choose a
production cache policy or establish a speedup**. They are small correctness
fixtures, and the timing profile does not contain ordered router selections.

## What the existing sources establish

| Saved source | Included work | Simulation result and limit |
|---|---|---|
| [Five-token Q4 fixture](../2026-09-07-foundation/recorded-routes.jsonl) | One five-token prefill window across 48 layers | 2,400 router selections become 1,553 distinct expert-pass demands. Every demand is compulsory in this capture; every policy has zero cache hits. Within-pass grouping is not reported as a cache hit. There is no subsequent generation to measure reuse. |
| [Mixed-reference recovery trace](../2026-09-09-selector-qualification/run-03/recovery/report.json.trace.jsonl) | Two successive single-token setup windows, then a separate two-token recovery forward; one partial layer pass is excluded | 1,824 demands, including 1,728 compulsory misses across two cold segments. At 1GiB, CLOCK/SLRU miss all demands while the ideal policy retains 96. At 2–8GiB, all three retain 96. These are setup/recovery routes, not a normal coding conversation. |
| [Cached profile](../../../.cache/benchmarks/decode-compute/baseline-profile.json) | 144 captured dependency passes, none with complete routes | `insufficient_evidence`, no curve. The recorded limit is 48 passes and 8,192 read records per phase. Read completion records cannot reconstruct router selections. |

The generated answers retain original source SHA256, recorded native build and
artifact identities, incomplete coverage and explicit simulation assumptions:
[prefill curve](foundation-prefill.json), [recovery curve](recovery.json),
[missing-route answer](profile-coverage.json). Neither a short capture nor an
unchanged result at larger capacities proves that more cache would not help
ordinary requests. No source has been relabeled with the current native build.

## Model and verification

The simulator fixes ascending expert-ID order within each pass and deduplicates
token rows by expert. It preserves recorded layer order; it never converts a
prefill panel into a token-by-token replay. Only complete 48-layer windows are
included, and omissions or position/phase/session discontinuities reset the cache.
Missing explicit reset/session markers remain a limitation even when positions
are contiguous. Reads complete and release immediately in the simulation;
native hit-first admission, live leases and GPU/SSD overlap are outside its model.

Each Q4 expert charges 2,768,896 aligned slot bytes and 2,764,800 application read
bytes. The supported mixed artifact retains Q4 experts. The future-aware minimum
is valid only for this equal-size, fixed-order model with mandatory admission.
It is not a bound on arbitrary native scheduling or request latency. Expert-only
budgets do not approve total engine allocations.

**147 Python tests passed**, including 16 cache-query tests: an independent
exhaustive victim-choice solver checks MIN on every sequence of up to six demands
over three keys at capacities 0–3; an independent second-chance queue checks CLOCK;
tests cover SLRU demotion, bounded oracle storage, zero/full caches, layer identity,
within-pass grouping, interrupted windows, restart/phase gaps, missing routes,
corrupt expert IDs, inconsistent identity/precision, unfinished reports, source
hash changes and query work limits. [Test output](python-tests.log) and
[verification hashes](verification.json) bind the implementation and source files.
No native rerun was necessary because this change only adds offline analysis.

## Reproduce and next experiment

Import the original sources using the existing query importer. Then:

```sh
PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/query_evidence.py \
  cache docs/benchmarks/2026-09-07-foundation/recorded-routes.jsonl

PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/query_evidence.py \
  cache docs/benchmarks/2026-09-09-selector-qualification/run-03/recovery/report.json.trace.jsonl
```

Default expert-cache budgets are 0, 1, 2, 4, 6 and 8GiB. The CLI documentation
describes phase filtering, custom budgets, per-layer counters and query bounds.
SQLite remains disposable; source reports and generated answers are ordinary files.

Keep global CLOCK in the engine. The next cache experiment should first capture
32 successive normal generated tokens, after a prompt, with a fixed deadline,
complete routes and explicit phase/session/reset markers. Expand useful evidence
to 256 tokens and a retained-history append. Only then select one policy for a
short timing screen at equal admitted memory, followed by full qualification if
it survives. The separate selector diagnostic retains its fixed 12GiB admission
and phase-progress reporting; these simulations do not qualify it.
