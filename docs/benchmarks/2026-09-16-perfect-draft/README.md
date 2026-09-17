# Perfect verifier: exact recovery passes; observed speed below the gate

Follow-up: the [capacity screen](../2026-09-16-perfect-draft-capacity/README.md)
now observes fewer reads and 4.6655 verified tokens/s at 1,460 slots. It stops
early below the 5-token/s gate; the reverse timing round remains unrun.

Serial, two-token and four-token verification match full-vocabulary logits,
selected routes and persistent state exactly. Changed future proposals leave
earlier logits unchanged; zero acceptance and every partial accepted-prefix
replay recover the corresponding serial state. These checks use the short fixed
workload and Metal validation, not production session/RNG rollback qualification.

The [current screen](screen-04/summary.json) completed all three correctness
processes and four of six timing processes. Its [audit](audit-04.json) passes.
The next process failed fixed memory admission, so the screen remains **failed,
incomplete**. The completed observations already miss the protocol's absolute
5 verified tokens/s floor. Do not integrate the real draft head or repeat the
unchanged candidate merely to fill the missing timing rows.

## Observations

All completed timing processes have zero observed process compression and
decompression, unchanged swap, nominal thermal state, AC power and Low Power
Mode off. Peak physical footprint is approximately 8.52–8.55GiB. Each verifies
16 tokens after the same 72-token prime; proposals are free and always accepted.

| Timing order | Width | Verified tokens/s | Live cache hits | Expert MiB/token | GPU ms/token |
|---|---:|---:|---:|---:|---:|
| First round | 1 | 4.0760 | 47.80% | 660.66 | 130.33 |
| First round | 2 | 3.6139 | 31.79% | 693.79 | 153.56 |
| First round | 4 | 4.1240 | 0% | 813.59 | 124.50 |
| Reverse round | 4 | 4.1061 | 0% | 813.59 | 127.35 |

There is no complete alternating-pair result, confidence interval or qualified
speedup. The [offline diagnostic](partial-diagnostic.json) binds these calculations
to raw report hashes and keeps `source_complete: false`. Its own `complete: true`
describes completion of the analysis only.

Both width-4 runs read 4,937 expert records with zero hits. Their mean distinct
working set per block is 1,234.25 records, above the 1,072-slot cache. They read
23.15% more expert bytes per token than serial. This supports cache thrashing as
a hypothesis; the aggregate mean is not a per-block route trace or a cache
simulation. A controlled capacity change is needed to establish its effect.

Width 4 reduces dispatches from 3,175 to 1,359.19 per token, but first-round GPU
time improves only about 4.5%. Aggregate GPU counters cannot attribute this to a
particular kernel. Checkpoint copies cost approximately 0.7ms/token at width 4;
they are not the dominant measured cost. Summed overlapping read-service and
queue durations must not be treated as the request's critical path.

The interrupted process requested 12,884,901,888 bytes (12GiB), but the engine
admitted 12,811,026,432 bytes. Its 1,072 slots were unchanged. The guard stopped
before priming rather than changing the experiment's allocation. The sealed
report retains its original `failed` status and records no timing for that arm.

## Benchmark repairs and preserved attempts

- `screen-02`: serial validation completed with clean memory, but native host
  counters reported battery power and Low Power Mode on. The stage stopped.
- `screen-03`: after AC/LPM conditions changed, serial and width-2 arithmetic,
  state and recovery matched, but starting cache hashes differed. No timing ran.
  The [cache analysis](cache-mismatch-analysis.json) preserves that distinction.
- `screen-04`: the [predeclared setup revision](protocol-stable-prime.md) uses
  existing fixed batches of up to 32 expert leases only during priming. The
  completion-driven coordinator is restored on every exit and used for measured
  decode. New prime logits/state/routes match the earlier clean serial prime;
  every successful process now has the same initial cache hash.

A separate native host preflight now checks power and thermal observations
before loading any model. It reports the reason and stops immediately when Low
Power Mode or thermal conditions violate the protocol. It loads no model and
creates no Metal device. The helper is frozen with the experiment's source and
binary identity. All 18 focused verifier/builder/preflight tests pass, as does
the native checkpoint self-test.

Production source and executable are unchanged by these benchmark repairs. The
native base fingerprint remains
`51877f96f015386dd642fd4e63fbc39ff2e75c1156f5ae1bcba764f546963594`;
the production binary SHA256 remains
`05495d0baf993b1f7f1bc4b9365753e88603909ac66db3309a1380ea8c7c519f`.
The separate revised verifier executable SHA256 is
`741e5f150c81efcdc2eb83a5c50b48fb31c7e85cc7aaa6c45e8b6db6fae4549b`.
Artifact, prepared-record, workload and tool identities are bound in each stage.
`initial-sources/`, `capture-sources/` and `screen-sources/` preserve the tools
used by the respective attempts; earlier audits retain their original versions.

## Next bounded experiment

Compare width-4 verification with 1,072 and 1,460 target expert slots at the same
12GiB total budget. This tests the block working-set hypothesis, a different
workload from the earlier negative serial capacity experiment. Declare the
protocol before running, then:

1. Validate logits, routes, persistent state and recovery at the larger capacity
   against the serial reference. Require identical model/workload/arithmetic.
2. Require repeatable initial cache state within each capacity; cache contents
   across different capacities are intentionally allowed to differ.
3. Run fresh alternating capacity pairs with the same width, proposals,
   checkpoint, instrumentation and host/memory gates. Include serial at the
   larger capacity before claiming any advantage from block verification.
4. Record hits, bytes, wall time and GPU time. Stop early if a clean candidate
   observation again misses the 5-token/s floor. No old timing samples are pooled.

The additional 388 slots cost 1,074,331,648 bytes (about 1.001GiB). The verifier
allocation remains below 12GiB on paper, but adding the currently planned fully
resident MTP head would exceed that budget. A positive cache result would require
a new joint target/draft allocation plan and measured drafting cost. It would not
authorize silently growing memory, changing precision or claiming production
5-token/s performance. Long contexts, real acceptance, session/RNG rollback and
sustained coding workflows remain unqualified.
