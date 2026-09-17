# Reference decode with observed compression

The new diagnostic completed in **51.836 seconds**, preserving every token and
the existing qualification rules. It identifies substantial resident GPU work
and expert-read waits; it does not establish a speed improvement or a causal
explanation of the earlier packed-Q4 regression.

The [raw capture](capture-01/summary.json) uses unchanged native build
`51877f96f015386dd642fd4e63fbc39ff2e75c1156f5ae1bcba764f546963594`, mixed artifact
`b2c422f3c643e36f04227a64d61796b44a4b1029` and prepared manifest
`c4bb4db3220a0de2218128738b2a6ac650a3d65086690c8dc2e8c23d097160da`.
Reference Q4, 1072 expert slots, core-cache residency and the 12GiB budget stayed
fixed. The [protocol](protocol.md) permits a bounded compression observation
only for this diagnostic. Earlier blocked reports retain their status.

## Where the time went

The [timeline](capture-01/timeline.json) covers sixteen decode forwards after
the 72-token prompt and sixteen after a retained 128-token append. These rows
are mutually exclusive intervals and sum to measured forward time.

| Observed interval, ms/token | Initial | Append |
|---|---:|---:|
| GPU executing | 126.51 | 145.44 |
| GPU idle with pending expert reads | 58.37 | 61.83 |
| Other GPU idle | 37.78 | 37.64 |
| GPU idle with submitted work | 30.12 | 31.19 |
| GPU idle with ready experts | 11.86 | 11.96 |
| GPU idle awaiting callbacks | 9.43 | 9.69 |
| **Total forward time** | **274.06** | **297.74** |
| Distance above 200ms/token | 74.06 | 97.74 |

This is an instrumented reference run, with observed compression. The distance
to 200ms is not an attainable savings estimate or a normal-throughput comparison.
The [ranked command populations](capture-01/opportunities.json) retain whole
stage sets and layer sets. The largest GPU class includes GDN projections and
update, hyper-connection input, residuals, preceding expert reduction and router:
54.08/61.70ms per token. Its corresponding attention class is 17.07/19.97ms.
Routed-only commands total 27.77/33.23ms and mixed shared/routed commands
20.39/22.41ms. These command sums describe subsets of the GPU row; they are not
additional time or isolated kernel costs.

Expert application reads are 658.03/722.13MiB per token. Median queue delays are
0.0268/0.0255ms, versus median read service of 2.388/2.462ms. More workers have
little support from this evidence. The previous-token top-two route heuristic
intersects only one initial and zero append new misses in the eligible trace;
this does not justify live prefetch. Prior whole-read coalescing, duplicate
ownership and Q8 load-ahead results remain negative evidence.

## Memory and exactness

[Memory annotations](capture-01/memory.json) retain all 76 chronological
observations and all 32 forwards. Physical footprint peaked at **8.604GiB**;
compressed lifetime peak was **108.75MiB**, inside the declared 512MiB
diagnostic allowance. Observed system swap remained unchanged at 1876819968
bytes. The strict clean-memory result is still false.

Of 561 observed decompressions, 354 occurred inside initial decode forwards
and two inside append forwards; the remaining changes are explicitly assigned
to ingestion or boundary gaps. Append was slower despite fewer decompressions.
This does not establish compression as harmless, but it does not support
blaming it alone for the phase difference. Sampling calls consumed about
0.121/0.119ms per token; these measure only explicit observation calls, not all
tracing overhead. No timing correction or token filtering was applied.

Outputs and normalized dispatch work match the sealed previous reference at
the same short history. The append reused 88 computed tokens and ingested
129 including the pending generated token. Full persistent-state evidence is
reused from the compatible sealed check; it was not newly recomputed.

## Selected experiment

Follow-up: the [completed fusion screen](../2026-09-15-hyper-fusion/README.md)
preserves exact outputs but makes the complete-block cycle 1.10% slower. It is
rejected without native integration or long qualification.

The [next experiment](next-experiment.md) tests decode-only fusion of each
layer's hyper up-projection, sigmoid and four-stream mix, preserving the
unchanged injection branch and exact arithmetic. The trace verifies 96 such
layer projections per token. A fused path can remove 192 dispatches and two
temporary vectors per block. **That is not a claim that the 54–62ms command
class disappears.** The fusion targets only part of that class and may fail
the material-gain gate; test complete blocks cheaply before any integration.

The 5 tokens/s objective remains open. No production or precision change is
promoted, and no second inference process was started in this diagnostic.

## Verification

- **354 Python tests passed**, including fourteen new diagnostic/ranking tests.
- [Offline reconstruction](capture-audit.json) and
  [independent raw audit](independent-review.json) pass.
- Verified 101600 dispatches, 1536 layer passes, 15360 selected-expert records,
  exact outputs, retained reuse, buffer ownership and complete trace coverage.
- Native source and tooling snapshots are archived alongside the sealed run.
  The SQLite ledger remains disposable; the raw reports remain authoritative.

```sh
.cache/qwen-reference-venv/bin/python scripts/qwen/request_diagnostic.py run \
  --output FRESH_DIRECTORY
.cache/qwen-reference-venv/bin/python scripts/qwen/request_diagnostic.py verify \
  --source FRESH_DIRECTORY --output AUDIT.json
```
