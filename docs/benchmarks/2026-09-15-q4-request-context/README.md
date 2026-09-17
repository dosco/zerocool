# Actual Q4 request context

The runner and offline trace analyzer are implemented. **The retry passed
admission and completed the reference conversation, but process compression
stopped the comparison before the packed variant.** The first attempt remains
resource-blocked before inference. No native inference source, weights,
precision, cache capacity or defaults changed.

The [declared protocol](protocol.md) runs one normal reference/packed pair at
1072 slots, with sixteen decode forwards in each of two request phases. A
directional slowdown triggers a reverse-order traced pair. The analyzer checks
32 forwards, 101600 dispatches, 1536 layer passes and 15360 expert records,
then ranks exclusive GPU/waiting buckets and complete command classes. Mixed
commands remain mixed. This can locate the actual-request discrepancy and the
remaining distance to 5 tokens/s without attributing overlapping costs twice.

## Attempt and admission

[capture-01](capture-01/summary.json) stopped after 7.893 seconds. The native
build is `51877f96f015386dd642fd4e63fbc39ff2e75c1156f5ae1bcba764f546963594`,
mixed artifact `b2c422f3c643e36f04227a64d61796b44a4b1029`, prepared manifest
`c4bb4db3220a0de2218128738b2a6ac650a3d65086690c8dc2e8c23d097160da`.

| Admission observation | Reported reclaimable memory | Outcome |
|---|---:|---|
| Initial | 11.866GiB | 1072 slots fit, but only a 10.366GiB limit was admitted |
| Retry 1 | 10.921GiB | Requested expert capacity rejected |
| Retry 2 | 9.421GiB | Requested expert capacity rejected |

The matched experiment requires the full 12GiB admitted limit plus the native
1.5GiB system headroom, or 13.5GiB reclaimable. The last observation was
4.079GiB short. The requested plan totals 9.998GiB including reserves; the first
admission therefore fit the requested slots but failed the fixed-budget
comparison contract. The later observations also failed slot capacity.

A subsequent read-only process snapshot found Brave the largest application
group at approximately 11.9GiB summed RSS. Summed RSS includes shared pages and
is neither unique physical memory nor guaranteed reclaimable memory. That
later snapshot does not establish why memory changed during admission. See
[admission context](admission-context.json). No other applications were stopped.

## Retry after Brave closed

[capture-02](capture-02/summary.json) was explicitly requested after Brave was
closed. Admission reported 13.804GiB reclaimable and retained the full 12GiB
limit and 1072 slots. The reference process completed both requests; the stage
then stopped after 50.126 seconds at its declared clean-memory gate.

| Reference phase | Observed generation | First token | Whole request |
|---|---:|---:|---:|
| Initial 72-token prompt | 3.709 tokens/s | 13.885s | 18.199s |
| Retained 128-token append | 3.464 tokens/s | 23.916s | 28.535s |

These timings are memory-disturbed observations, not a comparable clean pair or
evidence of an improvement. The packed variant and conditional traces never
started. Both reference phases completed seventeen outputs and sixteen decode
forwards, totaling 101600 dispatches. The append reused 88 computed tokens and
ingested 129, including the pending generated token.

Process compression first appears at the end of initial ingestion, before the
first measured decode window: 0 → 103.891MiB. Its cumulative peak reached
104.547MiB; decompressions increased from 0 to 1036. Physical footprint peaked
at 8.563GiB, below the 12GiB bound, and observed system swap stayed unchanged at
1893597184 bytes. This was a compression stop after successful admission, not
an allocation overrun or swap-growth failure. Admission alone did not guarantee
uncompressed execution.

The [retry audit](capture-02-audit.json) and
[independent raw review](capture-02-independent-review.json) preserve that
distinction and verify the complete reference work and unchanged identities.
No new persistent-state comparison or candidate correctness claim follows from
this single reference process. A later
[host snapshot](capture-02-host-snapshot.json) confirms no Brave process was
present; it does not identify which host activity caused compression during
inference. No other applications were changed.

## Verification and next step

- All **340 Python tests passed**, including thirteen new runner/analyzer tests.
- The [offline audit](capture-audit.json) reconstructs the sealed incomplete
  outcome and source identity. The compatible existing full-state proof was
  checked before admission; no new full-model execution occurred.
- An [independent review](independent-review.json) verifies the frozen files,
  artifact fingerprints, prior state proof and zero inference work. The sealed
  capture remains unchanged.
- Current tooling is archived under `screen-sources`; `initial-sources`
  preserves the native source fingerprint. The original request rejection and
  every earlier blocked or inconclusive diagnostic retain their decisions.

When sufficient headroom is available, run the unchanged protocol into a new
`capture-03` directory. Preserve both blocked attempts; do not shrink their
cache/budget or infer a paired performance result from either block. Stop
expanding isolated fixtures. The next
useful evidence remains an actual normal-request pair and, if triggered, the
complete decode dependency comparison. No 5 tokens/s or production promotion
has been established by this stage.

```sh
.cache/qwen-reference-venv/bin/python scripts/qwen/q4_request_context.py run \
  --output docs/benchmarks/2026-09-15-q4-request-context/capture-03
.cache/qwen-reference-venv/bin/python scripts/qwen/q4_request_context.py verify \
  --source docs/benchmarks/2026-09-15-q4-request-context/capture-03 \
  --output docs/benchmarks/2026-09-15-q4-request-context/capture-03-audit.json
```
