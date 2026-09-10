# Sparse-attention stage: M1 Pro evidence

Implementation and qualification are separate. The new GPU selector and masked
score-tile path are opt-in benchmark candidates. CPU selection and full score
tiles remain the production defaults. No model bytes or precision changed.

The subsequent [implementation audit](audit/README.md) fixes two qualification
validator gaps and benchmark cancellation recovery, records fresh verification,
and identifies the recovery check still blocked by current memory availability.
The measurements below retain their original build identities.

## Initial measurements (superseded build)

Actual Apple M1 Pro, 32GiB physical memory, 12GiB requested and admitted budget.
Mixed artifact revision `b2c422f3c643e36f04227a64d61796b44a4b1029`, prepared Q4
expert/ngram records. The complete experiment settings are in
[configs.json](configs.json).

Initial measured native build:
`504ba7664f83ee75b9fbfa67897a1492e6f3e62b4e060b03b77dd934f08dac98`.

| Cached prompt | CPU selection median | GPU selection median | Median paired ratio | Paired 95% interval |
|---|---:|---:|---:|---:|
| 2,048 | 184.860ms | 178.868ms | 0.96403 | 0.95457–0.99952 |
| 4,096 | 192.292ms | 189.740ms | 0.98673 | 0.98228–0.99443 |
| 7,168 | 218.518ms | 215.564ms | 0.97258 | 0.63384–1.04786 |

The 7K interval includes a regression; it is inconclusive. A subsequent numerical
test found that the initial floating-point sort treats subnormal inputs as zeros
on the M1, changing selection for both positive and negative tiny scores. The
test failed twice, while these ordinary model traces still matched exactly.
The selector now orders integer keys derived from the original IEEE bits,
canonicalizing signed zeros and rejecting NaN/+infinity by their bit patterns.
The three measurements above are retained as historical evidence and **do not
qualify the corrected build**. The source-change guard stopped the remaining
matrix before measurements from different builds could be combined.

The selector-only correction was build
`b2a95caa8c0c8d6f1d78d39b70a3cf1c4d602538fa1d89f0a6afcac114bce387`.
Its real 4K and retained-append captures matched all three arms. Its 7K prefill
then failed with `temporary workspace capacity exceeded`, before capture.

An 8MiB allocation-order regression reproduced the workspace failure six times:
first-fit reuse consumed large buffers for small requests and stranded later
allocations. The pool now reuses exact physical sizes and evicts unused shapes
within its existing capacity. No workspace or engine limit was raised.

Measured native build, including both numerical/workspace fixes:
`84229cdf0f0b35dd3f7028373a17c7b8a73b45e299060da1c884aed77772e323`.

Each comparison has five alternating pairs. Logits, all router selections and
all persistent layer state match the original reference exactly. Each timed
forward has 480 ready expert hits and zero application reads. Snapshot restore,
reference prefill, preload and equality checks are outside the timed region.
These are cached single-token experiments; they cannot establish normal request
throughput or the product's 5-token/s target.

Raw measurements and validated summaries are retained beside this report.
[cached-sources.json](cached-sources.json) records their local source paths and
SHA-256 hashes. Confidence intervals bootstrap the median of paired latency
ratios; the ratio of the two displayed medians is a different statistic.

## Qualification status

The measured build passes all **49 native tests / 6,873 assertions** under
Metal API and shader validation, in 3.85 seconds with an external 600-second
timeout. This includes future-block, subnormal and arbitrary finite-bit scores,
heavy expert microbatches, workspace shape transitions and the completion/
cancellation stress sequence using real uncached file reads.
The Python tooling suite passes **65 tests**.

[provenance.json](provenance.json) binds the native binaries, numerical tests,
fixture replay, benchmark/qualification tools, locks, configurations and workload
source by SHA-256. The failing pre-fix regression and successful final validation
logs are retained alongside it.

All **eight real-model captures** pass on the measured build, totaling 181,934,336
bytes (173.5MiB). Layers 3 and 47 agree bit for bit across CPU/full, GPU/full and
GPU/skip-masked for masks, attention scores, probabilities and final attention
outputs. Each instrumented profile contains only its own arm and layer.

| Captured work | Wholly masked tiles, layers 3 and 47 | Exact across all arms |
|---|---:|---|
| 4K next token | 34.1–35.3% | Yes |
| 7K next token | 59.8–60.3% | Yes |
| 128-token append to retained 4K | 11.9–16.5% | Yes |
| First decode after that append | 37.4–38.2% | Yes |

Counts refer to spatial query/key tiles per head; all 24 heads share each mask.
The 7K run completed with 11,169,939,456 peak Metal bytes (10.4GiB), a 12GiB
admitted budget and both temporary pools released after ingestion. This is a
profiled capture run, not a latency qualification or sustained-memory test.
[capture-summary.json](capture-summary.json) binds the replay reports, manifests
and native capture reports. The tensor payloads remain in the referenced local
cache directories. [fixture-rejection.json](fixture-rejection.json) records all
eight stale, malformed, oversized, hash-mismatched or truncated fixtures rejected
without a passing output report.

## Final-build cached comparison

At 4,096 prompt tokens, GPU selection with full score tiles versus GPU selection
with tile skipping completed five alternating pairs. Both arms match the original
CPU reference's complete logits, all routes and all persistent layer state. Every
timed forward has 480 ready expert hits and zero application reads.

| Full tiles median | Skip-masked median | Median paired ratio | Paired 95% interval |
|---:|---:|---:|---:|
| 196.046ms | 196.215ms | 0.99708 | 0.97496–1.01480 |

**Inconclusive:** tile skipping has not demonstrated a complete-token latency
gain in this cached 4K experiment. This does not support promotion. Smaller
attention-kernel work alone is insufficient evidence of a faster request.
The [validated summary](attention_score_tiles-4096-summary.json) and
[raw paired measurements](attention_score_tiles-4096.json) retain the evidence.

## Continued-session qualification

The 2,053-token prefix, 129-token append and two continuation tokens pass on the
measured build. GPU selection / tile skipping / the optimized schedule match the
original CPU-selection schedule exactly at every recorded stage: all 48 layers'
state, all route identities, all logits, positions and history. Each schedule also
matches its own fresh replay of the complete 2,184-token history. All nine native
state/failure checks pass in each arm, including wrong-artifact rejection,
partial-panel failure, cancellation, draining and refusal to reuse invalid state.
The requested 12GiB budget was retained.

[boundary-state-summary.json](boundary-state-summary.json) binds the complete
[reference](boundary-state-reference.json) and [candidate](boundary-state-candidate.json)
reports. This is native schedule parity; it is not an independent long-context
MLX oracle or a coding-quality evaluation.

The complete cached-context matrix, 4K/7K continued-versus-fresh qualification,
normal screen and five-pair request qualification remain pending. No normal-
request latency or production-promotion gate is claimed as passed. The neutral
4K cached tile result supports keeping the feature experimental.

See [the stage protocol](../../qwen_sparse_attention_stage.md) for exact commands,
required correctness coverage, selection rules and acceptance thresholds.
