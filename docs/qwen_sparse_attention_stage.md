# Exact sparse attention on the M1 Pro

The current follow-up is the [bounded selector qualification](qwen_selector_qualification_stage.md).
Use `--track selector` for that stage. The three-arm procedure below remains
available for historical reproduction and does not control the new stage's gates.

This stage removes the CPU sparse-selection dependency and optionally skips
wholly masked attention-score tiles. Both changes are explicit developer
experiments; production defaults and the pinned quantized weights are unchanged.
The reference remains the original CPU mask, score arithmetic, softmax, and
value aggregation. Every router-selected expert still executes.

## Native behavior

`--sparse-selection cpu|gpu` defaults to `cpu`.
`--attention-score-tiles full|skip-masked` defaults to `full`.
Both candidates are restricted to `bench` and `inspect` in the CLI. The native
library exposes the same explicit options. Automatic kernel selection does not
promote them.

GPU selection uses one 256-thread group per query and a bitonic sort of at most
2048 score/index pairs in 16KiB of threadgroup storage. Integer keys derived from
IEEE score bits preserve subnormal ordering even when GPU floating comparisons
flush tiny operands to zero. Scores descend; tied
scores, including signed zeros, use ascending indices. As in the CPU oracle,
ranking happens before finite-score and causal filtering. The first 512 ranks
are considered and each query's incomplete four-token tail is preserved.
Every mask byte is initialized. Negative infinity is valid; NaN and positive
infinity anywhere in the scores set a sticky error flag.

The flag is one persistent allocation outside either scratch pool. Its physical
16KiB charge is reported as `runtime_control_bytes` in all configurations;
resident-weight and session-snapshot sizes retain their meanings. The host checks
it at the existing router completion boundary, before route consumption or
expert admission. A forward starts only after prior users have drained. Invalid
scores abort the update, drain outstanding work, and leave partially changed
session state invalid. Diagnostic exports also check completed status.

Tile skipping uses the original 8x8 grid. The entire SIMD group returns before
Q/K loads only when all valid output positions are masked. It writes negative
infinity for those positions. Partly visible tiles keep the existing fragment
layout, accumulation, and BF16 rounding. Softmax and value reduction are unchanged.

## Correctness and fixture capture

Native tests compare CPU/GPU masks across sorting boundaries, ties, all four
tail alignments, irregular query rows, future scores and nonfinite inputs. Score,
probability and value outputs are bitwise compared, including fully masked tiles.
Sticky status is checked across both scratch pools.

Workspace reuse matches physical allocation sizes exactly. Reusing a large score
buffer for an earlier small allocation can exhaust the pool even when the actual
requested sizes fit; unused shapes are evicted instead. A bounded allocation-order
regression reproduces that failure in both pools, independently of model loading.

The production expert microbatch encoder is shared by both schedules. Tests use
projection-distinct packed codes/scales/biases, actual expert dimensions and
127/128/129, 255/256/257, and 513 selected rows. Unchanged single-row kernels form
the independent scheduling/scatter oracle; existing CPU tests cover arithmetic.
Every written contribution, reduction, untouched destination and guard is checked.
The integration test cycles 24 row-count steps through actual reads, three cache
slots, five experts, two scratch pools, eviction, reversed completion, cancellation
and recovery. Run the native test executable under a 600-second external timeout.

`bench --sparse-capture DIRECTORY` writes `sparse_attention_fixture_v1` data from
layers 3 and 47, capturing the first decode or retained-append case per layer.
Prompt microchunks do not fill the capture. The manifest binds build, artifact,
absolute positions, byte lengths and SHA-256 hashes. Captures are diagnostic and
limited to 16 cases / 256MiB. Existing manifests are never overwritten.
`qwen_sparse_replay MANIFEST REPORT` rejects changed build/artifact identity,
geometry, lengths or hashes and compares all three arms. It reports wholly masked
tile counts offline and separate instrumented GPU passes; neither is a normal
latency result. Tile counts are per head; all 24 heads share the same mask.
Required real-model fixtures cannot be skipped to produce a pass.

## Reproducible experiment

Use the existing pinned mixed artifact and prepared Q4 records, actual 32GiB M1
Pro, 12GiB total budget, panel 512, chunk 128, eight readers and packed Q8 rows 2.
All remaining settings are in
[configs.json](benchmarks/2026-09-09-sparse-attention/configs.json).

Arms: A = CPU/full; B = GPU/full; C = GPU/skip-masked. A/B cached comparisons cover
2K, 4K and 7K; B/C cover 4K and 7K. Each has five alternating pairs, original
reference logits/routes/all-state comparison, 480 ready expert hits and zero
application reads. `--cached-compare-axis` explicitly selects `q8_decode_rows`
(the backward-compatible default), `sparse_selection`, or `attention_score_tiles`.
Both host and Metal configurations switch per arm. Validators reject any other
configuration change, instrumentation, or incomplete pairing.
Cached timings must be positive and finite. Session qualification also verifies
the requested reference/candidate settings against both continued and fresh
execution, so accidentally running the reference twice cannot pass.
Cached replay restores both host options and Metal configuration on failure;
`qwen_cached_recovery` is the explicit real-model cancellation/reuse check.
Its latest admission limit and the audit evidence are recorded
[here](benchmarks/2026-09-09-sparse-attention/audit/README.md).

Use a new output directory for each command and run only one GPU workload:

```sh
PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/benchmark_sparse_attention.py \
  --phase cached --output .cache/benchmarks/sparse-attention/cached
PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/benchmark_sparse_attention.py \
  --phase capture --output .cache/benchmarks/sparse-attention/capture
PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/benchmark_sparse_attention.py \
  --phase qualify --output .cache/benchmarks/sparse-attention/state
PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/benchmark_sparse_attention.py \
  --phase screen --output .cache/benchmarks/sparse-attention/screen
PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/benchmark_sparse_attention.py \
  --phase paired --screen .cache/benchmarks/sparse-attention/screen/normal/summary.json \
  --output .cache/benchmarks/sparse-attention/paired
```

The normal screen uses two alternating pairs, 64 outputs, 2K/4K/7K prompts and
128-token appends to retained 4K. Freeze the candidate with the lower geometric
mean complete-request ratio on 4K and append; prefer B when within 1% of C.
Then run five alternating 256-output pairs against A on 2K, 4K and append.
Metadata-only admission retries never reduce budget or restart inference.

Stage latency acceptance requires at least 3% median complete-request improvement
on 4K or append, with its paired 95% upper ratio below 1. Every required workload's
first-token, generation and complete-request upper ratio must be at most 1.03.
Record negative and inconclusive evidence. GPU-pass times and CPU waits overlap
and must not be added as a wall-time breakdown.

Neither a stage timing pass nor fixture correctness is production promotion.
The original 5 tokens/s, 60-second 2K first-token, 10-second append, 7K reporting
and sustained 20-minute coding/recovery requirements remain separate gates.

## Compression follow-up

The [main roadmap](qwen_plan.md) now requires fusion-aware gate/up precision,
exact deployment-shaped calibration traces and disjoint quality evaluation.
Q3 conversion, EXL3 decoding, speculation, CPU KV offload and replacement-policy
experiments are not part of this implementation.
