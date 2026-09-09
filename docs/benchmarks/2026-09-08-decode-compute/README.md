# Cached decode computation

This stage implements diagnostic GPU-pass timing and a packed single-token Q8
matrix kernel. Both are experimental. Artifacts, quantized weight bytes, expert
selection, and production defaults are unchanged.

## Attribution

The three-repetition 2K cached-token profile measures 117.53ms/token in
`q8_mm_r2_t1`, compared with 32.25ms in routed expert work. These are instrumented
GPU-pass durations, not additive wall-time components. The profile uses separate
compute encoders per dispatch while preserving submission boundaries. Its total
forward time is therefore not a normal latency measurement.

See [attribution](baseline-attribution.json) and [source hashes](profile-sources.json).
The profiling build is recorded separately from the later candidate build.

## Exact packed Q8

The candidate uses packed word loads, compile-time lane widths and independent
output accumulators. It retains the existing scalar accumulation and reduction
order. `--q8-decode-rows 2` is the selected experiment; rows 4/8 remain unselected.
All six captured projection cases improved with row two in the five-pair
operator screen. Some small-projection results for rows four/eight were
inconclusive or slower. See the [operator screen](operator-screen.json).

After selecting two rows, fresh [held-out operator checks](heldout-operators.json)
on the final native build passed all six cases with exact output. Paired latency
ratios ranged from 0.315 to 0.857; all six bootstrap upper bounds were below one.
Their token window and captured activations differ from tuning, and all payload
hashes were rechecked. See [held-out provenance](heldout-sources.json). These
windows come from the same fixed coding workload; this is numerical/operator
evidence, not independent coding-quality evaluation.

On build `a6e5e9cd4b744d3cf0a3f7cf6fe1d45763048876bcedfddc26e19eb0565b1aba`,
five alternating cached full-token comparisons at 2K produced:

| Measurement | Existing control | Packed Q8, two rows |
|---|---:|---:|
| Median forward latency | 276.67ms | 192.40ms |

The median paired latency ratio is 0.690, with a bootstrap interval of
0.676–0.711. Every forward has 480 ready expert hits, zero model reads, and
bit-identical logits, selected routes, and all persistent state versus the
original arithmetic. The maximum observed process footprint in these replays
is 7.37GiB within the fixed 12GiB diagnostic budget.

See the [paired summary](cached-paired-summary.json) and [native report](cached-paired-2048.json).
This clears 200ms for the observed cached path, while the proposed 150ms target
remains unmet. It does not establish 5 tokens/s with cache misses.

## Verification and remaining work

The final native build passes 44 tests and 6436 assertions with Metal API and
shader validation enabled. Forty-two focused Python tests pass, including complete
dispatch coverage, instrumentation rejection, and alternating cached-pair
validation with fixed execution and memory configuration. Operator comparison
tests reject unmatched, duplicate and inexact samples. Native fixtures cover Q8 lane/output tails and exact FP32/BF16
outputs for all three packed row configurations.

The final-build [mixed session qualification](state-qualification.json) passed
with a 257-token prefix, 129-token append, and two continuation tokens. All 48
layers' persistent state, routes and logits match the original implementation
at the checked stages; continued state also matches fresh replay. Both arms
pass all nine state/failure checks, including artifact mismatch rejection,
partial-panel failure invalidation, refusal to reuse failed state, cancellation,
and draining outstanding GPU users. Raw reports remain under
`.cache/benchmarks/decode-compute/qualify-mixed/`, bound by the summary hashes.
This compares native arithmetic; it is not a new independent long-context oracle
or a coding-quality evaluation.

## Normal requests

The [normal-request comparison](normal-screen.json) uses [these fixed configurations](configs.json),
two alternating screens, fresh processes, the same pinned mixed artifact,
12GiB memory, panel 512, chunk 128, and 64 generated tokens. Cases are an initial
2K prompt and a 128-token append to sample-free retained 4K history.

| Mean of two runs | Existing control | Packed Q8, two rows |
|---|---:|---:|
| Generation after 2K prompt | 2.681 tokens/s | 3.531 tokens/s |
| Generation after retained-history append | 2.639 tokens/s | 3.213 tokens/s |
| 2K first token | 147.054s | 147.114s |
| Append first token | 9.903s | 9.899s |
| Complete 2K request, 64 outputs | 170.610s | 164.964s |
| Complete append request, 64 outputs | 33.782s | 29.509s |

This is a 31.7% generation increase on the 2K workload and 21.8% after the append.
Complete-request latency improves by 3.3% and 12.6% respectively; the append
request measurement excludes the separate 4K priming request. Initial prompt
processing remains the dominant cost of the 2K request.
Candidate generation ranges were 3.470–3.591 and 3.199–3.226 tokens/s respectively.
All eight runs emitted identical tokens within each workload. Appends reused
exactly 4096 tokens and ingested exactly 128, with no pending sampled token.
Observed process footprint stayed at or below 10.42GiB, with no emergency memory
resizes. Reads and hit rates were similar across arms; the measured gain comes
from the changed computation path. This short screen does not establish sustained
swap behavior or confidence bounds for full-length requests.

[Source hashes](normal-sources.json) identify the local raw reports and admitted
budgets. A metadata-only admission retry was needed before one candidate append;
the requested budget stayed at 12GiB, and no inference request was retried.
The 5 tokens/s and 60-second initial-prompt targets remain unmet.

Production promotion, the 150ms cached milestone, five-pair 256-token product
qualification, 7K reporting and the sustained coding/recovery workflow are not
claimed. See the [stage contract](../../qwen_decode_compute_stage.md).

## Reproduction

Build with `./build.sh`. Run only one GPU workload at a time on the actual
32GiB M1 Pro. The native reports record the prepared manifest and pinned mixed
artifact identities; metadata admission must accept the full requested 12GiB.

```sh
PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/benchmark_exact.py \
  --output .cache/benchmarks/decode-compute/normal-repeat \
  --config docs/benchmarks/2026-09-08-decode-compute/configs.json \
  --mode screen --comparison-purpose experiment --pairs 2 --memory-gb 12 \
  --cases prompt_2k append_128
```

Choose a new output directory for each run. This is an engineering screen;
the runner's promotion mode has separate workload and repetition requirements.

Capture independent token windows, then replay only the already-selected Q8
variant. Native replay requires the original manifest and payload hashes from
the current build:

```sh
PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/capture_phase_inputs.py \
  --output .cache/benchmarks/decode-compute/heldout-repeat --split heldout \
  --cases decode-0-gdn decode-31-attention --repetitions 5
PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/benchmark_q8_decode.py \
  .cache/benchmarks/decode-compute/heldout-repeat/decode-0-gdn/manifest.json \
  .cache/benchmarks/decode-compute/heldout-repeat/decode-31-attention/manifest.json \
  --rows 2 --repetitions 5 --output .cache/benchmarks/decode-compute/heldout-operators-repeat
```
