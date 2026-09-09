# Q4 audit, mixed Q8 operators, and first normal 2K baseline

Native build: `0b92485a51ee9558bcf74e0e79e05d8e37054763863f663ed8199bcb30e7cddb`.
Actual 32GiB M1 Pro and internal SSD. These measurements preceded the panel
implementation and remain evidence for this build only.

## Correctness

- [Native tests](native-tests.txt): 25 cases, 855 assertions, Metal API and shader validation enabled.
- [Full-model comparison](q4-parity.json): all 48 layer outputs and 248,320 final
  logits match the independently generated original MLX five-token fixture bit
  for bit. Prepared/completion/chunk=8/cache=32 and source/batched/chunk=1/cache=64
  also produce identical retained history and all 109 recurrent, convolution,
  attention, and index buffers. This does not qualify longer contexts or coding quality.
- [Storage replay](storage-correctness.json): all 1,553 selected expert records
  remain byte-identical to source. All 48 MoE outputs and four 130-token ngram
  cache passes remain identical under cold/hot/evicting caches.
- Invalid route traces now reject empty counts, fractions, overflowing IDs,
  duplicates, inconsistent positions/builds, and incomplete passes before GPU work.
  Paired dependency summaries retain all recorded passes and compare ordered
  expert-output hashes across layouts/schedules.
- Failed forward passes invalidate direct-library state. History commits only
  after output allocation and GPU completion succeed. Cancellation stops further
  expert admission/encoding and drains outstanding users.

## Mixed Q8 reference work

[Thirty-five real Q8 cases](q8-operators.json), totaling 110,566 output values,
match fixed per-token MLX 0.31.1 QMV bit for bit. Cases cover shared gate/up/down,
attention, Gated DeltaNet projections, hyper connections, embedding, output rows,
irregular dimensions, token batches, and repeated/reversed gathers. Peak native
Metal allocation was 3,784,704 bytes.

This is native affine Q8/64 matmul, fused gate/up, and embedding support. It
does not enable or qualify the complete mixed checkpoint. Only one 4.877GB shard
was downloaded and SHA256-verified against the pinned mixed revision; bounded
row fixtures preserve the source codes/scales/biases. The fixture manifest and
source lock are saved with the report. Ordinary batched MLX QMM is a separate
comparison: its different reduction geometry gives up to 0.003555 relative L2
on these fixtures. The fixed QMV comparison is the exact arithmetic control.

[Metadata verification](mixed-metadata.json) confirms identical tokenizer,
template, generation configuration, and architecture implementation files.
Only quantization fields differ in the configuration. Large tokenizer identity
uses the LFS payload SHA256, not the Git pointer blob hash. Expert/ngram payload
identity across complete artifacts has not yet been established.

```sh
.cache/qwen-reference-venv/bin/python scripts/qwen/reference_q8.py \
  --download --output /private/tmp/freellm-q8-reference
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 build/qwen/qwen_q8_check \
  /private/tmp/freellm-q8-reference q8-reference.lock.json q8-check.json
```

The offline reference uses MLX; production inference is C++/Metal and does not.

## Normal request baseline

The [normal 2K run](normal-2k.json) completed with tracing and Metal validation
disabled. It used an exact 2,048-token chat-formatted code excerpt and generated
256 tokens with greedy sampling. The input retains the beginning and end of a
longer rendered excerpt; [the actual prompt](normal-2k-prompt.txt) and all token
IDs are saved. Its output is not an independently scored coding workflow.

| Metric | Observed | Target |
|---|---:|---:|
| Time to first token | **461.55s** | ≤60s |
| Generation, 255 measured intervals | **1.753 tokens/s** | ≥5 |
| Median token latency | 345.44ms | report |
| 95th-percentile token latency | 443.82ms | report |
| Prepared application reads | **604,813,276,600 bytes** | report |
| Admitted engine budget | 8,075,902,976 bytes | ≤22GiB |
| Expert slots | 1,043 | admitted from same budget |

The requested 12GiB was reduced by live system availability. The reported
mean includes long latency outliers, so it is lower than the reciprocal of the
median token latency. This is one baseline, not five alternating paired runs,
and it fails both headline latency targets. Cumulative application reads include
prefill and generation; they must not be presented as prefill-only traffic or
physical SSD traffic. Device counters include other processes.

An earlier [short control](normal-small.json) had a 15.39s first token for a
128-token prompt, then 0.980 tokens/s over four decode intervals before EOS.
A 128-token append reused 132 computed tokens, required 129 new computations,
and returned EOS after 14.49s. It has no decode interval from which to estimate
throughput, and its retained history was shorter than the 4K acceptance target.

Repeated expert reads across 128-token chunks are a measured reason to implement
the planned layer-major panels. Panels alone cannot establish the generation
target; cache locality, GPU execution, and the remaining dependency path still
require request-level measurement. No 4K/7K, 20-minute stability, or coding-quality
gate has passed.
