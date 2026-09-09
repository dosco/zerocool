# Q4 storage and completion foundation — September 7, 2026

The native Q4 foundation is implemented and passes the bounded real-weight
checks below. **The complete plan and first-deliverable qualification are not
finished:** current memory pressure prevents full-model requalification and
normal initial/append/generation baselines. Mixed/Q8, layer-major panels,
calibration/Q3, and prediction remain subsequent work.

Native build: `8358867f2e1dfe9964864bd1697755de0453554dfa71a3b781a49e99285a202c`.
Q4 source: `aa7c790e804bbf9d491ddb109c3d61bc4a555f7c`.
Prepared manifest: `c4bb4db3220a0de2218128738b2a6ac650a3d65086690c8dc2e8c23d097160da`.
[Evidence identity](evidence-identity.json) binds the native build, artifact,
recorded routes, tool sources, and report hashes.

## Implemented

- Lossless expert records containing all nine original code/scale/bias pieces:
  2,764,800 payload bytes, 2,768,896-byte stride, 16KiB alignment. The source
  layout remains selectable; `--prepared` explicitly selects the sidecar.
- Contiguous 100-byte ngram rows, repeated-row joins and bounded same-page
  coalescing, with decoded values cached in BF16 without additional rounding.
- Eight persistent readers, current-demand priority, and half of queue admission
  reserved for demand rather than future ngram work.
- Completion notifications, a rolling window of at most 32 expert leases,
  at most two submitted GPU groups, and release/admission after completion.
  Ready work scatters to original positions before the unchanged reduction.
- Global CLOCK with survivor-preserving shrinkage; explicit cancellation drain.
- Native per-layer tracing and real-expert route replay; group limits 1/2/4/8
  and the former batch schedule are selectable for comparison.
- Atomic resumable preparation, full payload verification, pinned manifest,
  source/range/format checks, and cached file-identity validation at startup.

The prepared sidecar occupies **100,048,541,696 bytes (93.18GiB)** in addition
to the unchanged source checkpoint. This preparation does no quantization.
The mixed 4/8-bit quality reference and original source revisions are recorded
in `recipes.lock.json`; neither is claimed as a supported runtime artifact.

## Paired M1 measurements

Actual 32GiB M1 Pro, internal SSD, eight workers, F_NOCACHE. These are **isolated
48-layer expert dependency traversals**, using the saved real routes' last token
and real Q4 gate/up/down GPU work with fixed BF16 inputs. Ready hits are seeded
before each measured layer. The 0/5/10-hit scenarios are controlled experiments,
not observed hit rates from a coding session. Attention, routing, ngrams, shared
experts, and cache warmup reads are outside the timed interval.

Five paired repetitions, alternating case order, gave these medians:

| Storage and schedule | 0/10 hits | 5/10 hits | 10/10 hits |
|---|---:|---:|---:|
| Source, batched | 519.69ms | 268.78ms | 157.30ms |
| Source, completion | 391.48ms | 249.77ms | 143.30ms |
| Prepared, batched | 389.04ms | 210.89ms | 149.80ms |
| Prepared, completion (group 4) | **290.27ms** | **180.02ms** | 152.82ms |

Combining layout and scheduling reduces this all-miss median by **44.1%** and
half-hit median by **33.0%**. Both contribute independently. All-hit runs show
no reliable benefit from storage rearrangement. These are not tokens/s or
complete-request speedup claims.

All-miss runs read 1,327,104,000 application bytes in the measured region;
half-hit runs read 663,552,000; all-hit runs read zero. Seeded-hit reads still
access SSD outside that region. Device counters are recorded around each layer
and include unrelated system activity. They are not attributed solely to FreeLLM.

| Prepared completion group limit | 0/10 hits | 5/10 hits | 10/10 hits |
|---|---:|---:|---:|
| 1 | 288.73ms | 184.09ms | 179.73ms |
| 2 | 292.29ms | 179.45ms | 170.80ms |
| 4 | 290.27ms | 180.02ms | 152.82ms |
| 8 | 289.81ms | 178.73ms | 156.56ms |

Keep the approved starting limit of four. This sweep does not justify a larger
fixed limit using complete-request evidence. Five repetitions characterize
variability; they do not establish statistical significance or coding quality.
See [summary and ranges](dependency-summary.json) and the
[105 raw timing reports](dependency-runs.tar.gz).

The 5 tokens/s budget is **200ms for the whole token**; 8 tokens/s allows 125ms.
A half-hit expert path already consumes about 180ms. Even the all-hit path uses
about 150ms in this replay. Thus the target still requires measured cache locality
and substantially cheaper remaining computation; the 8 tokens/s aim is not
supported by these current kernels. Q3 payload savings alone do not resolve
this execution floor. These are engineering implications, not a full-model
performance projection: shared-work overlap and the live cache can differ.

Traces include admission, queue, read, encode, submit, GPU-group, and release
timestamps. GPU times apply to the group, not an individual expert kernel.
Group intervals can overlap; summed service/GPU times must not be treated as
exclusive contributions to token latency. The replay reports coordinator wait,
last-required-read latency, and ready-to-group-GPU delay separately.

## Correctness and integrity

- [Native suite](native-tests.txt): 22 tests, 741 assertions with Metal API and
  shader validation. Covers reversed readiness, rolling eviction, loading joins,
  priority, resizing, cancellation, and encoding/read failures.
- [Real-weight replay](storage-correctness.json): **1553 expert records** compared
  byte for byte to the source; all expert contributions across **48 layers**
  match the previously qualified five-token control fixture bit for bit.
  Fixture hashes are in the report. The unchanged final reduction/kernel is
  retained; this isolated test does not execute attention or reconstruct state.
- Ngram values for 130 tokens, including repeated tokens and EOS/history cases,
  are bit-identical across the source layout, prepared cold/hot cache, and
  forced eviction (four prepared passes).
- [Prepared verification](prepared-verification.txt) rehashed all 176 payload
  files. [Eleven altered or truncated artifact cases](prepared-rejections.json)
  are rejected. The source checkpoint and its existing weight hashes are intact.
- [Release rejection tests](release-rejections.txt): four pass. Missing assets,
  stale build evidence, diagnostic inference, and fewer than five performance
  repetitions cannot produce a passing release check.

The earlier full-model control matched all 48 layer outputs and 248,320 logits
against MLX. That historical result belongs to build `8c56e73e…`; it has **not**
been transferred to this build as a new full-model pass. Reversed readiness and
fixture contributions are useful evidence, but longer-context logits and all
persistent state still require live comparison.

## Current admission blocker and next gates

Normal startup needs a minimum engine budget of 5,275,484,160 bytes plus
1.5GiB external headroom: approximately **6.41GiB reclaimable memory**.
The machine currently exposes roughly **2.6GiB**, so the normal baseline exits
before inference. The smaller full-model diagnostic also fails admission
(minimum engine 2,280,734,720 bytes, plus the same headroom). See
[normal admission](normal-admission.txt), [diagnostic admission](full-model-admission.txt),
and [memory snapshot](prepared-inspect.json). No OS limits were raised or apps
closed. System swap was already substantial before these runs; there is no
20-minute stability result to attribute to the engine.

Once sufficient memory is available, the next required work is:

1. Full-model logits and recurrent/attention-state comparisons, including forced
   cache eviction, continued sessions, changed history, and prepared/source paths.
2. Normal Q4 prompt/append/generation baselines, recorded real coding hit/miss
   distributions, and complete-request comparisons with profiling disabled.
3. Mixed Q8 validation and resident-memory accounting, then layer-major panels,
   source-derived Q3 calibration/quality evaluation, and trace-justified cache or
   prediction experiments, in the approved plan's order.

The engine remains experimental. No 5–8 tokens/s, 2K/append latency, 7K context,
sustained coding workflow, or mixed-artifact quality gate is marked passed.

## Reproduction

```sh
./build.sh -DCMAKE_MAKE_PROGRAM=/usr/bin/make \
  -DFETCHCONTENT_SOURCE_DIR_JSON="$PWD/build/_deps/json-src"
.cache/qwen-reference-venv/bin/python scripts/qwen/prepare_storage.py \
  --output .cache/prepared/q4-records-v1 --verify
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 build/qwen/test_qwen
build/qwen/qwen_storage_check .cache/models/qwen38-flash-next \
  .cache/prepared/q4-records-v1 /path/to/qualified/step_0 storage-check.json
python3 scripts/qwen/benchmark_dependencies.py \
  --prepared .cache/prepared/q4-records-v1 \
  --routes docs/benchmarks/2026-09-07-foundation/recorded-routes.jsonl \
  --output /private/tmp/freellm-dependency-reproduction
```

The saved route file is included. The larger qualified tensor fixtures remain
in the existing local fixture directory; missing them is an explicit error.
The reproduction needs real Metal access and sufficient ordinary memory.
