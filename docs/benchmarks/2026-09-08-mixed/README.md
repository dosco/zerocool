# Mixed artifact integration and PLE fidelity

The complete mixed 4/8-bit artifact is downloaded, pinned and integrated into
the native engine. Its routed expert/ngram tensors can reuse the existing Q4
prepared records: every byte of all 816 relevant tensors was compared. Native
prepared-record execution, Q8 operators and the corrected PLE operators pass.
**Full mixed-model numerical qualification remains open.** The initial complete
comparison exposed a BF16 reduction error; memory admission prevented the
complete comparison from running again after that error was fixed.

The final native build in this report is
`40e554b6a941b1c0d0dd6eaee0405c07dae7b557cdd08e4e8cc50d6349e980b4`.
All GPU checks used the actual 32GiB Apple M1 Pro with Metal API and shader
validation. These correctness diagnostics do not qualify request latency.

## Artifact and storage evidence

| Check | Result |
|---|---|
| Complete mixed download | 24 required files, 106,218,442,623 bytes, SHA256 verified |
| Layout correspondence | All 3,215 tensor names and decoded shapes compatible |
| Routed experts | 432 tensor components, 67,947,724,800 bytes identical to Q4 |
| Ngram tables | 384 tensor components, 32,000,153,600 bytes identical to Q4 |
| Resident precision | 498 affine Q8 matrices, with existing BF16 tensors preserved |
| Resident allocation | 5,362,515,968 bytes; Q4 uses 2,937,683,968 |
| Cache cost of mixed precision | 2,424,832,000 additional resident bytes (2.258GiB), about 876 expert slots at equal budget |
| Native prepared-record replay | 1,561 selected records, all 48 layers' expert contributions, four ngram replays over 130 tokens: bit identical |

[layout.json](layout.json) records bounded header inspection.
[payload-equivalence.json](payload-equivalence.json) records the full byte
comparison, both complete checkpoint locks, file receipts and all per-tensor
hashes. Its approximately 199.9GB of application reads cover both sources;
this is not a physical SSD bandwidth measurement. The checkpoint files were
hashed during download and their current identities checked before and after
the byte comparison; `source_files_rehashed: false` means that the comparison
did not repeat those whole-file hashes.

`mixed-payload-reuse.lock.json` pins that evidence and both source-lock hashes.
Native prepared startup also validates the unchanged Q4 manifest, file hashes,
current file identities, ranges and formats. A changed source lock requires
renewed equivalence evidence. Resident tensors and tokenizer are loaded from
the selected mixed checkpoint. The prepared manifest's source revision remains
Q4; `inspect` reports the consuming mixed revision separately.

[prepared-storage.json](prepared-storage.json) validates native storage reuse
on the routes and inputs recorded before the PLE fix. It proves expert and
ngram computation on those inputs; it does not stand in for the corrected
full-model forward pass. Reuse avoids a second 100,048,541,696-byte prepared copy.

## Numerical investigation and fix

The independent full-model oracle now applies each matrix's declared Q4/Q8
format. Its reader and module preparation pass 36 independent real Q8 fixture
checks. The updated oracle also preserves all 48 Q4 layer outputs and the
248,320-logit golden vector exactly. See [oracle-q8-check.json](oracle-q8-check.json)
and [oracle-q4-regression.json](oracle-q4-regression.json).

The mixed oracle executes the pinned checkpoint's original decoder code using
MLX 0.31.1, with bounded reads for selected experts and ngrams. Its five-token
run, `[760, 6511, 314, 9338, 369]`, is saved in [reference.json](reference.json),
[reference-identity.json](reference-identity.json) and
[reference_logits.f32](reference_logits.f32). It is offline reference tooling;
the production library, CLI and server remain C++23/Metal.

The initial mixed native build `c669e046...` failed full-model comparison:
relative L2 was 0.0986083 against the 0.02 limit. Layer zero was exact, with the
first discrepancy in layer one's PLE block. Its Q8 key/value projections and
normalizations were bit identical. The gate reduction differed: the original
BF16 partial sums produce -312, whereas the native FP32 sum rounded to -314.
The resulting gate changed from 0.0771484375 to 0.07568359375 and propagated
through later layers. The failed comparison is retained in
[before-ple-fix-full-parity.json](before-ple-fix-full-parity.json).

`ple_gate` now follows the pinned reference's row reduction: 640 threads, four
adjacent products per thread, BF16 local sums, twenty BF16 SIMD partials and a
BF16 final sum. The order remains fixed across native token chunk sizes.
All eight production PLE stages on the same five real token inputs are now
bit identical, including normalization, gate output and convolution:
[ple-parity.json](ple-parity.json). A checked-in real boundary fixture protects
this case in the native suite. The ordinary embedding path was also corrected
to select Q4 or Q8 from the artifact instead of assuming Q4.

The final build passes 29 native tests / 893 assertions and 35 real Q8 cases.
The developer-tool suite passes 23 tests, including rejection of changed
payloads, cross-artifact logit/performance evidence and obsolete native builds.
See [native-tests.txt](native-tests.txt), [q8-check.json](q8-check.json) and
[tool-tests.txt](tool-tests.txt).

## Full Q4 panel checks completed before the PLE change

These results belong to their named earlier builds, not the final build above.

* Build `86d30a8...`: all 48 layers, 257-token initial prompt, 129-token append
  and two single-token continuations. Panel sizes 0, 256 and 512 all matched
  continued and fresh-replay logits and all retained state exactly. The
  13 checks also cover cancellation and partial-panel failure.
* At a fixed 32 expert slots, the continued workload read 164,845,670,400
  expert bytes with the 32-token chunk control, 68,273,971,200 with panel 256,
  and 66,971,750,400 with panel 512: a 59.4% reduction for panel 512.
* Build `c669e04...`: 20 five-token full-model panel/state/failure checks passed,
  including rejection of session state from the other artifact. Initial Q4
  logits remained identical to the independent golden fixture.

See [panel-full-append-86d30a8.json](panel-full-append-86d30a8.json) and
[panel-full-small-c669e04.json](panel-full-small-c669e04.json). These used
diagnostic trunk streaming and different scratch allocations, with a checkpoint
download running concurrently. Read counts support panel design; elapsed times
cannot qualify equal-memory request performance. Longer panel checks must be
repeated after the PLE arithmetic change.

## Admission and outstanding gates

The final mixed diagnostic needs at least 2,973,270,016 admitted engine bytes
(2.769GiB), plus the separate 1.5GiB system allowance. Two attempts after the
fix could admit only 568,573,952 and 880,295,936 bytes. Both stopped before
loading the model. Normal mixed startup at context 8192 needs at least
7,700,168,704 engine bytes; [inspect.json](inspect.json) records its current
admission failure and complete allocation estimate. No memory limits were
raised and no unrelated applications were stopped.

Next qualification work:

1. Repeat the full mixed five-token comparison against the saved independent
   logits and layer hashes, using source and prepared records. Recheck Q4
   after the shared PLE arithmetic change.
2. Compare both artifacts' continued state with fresh replay across chunk,
   panel and cache configurations, including the long sparse-attention case.
3. Run the five alternating paired normal-request comparisons at equal
   admitted memory. `benchmark_panels.py` accepts an explicit `--artifact`.
4. Establish mixed coding quality before original-derived Q3 calibration.
   No Q3 conversion or precision switching has been introduced.

The previous normal Q4 baseline remains 461.55 seconds to first token and
1.753 tokens/s for the 2K/256-output workload. The 5–8 tokens/s goal,
2K/append latency limits, 7K performance, paired quality bounds and sustained
20-minute coding workflow are not qualified. `recipes.lock.json` keeps the
mixed artifact unpromoted until full validation passes.

## Reproduction

```sh
./build.sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 build/qwen/test_qwen
.cache/qwen-reference-venv/bin/python -m unittest discover \
  -s scripts/qwen -p 'test_*.py'

# Complete pinned mixed source, about 106GB. Existing verified files are reused.
.cache/qwen-reference-venv/bin/python scripts/qwen/download.py \
  --lock mixed-models.lock.json --model .cache/qwen-mixed-reference
build/qwen/bin/freellm inspect --artifact mixed-4_8bit \
  --model .cache/qwen-mixed-reference --prepared .cache/prepared/q4-records-v1

# Full-model diagnostics still obey live admission; this is not a speed benchmark.
build/qwen/bin/freellm bench --artifact mixed-4_8bit \
  --model .cache/qwen-mixed-reference --prepared .cache/prepared/q4-records-v1 \
  --stream-trunk --context 256 --chunk 8 --expert-slots 32 --memory-gb 4 \
  --tokens-file tests/fixtures/qwen/mixed-reference-tokens.json \
  --logits-file /private/tmp/mixed-logits.f32 --json /private/tmp/mixed-native.json
.cache/qwen-reference-venv/bin/python scripts/qwen/compare_logits.py \
  --native /private/tmp/mixed-logits.f32 --native-report /private/tmp/mixed-native.json \
  --reference docs/benchmarks/2026-09-08-mixed/reference_logits.f32 \
  --reference-report docs/benchmarks/2026-09-08-mixed/reference.json \
  --out /private/tmp/mixed-parity.json
```
