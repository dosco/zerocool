# Implementation audit

The review found no selection, attention arithmetic, buffer-lifetime or memory-
accounting defect in the new native paths. Three concrete verification/recovery
issues were fixed; complete-request performance qualification remains unfinished.

## Fixes

- Cached comparison summaries rejected neither negative durations on both arms
  nor nonfinite timings. The validator now requires positive finite numbers.
- Session qualification could compare the reference report against itself.
  Both continued and fresh runs now have to match their independently requested
  reference/candidate execution, kernel and scheduling configurations.
- Cancelling cached replay during original-kernel setup or a control arm restored
  host options without restoring Metal configuration. One exception handler now
  covers setup, preload, warmup and measurements and restores both configurations.

The download recipe/corruption unit test also depended on the machine having
5GiB free disk space. It now mocks sufficient space; the separate disk-admission
test and production reserve remain unchanged.

## Verification and limits

Before the recovery fix, all recorded source/binary provenance hashes matched.
All 49 native tests / 6,873 assertions passed again with Metal API and shader
validation. All eight saved real-model 4K, 7K and retained-append attention cases
were replayed exactly on that same measured build:
`84229cdf0f0b35dd3f7028373a17c7b8a73b45e299060da1c884aed77772e323`.
The replay profiles contain the new selector and masked-tile kernels, confirming
the candidate dispatches are executed.

After the recovery fix and addition of its developer check, native build
`857ee812709cba3817cfef46d431060744a529bd2d3fa802fa3ece6dd430cb5c`
passes all 49 native tests / 6,873 assertions under both Metal validators.
All 68 Python tooling tests pass. The previously recorded cached comparison and
boundary session reports pass the strengthened validators with their original
build identity. This does not relabel old performance evidence as current-build
qualification. No inference arithmetic or weight bytes changed during this audit.

The new `qwen_cached_recovery` executable builds and checks cancellation during
reference setup and a control forward, restored settings, GPU drain, and fresh
inference using the same model. Its real-model runs were refused by memory
admission before either version reached cancellation: available engine budgets
were 8,116,666,368 and 7,851,524,096 bytes, below the fixed 12GiB test budget.
**The end-to-end recovery regression is not yet qualified.** No budget or system
memory limit was changed, and no passing recovery report was generated.

When the fixed budget can be admitted, run this check alone with an external
600-second timeout and a new output path:

```sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 build/qwen/qwen_cached_recovery \
  .cache/qwen-mixed-reference .cache/prepared/q4-records-v1 \
  .cache/benchmarks/sparse-attention/recovery-qualified.json
```

The GPU selector's invalid-score status is tested and its propagation into
forward-state invalidation was traced in source. A dedicated injected-invalid-
score full-model failure/recovery test is still absent. Independent full-model
fidelity, normal request latency and sustained coding-session acceptance remain
separate outstanding gates.

Tile skipping removes only wholly masked QK score tiles. Softmax and value
aggregation still scan the complete context, and partially visible tiles retain
all their original work. A correct implementation can therefore produce the
recorded neutral full-token timing result.

[Provenance and logs](provenance.json) bind the reviewed fixes, both builds and
the newly run checks. The parent report's provenance is retained as historical
evidence.
