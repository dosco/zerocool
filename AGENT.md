# Agent instructions for ZeroCool

ZeroCool runs Qwen3.8-Flash-Next on a 32GiB M1 Pro. There is one product: the
native C++23/Metal engine in `src/engine`, `include/engine` and
`kernels/metal/qwen.metal`. The earlier educational CPU/CUDA/TinyLLaMA tree has
been removed; do not reintroduce a second model, backend or tokenizer path.

## Scope

- The pinned affine-Q4 checkpoint is the unchanged control. The pinned mixed
  4/8-bit artifact is the quality target and must pass native validation before
  it can become a runtime default.
- In scope: contiguous prepared storage, completion-driven expert execution,
  next-layer expert prefetch, exact single-token kernels, layer-major prefill,
  and offline calibrated affine-Q3 experts.
- Keep the engine budget at or below 22GiB and the context at or below 8192
  tokens (`MaxContext` in `include/engine/storage.hpp`).
- Preserve each artifact's bytes, its exact router-selected experts, and its
  recurrent state. Never silently switch precision, substitute a smaller model,
  drop experts, or change OS memory limits.
- Offline candidate quantization must not overwrite the Q4 control. A different
  recipe may produce different routes and needs fresh session state.

## Naming

`src/engine`, `include/engine` and `zerocool::engine` are the model-independent
engine. `kernels/metal/qwen.metal`, `scripts/qwen/`, the `qwen_*` diagnostic
targets, `build/qwen` and `cmake/qwen.cmake` are Qwen-specific; a second model
family sits beside them rather than replacing them.

The prepared-storage format id is `zc-affine-records-v1`. Changing it moves four
pins: `manifest.json`'s SHA-256, its copies in `verification.json` and
`models.lock.json`, and then `file_locks_sha256` in
`mixed-payload-reuse.lock.json`, which pins `models.lock.json`'s own bytes. That
last pin forces renewed evidence whenever a lock file changes, so clearing it
means re-running `verify_mixed_payloads.py` (both checkpoints, ~190GiB of reads,
about two and a half minutes locally). Re-run it rather than hand-editing a
recorded hash.

## Working rules

- Build with `./build.sh` or `cmake --preset release`. Every target compiles
  with `-Wall -Wextra -Wpedantic -Werror` and `-fno-fast-math`.
- Run `ctest --preset release`: the native tests, the chat transport tests and
  the model-free Python checks. All three must pass before a commit.
- Use the existing nlohmann/json types and the existing Objective-C++ Metal
  bridge. Do not add a second JSON or GPU abstraction.
- CPU references and model-free tests support correctness. A release claim
  needs the explicit real-model and M1 performance gates.
- Report measured performance separately from targets, and say which run a
  number came from.
- New C++ follows `.clang-format`; new Python must pass `ruff check scripts`.
- Document a complex feature in `docs/`, and keep `docs/qwen_engine.md`
  describing what is implemented rather than what is planned.
- The decode defaults are all exact:
  - GPU keep-warm, single-token row kernels and SIMD route selection;
  - next-hyper expert prefetch and decode scratch reuse;
  - residency `auto`.

  `Options::resolve()` turns each `auto` into a concrete choice for the
  schedule, and every default has an off switch for control arms (see
  `zerocool --help` and
  [docs/qwen_decode_speed_stage.md](docs/qwen_decode_speed_stage.md)).

## Experiment discipline

Kernel, residency, memory and decode changes start behind an explicit option. A
change becomes a default only when it is exact within an artifact and a measured
complete-request gain is explicitly promoted, as the September 25 decode changes
were. Specifically:

- The `auto` kernel *policy* (shape tables, forced tiles) stays on the original
  kernels until a measured, exactly-validated rule is explicitly promoted.
- Bit-for-bit logits, routes and persistent state must hold within an artifact.
- Prove a new kernel exact at two levels:
  - Per operator, compare its raw FP32 sums, before output rounding, with the
    reference kernel on every shape the dispatch admits. BF16 output rounding
    hides most arithmetic differences, so comparing rounded outputs is not
    enough.
  - Per model, compare full-model logits and every layer's state against the
    reference configuration with `bench --chunk 1 --tokens-file ...
    --logits-file ...`. `--chunk 1` sends every token through the
    single-token path.
- Use fresh processes for independent measurements. A profiled timing, a cached
  replay, or a single screen cannot qualify a promotion.
- Time kernels in the engine, or with the GPU held busy. Decode's short,
  CPU-dependent command groups keep the GPU at a lower clock than a tight
  benchmark loop. One projection measured 363µs dispatched alone per command
  buffer and 108–119µs under sustained load.
- Pin `--expert-slots` in paired comparisons. Admission follows the memory other
  applications leave free, and cache capacity dominates generation once the GPU
  is fast.
- A run with compressed engine pages is disturbed: check each phase's
  `process.compressed_bytes` and decompressions.
- Treat prompt and generation latency equally.
- Allocation and state-lifetime checks are mandatory for any residency change.

Current stage documents live in `docs/` as `qwen_*_stage.md`; the plan of record
is [docs/qwen_plan.md](docs/qwen_plan.md) and the index is
[docs/README.md](docs/README.md).

## Safety-relevant invariants

- Message text from a client is never allowed to become a control token.
  `Tokenizer::encode_chat` encodes message and tool text as ordinary bytes; only
  the chat template itself emits turn boundaries. Do not call
  `encode(render(...))` on untrusted messages.
- The HTTP listener binds `127.0.0.1`, refuses a cross-origin request, refuses a
  non-loopback `Host`, and requires `Content-Type: application/json` on POST.
  Keep those checks when adding an endpoint.
- Retained session state is discarded only when it can no longer be continued
  (`Session::reusable`). A rejected or cancelled request must not force a full
  prompt replay.
- Expert prediction only warms the cache; it never decides which experts are
  computed. Speculative reads have four rules:
  - They are issued only after every selected expert of the current layer holds
    a lease.
  - They never evict a leased or loading slot.
  - They enter CLOCK unreferenced.
  - A demand acquire that joins a queued guess promotes it to demand priority.
- The GPU keep-warm runs model-independent work on its own queue. It holds only
  its accounted sink buffer and is joined before the Metal device is released.
