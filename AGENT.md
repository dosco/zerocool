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
  layer-major prefill, and offline calibrated affine-Q3 experts.
- Keep the engine budget at or below 22GiB and the context at or below 8192
  tokens (`MaxContext` in `include/engine/storage.hpp`).
- Preserve each artifact's bytes, its exact router-selected experts, and its
  recurrent state. Never silently switch precision, substitute a smaller model,
  drop experts, or change OS memory limits.
- Offline candidate quantization must not overwrite the Q4 control. A different
  recipe may produce different routes and needs fresh session state.

## Naming

The project was renamed from FreeLLM to ZeroCool in September 2026. The
directory split now carries meaning, so keep it:

- `src/engine`, `include/engine` and the `zerocool::engine` namespace are the
  model-independent engine.
- `kernels/metal/qwen.metal`, `scripts/qwen/` and the `qwen_*` diagnostic
  targets are Qwen-specific and keep that name. A second model family would sit
  beside them, not replace them.
- `build/qwen` and `cmake/qwen.cmake` also keep their names; they are disposable
  output and a build definition.

The prepared-storage format id is `zc-affine-records-v1`. If it ever changes
again, four pins move with it and the last one is not local arithmetic:
`manifest.json`'s SHA-256, its copies in `verification.json` and
`models.lock.json`, and then `file_locks_sha256` in
`mixed-payload-reuse.lock.json`, which pins `models.lock.json`'s own bytes.
That last pin exists to force renewed evidence whenever a lock file changes, so
clearing it means re-running `verify_mixed_payloads.py` (both checkpoints, about
190GiB of reads, roughly two and a half minutes locally). Re-run it; never
hand-edit a recorded hash to make the check pass, because the pin's only value
is that it has never been typed by hand.

Record new evidence at a new dated path under `docs/benchmarks/` and re-point
the lock at it. Do not edit files under `docs/benchmarks/` or `docs/experiments/`
in place; they record what a run actually observed.

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

## Experiment discipline

Kernel, residency, memory and decode experiments are benchmark-only until
paired normal-request evidence qualifies them. Specifically:

- `auto` kernel policy stays on the original kernels until a measured,
  exactly-validated rule is explicitly promoted.
- Bit-for-bit logits, routes and persistent state must hold within an artifact.
- Use fresh processes for independent measurements. A profiled timing, a cached
  replay, or a single screen cannot qualify a promotion.
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
