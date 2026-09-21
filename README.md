# ZeroCool

A native C++23/Metal inference engine for Apple Silicon. No Python in the
serving path, no cloud, and one process that holds the model for as long as
you keep talking to it.

It is being proved on a deliberately hard case first. ZeroCool runs pinned
Qwen3.8-Flash-Next checkpoints of about 104–106GB on a 32GiB M1 Pro. Routed
experts and ngram tables stay on the internal SSD. The engine holds a bounded
working set of at most 22GiB, reduced further by the memory actually available.

Two artifacts are pinned: an affine-Q4 control, which is the default, and a
mixed 4/8-bit quality reference selected with `--artifact mixed-4_8bit`.

**Status: the acceptance plan has not passed.** Full-model checks match all 48
layers and all 248,320 logits exactly against the original MLX reference for
both artifacts on the saved five-token fixture, and native generation and API
smoke tests pass. Long-context numerical agreement, sustained coding quality,
and 5–8 tokens/s at the specified contexts remain open gates. Measured results
and their limits are in [docs/README.md](docs/README.md).

## Requirements

Apple Silicon, macOS 15 or later, a C++23-capable Apple toolchain, CMake 3.20 or
later, and libcurl from the SDK. Metal source is embedded and compiled at
runtime, so no standalone `metal` command is needed. About 104GB of free disk is
required for the checkpoint.

## Build

```sh
./build.sh
```

`build.sh` configures `build/qwen` in Release and builds everything. CMake
presets are also available:

```sh
cmake --preset release && cmake --build --preset release --parallel 8
```

Presets: `release`, `debug`, `asan`, `tsan`, `no-tui`. A build with a sanitizer
or without optimization reports a different build fingerprint, so its timings
can never be mistaken for evidence. Use `-DZEROCOOL_BUILD_TUI=OFF` to build
without the terminal client, and `-DZEROCOOL_BUILD_DIAGNOSTICS=OFF` to skip the
replay and probe executables.

## Get the model

```sh
bash scripts/qwen/download.sh
python3 scripts/qwen/verify_checkpoint.py
build/qwen/bin/zerocool inspect --model .cache/models/qwen38-flash-next
```

`models.lock.json` pins every required file and hash. Verification writes a
local receipt, and startup rejects files that changed since it was written.

## Use it

Interactive terminal chat, which starts its own local server:

```sh
build/qwen/bin/zerocool chat
```

Chat defaults to `.cache/models/qwen38-flash-next` and
`.cache/prepared/q4-records-v1`, requests 12GiB, and keeps the Q4 and native
execution defaults. Enter inserts a newline, Ctrl+D sends, Ctrl+C cancels, and
Ctrl+Q quits. To attach to a server you already started, use
`zerocool chat --connect http://127.0.0.1:8080`.

One-shot generation and the local API server:

```sh
build/qwen/bin/zerocool run --model .cache/models/qwen38-flash-next
build/qwen/bin/zerocool serve --model .cache/models/qwen38-flash-next --port 8080
```

The API exposes `/v1/models` and `/v1/chat/completions`, including streaming,
sampling, reasoning text and structured tool calls. Model IDs are
`qwen3.8-flash-next:4bit` and `qwen3.8-flash-next:mixed-4_8bit`. The listener
binds `127.0.0.1`, runs one conversation at a time, refuses cross-origin and
non-loopback requests, and requires `Content-Type: application/json` on POST.
ZeroCool never executes a tool; a coding client does that. See
[docs/qwen_usability.md](docs/qwen_usability.md) for the full contract.

Memory values are GiB. Context covers input plus output and cannot exceed 8192
tokens. No code raises macOS wired-memory limits; if admission fails, `inspect`
reports the current limit.

## Test and measure

```sh
ctest --preset release
```

That runs the native engine tests, the chat transport tests, and the model-free
Python checks. The Python checks need NumPy; create the environment with
`bash scripts/qwen/setup_reference.sh`. Model-free tests are not a release
check. `scripts/qwen/release_check.py` fails when numerical, real-M1
performance, session or coding evidence is missing or outside its limits.

```sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 build/qwen/test_qwen
build/qwen/bin/zerocool bench --storage --repetitions 3 --json storage.json
build/qwen/bin/zerocool bench --prompt 'The capital of France is' --max-tokens 256 --repetitions 3 --json generation.json
```

## A note on the name

This project was called FreeLLM until September 2026, and the rename went all
the way down: the `zerocool::engine` namespace, the `zerocool` binary, the
`ZEROCOOL_*` variables, the HTTP surface, and the prepared-storage format id,
which is now `zc-affine-records-v1`.

Renaming that format id meant rewriting a hash chain, because the manifest it
lives in is hash-pinned. All of it was regenerated locally, and the 93GB of
prepared records were never touched:

1. `manifest.json` carries the format id, so its SHA-256 moved to `3ceca18e…`.
2. That hash is pinned in `verification.json` and in `models.lock.json`.
3. Editing `models.lock.json` moved its own SHA-256 to `bc631072…`, which is
   pinned in turn by `file_locks_sha256` in `mixed-payload-reuse.lock.json`.
4. That last pin exists so a lock file cannot change without renewed evidence,
   so the payload-equivalence proof was re-run: all 816 routed-expert and ngram
   tensors compared byte for byte, 186.17GiB of reads, 143.7s. It is recorded in
   `docs/benchmarks/2026-09-21-rename-revalidation/`.

Nothing under `docs/benchmarks/` or `docs/experiments/` was edited. The
September 8 proof still records what it measured; the re-run was added beside it
at a new path, and the lock now points at the new one.

Old material refers to paths under `.../src/freellm`, and the build fingerprint
is now `82cb0373…`, matching none of the nine recorded in `docs/`. That is the
fingerprint working as designed: the real-model numerical, M1 performance and
session gates still have to be re-run before any of them count as evidence.

## Layout

| Path | Contents |
|---|---|
| `src/engine`, `include/engine` | The engine: storage, Metal, model, pipeline, session, server, CLI |
| `kernels/metal/qwen.metal` | Every compute kernel, embedded at build time |
| `tests` | Native and transport tests, plus captured numerical fixtures |
| `scripts/qwen` | Developer tooling: download, verification, screens, references |
| `docs` | Plan, engine reference, stage reports and measurement evidence |

For how the engine works and why it is built this way, including the Apple
Silicon constraints, the Qwen architecture and which optimizations were measured
and kept, see [docs/building-the-engine.md](docs/building-the-engine.md).

The original MLX oracle uses `mlx==0.31.1` and `mlx-lm==0.31.1`. Python is not a
production inference dependency. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for model and adapted-code
licenses.
