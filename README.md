<div align="center">

# ZeroCool

### A 105GB model. A 32GB laptop. No cloud.

[![checks](https://github.com/dosco/zerocool/actions/workflows/ci.yml/badge.svg)](https://github.com/dosco/zerocool/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![platform](https://img.shields.io/badge/platform-Apple%20Silicon-111111?logo=apple&logoColor=white)](#requirements)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-00599C.svg)](#requirements)

</div>

---

ZeroCool is a native **C++23 / Metal** inference engine for Apple Silicon. One
process holds the model, serves on a loopback socket, and hands the memory back
when you close it. No Python in the serving path. Nothing leaves the machine.

It is being proved on a deliberately unreasonable case: **Qwen3.8-Flash-Next —
about 104–106GB of weights — on a 32GiB M1 Pro.** Routed experts and ngram
tables stay on the internal SSD and stream in on demand, so the engine never
holds more than a bounded working set. Everything smaller is the easy case.

| | |
|:--|--:|
| Checkpoint on disk | ~104 GB |
| Prepared expert records | 100,048,541,696 bytes |
| **Peak working set** | **≤ 22 GiB** |
| Machine it targets | 32 GiB M1 Pro |
| Context window | 8192 tokens |

## Status: not done yet

This is an engineering log, not a launch. Full-model checks match **all 48
layers and all 248,320 logits exactly** against the MLX reference, for both
artifacts, on the saved five-token fixture. Native generation and API smoke
tests pass. The rest is open:

| Gate | State |
|:--|:--|
| Bit-exact logits vs MLX (five-token fixture) | ✅ passing |
| Native generation + API smoke | ✅ passing |
| Long-context numerical agreement | ⬜ open |
| Sustained coding quality | ⬜ open |
| 5–8 tokens/s at the specified contexts | ⬜ open |

Measured results and their limits are in [docs/README.md](docs/README.md). A
number counts here only when a build fingerprint and a real-model run back it.

## Quick start

> **Heads up:** this wants ~200GB of free disk and a long afternoon. The engine
> is a 3MB binary. The model is all the rest.

**1. Install the engine** — this repo is its own Homebrew tap:

```sh
brew tap dosco/zerocool https://github.com/dosco/zerocool
brew install --HEAD zerocool
```

<details>
<summary>or build from source</summary>

<br>

```sh
git clone https://github.com/dosco/zerocool.git && cd zerocool
./build.sh                      # binary lands in build/qwen/bin/zerocool
```

</details>

**2. Get the model.** No Python, no repository checkout — the lock is compiled
into the binary, so this works from any directory:

```sh
zerocool setup                     # download, verify, then prepare: one command
```

`setup` fetches the ~104GB checkpoint four files at a time, verifies every
pinned hash, and repacks the routed experts and ngram tables into the ~100GB of
contiguous records the engine reads. Interrupt it and rerun: a partial transfer
continues from where it stopped and a finished record is never rebuilt. The
steps are separately available when you want them:

```sh
zerocool download --jobs 8         # more concurrency if your link allows
zerocool verify --check-receipt    # recheck without rehashing 104GB
zerocool prepare --output DIR      # repack only
```

**3. Talk to it:**

```sh
zerocool chat
```

`Enter` inserts a newline, `Ctrl+D` sends, `Ctrl+C` cancels, `Ctrl+Q` quits.

> The 104GB fetch and the 100GB preparation pass are the real cost, not the
> install. They stay explicit and resumable rather than buried in a package
> manager's install hook, so interrupting one loses nothing.

## Use it

A Homebrew install puts `zerocool` on `PATH`. From a source checkout, alias it
first with `alias zerocool=$PWD/build/qwen/bin/zerocool`.

```sh
zerocool run     --model .cache/models/qwen38-flash-next   # one-shot generation
zerocool serve   --model .cache/models/qwen38-flash-next --port 8080
zerocool chat    --connect http://127.0.0.1:8080           # attach to a server
zerocool inspect --model .cache/models/qwen38-flash-next   # identity and budgets
```

The server speaks an OpenAI-shaped API — `/v1/models` and
`/v1/chat/completions`, with streaming, sampling, reasoning text and structured
tool calls. Model IDs are `qwen3.8-flash-next:4bit` and
`qwen3.8-flash-next:mixed-4_8bit`. Two artifacts are pinned: an affine-Q4
control (the default) and a mixed 4/8-bit quality reference via
`--artifact mixed-4_8bit`.

**ZeroCool never executes a tool.** It emits structured tool calls; a client
decides what to do with them. The listener binds `127.0.0.1`, serves one
conversation at a time, and refuses cross-origin and non-loopback requests. Full
contract in [docs/qwen_usability.md](docs/qwen_usability.md).

## Requirements

Apple Silicon · macOS 15 or later · a C++23-capable Apple toolchain · CMake
3.20+ · libcurl from the SDK. Metal source is embedded and compiled at runtime,
so no standalone `metal` command is needed.

## Build and test

```sh
cmake --preset release && cmake --build --preset release --parallel 8
ctest --preset release
```

Presets: `release`, `debug`, `asan`, `tsan`, `no-tui`. A sanitized or
unoptimized build reports a different fingerprint on purpose, so its timings can
never be mistaken for evidence.

`ctest` runs the native engine tests, the chat transport tests and the
model-free Python checks. Checks that read recorded benchmark evidence skip
cleanly on a fresh clone, naming exactly what they'd need. None of it is a
release pass — [`scripts/qwen/release_check.py`](scripts/qwen/release_check.py)
is, and it fails closed without real-model numerical, M1 performance and session
evidence.

## Layout

| Path | Contents |
|:--|:--|
| `src/engine`, `include/engine` | The engine: storage, Metal, model, pipeline, session, server, CLI |
| `kernels/metal/qwen.metal` | Every compute kernel, embedded at build time |
| `scripts/qwen` | Download, verification, screens and CPU references |
| `tests` | Native and transport tests, plus captured numerical fixtures |
| `docs` | Plan, engine reference, stage reports and measurement evidence |

For how it works and why — the Apple Silicon constraints, the Qwen
architecture, and which optimizations were measured and kept — see
[docs/building-the-engine.md](docs/building-the-engine.md).

## Why "ZeroCool"

Dade Murphy's handle in *Hackers* (1995). It also happens to describe the
engine twice over:

- **Zero-copy** — prepared expert records are mapped straight out of contiguous
  SSD storage. Weights are not unpacked into a second buffer on the way in.
- **Cool** — a bounded working set means the machine isn't swapping itself to
  death or spinning its fans to keep 105GB resident.

It was called FreeLLM once. "Free" said nothing true about it.

## License

[Apache License 2.0](LICENSE). The adapted kernels and tooling keep their
upstream MIT notices, collected in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights are downloaded
separately and are **not** covered by this license — the checkpoint is governed
by the [Qwen Community License](docs/licenses/qwen.txt).

The MLX oracle used for verification pins `mlx==0.31.1` and `mlx-lm==0.31.1`.
Python is not a production inference dependency.
