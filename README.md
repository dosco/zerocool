# FreeLLM: one large model on a 32GB M1 Pro

FreeLLM is an experimental C++23/Metal engine for pinned Qwen3.8-Flash-Next
checkpoints: a Q4 control and a mixed 4/8-bit reference, about 104–106GB each.
Routed experts and ngram tables stay on the internal SSD; the engine
uses a bounded working set of at most 22GiB, reduced by current system limits.

The [current plan](docs/qwen_plan.md) selects a pinned mixed 4/8-bit quality
reference, then calibrated affine-Q3 experts. The unchanged Q4 control remains
the default. Lossless prepared records and completion-driven expert execution
are implemented. The complete mixed artifact is now integrated behind explicit
`--artifact mixed-4_8bit` and can reuse the verified Q4 expert/ngram records.
[Full-model checks](docs/benchmarks/2026-09-08-validation/README.md) now match
all 48 layers and all 248,320 logits exactly for both artifacts on the saved
five-token fixture. Native retained state also matches across storage layouts,
cache sizes, token batching and short session continuations. These checks fixed
PLE gate and GDN normalization rounding discrepancies. Q4 remains the default;
mixed long-context correctness, coding quality and performance are still experimental.
Larger prefill panels are an explicit `--panel` experiment. An earlier 48-layer
257-token prompt/129-token append check preserved retained state and reduced
expert read bytes by 59.4%; equal-memory request timing remains unqualified.
[A normal mixed 2K run](docs/benchmarks/2026-09-08-validation/README.md)
with a 12GiB budget measured 470.6s to first token and 1.99 tokens/s, without
system swap growth. This single run misses the latency targets.

The next [exact-arithmetic kernel stage](docs/qwen_next_stage.md) adds packed
weight reuse across prompt tokens and precomputed recurrent gates, selectable
in benchmarks. A fresh-process mixed 2K/64-token screen at 8GiB reduced first
token time from 403s to 231s, with identical generated tokens; generation stayed
near 1.63 tokens/s. This is one screen, not paired qualification or a new
runtime default. See the [stage report](docs/benchmarks/2026-09-08-exact-kernels/README.md).

The [prompt-memory and matrix stage](docs/qwen_memory_compute_stage.md) adds
explicit workspace reclamation after ingestion, lazy expert-cache regrowth,
exact output-row blocking and shared gate/up input loads. These remain
benchmark-only. Filtered real-input capture covers later layers and separates
prefill, append and generation; paired promotion still requires normal requests.

The [cached decode computation stage](docs/qwen_decode_compute_stage.md) adds
GPU-pass diagnostics and exact packed Q8 decode. Five alternating 2K cached-token
pairs reduced median latency from 276.7ms to 192.4ms, with identical logits,
routes and persistent state. This is a zero-read diagnostic, not normal inference
throughput. Two alternating normal-request screens at 12GiB improved generation
from 2.68 to 3.53 tokens/s after a 2K prompt, and from 2.64 to 3.21 after a
128-token append to retained 4K history. Generated tokens matched; first-token
times stayed near 147s and 9.9s respectively. These are 64-output-token screens,
and the candidate remains benchmark-only. See the [stage report](docs/benchmarks/2026-09-08-decode-compute/README.md).

**The complete acceptance plan has not passed.** Earlier native generation and API
smoke tests passed, and both saved five-token full-model fixtures now match the
original MLX reference bit for bit. Longer-context numerical agreement,
sustained coding quality, and 5–8 tokens/s at the specified contexts remain
qualification gates. See [implementation and measurements](docs/qwen_engine.md).

## Build and assets

Apple Silicon, macOS 15 or later, a C++23-capable Apple toolchain, and CMake
are required. Metal source is embedded and compiled at runtime, so building
the native engine does not require the standalone `metal` command.

```sh
./build.sh
bash scripts/qwen/download.sh
build/qwen/bin/freellm inspect --model .cache/models/qwen38-flash-next
```

The source download is approximately 104GB. `models.lock.json` pins all required
files and hashes. Verification creates a local receipt; startup rejects
files changed since verification. Full release checks rehash all files.

```sh
python3 scripts/qwen/verify_checkpoint.py
build/qwen/bin/freellm run --model .cache/models/qwen38-flash-next
build/qwen/bin/freellm serve --model .cache/models/qwen38-flash-next --port 8080
```

The local API exposes `/v1/models` and `/v1/chat/completions`, including SSE,
sampling, reasoning text, and structured tool-call output. Model ID:
`qwen3.8-flash-next:4bit`, or `qwen3.8-flash-next:mixed-4_8bit` for the explicitly
selected experimental mixed artifact. Bind address: `127.0.0.1`. One conversation executes
at a time. A coding client executes tools; FreeLLM provides inference.

Memory values are **GiB**. Context includes input and output and cannot exceed
8192 tokens. No code raises macOS wired-memory limits. If admission fails,
`inspect` explains the current memory limit; closing unused apps may help.

## Verification and measurement

```sh
ctest --test-dir build/qwen --output-on-failure
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 build/qwen/test_qwen
build/qwen/bin/freellm bench --storage --repetitions 3 --json storage.json
build/qwen/bin/freellm bench --prompt 'The capital of France is' --max-tokens 256 --repetitions 3 --json generation.json
```

Set up an isolated reference environment with `bash scripts/qwen/setup_reference.sh`.
Python reference/benchmark tooling additionally uses NumPy, tokenizers, and
Jinja2. The original-checkpoint oracle uses `mlx==0.31.1` and
`mlx-lm==0.31.1`; Python is not a production inference dependency.
Model-free tests are not a release check. `scripts/qwen/release_check.py`
fails when numerical, real-M1 performance, session, or coding evidence is
missing or outside its limits.

## Preserved legacy work

The previous models and backends remain available explicitly:

```sh
FREELLM_BUILD_DIR=build/legacy ./build.sh -DFREELLM_BUILD_LEGACY=ON
```

The [previous README](docs/legacy_readme.md) documents that implementation.
It is outside the default build. See [third-party notices](THIRD_PARTY_NOTICES.md)
for model and adapted-code licenses.
