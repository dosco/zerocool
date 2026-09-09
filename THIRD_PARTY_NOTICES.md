# Third-party notices

The specialized runtime and its verification tools consult the following
sources. The model files are downloaded separately and are not committed.

- Qwen3.8-Flash-Next checkpoint and `qwen4_exp.py`, revision
  `aa7c790e804bbf9d491ddb109c3d61bc4a555f7c`: architecture equations used by
  `src/qwen/model.cpp`, `kernels/metal/qwen.metal`, and the independent CPU
  reference. [Qwen Community License 1.0](docs/licenses/qwen.txt).
- Slotstream, revision `9342e70cec78db060f61098ffe011f722ef77cf0`:
  checkpoint manifest, layout descriptions, and the adapted gated-delta kernel
  in `Sources/Slotstream/Vendored/GatedDelta.swift`.
  [MIT, Carlos Galarza](docs/licenses/slotstream.txt).
- The gated-delta implementation traces to mlx-swift-lm and mlx-lm.
  [MIT, ml-explore](docs/licenses/mlx-swift-lm.txt) and
  [MIT, Apple](docs/licenses/mlx-lm.txt). Those notices apply to the adapted
  recurrence implementation in `kernels/metal/qwen.metal`.
- MLX 0.31.1: affine Q4/Q8 accumulation, BF16 activation boundaries, and
  weighted normalization/reduction behavior in `kernels/metal/qwen.metal`
  are adapted from `quantized.h`, `utils.h`, `unary_ops.h`, and
  `rms_norm.metal`, and `reduction/reduce_row.h`/`reduction/ops.h` (PLE BF16
  partial sums). [MIT, Apple](docs/licenses/mlx.txt).
- minja, revision `021c2293c187789ef13d56c6cfd89c9b134fd80f`, provides Jinja
  parsing. [MIT, Google](docs/licenses/minja.txt).
- nlohmann/json 3.11.3 is fetched by CMake under its MIT license. The fetched
  source retains its `LICENSE.MIT` notice. Existing doctest and legacy
  dependencies retain their own notices.

ds4 (`f62ca29a308724cde5bc99134ede19104b2a3260`) informed cache ownership,
bounded reads, and scheduling experiments. No ds4 source was copied. Lily's
packed-weight and fusion ideas informed the fused gate/up design; no Lily
source was copied. Future source reuse must carry the corresponding notices.
