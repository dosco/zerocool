# Decode speed: GPU clock, row kernels and next-layer prefetch

**Decision.** On 2026-09-25 these changes became the defaults for every command:

- a GPU clock keep-warm;
- exact single-token row kernels;
- SIMD route selection;
- next-hyper expert prefetch;
- decode scratch reuse;
- residency `auto`.

On the Q4 control they took generation from 2.40 to 5.29–5.40 tokens/s after the
72-token coding prompt, and from 2.27 to about 4.89 after the retained
128-token append. One clean run of the 2K/256 acceptance workload measured
5.86 tokens/s. Generated tokens, all final logits and every layer's state are
bit-identical to the previous defaults.

These are single screens on a busy laptop, not the repeated 2K/4K
qualification. The design and the reasoning are in
[the stage report](../../qwen_decode_speed_stage.md).

## Setup

- **Machine.** 32GiB M1 Pro with the internal SSD. Other applications were
  running (browser, containers, chat clients), so a 12GiB request was admitted
  at about 6.5–8.8GiB.
- **Artifact.** Q4 control `aa7c790e`, with prepared records.
- **Common settings.** Context 8192, temperature zero, seed zero, fresh
  processes.
- **Short workload.** The 72-token coding prompt from the earlier screens with
  96 generated tokens, then a retained 128-token append with 48 generated
  tokens. Expert slots pinned at 850 so both arms are comparable.
- **2K workload.**
  [`2026-09-08-validation/workload-2k.json`](../2026-09-08-validation/workload-2k.json):
  a 2048-token prompt and 256 generated tokens, with panel 512.

```sh
zerocool bench --model .cache/models/qwen38-flash-next --prepared .cache/prepared/q4-records-v1 \
  --memory-gb 12 --context 8192 --temperature 0 --seed 0 --repetitions 1 \
  --workload-file WORKLOAD --expert-slots 850 [--panel 512]
```

The previous defaults are reproduced on any later build with
`--route-selection serial --q4-rows off --decode-scratch none
--expert-prefetch off --gpu-warm off --residency off`.

## Results

Tokens/s is `zerocool bench`'s decode rate; GPU is command-group time per
decode token. Build `736c023b` is the committed engine. `faabb18a` and
`985f9861` are earlier builds of the same change. They differ only in the review
fixes and the residency default, not in any arithmetic.

| Build | Configuration | Slots | Request | Tokens/s | GPU ms/token | Compression |
|---|---|---:|---|---:|---:|---|
| faabb18a | Previous defaults | 850 | 72-token prompt | 2.398 | 279 | none |
| faabb18a | Previous defaults | 850 | 128-token append | 2.269 | 291 | none |
| faabb18a | New defaults, residency off | 850 | 72-token prompt | 5.395 | 65 | none |
| faabb18a | New defaults, residency off | 850 | 128-token append | 4.894 | 66 | none |
| 736c023b | New defaults | 850 | 72-token prompt | 5.287 | 65 | none |
| 736c023b | New defaults | 850 | 128-token append | 4.885 | 68 | none |
| 985f9861 | New defaults, residency off | 900 | 2K/256 | 4.855 | 73 | 3.34GiB during ingest |
| 985f9861 | New defaults, residency on | 900 | 2K/256 | 4.893 | 73 | 0.09GiB |
| 736c023b | New defaults, unpinned | 1977 | 2K/256 | 5.855 | 72 | 0.10GiB |

First-token time was 7.7–7.8s for the short prompt and 12.7–13.3s for the append
in both configurations. With the new defaults it was 194–199s for 2K; no 2K run
used the previous defaults. Output tokens are identical in every row sharing a
request (see `output_token_ids_sha256`).

With residency off, the 2K runs compressed 3.34GiB of engine pages during
ingestion and made 65 thousand decompressions during decode, even with 5.9GiB
reported reclaimable. That is why residency now defaults to `auto`.

## Exactness

The runs took 12 prompt tokens one at a time (`--chunk 1`, so every token uses
the single-token path) with `--logits-file`. The previous defaults and the final
defaults produced the same logits file, SHA-256 `9d344a29…`, and the same
SHA-256 for every layer's recurrent, convolution, attention and index state.

The native suite adds a test comparing each row kernel, `plain_mv` and
`norm_wide` with its reference kernel. It compares raw FP32 sums before output
rounding as well as rounded outputs.

## Diagnostic screens behind the decision

These screens are not request timings.

- **Kernel profile.** Per-dispatch GPU time was about 300ms per token, most of
  it in matrix-vector kernels running at 12–25GB/s. Kernels timed inside the
  engine were about three times slower than the same kernels timed alone.
- **Clock probe.**
  - One 14.7MB Q4 projection per command buffer took 363µs.
  - The same dispatch took 108–119µs with the GPU busy, including when the only
    other work was a single 32-thread SIMD group spinning on a second queue.
  - Idle gaps between dispatches made no further difference.
  - With that spinner running in a separate process, the unchanged engine went
    from 3.04 to 3.89 tokens/s, and the new kernels with prefetch from 3.37 to
    4.96.
- **Isolated row kernels,** with random codes on real shapes.
  - Every admitted shape matched raw FP32 sums bit for bit, and deliberately
    wrong lane widths were caught.
  - Speed-ups: resident projections 2.6–3.1× (lm_head 7.86 → 2.51ms), expert
    gate/up and down about 2×, `plain_mv` about 3× for N ≤ 48, and `norm_wide`
    8× (48 → 5.9µs).
- **Predictor accuracy,** over 96 decode tokens.
  - The `next-hyper` guesses matched 77% of the next layer's experts, against
    63% for `next`.
  - Precision by rank ran from 99% for the first guess to 44% for the tenth.
  - Demand misses fell from 263 to 86 per token at 900 slots.
- **SSD.** An uncached 2.76MB expert read takes about 0.9ms whether split or
  not, and five concurrent reads saturate at 4.9GB/s. Sixteen readers measured
  the same as eight.
- **Per-layer LFU.** A replay of captured routes predicted about 10% fewer misses
  than CLOCK without prefetch. With prefetch it raised demand misses from 88 to
  113 per token and lost both pairs. It was removed.

## Files

`summary.json` condenses each raw report:

- build fingerprint, report hash and command;
- resolved configuration and memory plan;
- per request: rates, GPU time, expert hits, misses and prefetch counters, the
  output-token hash, and memory for each phase;
- the exactness hashes.

The raw reports stay local, as the evidence policy in
[`docs/README.md`](../../README.md) describes.
