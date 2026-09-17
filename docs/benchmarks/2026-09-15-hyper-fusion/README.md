# Complete hyper-connection fusion: reject this candidate

The exact fusion passed correctness checks but made the complete-block cycle
**1.10% slower**. All five primary paired wall comparisons were slower. Keep
the existing native path; no engine integration or long qualification follows.
The 5 tokens/s product target remains open.

The [declared protocol](protocol.md) required a whole-block wall gain and at
least 20ms/token in a frequency-weighted screening projection. The actual
projection is **−0.316ms/token** (negative means slower). It is operator evidence,
not a measured or predicted normal-request change.

## Results

[Sealed screen](screen-01/summary.json), [raw timing](screen-01/timing.json),
[separate Metal validation](screen-01/validate.json), [offline audit](screen-audit.json),
and [independent source/data audit](independent-review.json).

| Complete cycle, eight blocks | Reference | Fused |
|---|---:|---:|
| Median wall time | 2.416ms | 2.438ms |
| Median GPU time | 2.390ms | 2.418ms |
| Median CPU encoding | 14.84µs | 11.16µs |
| Dispatches | 64 | 48 |

The primary paired geometric wall ratio is **1.01101**, with 95% interval
**1.00457–1.01750**. The individual pair wall savings, multiplied by 12 to
represent 96 blocks, are −0.316, −0.421, −0.485, −0.097 and −0.276ms/token.
This preserves every pair, including the initially slower individual-fixture
reference measurement; individual fixtures are not independent pair samples.
Five pairs in one process may be correlated, so this remains a screening result.

The reference graph includes norm, Q8 down projection, activation, Q8 up
projection, sigmoid, mix, BF16 injection and its activation. The candidate
fuses only up/sigmoid/mix. Reducing dispatches lowered encoding time, while
GPU time increased enough to erase that saving. The experiment does not
identify whether register use, memory access or another GPU effect caused it.

The screen completed in **3.106 seconds**, including source verification,
validation and timing. The timing process used **28.43MiB** actual shared
buffers and peaked at **63.77MiB** physical footprint. Compression and
compressed lifetime peak were zero, decompression stayed zero, and system
swap remained unchanged. Post-compilation timing-process work took 1.314 seconds,
inside the separate 30-second process and 60-second combined bounds.

## Real inputs and exactness

The separate [capture](capture-01/summary.json) completed in 18.227seconds.
It processed the existing 72-token initial prompt with two outputs, capturing
one true decode forward at position 72. The four fixtures are attention/MLP
hyper blocks from GDN layer 0 and attention layer 3. Both full block output and
injection are stored alongside each raw input. No normalized input was
reconstructed or relabeled as an original input.

A developer copy of Model::hyper adds only bounded capture waits/file writes.
Production source, objects, library and executable remain unchanged. The
[producer receipt](producer-build.json) records actual capture binary/source
hashes separately from base native fingerprint
`51877f96f015386dd642fd4e63fbc39ff2e75c1156f5ae1bcba764f546963594`.
The capture's instrumentation makes its timings unsuitable for performance claims.

Prepared tensors retain original bytes from mixed artifact
`b2c422f3c643e36f04227a64d61796b44a4b1029`. Selected weight payloads are
rehashed against verified source ranges; header checks establish the same
Q8 group 64/BF16 layout in all 96 target blocks without reading every weight.
Fixture frequency weights are 36, 36, 12, 12; the primary cycle is 0, 1, 0, 1, 0, 1, 2, 3.

All four standalone reference outputs and injections match their native
captures byte-for-byte. The fused graph matches both, including twelve separate
synthetic tail cases covering signed zero, tiny values, BF16 ties, cancellation
and saturation. Input hashes and surrounding guards remain unchanged. These
checks are not full-model persistent-state or exhaustive BF16-domain proof.

Both probe arms coexist with preallocated buffers. Removing intermediates
would eliminate 7.5MiB logical payload across 96 blocks; native 16KiB rounding
could charge 9MiB if all were independently retained. Neither number is a
measured engine footprint saving. No request speed or physical-memory
improvement is claimed.

## Reproduce and retain

The current probe and capture tools are archived in `screen-sources`, with a
hash manifest. The source-copy capture build is external to production build
outputs. The four raw fixtures and their prepared bytes are preserved under
`capture-01`; raw reports remain the source of truth.

```sh
.cache/qwen-reference-venv/bin/python scripts/qwen/capture_hyper_inputs.py build \
  --output FRESH_BUILD_DIRECTORY
.cache/qwen-reference-venv/bin/python scripts/qwen/capture_hyper_inputs.py run \
  --output FRESH_CAPTURE_DIRECTORY --producer FRESH_BUILD_DIRECTORY/producer.json
clang++ -std=c++23 -O2 -Wall -Wextra -Werror -fobjc-arc \
  -framework Foundation -framework Metal scripts/qwen/probe_hyper_fused.mm \
  -o /tmp/freellm-hyper-fused
.cache/qwen-reference-venv/bin/python scripts/qwen/screen_hyper_fusion.py \
  --binary /tmp/freellm-hyper-fused --fixtures FRESH_CAPTURE_DIRECTORY/fixtures \
  --output FRESH_SCREEN_DIRECTORY
.cache/qwen-reference-venv/bin/python scripts/qwen/screen_hyper_fusion.py \
  --output FRESH_SCREEN_DIRECTORY --audit
```

[364 Python tests pass](python-tests.log); the ten focused hyper tests also
passed after final source preparation. The standalone probe compiles with
warnings treated as errors. No expensive full-model qualification is warranted.
Retain this composition as a negative experiment, alongside the distinct Q8
load-ahead, narrow BF16 and packed-Q4 request results. Return to the dependency
budget before selecting a different candidate; do not repeat this fusion merely
because its dispatch count is smaller.
