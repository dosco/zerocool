# Four-token packed-Q8 screen

Test one candidate: 32-bit packed weight loads, four token accumulators and one
output row, on the three already verified layer-0 GDN inputs. Preserve the
reference width-eight lane partition, scalar accumulation order, SIMD reduction,
affine scales/biases and BF16 rounding. Do not combine output-row pairing or
other selector changes. Scope is K/N = 2560/10240, 2560/6144 and 6144/2560, group
64, four rows, aligned contiguous weights. All production sources stay unchanged.

Revalidate the original source capture's full logits/routes/state, build/artifact
identity and all packed tensor hashes against the pinned mixed checkpoint. Reuse
only tensor bytes and numerical identity from that memory-disturbed capture;
never its timing, memory samples or status. Keep the original failed attempt.

Use an independently hashed developer build with native compiler/math flags and
explicit native dispatch-count evidence. Run fresh Metal API/shader validation,
then five alternating pairs of 32 dispatches per shape. Warm up each arm once;
exclude warmup from timing. Compare every output to the scalar reference and
require identical reference hashes between the validation and timing processes.

Native preflight, exclusive GPU lease, nominal thermals, AC power, Low Power Mode
off, zero process compression, stable decompression/swap counters, maximum
256MiB Metal buffers and 2GiB physical footprint are unchanged. Stop on any
disturbance. Bound each process to 60 seconds and the complete stage to 180.

Advance only if summed paired GPU ratio upper 95% bound is below 1 and the median
shape-frequency projection saves at least 10ms per input token (36 calls of each
shape per four-token block). This is a screening heuristic, not an end-to-end
latency prediction. A survivor needs a fresh full-verifier exactness/recovery and
normal timing comparison at 1,460 slots and the existing 12GiB total budget before
promotion. No result here establishes real draft acceptance, five tokens/s or
production speculative decoding. Do not run expensive verifier timing for a
candidate that fails this screen.
