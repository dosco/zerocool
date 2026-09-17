# Complete hyper-connection fusion screen

Declared before fixture capture and GPU timing. This stage implements the
selected reference-diagnostic experiment. No production kernel or default changes.

## Fixed scope

One appended developer Metal kernel fuses Q8 up-projection, sigmoid and hc_mix
for one token. Reference uses the unchanged native Q8 packed two-row kernels:
down K10240,N320,width8; up K320,N10240,width4, group64. Norm and BF16 injection
remain in their original positions. Each reference block has eight dispatches;
the candidate has six. All BF16 boundaries and stream reduction order remain.

Capture four actual decode inputs and full block/injection outputs from layers
0 (GDN) and3 (full attention), attention and MLP hyper-connections respectively.
Use a separately compiled developer copy with a narrow capture hook. The
production source and executable remain unchanged. Record the original source,
instrumented source, actual executable hashes, base fingerprint, and complete
capture identity. The base fingerprint does not identify the instrumented
capture binary by itself. Run a 72-token prompt with two generated outputs,
providing one true decode forward; no append or long-context qualification.
Retain failed capture attempts. The capture has its own 240-second stage budget.

Prepare original mixed-artifact packed tensors losslessly after verifying the
pinned artifact's receipt and selected tensor ranges. Every input, native result
and weight payload is hashed. Check shape/format eligibility of all96 layer
blocks. Four fixtures prove four blocks, not full-model correctness.

## Validation and timing

Separate Metal API/shader validation from uninstrumented timing. Compare fused
and reference full output plus injection byte-for-byte, and reference versus
the actual native captured results. Check read-only payloads and neighboring
buffer guards. Twelve additional synthetic tail cases cover signed zero, BF16
ties, tiny values, cancellation and sigmoid saturation; label these separately
from the real fixtures. They are finite arithmetic coverage, not exhaustive
BF16-domain verification.

Five fresh alternating pairs, 32 complete-block repetitions per sample.
Individual fixture timings are diagnostic. Primary timing is one complete
cycle in fixture order0,1,0,1,0,1,2,3. This represents frequency weights
36,36,12,12 after multiplying cycle savings by12. It includes unchanged norm,
down projection/activation and injection. Pair is the statistical unit.
Report complete-cycle wall, CPU encoding and GPU time. Reject component times
exceeding wall time by more than2% plus1microsecond (cross-clock tolerance). Do not subtract unchanged
work or use an isolated-kernel gain as the acceptance metric.

Bound actual shared Metal buffers including guards/padding to256MiB. Physical
footprint must be positive, no greater than its recorded lifetime peak, and
within the existing12GiB experiment budget. Bound
post-compilation GPU stage time to30seconds per process and60seconds combined.
The runner independently rejects timing exceeding the bounds. Capture is
separate from this small resident operator timing budget. Own the exclusive
GPU experiment lease; seal reports and source identities, preserving failures.

Reject if outputs differ, a required observation is absent, compression or
lifetime compressed peak is nonzero, decompression changes, or swap changes
between any recorded timing-process boundaries. Physical memory is reported
at boundaries and lifetime peaks. These are acceptance observations, not a
claim of continuous system-memory monitoring.

Advance only if primary complete-cycle wall ratio's two-sided paired-log
Student-t95% upper bound is below1 AND median frequency-weighted wall saving
is at least20ms/token. GPU-only gains cannot pass. Keep all five pairs and
outliers; no pooling earlier experiments. A survivor still needs a short native
normal-request screen and fresh state/failure checks before expensive full
qualification. This stage can never promote production or claim request gains.

## Limits

Four repeated resident fixtures differ from48 layers, SSD arrivals and native
allocation lifetimes. The frequency projection is only a screening heuristic.
Eliminated up/sigmoid intermediates total7.5MiB logical payload across96 blocks;
16KiB native allocation rounding could charge9MiB if independently retained.
Neither is a measured physical-footprint saving. The standalone probe keeps
both arms allocated and does not measure engine allocation reduction.
