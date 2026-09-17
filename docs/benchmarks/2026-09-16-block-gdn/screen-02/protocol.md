# Four-token GDN Q8 row-pair screen

The completed clean block profile identifies three recurrent-layer Q8 shapes
before expert routing: 2560x10240, 2560x6144 and 6144x2560. Their instrumented
GPU pass costs sum to 41.17ms per input token. This is a ranking opportunity,
not a prediction that all of that time can be removed.

Capture actual inputs and unchanged weights from layer 0 at offset 72 of the
existing four-token verifier. Run the complete 72-token prime and sixteen input
continuation, retaining all blocks and exact output/state checks. Use the same
CLOCK 1,460 slots, 8K state and 12GiB admission. The existing 512MiB diagnostic
allowance remains inside that budget. Rehash all nine weight/scale/bias tensors
against the pinned mixed checkpoint and verify captured input payloads.

One candidate: existing Q8 affine kernels with two output rows and four token
rows, compared with the existing one-output-row/four-token kernels. Preserve
lane partition, arithmetic and reduction; no new quantization or kernel source.
Only these three Q8 GDN shapes are eligible for a subsequent integration.

The isolated probe uses at most 256MiB of Metal buffers. Validate all outputs
against the scalar reference under Metal API and shader validation first. Run
one explicit warmup per arm, then five alternating timing pairs of 32 identical
dispatches per arm/shape. Keep warmups in evidence and outside paired timing.
Include host power/thermal, cumulative compression and physical memory checks.
Any compression, decompression growth, swap change or host disturbance stops
the stage. Capture is bounded to 120 seconds; each probe to 60; stage to 300.

Advance only when the paired 95% upper bound on the summed GPU-time ratio is
below one and the median shape-frequency projection saves at least 10ms per
input token. Projection multiplies each shape's four-token duration by 36 and
divides by four. It is screening evidence, not whole-request latency. No adaptive
variant sweep or promotion occurs here. A passing candidate still needs exact
full-model recovery checks and fresh uninstrumented verifier timing at the
original absolute 5-token/s early gate before real drafting can advance.

The source profile and this new experiment retain separate raw reports and
seals. Incomplete and disturbed attempts are never pooled or relabeled complete.
