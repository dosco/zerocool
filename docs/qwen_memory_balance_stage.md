# Bounded memory-balance stage

This stage diagnoses varying GPU execution time and screens one cache-capacity
change. It does not change production defaults or establish product acceptance.
Native inference remains C++23/Metal with the pinned mixed 4/8-bit artifact and
unchanged prepared Q4 experts/ngrams.

```sh
python3 scripts/qwen/screen_capacity.py --output .cache/benchmarks/memory-balance-NEW
```

The output must be a new directory. The runner freezes build/artifact/workload
identity, owns the existing exclusive workload lock, and retains admission
failures, partial reports and negative results. It checks exact real-model
probe-on/off logits, routes, persistent state, fresh replay, cancellation and
failure first, with 32 expert slots and two 180-second process limits.

## GPU reference

`bench --workload-file FILE --gpu-reference resident-q8-v1` enables a boundary
probe; `off` is the default. The probe requires the mixed artifact, resident
trunk, serial prefill and fixed phase memory, without other detailed profiling.
It executes the reference `q8_mm` operation on the existing layer-zero
`linear_attn.in_proj_z` matrix (K=2560, N=6144, group 64), using deterministic
BF16-representable input. The first sample contains one dispatch; three later
samples contain eight dispatches each. Every sample and checksum is retained.

Probes run before the initial request and after the entire retained-append
conversation, never between those requests. Two temporary buffers occupy 48KiB
aligned within the existing transient scratch reservation, with disjoint
lifetimes from normal forward temporaries. GPU completion precedes buffer reuse
or release. The diagnostic checks unchanged cache counters/plans and restored
live allocation, and uses existing resident weights without extra model reads.

Raw CPU/GPU timestamps, valid derived queue delay, warm per-dispatch GPU median,
process memory, and OS-reported thermal/low-power/power-source information are
reported separately from request timers and phase deltas. Invalid or missing
GPU timing is not a zero duration. OS-reported nominal thermal or false low-power
state may also mean unsupported/unknown; no GPU frequency or power measurement
is inferred. Boundary samples describe endpoints, not the entire conversation.

Probes warm hardware and touch a resident matrix. Enabled reports have explicit
`gpu_reference_mode` and `gpu_references` fields and cannot pass ordinary timing
validation. Their timings are never normalized by probe ratios. Historical
reports remain readable under the legacy protocol; new capacity comparisons
require explicit probe mode. No historical timings are pooled with new runs.

`--bench-progress NEW_FILE` adds flushed model-load, initial-request, append,
pre/post-probe, and completion events outside request timers. The bounded runner
shows elapsed time and remaining stage time at least every ten seconds while
waiting. Cancellation retains partial results; outstanding users drain before
release. Deadline cleanup may continue beyond the measurement deadline.

## Fixed experiment

Every arm uses a 12GiB ceiling on the actual 32GiB M1 Pro, context 8192, CLOCK,
residency off, reference arithmetic, panel 512, chunk 128, four ready experts and
eight I/O workers. The unchanged workload consists of 72 input tokens + 33
outputs, then 128 new input tokens + 33 outputs, retaining 104 computed tokens.
The follow-up ingests 129 tokens including the previous pending output.

| Phase | Work | Total measurement deadline |
|---|---|---:|
| Diagnostic | Three fresh conversations, 1,848 slots, probes on |450s|
| Screen | Two alternating 1,848/1,460-slot pairs, probes on |600s|
| Confirmation | Only a survivor: five fresh alternating pairs, probes off |900s|

Each conversation process is limited to 150s within the stage deadline. Metadata
admission retains the existing bounded retries; inference is never retried.
Budgets, panels and cache caps are never silently reduced. A negative or
inconclusive screen stops the experiment without trying additional capacities.

Reducing 1,848 slots to 1,460 leaves 1,074,331,648 additional bytes unallocated.
Comparisons explicitly allow only expert slot count, expert bytes and total
planned bytes to differ. Every fixed allocation category and the ceiling must
match, and each arm's plan must remain fixed through both requests. Existing
policy/residency comparisons retain strict equal-allocation checks.

The screen requires both conversation ratios below 1, median at most 0.99,
and every request/TTFT/decode metric's median at most 1.03. Confirmation requires
conversation geometric mean at most 0.99, its paired log-ratio Student-t 95% upper
bound below 1, and every secondary upper bound at most 1.03. Five pairs provide
limited evidence; temporal dependence and outliers remain limitations.

## Evidence and decisions

```sh
python3 scripts/qwen/query_evidence.py import .cache/benchmarks/memory-balance-NEW
python3 scripts/qwen/query_evidence.py compare \
  .cache/benchmarks/memory-balance-NEW/screen/summary.json \
  --control control --candidate candidate --change expert_slots
```

The query reopens original reports by hash, checks complete alternating pairs,
rejects reused reports and instrumentation differences, and reconstructs the
same decision as the runner. The index is disposable; raw files and seals remain
authoritative. A successful confirmation is only a candidate for later long
context, sustained memory and coding/tool-recovery qualification. Neither this
stage nor its offline query promotes runtime defaults.
