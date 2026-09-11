# Isolate packed Q8 in steady generation

Use the existing packed Q8 two-row kernel as the sole computation change on the
current native build. Earlier results bundled several other optimized settings;
they select this candidate but are not pooled with this experiment. The kernel
preserves the mixed artifact's weight bytes and arithmetic. It does not perform
additional quantization.

```sh
python3 scripts/qwen/screen_q8_steady.py --output .cache/benchmarks/q8-steady-NEW
```

Both arms retain the original schedule: CLOCK with 1848 expert slots, a 12GiB
ceiling, residency off, serial prefill, fixed phase memory, reference expert
execution, panel 512, chunk 128, four ready experts, and eight I/O workers. The
context limit is 8192. The candidate changes `kernel_policy` from `reference`
to `candidate` to permit `q8_decode_rows=2`; all other kernel options stay at
their original defaults. No production configuration is changed.

The experiment has three bounded stages:

1. Capture fresh single-token decode inputs for the three Q8 projections at
   layer 0 (GDN) and layer 31 (attention). Use a 72-token prompt and two outputs.
   Replay each projection for five alternating operator pairs. These timings
   compare packed Q8 against the previous two-row kernel; both are also checked
   bitwise against the original reference. Each case must have a paired median
   bootstrap 95% upper ratio below 1 before model comparisons proceed. Total
   stage deadline: 240 seconds. No variant search over rows 4 or 8.
2. Compare original and candidate logits, routes, and persistent state across
   all 48 layers with forced eviction at 32 expert slots. Check fresh replay,
   continuation, failure, and cancellation with Metal validation. Two processes,
   each limited to 180 seconds. This is optimization parity, not an independent
   model oracle or long-context qualification.
3. Run two fresh alternating pairs of complete conversations: 72 prompt tokens
   plus 33 outputs, then a 128-token append plus 33 outputs. Retain 104 computed
   tokens and ingest the pending output with the append. Each conversation has
   a 150-second limit, within a 600-second stage deadline. Request timing uses
   the original reference as control and excludes profiling, validation,
   boundary probes, and operator capture.

The request screen requires both candidate/control conversation ratios below 1,
their median at most 0.99, and the median ratio for each request, first-token,
and generation metric at most 1.03. Two pairs have no confidence bounds. A
survivor proceeds to a separately declared five-pair confirmation later; it
does not become a production default. Long contexts and sustained coding still
require separate qualification.

The first five-pair operator pilot stopped after 36 seconds. All outputs were
exact and five large projections passed; the smallest projection's CPU wall
ratio had an upper bound above 1 despite lower GPU time in every pair. Preserve
that stop. One separately recorded follow-up uses `--operator-repetitions 20`
on fresh captures to reduce sensitivity to dispatch-time jitter, with the same
per-case gate and stage deadlines. It does not pool the pilot samples, change
the candidate, or retry until success. These operator intervals are screening
statistics; independent request confirmation remains necessary.

The runner freezes native and harness identity, hashes fresh captured payloads,
verifies artifact receipts, owns the GPU workload lock, and saves partial and
negative evidence. Memory admission may retry metadata checks within its
existing bounded policy; inference is not retried. It never reduces the
declared cache, panel, or budget to rescue a comparison. Cleanup can extend past
a measurement deadline.

```sh
python3 scripts/qwen/query_evidence.py import .cache/benchmarks/q8-steady-NEW
python3 scripts/qwen/query_evidence.py compare \
  .cache/benchmarks/q8-steady-NEW/summary.json \
  --control control --candidate candidate \
  --change kernel_policy --change q8_decode_rows
```

Queries reopen original correctness, operator, and request JSON by hash and
reconstruct the decision. Raw payloads remain in the local capture directory;
their recorded hashes bind the native replay. The disposable index never turns
an unfinished stage or duplicated report into another measurement.

## Five-pair confirmation

```sh
python3 scripts/qwen/confirm_q8_steady.py --output .cache/benchmarks/q8-confirm-NEW
```

This imports and revalidates the completed 20-operator-pair short screen, with
its current-build exact state checks. It runs ten fresh conversations in five
alternating pairs, with identical artifact, allocation, workload and kernel
configuration. Prior request timings are excluded. No early success stopping or
inference retry is allowed. The full stage has a 900-second deadline and each
conversation a 150-second deadline; cleanup can extend past measurement limits.

The predeclared gate requires a conversation geometric mean ratio at most 0.99,
its paired log-ratio Student-t two-sided 95% upper bound below 1, and each
request/first-token/generation secondary upper bound at most 1.03. These small
sample intervals assume independent, approximately normal pair log-ratios;
host drift can violate that assumption. A passing result supports the next
longer-context experiment, without claiming the 5 tok/s product target or
promoting defaults. `compare` accepts `q8_steady_confirmation_v1` and validates
the prerequisite and all ten original runs; it rejects pooled or duplicate runs.
