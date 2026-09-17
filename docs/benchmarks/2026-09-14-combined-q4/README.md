# Native packed-Q4 decode and expert-cache comparison

Native build `51877f96f015386dd642fd4e63fbc39ff2e75c1156f5ae1bcba764f546963594`.
Mixed artifact `b2c422f3c643e36f04227a64d61796b44a4b1029`, unchanged prepared
Q4 records `c4bb4db3220a0de2218128738b2a6ac650a3d65086690c8dc2e8c23d097160da`.
Actual 32GiB M1 Pro; normal requests retain the 12GiB maximum and core-cache
residency. No weight precision, expert selection, or default option changed.

## Implementation and correctness

`--q4-decode packed-r2` requires experimental candidate policy and is recorded
in the native kernel identity. Selection requires decode, one token, affine
Q4/group64 and the fixed expert dimensions. Gathered gate inputs, other shapes,
Q8 weights and ingestion retain their existing dispatch. The contiguous down
activation can use the candidate independently of how its gate input originated.
The arithmetic body is copied verbatim from the previous probe, with only its
entry-point names changed. `integration-origin.json` binds that source; the
preceding source is retained under `pre-change-sources`.

`state-01` completed in 50.98 seconds:

- All 72 native tests, 49,201 assertions, zero skipped, under Metal API/shader validation.
- 64 real original-Q4 expert/input cases across layers 0, 16, 32 and 47, with
  exact gate activations and BF16/FP32 down outputs through native dispatch.
  Guard regions around nonzero output destinations remained intact.
- Both all-48-layer state arms passed 12 checks each: identical logits/routes/
  convolution, recurrence, attention and index state; retained continuation
  versus fresh replay; forced eviction at 32 slots; cancellation/failure drains.
- The state harness now explicitly labels its continuation as decode. Dispatch
  counts prove both continuation steps used all 480 gate/up and down expert
  operations per step. Fresh replay uses the ingestion fallback.
- 265 Python tests passed, including factorial gates and ignored-option rejection.

This establishes parity with the native reference on the checked inputs and
short full-model sequence, not a new independent model oracle or coding quality
qualification. State validation uses short context; the timing screen below
uses the normal 8192-token allocation.

## Request screen: rejected

All eight conversations completed in **519.14 seconds**, in the declared
`A B C D`, then `D C B A` order. State and screen evidence audits passed.
There was no observed compression, decompression, swap growth, budget change,
or output/reuse mismatch. All decode forwards made the expected 480 gate/up
and 480 down expert calls. No prior samples were pooled.

| Arm | Slots | Q4 decode | Initial tokens/s | Append tokens/s | Conversation seconds |
|---|---:|---|---:|---:|---:|
| A | 1072 | reference | 3.43 | 3.20 | 60.83 |
| B | 1072 | packed-r2 | 3.30 | 2.98 | 61.72 |
| C | 1460 | reference | 3.69 | 3.46 | 61.04 |
| D | 1460 | packed-r2 | 3.57 | 3.18 | 62.05 |

Values are medians of two fresh runs per arm. Each conversation uses 72 prompt
tokens and 33 outputs, followed by a retained 128-token append and 33 outputs.
The first output comes from ingestion; each phase times 32 decode forwards.
This is short-context screening, not 2K/4K or sustained-use acceptance.

Paired median effects:

- **Q4 alone (B/A):** initial decode 3.72% slower, append decode 7.27% slower;
  whole conversation 1.47% slower. The expert-read bytes are identical.
- **Q4 at the larger cache (D/C):** initial decode 3.39% slower, append decode
  8.60% slower; whole conversation 1.66% slower. Both run orders lose.
- **Cache alone (C/A):** saves 20.91/23.18ms per decode token, but append TTFT
  rises 7.00%; whole conversation is 0.32% slower. This also fails promotion.
- **Combined (D/A):** saves 11.74ms initially and loses 1.54ms after the append;
  whole conversation is 1.98% slower and append TTFT rises 6.80%.

These are directional two-pair results, without confidence-qualified benefit
or regression claims. The unchanged gates reject the candidate: material
initial/append savings, whole-conversation improvement, secondary latency,
and kernel benefit at both cache sizes all fail. Only clean memory passes.
**Do not run five-pair or long qualification for this unchanged candidate.**
Keep the experimental reference at 1072 slots with `q4_decode=reference`;
the packed path stays opt-in for investigation. Production defaults are unchanged.

The allocations were exactly 10,735,026,176 and 11,809,357,824 planned bytes,
under the identical 12,884,901,888-byte maximum. Maximum sampled physical
footprint was 8.497GiB for the smaller cache and 9.498GiB for the larger cache.
Memory figures describe sampled boundaries and native allocation counters,
not a continuous OS trace.

The isolated 43–47% expert GPU reduction did not transfer to requests.
Median append GPU-command duration also rises with packed Q4: 187.47→206.96ms
per token at 1072 slots and 175.26→189.81ms at 1460. This screen does not isolate
which operations or GPU scheduling effects account for that change. GPU
command durations overlap reads and cannot be added to request wall time.
Any follow-up should explain this runtime difference before trying another
long kernel qualification. The previous individual operator/cache gate
failures remain preserved.

## Reproduction and evidence

After building, run:

```sh
.cache/qwen-reference-venv/bin/python scripts/qwen/combined_q4.py state --output FRESH_STATE
.cache/qwen-reference-venv/bin/python scripts/qwen/combined_q4.py screen --state FRESH_STATE --output FRESH_SCREEN
.cache/qwen-reference-venv/bin/python scripts/qwen/combined_q4.py verify --source FRESH_SCREEN --output AUDIT.json
```

State checking needs the pinned original Q4 records and captured activations;
missing assets fail. Both live modes share the GPU lease, freeze sources and
binaries, retain incomplete evidence, and drain on cancellation. The screen
uses 150 seconds per process and 900 seconds overall; profiling and Metal
validation are disabled for timing. The state gate enables both.

- `state-01/`, `screen-01/`: original sealed reports, configurations and logs.
- `state-audit.json`, `screen-audit.json`: read-only source/result reconstruction.
- `metrics.json`: compact derived timings, application bytes and memory; raw
  reports and their hashes remain authoritative.
- `screen-sources/`: source corresponding to these native binaries and tools.
- `python-tests.log`, `build.log`: supporting verification.

See [the stage plan](../../qwen_combined_decode_stage.md).
