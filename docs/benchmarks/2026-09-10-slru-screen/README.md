# Native SLRU implementation and bounded timing screen

Experimental SLRU is implemented and passes native and real-model correctness
checks. It consistently reduced expert reads, but **did not demonstrate a
repeatable whole-conversation speedup** in the two-pair screen. Keep CLOCK as the
default. No five-pair or longer qualification was launched.

The complete correctness-plus-timing experiment finished in **316.14 seconds**,
within its ten-minute deadline. Both policies used the same native build,
mixed 4/8-bit artifact, reference arithmetic, 12GiB allocation and 1,848 expert
slots during timing. The initial 72-token coding prompt generated 33 outputs;
a 128-token follow-up reused 104 computed tokens, ingested the pending output
plus new input, and generated another 33 outputs. Each request contains 32
single-token decode steps. This is shorter history than the prior 256-step
locality capture and does not qualify the 4K-history append requirement.

## Measured timing

Each row is a separate native process. Each process starts with empty runtime
caches and retains its live session for the follow-up. Order is CLOCK/SLRU,
then SLRU/CLOCK. Timing runs disable route/profile capture and Metal validation.

| Pair / policy | Initial request | Follow-up request | Initial decode | Follow-up decode | Follow-up first token |
|---|---:|---:|---:|---:|---:|
| 0 / CLOCK | 38.44 s | 37.25 s | 1.44 tokens/s | 2.42 tokens/s | 24.02 s |
| 0 / SLRU | 27.73 s | 35.87 s | 2.58 tokens/s | 2.69 tokens/s | 23.98 s |
| 1 / SLRU | 33.01 s | 35.75 s | 1.86 tokens/s | 2.71 tokens/s | 23.95 s |
| 1 / CLOCK | 28.37 s | 37.07 s | 2.48 tokens/s | 2.45 tokens/s | 24.02 s |

Whole-conversation SLRU/CLOCK ratios were **0.8403** and **1.0508**: 15.97%
faster, then 5.08% slower. The median ratio of 0.9456 must not be presented as a
reliable speedup. The predeclared gate required both pairs to favor SLRU, at
least 1% median improvement, and no individual metric median regression beyond
3%. It failed the first condition. Two pairs provide no confidence bounds.

The follow-up request was 3.69% and 3.57% faster, with 10.16% and 9.53% lower
decode time. Its first-token wait barely changed. These are observations from
two runs per policy, not promotion evidence or a solution to the 10-second target.

## Expert reads and memory

Native hit/miss counts were identical between repetitions of each policy:

| Phase | CLOCK misses | SLRU misses |
|---|---:|---:|
| Initial ingestion | 7,688 | 7,688 |
| Initial generation | 6,534 | 5,845 |
| Follow-up ingestion | 8,527 | 7,571 |
| Follow-up generation | 7,515 | 6,217 |
| Total | 30,264 | 27,321 |

SLRU issued **9.72% fewer expert reads overall**. These are observed application
reads, not simulated misses or device traffic. Every selected expert still ran.
The stable read counts contrast with variable generation latency: neither
aggregate read bytes nor a cache simulation explains the complete wait.

All processes retained the same 12GiB allocation plan. Reported compressed
process memory after initial generation ranged from approximately 4.35 to
6.16GiB. Initial requests recorded 2,426,575 / 520,247 decompressions for
pair 0 CLOCK / SLRU, then 1,497,850 / 604,339 for pair 1 SLRU / CLOCK.
That variation is a concrete reason to inspect memory behavior before attributing
the timing difference to cache policy; these snapshots alone do not establish
causation. Recorded physical footprints remained within budget, and system swap
usage did not grow across any measured request. This is not a continuous peak
or sustained-session measurement; system counters include other processes.

## Implementation and correctness

`bench` and `inspect` accept `--cache-policy clock|slru`. CLOCK remains the default
and the policy for `run` and `serve`. SLRU inserts new entries into probation,
promotes hits to protected MRU, and demotes protected LRU when the protected count
exceeds floor(75% of capacity). Both queues remain evictable. If all probation
entries are busy, an eligible protected entry can be evicted; loading or leased
buffers are never overwritten. Shrinkage retains surviving entry order; clear
resets both queues. Embedded links require no separate queue buffers and use the
same entry layout and common CPU/driver reserve for both policies.

- **55 native tests, 45,673 assertions passed**, with Metal API and shader
  validation and no skips. An independent demand replay checks SLRU hits and
  resident sets at capacities 1–8. Tests cover loading joins, pinned buffers,
  protected fallback, resizing, cancellation, read failures and GPU completion.
- **164 Python tests passed**, including request identity, policy forwarding,
  output/reuse and memory validation, missing phases, state proof and the
  alternating-pair decision. The suite emitted two existing SQLite resource
  warnings; there were no test failures.
- The real mixed model ran a nine-token prefix/append/continuation fixture with
  only 32 slots. CLOCK and SLRU matched **every logit, routed expert and persistent
  state hash across all 48 layers**. Continued state matched fresh replay, and
  both policies passed partial-state failure and cancellation checks. This gate
  used Metal validation and preceded timing.
- All four normal conversations generated identical output token IDs and reported
  identical computation reuse. This is native cache-policy parity, not an
  independent model implementation or coding-quality evaluation.

Native build:
`35f708489e0f1bb159bc10af85ca5ca5c42bb4516a1b4a0db87e6ea45174006e`.
Artifact: `b2c422f3c643e36f04227a64d61796b44a4b1029`, using the existing prepared
Q4 expert records. Timing uses panel 512, microchunks 128, four ready experts per
group and eight I/O workers. No weights, routing rules, arithmetic or reduction
order changed.

[Raw summary](raw/summary.json), [derived observations](observations.json),
[correctness control](raw/clock-correctness.json),
[correctness candidate](raw/slru-correctness.json), [native test output](native-tests.log),
[Python test output](python-tests.log), [ledger result](ledger-result.json) and
[verification hashes](verification.json) retain the source evidence. Original and
copied raw directories match the same immutable seal. The final source/artifact
identity, correctness gate, request validation and decision were rechecked offline.

## Next decision

Keep SLRU available for diagnosis, without tuning its protected fraction or starting
long qualification. First use the recorded per-request memory and dependency
measurements to design a small test of the variable initial-generation latency.
Check whether compression or another exposed dependency explains it; do not infer
a cause from decompression totals alone. A future cache timing experiment must
also revisit longer generation and the post-append phase where the earlier trace
showed a possible regression. Full acceptance remains unproven.
