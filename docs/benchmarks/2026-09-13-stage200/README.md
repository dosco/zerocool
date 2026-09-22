# Bounded memory, dependency and Q3 experiments

This stage adds observed/shrinking memory-pressure policies, current-build
decode-only dependency capture and an isolated native Q3 gate/up probe. The
production model and default arithmetic remain unchanged. The 5 tokens/s target
is not qualified by these measurements.

## Capacity decision

All normal comparisons use mixed 4/8-bit, original prepared Q4 expert records,
12GiB, scratch reuse, packed Q8 rows 2, SIMD routing, CLOCK, eight readers,
ready groups of four, panel 512 and microchunks 128. The workload has 72 initial
tokens, 33 outputs, a 128-token retained-history append and 33 more outputs.
Each comparison contains two fresh alternating pairs, capped at 600 seconds
total and 150 seconds per process. No samples are pooled between comparisons.

| Comparison | Clean control processes | Clean smaller-cache processes | Median smaller/control conversation ratio | Decision |
|---|---:|---:|---:|---|
| [1848 vs 1460 slots](capacity-01/summary.json) | 0/2 | 0/2 | 1.07418 | Unresolved compression |
| [1460 vs 1072 slots](capacity-fallback-01/summary.json) | 2/2 | 2/2 | 0.97672 | Keep 1460 provisionally |

The prescribed single fallback finished with both sizes clean. The policy
therefore retains the larger of those two sizes for subsequent diagnostics.
This does not erase the earlier disturbed 1460-slot observations or prove
stability under arbitrary host conditions. Clean means zero compression at
recorded phase boundaries and unchanged decompression counters; it is not
continuous physical-residency monitoring.

Every output token matched the previous control. The final fallback's median
initial/append generation rates were 3.279/2.912 tokens/s at 1460 slots and
3.434/3.163 at 1072. These are short screens, with no confidence-based speed
qualification or production capacity change.

## Fresh short-request dependencies

[Normal and traced requests](profile-short-02/summary.json) use 17 outputs, covering 16 decode forwards in each
phase. Both processes stayed clean, and outputs matched. Normal initial/append
decode took 343.10/338.01ms per token. Traced decode took 323.52/349.34ms: ratios
0.94296/1.03352. A faster traced arm is evidence of run variation, not negative
instrumentation cost. No timing adjustment is applied.

All 48 layers and ten selected experts per layer were [joined](profile-short-02/timelines.json) for each captured
token. Exclusive mean forward intervals were:

| Observed interval | Initial ms/token | Append ms/token |
|---|---:|---:|
| GPU command execution timestamps | 129.47 | 158.36 |
| GPU idle with submitted work | 76.44 | 75.68 |
| GPU idle with ready experts | 15.45 | 15.82 |
| GPU idle with pending reads | 50.04 | 46.27 |
| GPU idle with outstanding completion callbacks | 12.18 | 11.82 |
| Other GPU idle intervals | 39.20 | 40.63 |

The rows sum to 322.78/348.59ms forward time. They identify overlap, not why
the coordinator or device waited. GPU command intervals can include stalls or
preemption. Submitted-work delay is not automatically removable engine work.
Layer-boundary submission gaps of 36.06/36.79ms overlap this table and must not
be added to it. The previously failed expert-tail and grouping experiments
remain negative evidence; these intervals alone do not justify repeating them.

## Preserved trace-limit failure

The [first capture](profile-short-01/summary.json) required 101,600 dispatch records and exceeded the historical
100,000-entry bound. Analysis rejected it as incomplete. Decode-only capture
now has a fixed 120,000-entry allowance; historical capture modes retain their
old limit. The collector also checks the normal request's dispatch count before
starting the traced companion. The first attempt remains failed, never promoted
or substituted for complete coverage.

Capacity and the incomplete trace used native build `b6d6d6b7`. The corrected
capture uses `d95cddfa`. This build change only increases diagnostic capacity
and exposes its bound; normal inference arithmetic is unchanged. Earlier source
versions are retained in `initial-sources` with their original bytes.

## Bounded 2K attempt

The [normal 2K conversation](profile-2k-01/summary.json) reached its 300-second process cap during the
initial request. Cancellation drained the process; the request phase ended
after 297.57 seconds with no completed request row. The traced companion and
append were not run. This is a timed-out diagnostic, not a throughput sample.
The macro progress events do not expose the first generated token, so exact
TTFT is missing. Do not infer it from the interrupted request duration.

This attempt cannot support a 2K-versus-short dependency comparison. It also
cannot qualify the product targets. The trace's large observed intervals alone
do not establish a concrete recoverable 20ms/token improvement; no new exact
kernel or scheduling change is selected from this evidence. The previous
failed expert-tail and grouped-expert screens remain closed.

## Q3 fixture protocol correction

The [initial capture](q3-01/summary.json) used a truncated prompt ending inside a user message. The
model stopped before nine generated tokens, and the harness rejected missing
activation coverage. That failed attempt is preserved. The revised fixture
protocol replays 72 source tokens and eight recorded continuation tokens
through all 48 layers. It records its teacher-forced mode and uses disjoint
source windows for tuning and held-out evaluation. Production EOS behavior
and sampling remain unchanged. This capture proves operator input provenance,
not coding quality or freely generated request performance.

## Q3 operator result: inconclusive latency

The corrected [tuning capture and screen](q3-02/summary.json) finished in 28.65s.
Eight records and real layer inputs cover two selected experts in each of
layers 0, 16, 32 and 47. All 32 one/two/four/eight-row cases passed separate
Metal API/shader validation and exact gate/up/down comparison against Q3 codes
expanded into the existing Q4 arithmetic. This tests the new decoder's arithmetic;
it does not claim that quantizing Q4 weights preserves the Q4 model output.

| Measurement | Result |
|---|---:|
| Padded Q4 expert record | 2,768,896 bytes |
| Padded Q3 gate/up + Q4 down record | 2,359,296 bytes |
| Record saving | 14.79% |
| One-token candidate/control geometric latency ratio | 1.05798 |
| Paired 95% interval | 0.92736–1.20700 |
| Maximum accepted upper bound | 1.03 |
| Probe process peak | 21.44MiB |
| Probe tracked GPU peak | 6.125MiB |

The five alternating pairs time complete gate/up/down chains, including host
dispatch and completion, with equal preallocated outputs. The pair is the
statistical unit, aggregating the eight experts; fixtures within a pair are not
treated as independent repetitions. The interval uses paired log ratios and
Student-t with four degrees of freedom. Temporal device effects may violate
the independence assumption. No measured arm showed compression.

The latency evidence does not pass non-inferiority, but also does not establish
a regression: the interval includes ratios below one. The original collector
used `operator_regression` for any failed gate. The corrected classifier and
[independent reconstruction](verification.json) report
`operator_latency_inconclusive`; original timings and the original summary are
preserved. The advancement decision is unchanged: held-out capture, full source
conversion and model-quality evaluation were not run. No Q3 runtime artifact
or coding-quality claim is produced.

## Pressure result: live release not exercised

The [observe/shrink comparison](pressure-01/summary.json) completed all four
processes at 1460 slots. Each stayed clean at recorded boundaries, every output
matched the control, and each received zero normal/warning/critical notifications.
The result is `not_exercised`. Synthetic policy and leased-buffer tests pass,
but no empirical claim about live warning response or stability benefit follows.
Release accounting measures buffer ownership; physical reclamation can lag.

## Verification and reuse

- [68 native tests / 48,994 assertions](native-tests.log) passed with Metal API
  and shader validation; none skipped. The final native build passes, and the
  separate real-weight Q3 validation covers all 32 operator cases.
- [240 Python tests](python-tests.log) passed, including missing evidence,
  incompatible pressure transitions, trace coverage and inconclusive confidence
  classification.
- The [audit](verification.json) verifies all eight raw seals, reconstructs both
  native source versions and checks matching current or archived tooling bytes.
  It revalidates 28 completed request rows, the full corrected timeline and Q3
  fixture hashes/operator results. This is not an independent full-model
  logit/state or coding-quality qualification.
- Four source-bound experiment ledger entries record
  [capacity](../../experiments/24cf40565de3f4475c04b0f2ab592e99f909c2e0239a6e47bfe1729def722d6b.json),
  [pressure](../../experiments/30ca4feea9d027a890bffab8e5c206377cf0660d789891e682549d32ae739626.json),
  [dependencies](../../experiments/ec0081a910737faecd2884c5bd5b559573433a8f228c0231616f616b177d8e1d.json)
  and [Q3](../../experiments/436dea78bfc7101b9be42609b9c4e49bf199f007b87aebb76766d95666dc826c.json)
  as inconclusive results. They are indexed for agent queries.
  Historical native binaries are not archived; their captured hashes remain in
  each identity file. Raw model/prepared artifacts remain pinned and unchanged.

Reconstruct the report without running inference:

```sh
.cache/qwen-reference-venv/bin/python scripts/qwen/verify_stage200.py \
  docs/benchmarks/2026-09-13-stage200 --output /tmp/stage200-verification.json
.cache/qwen-reference-venv/bin/python scripts/qwen/query_evidence.py history \
  --search 'Q3 gate/up' --limit 20
```

The next decision needs a bounded measurement that distinguishes engine work
from the approximately 76ms submitted-command interval before selecting another
exact optimization. Resolve the interrupted request's ingestion/decode progress
before allocating another long 2K run. Q3 can only progress through a separately
declared operator timing experiment; these uncertain measurements must not be
pooled into a later passing result. The 5 tokens/s and complete coding-session
targets remain open.
