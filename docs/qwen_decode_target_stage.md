# Generation target: 200ms per token

Continuation: [current-build implementation and experiment gates](qwen_stage200.md).
The [submission/residency follow-up](benchmarks/2026-09-14-submission/README.md)
confirms a 4.35% short-request gain, but misses the 20ms/token append gate and
does not reach 5 tokens/s. The [capacity follow-up](benchmarks/2026-09-14-cache-residency/README.md)
finds 1.77% lower conversation latency in two clean pairs, below the 20ms/token
stage gate. Keep 1072 slots as the experimental reference; do not extend this
capacity screen into expensive qualification. The clean resident-decode trace
now supports isolating the GPU work before routing: its largest command class
takes about 64–67ms/token and precedes expert read admission. Profile its
constituent real-weight operators before selecting a kernel change. The planned
previous-token top-two predictor has almost no observed uncached demand here.
The [resident-operator screen](benchmarks/2026-09-14-resident-operators/README.md)
now rejects Q8 load-ahead: exact outputs, but only a 0.47ms/token rough isolated
projection and no clear combined-kernel gain. The subsequent four-step capture
completed at the unchanged allocation. [Packed Q4 expert probes](benchmarks/2026-09-14-q4-packed/README.md)
then cut isolated GPU time by 43–47%, preserving exact outputs, but project only
about 14ms/token. The next [combined Q4/cache stage](qwen_combined_decode_stage.md)
now tests each component and the combination on normal requests. Its
[completed screen](benchmarks/2026-09-14-combined-q4/README.md) fails: packed Q4
is slower at both cache sizes, the combined conversation regresses 1.98%,
and cache alone regresses append TTFT despite faster decode. Keep reference
Q4/1072 slots; no five-pair or long qualification for this candidate. Prior
individual failures remain unchanged. Investigate the isolated-versus-runtime
GPU difference before another kernel candidate.
Small attention-history savings, sliding-window replacement and purgeable-buffer
experiments are deferred. Q3 remains an isolated format/operator experiment.

This is the current near-term priority. Pause the narrow-projection experiment
and defer separate append optimization while identifying a credible path to
5 tokens/s. Preserve exact arithmetic, all selected experts, artifact identity,
memory safety and the existing first-token regression guards.

## Measured gap

The [five-pair selector result](benchmarks/2026-09-12-route-five-pairs/README.md)
provides the unchanged experimental baseline: packed Q8 rows 2, SIMD expert
selection, CLOCK, 1848 expert slots, original execution, eight I/O workers,
and ready groups of four at 12GiB on the M1 Pro. It is not production promotion.

| Candidate phase, median over five runs | Current | Target |
|---|---:|---:|
| Initial decode wall time per token | 278.50ms | 200ms |
| Retained-append decode wall time per token | 296.19ms | 200ms |
| Initial expert application reads per token | 564.45MB | Measure the critical dependency |
| Append expert application reads per token | 649.38MB | Measure the critical dependency |

Median per-token GPU command duration is about 121–129ms in the phase deltas;
the coordinator's expert-pipeline waits are about 111–125ms. These overlap
each other and reads, so adding them cannot explain token wall time. The
required reduction is 79–96ms per token. Multiplying the projection probe's
synthetic saving by 96 calls suggests only about 4.4ms, even if it transfers.
That is not the leading experiment for this milestone.

## First measurement

Run `scripts/qwen/trace_decode_target.py --output FRESH_DIRECTORY`.

It verifies and freezes the confirmed baseline and runs one uninstrumented
72-token initial prompt with 17 outputs, then the identical request with
existing command-group profiling, expert dependency records, and decode
boundary timestamps. Keep the first 16 committed decode steps: approximately
75,600 dispatch records fit the existing 100,000-record profile bound. A
32-step capture would exceed that bound. Truncation cannot pass.

The total deadline is 300 seconds; each inference process is capped at 150.
Each arm uses bounded metadata admission, the shared GPU lease, the same
1848 slots and panel, and temperature zero. Validation and per-kernel timing
passes stay off. Interrupted or resource-blocked attempts stay unfinished;
do not reduce the budget or repeatedly retry unchanged host conditions.

`decode_timeline.py` joins the existing Mach-uptime clocks and verifies all
48 layers and 480 selected-expert records for every captured token. The
expert's submitting timestamp links it to its actual command group. It
partitions time into mutually exclusive observed states, in precedence order:

1. GPU active.
2. GPU idle with a submitted command awaiting execution.
3. GPU idle with expert data ready but not submitted.
4. GPU idle with outstanding required expert reads and no ready expert.
5. GPU idle with a completed GPU command awaiting its completion callback.
6. Remaining GPU idle time.

These are overlap observations, not proof of a blocking cause. Ready work
can still be constrained by capacity; unrelated CPU work can occur within a
bucket. The report separately measures the gap from the last expert's GPU
completion to submission of its reduction group for the first 47 layers.
Measuring all the way to the next router would include intervening attention
GPU work on some layers and overstate the submission gap. That gap overlaps the
buckets and must never be added to them. Trace writing and collection add
overhead; report the traced/normal ratio without treating one pair as an
overhead correction or a candidate speedup.

## Choose and screen the next change

Choose one candidate from the largest measured exposed interval. Write down
how many milliseconds it could plausibly save before implementing it. Favor
an opportunity on the order of 20ms/token or more; this is a prioritization
heuristic, not permission to claim predicted savings as measurements.

- Large gaps between expert completion and the next submission motivate
  testing removal of the end-of-layer CPU barrier while retaining leases
  until the final GPU user finishes. Command ordering and bounded ownership
  must remain intact. The first screen below failed to demonstrate a repeatable gain.
- Predominantly read-dependent idle time motivates an exact read scheduling
  experiment. Preserve previous negative cache results; additional memory
  and fewer read bytes have not by themselves improved complete requests.
- Predominantly active GPU time motivates profiling the largest expert or
  resident operation, with a short operator test before runtime integration.

Keep existing output/state checks, then use two alternating short normal
request pairs to screen the survivor. Rank by measured decode wall-time
reduction while retaining the request and first-token regression guards.
Do not run expensive full-model qualification for every weak candidate.
If no single experiment closes the gap, record the remaining millisecond
budget and choose the next supported dependency.

## Acceptance remains the real workload

Ultimately measure at least 5 tokens/s over 256 generated tokens with 2K and
4K prompts, with repeated comparable runs and confidence reporting. Then
complete the existing 7K report and sustained coding/session checks. A
16-token trace, cached replay, synthetic operator result, or short-history
conversation cannot qualify that target. Existing mixed-Q8 first-token
uncertainty remains open. No precision change or production promotion is
part of this measurement stage.

## Current experiment

The [completed capture](benchmarks/2026-09-13-decode-target/README.md) shows a
71.09ms/token submission gap after final expert GPU completion, with substantial
trace overhead and separate GPU-speed variation. The next measured candidate
is `--expert-tail overlap`: encode the reduction and next attention work before
waiting for the last submitted expert groups, retain their leases, and drain
before the next expert admission. It is implemented behind an explicit option;
`wait` remains the default. See the [bounded screen and acceptance rules](benchmarks/2026-09-13-expert-tail/README.md).

The initial tail-overlap screen failed: one initial-generation win, one loss,
and no append-generation win. Preserve it as negative evidence; do not run
five-pair or long-context qualification on this result. Next measure temporary
buffer allocation/retirement cost before implementing bounded reuse. Both arms
still allocate 3238 temporary/state buffers per token. The failed screen also
recorded a 7.56GiB compression peak in one append despite no net swap growth;
keep these observations explicit in later comparisons.

The [buffer-cost capture](benchmarks/2026-09-13-buffer-costs/README.md) found
54–59ms/token in measured allocation/retirement CPU intervals, but its compressed
model pages keep the full-model opportunity result inconclusive. A separate
bounded probe of the existing scratch pool saved a median 51.78ms per synthetic
iteration over five alternating pairs, with no observed process compression.
This supports testing single-token temporary reuse within the admitted scratch
allowance. It does not predict the inference saving. Keep 1848 expert slots,
unchanged arithmetic, and include cold allocation and pool release in normal
initial/append comparisons. Release retained temporaries before multi-token
prefill; cancellation must drain all GPU and I/O users before dropping ownership.

The [bounded decode reuse screen](benchmarks/2026-09-13-decode-scratch/README.md)
now passes both alternating timing comparisons: initial decode 4.03/4.43 tokens/s,
append decode 4.06/4.09, with unchanged output tokens and 3238→137 physical
allocations per token. Retained workspace is about 74MiB within the admitted
scratch allowance. Conversation ratios were 0.90281/0.93681. Compression and
decompression were observed, so the declared memory gate keeps this result
inconclusive and the option disabled by default. Preserve all measurements;
do not run the later expensive qualification until a fresh clean-memory screen
passes. Candidate medians leave roughly 37–45ms/token to the short-workload
200ms goal; 2K/4K and sustained-use acceptance still require their own evidence.

The [confirmation preparation](benchmarks/2026-09-13-scratch-confirmation/README.md)
adds the five-pair runner and corrects the fresh-replay workspace check for a
one-token remainder. A four-layer real-weight diagnostic passed all 20 control
and candidate state/recovery checks at 4GiB. It computes no model logits and
cannot replace the full-model gate. The initial 12GiB admission attempt was
blocked at 7.66GiB reclaimable. After memory was freed, a fresh screen completed:
reuse won every decode comparison (3.78–4.35 tokens/s), but compression in the
first control keeps the entire batch inconclusive. The last three processes
showed clean decode memory, supporting one fresh unchanged screen from that
settled state. That second screen again favored reuse (3.94–4.39 tokens/s), but
compression recurred. Both complete batches remain inconclusive and separate;
no five-pair confirmation or full-state qualification ran for them.

Stop repeating unchanged screens. Capture memory within ingestion and at
decode/drain/destruction boundaries in one bounded diagnostic with the same
12GiB budget and 1848 slots. Existing snapshots locate initial compression
between the start and end of ingestion, despite 18.64GiB reported reclaimable
at its start in the second screen. Tracked Metal allocation peaked at 10.433GiB;
there is no continuous process-footprint or buffer-specific compression trace.
Resolve that evidence gap before changing admission, residency or ownership.
Prior core-residency confirmation did not establish a latency benefit, so it
is not an assumed remedy. Defaults, arithmetic and weights remain unchanged.

The [ingestion and cleanup diagnostic](benchmarks/2026-09-13-ingestion-memory/README.md)
now brackets compression onset at layer 9 during expert-cache growth in one
control, with process peaks at 10.559–10.561GiB inside the 12GiB plan. A separate
versioned follow-up showed an immediate 1.51GiB post-destruction footprint fall
to 31.7MiB within 250ms; the earlier compressed control lacks delayed samples.
Do not infer a leak or allocation overrun from its immediate footprint alone.
All eight diagnostic requests preserve prior output IDs and dispatch counts.
These instrumented captures do not advance the failed timing screens.

For the next bounded intervention, consider existing core residency while
holding scratch reuse on, capacity fixed and GPU timing reference enabled.
This tests the combined configuration; the older failed core-residency result
remains negative evidence. Promotion still requires complete-request improvement
with memory observations, followed by the established real-workload gates.
