# Generation target: 200ms per token

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
  must remain intact. This candidate is not implemented or justified yet.
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
