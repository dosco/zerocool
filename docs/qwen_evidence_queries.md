# Offline evidence queries

`scripts/qwen/query_evidence.py` gives a coding agent compact JSON answers from
existing benchmark reports. It uses Python's standard library and local SQLite;
it never runs inference, contacts a service, changes runtime instrumentation or
alters qualification reports. Native production inference is unchanged.

Raw JSON/JSONL and the authored ledger files remain authoritative. The database
at `.cache/evidence/index.sqlite` is disposable. Import reports and rebuild it:

```sh
PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/query_evidence.py \
  import docs/benchmarks .cache/benchmarks/decode-compute docs/experiments --rebuild
```

Pass `--db PATH` before the command to use a different index. Import checks file
stability and reports malformed or oversized files. The default maximum is 64MiB
per file and 100,000 JSONL lines; omissions are explicit. It does not parse model
weights, tensor captures, logs or prose. Supply an active qualification directory
only if a snapshot is wanted: unfinished data never becomes a completed result.
No writes occur in the imported directories.

## Commands

Every answer includes source paths/hashes and limitations. History has sources
per result. Use an indexed path or an unambiguous content-hash prefix as `SOURCE`.
Identical copies have one ID with multiple paths. Queries recheck selected files;
changed or missing evidence fails unless another exact copy remains available.
Unavailable aliases are listed. Adding observations never assigns new pair IDs.

| Command | Answer |
|---|---|
| `history --search TEXT --limit 20 --offset 0` | Search report identities, statuses, errors, decisions and ledger hypotheses; includes unfinished and negative results |
| `show SOURCE` | Compact identity, original status and phase outcomes |
| `compare SOURCE --control NAME --candidate NAME --change KEY [--change KEY] [--case NAME]` | Revalidated paired timings and correctness scope; reject unrecorded differences |
| `memory SOURCE --limit 10 --offset 0` | Recorded allocation plans, physical footprint, live GPU groups, retirements, scratch and cache snapshots |
| `timeline SOURCE [--phase PHASE] [--layer N] [--token N] --limit 10` | Captured expert read/encode/GPU/release timestamps, positions and trace limits |
| `cache SOURCE [--budget-mib N ...] [--phase PHASE] [--per-layer]` | Simulated expert reads for CLOCK, probation/protected SLRU and a future-aware bound at equal bytes; no predicted latency |
| `next SOURCE` | Evidence gap and smallest useful measurement, with no invented request benefit |
| `record ENTRY.json [--ledger DIRECTORY]` | Append a durable, source-bound experiment interpretation |

For example, query the historical Q8 experiment:

```sh
PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/query_evidence.py \
  compare docs/benchmarks/2026-09-08-decode-compute/normal-screen.json \
  --control decode-control --candidate packed-q8-r2 --change q8_decode_rows

PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/query_evidence.py \
  compare docs/benchmarks/2026-09-08-decode-compute/cached-paired-2048.json \
  --control control --candidate candidate --change q8_decode_rows

PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/query_evidence.py \
  timeline .cache/benchmarks/decode-compute/baseline-profile.json --layer 0 --token 0 --limit 1
```

The CLI is the thin agent interface: one subprocess, JSON stdout, no server or
model required. Exit 2 means an error or an incompatible comparison. Use history
and show to locate the right evidence instead of loading large reports into the
agent context. `--limit` is bounded to 1–100; paginated results give `next_offset`.

## Comparison contract

Normal comparisons accept one `exact_kernel_normal_requests` summary and locate
its original raw reports by SHA256, including copies imported from other paths.
They reconstruct metrics using the existing native-report validator and check:

- Complete experiment and complete, unique, alternating recorded pairs.
- Distinct original raw reports for each observation, with full SHA256 references;
  relabeling copies as new pairs cannot increase the sample count. Workload labels
  must match native reports, and declared workload coverage must be present.
- Exactly the declared configuration changes; same build, artifact, prepared
  payload, actual machine, admitted memory budget and instrumentation mode.
- Original prompt tokens, history primes, output lengths, sampling, exact
  generated tokens and retained computation. Summary metrics must match raw data.
- Positive finite timings. Five or more pairs get the existing deterministic
  paired bootstrap interval; shorter screens retain null confidence bounds.

Historical normal reports embed their complete workloads, so missing separately
hashed workload sidecars need not cause input tokens to be guessed. Legacy missing
counter-profile/environment details are disclosed. Across-build matching and
pooling separate experiments are deliberately unsupported in this first version.
The query never promotes a configuration, even if a confidence interval suggests
a benefit. Full state correctness and coding quality require their own evidence.

Failed or explicitly unfinished status overrides an inconsistent completion claim.
Cached comparisons use the existing strict replay validator, verify prepared
artifact and machine consistency, and require the user to name the recorded axis.
They remain cached comparisons. Their timings cannot be substituted for normal
request latency. Missing measurements remain missing; failed or partial reports
are searchable but cannot establish a winner.

## Memory and timeline limits

Memory plans describe capacity; they are not physical allocation measurements.
Nested categories overlap, so the query does not sum them. Physical footprint,
live command groups and retirements are returned only where recorded. No model
destruction or final-drain boundary is inferred from a zero allocation counter.

Timeline joins are limited to each recorded expert lifecycle. A source pointer
locates its original entry; it is not a new runtime event ID. Read timestamps,
encoding, GPU start/end and release remain in their original clock coordinates.
The query does not infer why the coordinator was blocked or add overlapping sums.
It exposes the existing 48-pass/8,192-read-record capture limits when present.
`truncated=false` on the command trace does not establish whole-request dependency
coverage. Each returned pass includes at most eight expert records; the full
captured list remains in the raw report. JSONL line capture is explicitly bounded.

Shared runtime IDs, classified blocking reasons and physical memory after
destruction remain targeted follow-ups when a decision requires them.

## Cache-size what-ifs

```sh
PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/query_evidence.py \
  cache docs/benchmarks/2026-09-07-foundation/recorded-routes.jsonl \
  --budget-mib 0 --budget-mib 1024 --budget-mib 2048 --budget-mib 4096
```

This is an offline simulation for choosing experiments, not a reconstruction of
native asynchronous execution. It accepts native route JSONL, or profile objects
whose `expert_dependencies` contain complete routes and artifact/build identity.
Timing records without routes return `insufficient_evidence`; read records and
aggregate hit counts cannot reconstruct missing router selections. Legacy JSONL files
lack terminal request markers, so their completion stays unknown. Explicitly
unfinished profile status remains attached to the simulation.

New `qwen_route_trace_v1` captures add explicit request, session, forward and cache
reset boundaries. Their reader includes routes only after `forward_commit`, checks
the committed input history against the request's output tokens, and preserves
unfinished status. Unfiltered curves keep simulated prefill warmth into decode;
`--phase decode` starts cold after excluded prefill. Legacy traces retain the
conservative phase-change resets and unknown completion described below.
Each policy also reports `request_phases`: demands, hits, misses and application
bytes for each recorded request/phase. These partition the same simulation while
retaining cache contents across uninterrupted prefill, generation and append.
Use these counters to compare a warm follow-up. A filtered phase restarts cold
and answers a different question. MIN minimizes the segment total; its phase
contributions are not independent per-phase minimums.

The model is deliberately fixed and inspectable:

- Only the pinned Q4 and mixed-4/8 revisions are supported. Both use fixed-size
  Q4 expert payloads: **2,764,800 read bytes** and **2,768,896 aligned slot bytes**
  per expert, matching `include/qwen/storage.hpp`. No lower-bit estimate is used.
- Preserve the file's layer-pass order. A pass groups its token rows by selected
  expert; repeated selections within that pass are grouped work, not cache hits.
  Each policy sees the same ascending expert-ID order within a pass. Native
  hit-first admission, GPU pins, loading joins and prefetch are not simulated.
- Include only complete windows of layers 0–47 with matching token range and
  recorded phase/session. Do not transpose a five-token prefill into five decode
  steps. Exclude incomplete windows and reset at gaps, rollback, repetition or
  phase/session changes. Contiguous positions imply continuation only as a stated
  assumption: missing session/reset markers prevent proving every boundary.
- Each segment starts cold. CLOCK uses second chance; SLRU gives the protected
  queue a target of floor(75% of slots), demoting its oldest entry on overflow.
  New entries enter probation, whose oldest entry is evictable. Belady/MIN has
  full future knowledge and mandatory admission. Its minimum miss count applies
  only to this equal-size, fixed-order, serial model, not arbitrary native
  schedules or future variable-size Q3 records.

Default expert-only budgets are 0, 1, 2, 4, 6 and 8GiB. Specify 1–12 distinct
integer MiB budgets between 0 and 22GiB. Aligned slots are rounded down; unused
budget is reported. These are **not admitted engine budgets**: resident matrices,
metadata, ngrams, state, scratch and driver reserve consume additional memory.
Zero slots is a bypass comparison, not a valid native cache configuration.

Results include source hashes, token/layer coverage, compulsory misses, within-pass
grouping, per-policy hits, byte hit fractions, application read bytes and the
CLOCK-to-oracle read gap. `--per-layer` adds all 48 layer counters. Capacity curves
are not forced to be monotonic. The query refuses oversized sweeps rather than
silently shortening them: 100,000 lines, 250,000 grouped demands, at most nine
million policy accesses, and the index's 64MiB source limit. Coverage detail lists
are limited to 20 entries and explicitly marked; counts and simulation still
include every admitted segment.

There is no SSD bandwidth conversion, speedup prediction, policy promotion or
quality claim. [Initial cache-query evidence](benchmarks/2026-09-10-cache-simulation/README.md)
shows why the existing prefill and recovery fixtures cannot select a production
policy. Next capture a deadline-limited normal 32-token continuation with explicit
boundaries; expand to 256 tokens and retained-history append only if the initial
evidence warrants it. This keeps the existing screen-first qualification order.

## Bounded normal route capture

```sh
PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/capture_routes.py \
  --output .cache/benchmarks/routes/NEW-RUN --time-limit 180
```

This helper makes one normal coding request using the pinned mixed artifact,
existing prepared Q4 expert records, reference kernels, a 512-token panel,
128-token microchunks and fixed 12GiB admission on the real M1 Pro. It renders a
short coding prompt with the native tokenizer and uses greedy sampling. The
request allows 33 output tokens to observe 32 single-token forward steps: the
prompt produces the first output. EOS can end the request early; no token is
forced to meet the target. This is an instrumented locality capture, not a latency
comparison, coding-quality evaluation or selector qualification.

After the short capture supports it, extend the same prompt and add a follow-up:

```sh
PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/capture_routes.py \
  --output .cache/benchmarks/routes/NEW-EXTENDED-RUN \
  --decode-steps 256 --append-128 --time-limit 300
```

The first request allows 257 output tokens for 256 decode steps. The follow-up
adds exactly 128 input tokens, including native-tokenized chat framing, and allows
33 outputs for 32 further decode steps. The preparation recipe records the full
user text and its bounded token prefix; only the body is truncated. It reuses the
actual first response, including its length-limited ending. A fixed closing chat
delimiter is retained even if that response ends at EOS. This is a controlled
locality workload, not a fully rendered client conversation or a quality test.

The last sampled output has not yet passed through the model. Thus the follow-up
reuses the prior input plus all but that final output, then computes the pending
output and 128 new input tokens. The validator checks the native pending-token
count, exact committed history, unchanged session and memory plan, request identity,
and per-request stopping rules. EOS remains a valid early stop but does not pass
the requested decode-step target. This short history does not qualify the 4K-history
append acceptance target.

The deadline includes prompt preparation, admission, model loading and inference.
Cancellation drains native work; process cleanup may extend elapsed time beyond
the deadline. The helper never retries inference or lowers the fixed budget.
It uses the existing GPU workload lock, artifact receipts, source/build identity
guard and storage admission. Native trace capture also refuses a reduced budget
if availability changes between inspection and model construction.

`summary.json` reports the current outer phase, final status, committed decode
count and last trace progress. `routes.jsonl` flushes each event and survives normal
cancellation. It records a header, request inputs/results, each forward's input
tokens/session/phase/position, layer routes, commits/aborts and termination. It is
bounded to 20,000 events, 64KiB per record and 16MiB total. Missing terminal markers
and partial final records remain unfinished, never inferred successful.

The native option is `bench --workload-file FILE --repetitions 1 --route-trace
NEW-FILE`. It uses an exclusive new output file and marks benchmark timings as
instrumented. It adds no GPU waits or changes to inference arithmetic. Only normal
full-model workloads are accepted. Detailed dependency timing traces and the
cached selector's progress reporting keep their existing formats.

On completion or interruption, the helper saves a conservative cache curve when
the raw evidence can be indexed, plus immutable evidence hashes. A partial curve
does not establish the 32-step target. Inspect progress during a live run with
the read-only `route_trace.progress(path)` helper; the SQLite importer continues
to require stable source snapshots.

## Bounded native cache-policy screen

```sh
PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/screen_cache.py \
  --output .cache/benchmarks/cache-policy/NEW-RUN --time-limit 600
```

This developer helper first checks CLOCK and experimental SLRU on a nine-token
real mixed-model fixture with 32 slots, unchanged reference arithmetic, exact
logit/route/state hashes, fresh replay, cancellation and failure checks. Only a
passing comparison proceeds to normal inference timing with the full admitted
12GiB allocation. Correctness uses Metal validation; timing disables it and all
route/profile capture.

The timing workload uses the earlier coding prompt with 33 outputs, followed by
128 new input tokens and 33 more outputs. Two fresh-process pairs run in order
CLOCK/SLRU, SLRU/CLOCK. The helper enforces equal memory plans, actual history
reuse and output tokens, and reports initial and follow-up latency independently.
Its phase field identifies each correctness run and timed policy/pair. The total
deadline includes setup and correctness; individual correctness and timing runs
are also bounded. Interrupted or blocked attempts keep their raw reports.

Both conversation ratios must favor SLRU, with at least a 1% median improvement
and no phase metric median regression beyond 3%, to justify five paired runs.
Two pairs provide no confidence bound or promotion. The 4K-history append,
2K/4K/7K latency and sustained-session requirements remain separate. `history`
and `show` can index this report; `screen_cache.decide` owns this specialized
two-pair decision instead of treating it as the longer exact-kernel protocol.

## Decode-startup diagnostics

```sh
PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/diagnose_decode_startup.py \
  --output .cache/benchmarks/decode-startup/NEW-RUN --time-limit 360
```

This helper uses the existing short coding prompt and 33 outputs, with CLOCK,
reference arithmetic and the fixed 12GiB plan. Two alternating fresh-process
pairs change only `--residency off|core`. It checks the same artifact, allocation,
tokens and actual registered core buffers. It never raises OS limits, forces
memory pressure, changes weights or starts longer qualification. Its historical
analysis preserves the earlier cache screen's first-four versus remaining-token
split and labels that split as selected after observing those earlier runs.

Native `bench --workload-file FILE --decode-diagnostics` records the first 32
completed decode forwards of each request. Each sample identifies its token,
absolute position and start/end time, with before/after process-memory, Metal
timing, expert-read and dependency counters. It adds no GPU wait or command
submission. Normal timings with this option are marked instrumented and cannot
qualify performance. Output reports captured, total and omitted decode steps;
prefill and sampling have no per-step coverage. Interrupted requests do not
produce a completed per-step report and cannot pass the helper's checks.

Counter snapshots have a measured `sample_ns` cost; this excludes outer sample
assembly and report serialization. Request wall time includes observation work.
Missing process counters, counter resets and wraps remain unknown. CPU waiting,
GPU execution and concurrent read durations overlap and must not be summed into
a critical-path explanation. Residency registration is a request to the driver,
not a guarantee that all enrolled bytes remain physically resident.

The [first completed diagnostic](benchmarks/2026-09-10-decode-startup/README.md)
captured all 128 expected decode intervals and reproduced the startup difference
in both pair orders. Its raw reports, derived windows and ledger entry remain
diagnostic evidence; the next timing screen must disable this option.

## Normal residency screen

```sh
PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/screen_residency.py \
  --output .cache/benchmarks/residency-screen/NEW-RUN --time-limit 480
```

This screen holds CLOCK, reference arithmetic, the mixed artifact and the 12GiB
allocation fixed. It runs off/core then core/off in four fresh processes, with
diagnostics and Metal validation disabled. Each process generates 33 outputs
from the existing 72-token coding prompt, appends 128 tokens to its live history,
then generates another 33 outputs. The follow-up must reuse exactly 104 computed
tokens and ingest the pending output along with the new input. It checks every
output against the prior CLOCK control and requires identical memory plans.

The predeclared gate requires both complete-conversation ratios to favor core,
at least 1% median improvement, and no per-request metric median regression over
3%. A passing short screen supports five paired repetitions; it does not itself
qualify latency or change defaults. The helper records core buffer enrollment
and retirement at request/phase boundaries, rejecting expert enrollment, extra
GPU profiling, unfinished requests and changed artifacts. Its native build must
match the completed startup diagnostic. The normal screen adds no per-token
observations; it retains the engine's existing request and phase measurements.

The total deadline defaults to 480 seconds, with a 150-second subprocess limit
per conversation. Admission uses at most three metadata checks, separated by
2 and 5 seconds and bounded by the remaining deadline. It retains rejected
checks and never retries inference or reduces the budget. Exhausted admission
or a deadline remains incomplete. Reports
are sealed on exit; no longer test is started automatically.

The [first completed normal screen](benchmarks/2026-09-10-residency-screen/README.md)
passed the two-pair gate with 4.94% and 6.06% lower conversation time. Its ledger
entry records a promising screen, without confidence or production qualification.

For five new confirmation pairs, use the same helper with `--pairs 5` and a
900-second deadline:

```sh
PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/screen_residency.py \
  --pairs 5 --output .cache/benchmarks/residency-paired/NEW-RUN --time-limit 900
```

This mode first revalidates the completed short screen's four original reports,
tokens, memory, residency and decision. Its ten fresh conversations alternate
off/core, core/off, off/core, core/off, off/core. Earlier pairs are not pooled.
The complete-conversation core/off ratio is the primary measurement. The gate
requires a geometric mean of at most 0.99 and a two-sided 95% interval upper
bound below 1.00. Each initial/follow-up request, first-token and decode metric
must have an interval upper bound at most 1.03.

Intervals use the five paired log-ratios and Student-t with four degrees of
freedom. They assume independent, approximately normal pair effects; changing
host conditions can violate those assumptions. Each interval has marginal
coverage, not simultaneous coverage across all metrics. The pair is the unit
of analysis, not the token. Five observations provide limited precision. A
positive result supports longer validation, not product acceptance or automatic
promotion. The helper does not stop early for apparent success. Missing pairs
from interruption, deadline or resource admission keep the run incomplete.

The [first completed five-pair confirmation](benchmarks/2026-09-10-residency-paired/README.md)
did not establish improvement: primary ratio 1.0228, interval 0.7707–1.3573.
Its earlier failed admission and all measured slow runs are preserved. The
short-screen result has not been promoted or pooled with the confirmation.

## Durable experiment ledger

Ledger JSON belongs in `docs/experiments/`; it is independent of SQLite. The
`record` input requires these fields:

```json
{
  "hypothesis": "The controlled change reduces exposed decode work.",
  "expected_effect": "Lower normal request latency at the same memory budget.",
  "controlled_change": "One named configuration difference.",
  "correctness": "What was checked and what remains unverified.",
  "outcome": "Measured result or concrete reason execution stopped.",
  "decision": "blocked",
  "smallest_experiment": "The next bounded test that resolves the uncertainty.",
  "limitations": ["No performance conclusion from an unfinished attempt."],
  "evidence": ["An indexed original report path or content hash"]
}
```

Decisions are `proposed`, `blocked`, `rejected`, `inconclusive`, `promising` or
`adopted`. These are authored interpretations, not automatic qualification.
Benefit or adoption requires at least one explicitly completed, nonfailed source.
Rejection can also cite an explicitly completed failing correctness check.
Unfinished sources plus unknown metadata cannot satisfy either rule. Original
statuses/hashes stay attached. Entries are new files;
corrections should be new entries that identify the earlier entry. Nothing rewrites
the original run. Retrospective entries must say so; expected effects must not be
presented as preregistered after seeing the outcome.

Two initial retrospective entries preserve the blocked selector screen and the
historical Q8 cached-versus-normal distinction. The existing qualification
`recommend()` function is untouched, preserving frozen-run tooling identity.
The new `next` query explicitly asks for normal dependency evidence before ranking
cached hotspots; it does not pretend existing wait sums reveal critical-path
savings. [Original verification and example answers](benchmarks/2026-09-09-evidence-queries/README.md)
bind the first implementation to its reports. The [follow-up audit](benchmarks/2026-09-10-evidence-audit/README.md)
records the current fixes and 116 passing tests. Incremental reimport refreshes
metadata projections; `show` supports JSONL metadata. Parsing rejects duplicate
keys and numeric overflow, and the read itself is bounded against growing files.
