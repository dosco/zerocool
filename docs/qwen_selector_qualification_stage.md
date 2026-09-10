# Bounded GPU selector qualification

This stage decides whether CPU/full → GPU/full sparse selection improves normal
requests on the actual 32GiB M1 Pro. It adds no inference optimization, changes
no weights and leaves production defaults unchanged. The earlier three-arm
[experiment](qwen_sparse_attention_stage.md) remains available with
`--track three-arm`.

Current implementation checks and qualification progress are recorded in
[the qualification report](benchmarks/2026-09-09-selector-qualification/README.md).
The latest [capped attempt](benchmarks/2026-09-10-selector-triage/README.md) passed
12GiB memory admission but exhausted its 600-second deadline before reporting
timings. The short request helper is ready and unexecuted. Cached phase progress
is now implemented as described below; older attempts have no such trace.

## Fixed experiment

The first two entries of
[configs.json](benchmarks/2026-09-09-sparse-attention/configs.json) must differ
only in `sparse_selection`. Both retain full score tiles, packed Q8 rows 2,
panel 512, chunk 128, ready group 2, eight readers, core/cache residency,
grouped decode, double prefill and phase memory reclamation. The engine budget
is exactly 12GiB with an 8192-token context. Sampling is greedy with seed zero;
generated tokens must agree across arms and repetitions.

Before model execution, `identity.json` freezes native source identity,
executables, qualification scripts, workload/configuration files, checkpoint
receipts, prepared manifest and verified payload fingerprints. Admission checks
the actual machine, build, artifact and unchanged budget. Source, executable or
verified asset changes invalidate the experiment and its resume.

The runner requires 7GiB free initially, preserves a 5GiB disk reserve and stops
if new evidence reaches 2GiB. It checks these limits every two seconds while
children run. It never lowers the engine budget, deletes files, changes OS
limits or substitutes diagnostic streaming. Missing assets, failed admission
and interrupted work leave the stage unfinished. The advisory lock excludes
other instances of this runner; run no other GPU workloads alongside it.

## Fast iteration is the entry point

Do not run the all-phase qualifier for every candidate. Eliminate weak candidates
before paying for long-context state and recovery runs:

1. Check operators and saved real-model attention inputs. Reuse completed checks
   only when their original dependencies and seals still verify. This build has
   already passed those checks; repeating them adds no evidence without a change.
2. Run five alternating CPU/full versus GPU/full **cached 4K pairs**, with a
   **600-second total deadline including setup**. The deadline stops new work;
   bounded child cleanup may extend elapsed time. Disable both Metal validators
   during timing. Require exact outputs/state, zero application reads, a median
   candidate/control latency ratio at most 0.99 and a paired 95% upper ratio below
   1 to proceed. Negative and inconclusive results stop this attempt. A timeout
   or failed memory admission is unfinished, not a performance rejection.
3. For a promising cached result, add a small normal-request screen: one CPU/GPU
   pair of a 128-token append to retained 4K history, eight outputs, at most
   15 minutes including both history primes. Check output equality and actual
   computation reuse. This is exploratory; it makes no confidence or promotion
   claim. `triage_requests.py` implements this screen after independently
   verifying a promising cached result. Retain the 4K history so sparse selection
   is exercised. Recommend full validation only if request latency improves by
   at least 1% and neither first-token nor decode latency regresses by over 3%.
   One pair is an exploratory filter, never a confidence or promotion claim.
4. Only survivors enter full boundary/append/7K state, cancellation and recovery
   qualification, then the established normal-request performance gates below.
   Keep the 20-minute coding workflow and quality acceptance separate.

The implemented entry point runs only step 2, after validating unchanged-build
step-1 evidence. It never launches a normal screen or full qualification:

```sh
PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/triage_selector.py \
  --prerequisites .cache/benchmarks/selector-qualification/2026-09-09-run-03 \
  --output .cache/benchmarks/selector-qualification/triage-01 \
  --time-limit 600
```

Use a fresh output directory. Its summary distinguishes `promising`,
`not_promising`, `inconclusive`, `resource_blocked`, `time_budget_exhausted`,
`interrupted` and `failed`. `complete` means cached triage completed; it never
means normal latency, full validation or production was qualified. Five samples
are fixed in advance; ambiguity does not trigger more samples.

The triage helper preserves original provenance and verifies every original
source, binary, tooling and asset dependency. Additional tooling files can exist,
but changed original dependencies invalidate reuse. Cross-build saved-input reuse
is future harness work: separate fixture/reference identity from candidate
execution identity and verify original goldens without relabeling them. The
current replay binary still requires its own build identity.

For a cached result that passes the gate, run the short normal-request screen:

```sh
PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/triage_requests.py \
  --cached-triage .cache/benchmarks/selector-qualification/triage-01 \
  --output .cache/benchmarks/selector-qualification/short-01 \
  --time-limit 900
```

The helper revalidates the original cached phase seal and measurements, freezes
its current tooling and prerequisite hashes, and holds the same GPU-workload
lease. Each arm starts a fresh process, primes 4,096 tokens without generating,
then appends 128 tokens to the live state and generates eight. Native reports
must confirm the actual 4,096-token reuse. Both history primes and admissions
are inside the single 900-second deadline; pending child work drains on exit.
Original normal-report, admission and workload hashes bind each observation.
Metal API/shader validation is disabled and recorded for timing.

`normal/summary.json` uses the existing normal-request schema with
`mode: short_screen`, one pair and eight outputs, so the offline query tool can
revalidate it without treating it as the longer performance gate. The outer
summary recommends whether to advance but never launches full validation.
Negative cached results cannot invoke this screen. A timeout or resource block
leaves the attempt unfinished and cannot reject the candidate on performance.

## Cached phase progress

The selector runner passes `--cached-progress FILE` to native cached replay and
retains `cached-4096/progress.jsonl` beside the timing report. For a direct cached
diagnostic, add this flag with a new file path. It is restricted to
`bench --cached-token-replay`; existing progress files are never overwritten.

The JSONL stream flushes on every event. Each record includes native build and
artifact identity, a sequence number, monotonic and elapsed times, phase elapsed
time and phase-specific details. Phases cover model loading, pipeline setup,
reference history priming, state snapshot, reference continuation, expert and
ngram preload, candidate setup, each warmup/paired replay arm, and report writing.
History updates identify the completed prefix and active token chunk; preload
updates count completed experts. Replay arms identify their repetition, variant,
restore/forward/verification step and completed pairs.

Read a running or stopped trace without loading the model:

```sh
PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/cached_progress.py \
  .cache/benchmarks/selector-qualification/triage-01/cached-run/cached-4096/progress.jsonl
```

The reader reports cumulative completed phase time and the last unfinished
phase, including its completed work. It ignores a partial final line, rejects
reordered or inconsistent records, and leaves missing progress missing. The
outer triage summary includes this diagnostic snapshot even after a deadline or
failure. No terminal record means unfinished; a quiet interval after the last
flush is not measured. Cancellation/failure elapsed time can include cleanup.

Progress writes occur outside the measured forward interval and add no GPU
waits. They can still affect total diagnostic duration and setup/cache timing;
phase durations are instrumented measurements. Partial pairs never qualify
performance or correctness. Reports remain the source of timing decisions.
Traces are bounded to 10,000 records of at most 4KiB each, with no tensor payloads.

This native instrumentation changes build identity. Old correctness evidence
retains its original build and cannot satisfy the unchanged-build prerequisite
for the rebuilt binary. An interrupted progress check is not full qualification.

## Full qualification for surviving candidates

The existing `benchmark_sparse_attention.py --track selector --phase all` retains
its original order below. It has **not** been reordered and is not the fast
iteration entry point. Start it explicitly only after the inexpensive screens
justify the work.

1. Run native tests with Metal API/shader validation, Python tests and the
   explicit real-model cached cancellation/reuse check. Cancellation must
   actually occur during reference setup (1–47 completed layer records) and
   warmup control (97–143 records). Preserve both trace hashes and validate their
   sequence/build/artifact. Drain GPU work, restore configuration and reuse the
   same model successfully. The intended watcher target alone cannot pass.
2. Capture fresh 4K, 7K and retained-128-token-append attention inputs at layers
   3 and 47. Replay the eight cases through the three-arm correctness harness.
   Verify positions, shapes, original tensor bytes and hashes. Older captures
   never receive new build identities.
3. Compare original-reference and candidate all-layer logits, routes and
   persistent state at the sparse boundary, retained append and 7K. Independently
   check the requested candidate so running the reference twice cannot qualify.
4. Run five alternating CPU/full versus GPU/full cached pairs at 4K first,
   then 2K and 7K. Require exact outputs/state, 480 ready expert hits, zero
   application reads and positive finite timings. These isolate computation;
   they do not qualify normal request speed.
5. Screen normal requests with two alternating pairs and 64 outputs for 2K,
   4K, retained 128-token append after 4K, and 7K. Advance only if 4K or append
   has both request ratios below 1 and median ratio at most 0.99, while all
   twelve screened metric medians are at most 1.03. This screen makes no
   confidence claim.
6. Only after a promising screen, run exactly five alternating pairs with 256
   outputs at 2K, 4K and append. Require at least 3% median request improvement
   on 4K or append, its paired 95% upper ratio below 1, and all nine first-token,
   decode and request upper ratios at most 1.03. Negative or inconclusive valid
   results finish the comparison; do not add samples to chase significance.
7. Profile CPU/full unless the paired latency gate passes, then profile GPU/full.
   Use three instrumented cached 4K repetitions for GPU attribution and
   eight-output 2K/append requests for dependency, memory and I/O evidence.
   Normal profiles preserve command groups and complete aggregate counts;
   detailed events are bounded and may truncate. Their output tokens must agree
   with the uninstrumented screen. Recommend one measured GPU opportunity with
   a conditional savings range and correctness work; retain request dependency
   measurements to assess whether SSD waits hide it. Do not implement another
   optimization in this stage.

GPU pass times, CPU encoding and waits overlap; their sums are not a wall-time
breakdown. Instrumented timings never qualify latency. The generated README
and `summary.json` retain phase outcomes, raw measurements, hashes, paired
confidence bounds and the next investigation. `complete` means the experiment
finished, including when improvement was not demonstrated. Production promotion,
absolute latency targets, coding quality and the 20-minute sustained workflow
remain separate gates.

## Full-qualification start and resume

For a selected survivor, build once and keep that build and the qualification sources fixed. Use a new
output directory for every invocation:

```sh
cmake --build build/qwen -j 4
PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/benchmark_sparse_attention.py \
  --track selector --phase all \
  --output .cache/benchmarks/selector-qualification/run-01
```

A 4K cached phase can also run independently, but this low-level command has no
600-second overall triage deadline. Prefer `triage_selector.py` for screening:

```sh
PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/benchmark_sparse_attention.py \
  --track selector --phase cached --contexts 4096 \
  --output .cache/benchmarks/selector-qualification/cached-01
```

Continue an interrupted stage into a fresh directory:

```sh
PYTHONPATH=scripts/qwen .cache/qwen-reference-venv/bin/python scripts/qwen/benchmark_sparse_attention.py \
  --track selector --phase all \
  --resume-from .cache/benchmarks/selector-qualification/run-01 \
  --output .cache/benchmarks/selector-qualification/run-02
```

Completed phases are copied with sealed file inventories and revalidated.
Normal comparisons reconstruct observations from hashed native reports,
admission records and exact workload files. Only whole alternating pairs across
all required workloads are reused. A partial pair is rerun in full; its surviving
arm cannot be matched with a new measurement. A hard interruption can use
`progress.json` when no final summary exists. Earlier evidence stays untouched.
Individual `paired` or `profile` phases require the completed prior decision
through `--resume-from` and never silently run a new screen.

Every required correctness, cached, normal and profiling phase must have valid
evidence to finish the stage. A resource-blocked report is a checkpoint, never a
performance result.
