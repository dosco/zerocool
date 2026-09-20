# Complete cache traces for wider verification windows

The developer trace producer and offline replay are implemented. Production
cache capacity, arithmetic and inference defaults are unchanged. No full-model
trace or new throughput result has completed in this stage.

The preceding [streamed-embedding screen](../2026-09-18-streamed-mtp/README.md)
saved 644–647MiB of peak physical memory with essentially flat latency. This
stage measures whether larger cache capacity could reduce required expert reads
before implementing a memory/workspace transition.

## Implementation

`build_horizon_cache_trace.py` builds isolated native source copies with explicit
trace-on/off modes. Both use exact streamed embeddings, 1460 target / 32 draft
slots, fixed draft priming, a 12GiB joint budget and four-row GPU tiles. The target
trace is scoped around target forwards so draft-cache traffic cannot enter it.
Lazy ngram initialization remains intact.

The capture is bounded to a prompt of at most 128 tokens and a known continuation
of at most 64 tokens, 100,000 events and 32MiB. Both arms reserve the same trace
workspace. Trace writes use uncached I/O. The records include selected experts,
token rows, acquisition order, slots, victims, pins/releases and a drained cache
snapshot after every forward.

`horizon_cache_replay.py` derives expected forward coverage from the input and
requested width, including single-token tails. It checks cumulative route hashes,
every native CLOCK decision, snapshots and application-read counts before
simulating 1460/1909/2048 slots in the recorded order. Row-use histograms distinguish
experts serving one token from experts shared across several tokens. Simulations
do not predict changed completion order or latency, and do not admit larger
allocations.

`capture_horizon_cache.py` first runs real embedding/kernel fixtures, then requires
trace-on/off equality for logits, routes, persistent state, admission and initial
caches. Instrumented throughput is excluded from its observations. Its offline
audit rebuilds completed comparisons and curves from sealed raw reports.

## Validation and current boundary

All **565 Python tests**, native tests and chat transport tests pass through
`ctest --preset release`; Ruff passes and the new C++ header follows the repository
formatter. Tests exercise both widths, irregular tails, omitted snapshots,
corrupted or missing events, draft leakage, unready releases and explicit trace
memory admission. The native source-copy producer compiles with repository flags.

[`capture-01`](capture-01/summary.json) stopped at host identity verification.
Recent CMake work had added source/configuration fingerprint inputs without
updating the Python evidence tool. Python now includes those inputs and the
configured compiler, build type, flags and sanitizer. CMake now reads the uppercase
configuration flag variable, so Release optimization flags are actually included.
An independent CMake/Python regression test checks their parity and distinguishes
changed flags, sanitizer settings and sources. This changes build identity, not
inference arithmetic; prior producer receipts remain historical evidence.

The rebuilt [`capture-02`](capture-02/summary.json) passes all ten real embedding
cases and four kernel cases under Metal validation with clean host/memory gauges.
It then stops before full-model loading: **7.952GiB available; 13.5GiB required**.
No four/eight-token model samples or capacity simulations are available yet.

The [saved-trace parser audit](parser-audit-01/summary.json) replays both earlier
1072- and 1460-slot native captures with the extended parser and reproduces their
saved curves. This checks backward compatibility; those old traces do not become
new wider-window evidence.

## Continue

With sufficient available memory, use a fresh output directory:

```sh
.venv/bin/python scripts/qwen/capture_horizon_cache.py run \
  --build .cache/horizon-cache-build-20260919-03 \
  --output docs/benchmarks/2026-09-19-horizon-cache/capture-03
```

The builder must be recreated if any frozen source or build input changes. After
a complete capture, independently rebuild its results:

```sh
.venv/bin/python scripts/qwen/capture_horizon_cache.py audit \
  --source docs/benchmarks/2026-09-19-horizon-cache/capture-03 \
  --output docs/benchmarks/2026-09-19-horizon-cache/audit-03/summary.json
```

Only a meaningful read reduction justifies a subsequent, separately admitted
cache-growth experiment. A 2048-slot allocation still needs verified workspace
release and memory headroom. The old 1536-slot and SLRU decisions remain unchanged.
See the [registered protocol](protocol.md) for the experimental boundary.
