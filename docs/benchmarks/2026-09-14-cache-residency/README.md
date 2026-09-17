# Cache capacity under core-plus-expert residency

Another 1GiB of expert storage helped both short timing pairs under residency,
reducing median paired conversation latency by **1.77%**. Initial/append
generation medians changed from **4.11/3.69 to 4.37/3.87 tokens/s**. Every
process was clean at recorded boundaries. The 14.78/12.65ms per-token savings
miss the 20ms stage gate, so this screen does not advance to expensive
confirmation. Keep 1072 slots as the existing experimental reference.

This is a two-pair directional result, not a confidence-qualified speedup.
The native build, artifact bytes and inference arithmetic are unchanged.

## Rechecked evidence

The [residency audit](prior-residency-audit.json) reconstructs the five fresh
pairs, exact-state proof and 4.35% complete-conversation gain at 1072 slots.
The [capacity audit](prior-capacity-audit.json) reconstructs the earlier clean
1460-versus-1072 comparison. In that comparison the smaller cache had 2.33%
lower median paired conversation latency. Its policy retained 1460 provisionally
because both sizes were clean, not because the larger cache won on speed.

Those native counters already show the tradeoff: the larger cache reduced
initial/append generation misses by 8.43%/8.21%, but the request comparison
favored the smaller allocation. The earlier
[12/18GiB experiment](../2026-09-11-memory-budget/README.md) also remains negative
evidence: 42–45% fewer expert reads did not overcome compression and slower
complete conversations. The new comparison tests a distinct residency setting;
it does not invalidate either earlier result.

## Saved routing replay

The [offline analysis](demand/summary.json) replays complete committed forwards
at exactly 1072 and 1460 slots, retaining the simulated cache across prefill,
generation and append. The additional 388 aligned slots cost 1,074,331,648 bytes
(1.00055GiB). All other allocations need separate native admission.

| Saved workload phase | CLOCK misses, 1072 | CLOCK misses, 1460 | Fewer application reads |
|---|---:|---:|---:|
| Short capture, 32 decode steps | 7,823 | 7,170 | 8.35% |
| Extended capture, 256 decode steps | 64,704 | 57,151 | 11.67% |
| Extended capture, retained append ingestion | 8,923 | 8,884 | 0.44% |
| Extended capture, 32 post-append decode steps | 9,293 | 8,466 | 8.90% |

Initial ingestion misses remain 7,688 at both sizes. The short and extended
captures share a prompt and are not independent workloads. These historical
builds use fixed ascending expert demand order in the simulation; native
hit-first admission, outstanding GPU leases, read overlap, OS caching and
compression are omitted. Predicted latency and tokens/s remain null.

## Native screen protocol

Run two fresh alternating pairs: 1072 then 1460, followed by 1460 then 1072.
Both arms use core-plus-expert residency, immediate expert execution, scratch
reuse, packed Q8 rows 2, SIMD routes, CLOCK, eight readers, ready groups of four,
panel 512 and microchunks 128. The maximum stays 12GiB; planned allocations are
approximately 10 and 11GiB. The prompt has 72 tokens, followed by 33 outputs,
a retained 128-token append and 33 more outputs. No profiling, diagnostic
replay, GPU reference probes or Metal validation runs inside timing.

The total cap is 600 seconds, with 150 seconds per process. Fixed admission,
artifact/source seals, exact output tokens, actual history reuse and arithmetic
dispatch counts are checked. The existing same-build all-layer state proof at
32 slots is revalidated; it is not a new state check at the larger capacity.

Stop the comparison after any process shows compression, changed decompression
counters, missing swap observations, or increased swap between recorded
boundaries. Observations are not continuous; system swap includes other
applications. No system memory limits are raised.

The existing stage gate stays fixed: both conversations must improve, median
conversation ratio must be at most 0.99, every request/first-token/decode median
ratio at most 1.03, every decode saving positive, and median savings at least
20ms/token in both phases. A directional two-pair gain below that gate is
reported separately from a confidence-based gain or a passing stage.

## Native result

The [screen](screen-01/summary.json) completed all four processes in 222.92s.
Both conversation ratios favor 1460 slots: **0.98113 and 0.98341**. All four
individual decode comparisons also favor it. Output tokens, actual 104-token
state reuse and arithmetic dispatch counts match. The
[audit](verification.json) reconstructs the observations and failed advancement
gate from sealed raw data.

| Median across two runs | 1072 slots | 1460 slots |
|---|---:|---:|
| Initial generation, tokens/s | 4.106 | 4.371 |
| Generation after append, tokens/s | 3.689 | 3.870 |
| Initial first token, seconds | 13.503 | 13.473 |
| Append first token, seconds | 23.241 | 23.206 |
| Initial decode, ms/token | 243.554 | 228.775 |
| Append decode, ms/token | 271.065 | 258.411 |
| Planned engine memory, GiB | 9.998 | 10.998 |
| Largest phase-boundary footprint, GiB | 8.486 | 9.494 |

Initial/append generation application reads fell 8.56%/8.18%. Initial
ingestion reads did not change, and append ingestion reads fell only 0.56%.
There was no observed compression, decompression-counter change or swap growth
within any process. These gauges are not continuous peaks or a sustained-use
qualification. Requests at other times must not be compared as if they were
paired controls; in particular, the previous residency study recorded slower
baselines and cannot be pooled with these measurements.

The extra cache does not close the gap: even its short-screen medians still
need about 29ms initially and 58ms after the append to reach 200ms/token. Its
first-token times barely changed. Further cache growth and five-pair capacity
confirmation stop here; production defaults stay unchanged.

## Clean dependency capture and next experiment

The [follow-up capture](profile-01/summary.json) uses the existing 1072-slot
reference with residency enabled. One normal and one traced conversation
finished with no observed compression, decompression-counter change or swap
growth. It covers all 48 layers and ten selected experts for sixteen decode
steps in each phase. Unlike the preceding driver capture, both processes are
clean. The [reconstruction](resident-analysis.json) verifies complete command,
expert and buffer-counter coverage.

| Exclusive traced interval, ms/token | Initial | Append |
|---|---:|---:|
| GPU execution | 156.34 | 150.42 |
| GPU idle with submitted work | 28.88 | 31.38 |
| GPU idle with ready experts | 10.63 | 11.59 |
| GPU idle with pending reads | 52.71 | 58.37 |
| GPU idle awaiting callbacks | 9.62 | 9.95 |
| Other GPU idle | 40.47 | 36.26 |

Within submitted-work idle, driver scheduling accounts for only 6.36/6.97ms;
another 18.88/20.41ms occurs after scheduling and before GPU execution. These
are subdivisions, not additional time. Allocation and group retirement
account for about 3.3ms/token of CPU intervals, also overlapping the table.
This does not support another buffer-lifetime optimization as a large saving.

Median required-read queue time is 0.026/0.028ms, versus 2.45/2.42ms in read
service. Queuing does not dominate typical reads. Changing worker count or
adding priority machinery is not selected solely from the pending-read bucket.

The traced runs are **24.83% and 14.98% slower** than their normal companions.
These are measured overhead/variation ratios; no subtraction or scaling is
used to predict normal latency. Different capture conditions also prevent
assigning the entire improvement over older traces to residency.

A retrospective check of the planned previous-token top-two predictor finds
799/660 predictions selected again, out of 1410 per phase. Only **1/0** are
actual new misses: the repeated experts are almost always already cached.
This excludes the first captured token and layer zero. It is not a live
prefetch simulation and does not model cache state at issuance, displacement,
extra bandwidth or earlier completion. It provides little support for that
specific predictor on this workload.

The next bounded experiment targets **resident computation before routing**.
The largest command class, containing GDN, residuals, projections, reduction
and routing, takes 66.64/63.54ms of traced GPU time per token. These are group
durations, not a claim that GDN alone takes that time. The current scheduler
finishes routing before issuing the layer's expert reads
(`src/qwen/model.cpp`, `Model::moe`), so faster work here can also start those
reads earlier. First isolate its constituent operators using saved real-weight
inputs and the existing replay tools. Select a change only after a cheap
operator screen supports a material gain; then test complete requests at the
same 1072-slot, residency-enabled allocation. Do not assume the whole group
duration is removable, or reopen rejected grouping/cache experiments without
new evidence.

## Reproduction

```sh
.cache/qwen-reference-venv/bin/python scripts/qwen/cache_residency.py analyze \
  --output FRESH_DEMAND_DIRECTORY
.cache/qwen-reference-venv/bin/python scripts/qwen/cache_residency.py screen \
  --output FRESH_SCREEN_DIRECTORY
.cache/qwen-reference-venv/bin/python scripts/qwen/cache_residency.py verify \
  --source docs/benchmarks/2026-09-14-cache-residency/screen-01 \
  --output /tmp/cache-residency-verification.json
.cache/qwen-reference-venv/bin/python scripts/qwen/profile_resident_decode.py capture \
  --output FRESH_PROFILE_DIRECTORY
.cache/qwen-reference-venv/bin/python scripts/qwen/profile_resident_decode.py analyze \
  --source docs/benchmarks/2026-09-14-cache-residency/profile-01 \
  --output /tmp/resident-decode-analysis.json
```

Raw reports remain authoritative; the source snapshot supports later audits.
The [253 Python tests](python-tests.log) passed. No new native code or kernels
were changed, so the prior native validation applies to the identical build.
Production defaults, long-context acceptance and the 5 tokens/s target remain
unchanged.

The experiment ledger preserves the
[capacity result](../../experiments/53758dab1158f00ed15aae0e94f441dfaad135ab6438d97fe3526d3951f92e5e.json)
and the [next resident-operator hypothesis](../../experiments/49f05235f2606702c2a97ef69f958a5c0d2763fe480a66f231c213ce38482d93.json),
including the failed advancement gate and weak prefetch overlap. Both entries
are available through the local evidence query tool.
