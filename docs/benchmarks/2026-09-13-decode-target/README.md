# Generation dependency capture on the 32GiB M1 Pro

The fixed 12GiB allocation was admitted on retry, with all 1848 expert slots.
The normal/traced capture completed in 55.29 seconds. No native source changed
between the earlier confirmed router baseline and these measurements.

| Measurement | ms per generated token | tokens/s |
|---|---:|---:|
| Earlier five-pair candidate, median initial decode | 278.50 | 3.59 |
| New ordinary 16-step request | 473.64 | 2.11 |
| Same request with dependency/command tracing | 579.69 | 1.73 |
| Subsequent unchanged request with boundary GPU probes | 283.73 | 3.52 |

The new capture's two arms produced identical tokens and used the same memory
allocation. Trace latency was 22.4% higher. The last row uses an additional
diagnostic protocol; it is not a replacement control or a speedup. All results
remain preserved. The different output lengths also preclude pooling these
measurements with the earlier 32-step confirmation.

The ordinary capture's GPU command duration was 323.83ms/token, versus about
121ms/token in the earlier confirmation. The traced command duration was
376.39ms/token. Model compression and decompressions were zero; sampled system
swap use remained unchanged. The subsequent fixed resident-Q8 reference took
666.13 microseconds per warm dispatch before the conversation and 494.69 after,
with identical checksums and released workspace. Its after value is close to
the previously recorded roughly 500-microsecond reference. OS observations
reported AC power, nominal thermal state and low-power mode off, with documented
availability limitations. There is no frequency/wattage measurement or proven
cause for the variation. Do not diagnose it as SSD pressure or thermal throttling.

## Captured overlap

All 48 layers and 480 selected-expert records were joined for each of the 16
captured decode steps. The mutually exclusive mean intervals are:

| Observed interval | ms/token |
|---|---:|
| Within a GPU command's execution timestamps | 376.39 |
| GPU idle with a submitted command | 56.66 |
| GPU idle with ready expert data not submitted | 26.55 |
| GPU idle with pending required reads and no ready expert | 14.60 |
| GPU idle with an outstanding completion callback | 10.97 |
| Remaining GPU idle time | 93.82 |

These sum to the measured 578.98ms forward interval; they are observations of
overlap, not attribution of why the coordinator was blocked. Command lifetimes
can include stalls or preemption and do not establish hardware utilization.
Expert reads were 607.91MB/token in both arms. Application bytes and observed
device traffic remain separate in the raw reports.

The gap from final expert GPU completion to submission of its reduction group
was 71.09ms/token over the first 47 layers. It overlaps the table and must not
be added to it. Trace writing and next-layer preparation occur there. This
supports a bounded [expert-tail overlap experiment](../2026-09-13-expert-tail/README.md)
that prepares the next layer before the expert tail finishes while preserving
leases. It does not predict that all 71ms can be recovered during normal inference.

[Capture summary](capture/summary.json), [joined timeline](capture/timeline.json),
[ordinary request](capture/normal.json), [traced request](capture/traced.json),
[GPU boundary diagnostic](gpu-boundary/request.json),
[earlier source-bound baseline](baseline.json).

## Preserved first attempt

The earlier [blocked attempt](blocked/summary.json) stopped at metadata admission:
about 7GiB reclaimable versus 13.5GiB required including the safety margin.
It contains no inference or timing samples. The collector subsequently gained
stricter source validation and an explicit 17-output validator; that attempt
predates those tooling refinements and remains unfinished.

The collector enforces a five-minute deadline, exact token outputs, native and
artifact identity, the same 12GiB allocation and complete trace coverage. The
209-test tooling suite now includes five new tail-screen/query checks in addition
to the earlier 204 tests; the original [204-test log](python-tests.log) remains
unchanged here. Neither this short trace nor the proposed scheduling change
qualifies the 5 tokens/s target at 2K/4K context.
