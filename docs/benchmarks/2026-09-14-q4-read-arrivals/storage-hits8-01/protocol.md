# Q4 arrivals with selected file ranges invalidated

Declared before these timing samples. Keep `protocol.md` and its completed
`hits8-01` / `hits2-01` reports unchanged. Those runs requested 14,155,776,000
application bytes per condition including preparation, but observed only
6,049,792 / 684,032 device-read bytes. They do not exercise the physical-storage
part of the hypothesis.

The new explicit `--invalidate-files` condition uses the same eight original
prepared records, resident weights, fixed expert buffers, arithmetic, inputs,
rotating hit mask, process separation, five alternating pairs, and group caps.
It is a developer diagnostic; no production source or default changes.

After draining GPU and I/O and clearing expert-cache metadata, invalidate only
the eight selected page-aligned file ranges through read-only shared mappings
and `msync(MS_SYNC | MS_INVALIDATE)`. Check mapping, synchronization and resident
page observations; require no resident pages immediately after invalidation.
Then prime the declared hits with the existing uncached prepared-record reads.
Never perform buffered file verification between invalidation and these reads.
Checks of already returned buffers do not reread their source files.

Invalidation happens outside the measured coordinator windows. Record its time
and eight calls per batch separately inside preparation time. A timing arm has
32 batches, 256 invalidations, and exactly 707,788,800 total application bytes
including preparation. The eight-hit arm reads all bytes during preparation;
the two-hit arm reads 530,841,600 bytes during the measured coordinator windows
and 176,947,200 during preparation. These are logical record bytes, not syscall
counts. No global cache purge, source-file edits, or memory-limit changes.

Preflight is CPU-only, checks returned record hashes and source metadata, and
compares device traffic before and after selected-range invalidation. Its read
loop includes hashing, so elapsed time is not an SSD service-time measurement.

Run each condition as fresh check/timing/trace processes. A sampled timing arm
qualifies as having exercised device reads only if counters are available and
observed read bytes are 90–110% of its total application bytes. Apply the test
to every arm; aggregate agreement alone is insufficient. This is a conservative
screen under systemwide counters, not proof of per-process attribution, cold
SSD-controller state, or NAND access. Missing/disturbed coverage cannot qualify
the storage hypothesis. Do not pool old timing samples with these samples.

Keep the preceding diagnostic performance gate: paired GPU-time upper 95% bound
below one and coordinator-wall upper bound at most 1.03 at both group caps,
with exact outputs, no new Metal allocations, clean memory and host samples.
Report a separate storage-unqualified disposition when the explicit condition
fails coverage; an inconclusive performance interval remains inconclusive.
Each stage retains its 180-second deadline, 60 seconds per process, shared lease,
12GiB budget, and existing disk/resource admission and immutable report seals.

Neither result qualifies normal-request throughput. The decision is whether
actual device-read arrivals remove the prior cached-kernel gain, and whether
the smallest next experiment should target scheduling or surrounding compute.
