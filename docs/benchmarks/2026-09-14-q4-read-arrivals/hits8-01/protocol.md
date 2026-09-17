# Q4 gains under controlled prepared-record arrivals

Declared before timing. This extends the native all-hit coordinator diagnostic;
it does not change the inference library, model bytes, quantization or defaults.
The normal-request packed-Q4 regression remains the performance control.

Run two separate conditions, each with its own five alternating reference/packed
pairs and group limits one/four. Do not pool conditions or reuse previous timing
samples. Both retain actual mixed-artifact resident weights, eight ExpertStride
shared buffers, eight I/O workers, at most two live command groups, and the
12GiB ceiling. Use the same eight original Q4 records and 64 saved expert/input
cases as the preceding replay. Four repetitions give 256 expert executions per
measured arm. One repetition warms each arm equally.

- **Eight hits:** all selected experts are ready before each measured batch.
- **Two hits:** two selected experts are ready and six must arrive through
  `PreparedArtifact::expert` and the normal `execute_experts` coordinator.

Rotate the hit indices as `(batch + j) % 8`; both kernel variants receive the
same mask. This is a declared diagnostic pattern, not recorded normal routing.
The eight experts span four layers; numerical outputs stay at their original
destinations. All selected expert contributions execute with fixed arithmetic.

Between batches, drain GPU and I/O, release leases, clear cache metadata, reset
the fixed-pool cursor and preload exactly the declared hits. These operations
occur outside the measured coordinator windows. The pool allocator accepts only
ExpertStride and cannot wrap within a batch; clear/reprime cannot add physical
Metal allocations. Report preparation time, bytes and logical load calls
separately. The eight-hit control uses the same preparation mechanism.

Per four-repetition arm, require 256 chains, 32*K ready hits, 32*(8-K) misses,
zero joins and zero new physical Metal allocations. Timed application bytes must
equal misses * 2,764,800; the two-hit condition reads exactly 530,841,600 bytes.
Logical expert-load calls must equal misses. These are not physical pread-call
counts. Whole-device counters span preparation as well as timed work and other
processes; do not divide them by the coordinator-only wall time. F_NOCACHE does
not establish cold SSD/controller state.

Validate original fixture hashes against actual prepared bytes before running.
Establish expected output bytes from the original fixture records independently
of arrival order. Use separate native check, timing and trace processes. Check
enables Metal validation and compares every batch. Timing disables validation
and detailed tracing, checks the actual final timed batch, and rechecks all 64
outputs outside measurement. Both preserve untouched destination guards.

Trace captures one repetition per variant/group, with eight batches each and
no per-dispatch profiling. Verify each selected expert occurs once, its primed
acquisition class matches the mask, timestamp ordering is valid, group size is
within the cap, and outstanding ownership ends after GPU completion. Derive
group occupancy and read-completion-to-submission delays from these records.
Do not presume trace occupancy equals uninstrumented timing occupancy or
combine overlapping GPU, read and host waits as exclusive wall-time costs.

The diagnostic gain criterion remains GPU-time upper paired 95% bound below one
and wall-time upper bound at most 1.03 at both group limits, with clean sampled
memory and host observations. Retain inconclusive and disturbed outcomes.
Each condition has a 180-second stage deadline and 60 seconds per process; the
shared GPU lease, existing disk/memory checks and immutable evidence seals apply.

This stage ends with a narrow conclusion about read arrivals and one supported
next experiment. A diagnostic gain does not qualify normal requests, 5 tokens/s,
long-context inference or sustained coding. If reads reproduce the regression,
use the separate trace to identify the smallest scheduling change; if not,
preceding resident computation and full-token buffer lifetime remain open.
