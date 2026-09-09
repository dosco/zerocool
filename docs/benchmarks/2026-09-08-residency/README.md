# Residency and execution experiments on the 32GiB M1 Pro

The native residency, direct/grouped decode, two-workspace prefill, complete
cached-token replay, and captured-operator policy paths are implemented as
explicit benchmark candidates. Production execution remains on its reference
defaults. These measurements do **not** qualify promotion or establish the
5 tokens/s product target.

## Identity and method

This is the actual Apple M1 Pro with 32GiB RAM and its internal SSD, running
macOS 15.6.1. The normal screens below use the pinned mixed 4/8-bit artifact
`b2c422f3c643e36f04227a64d61796b44a4b1029`, unchanged Q4 expert/ngram records,
8192-token context capacity, a 12GiB engine budget, 512-token panels,
128-token microchunks, and eight I/O workers. No OS memory limits were changed.

Normal screens are fresh processes, a 2048-token prompt and 64 output tokens,
fixed greedy sampling, and no Metal validation or diagnostic profiling during
timing. The first output is included in TTFT; throughput measures the remaining
63 generation intervals. The runner checks prepared-artifact identity, exact
output tokens, admission and execution options. A single screen has no paired
confidence bound.

[Build identities](build-identities.json) bind source fingerprints to preserved
native binaries and source archive hashes. The local archives are in
`.cache/controls/`. Reports retain their original build IDs:

| Build | Evidence |
|---|---|
| `402c3a65` | Preserved pre-stage native control |
| `a019141b` | Initial implementation, both short artifact/state checks and cached-token diagnostic |
| `f412c50f` | Residency normal-request screens; added diagnostic cancellation/admission guards |
| `b3d9e55d` | Current implementation; also rejects duplicate shape rules and ambiguous diagnostic selectors |

Older reports are not relabeled as current-build verification. The exact
configuration and workload are retained beside each report.
[Final tooling identities](final-tool-identities.json) also bind the updated
session checker and admission wrappers; the engine binary is unchanged from
the preserved `b3d9e55d` build. A current tooling archive is retained at
`.cache/benchmarks/residency-stage/qualification-tools.tar.gz`.

## Residency screen

All three arms use the same 1848 expert slots, reference expert execution,
serial prefill, tile-8 candidate affine kernels and precomputed GDN gates.
Only requested residency changes.

| Residency | 2K TTFT | Generation | p95 token | GPU command time/token |
|---|---:|---:|---:|---:|
| Off | 229.71 s | 2.447 tokens/s | 490.45 ms | 235.18 ms |
| Core | 227.22 s | 2.428 tokens/s | 503.43 ms | 238.23 ms |
| Core + expert cache | 222.17 s | 2.659 tokens/s | 431.21 ms | 230.47 ms |

The three outputs match. Each arm has 54.49% ready-hit rate during generation
and 38.05GB of application reads over the 63 measured generation intervals.
Process footprint at completion is about 10.42GiB. Process decompressions,
compression peak and change in system swap usage are all zero in these screens.
The approximate 9% generation improvement for core plus cache is a screening
observation; order, host activity and storage caching remain possible influences.
There is no observed compression event explaining the gain.

The initial run completed off and core, then rejected the third arm because
live memory had not yet recovered after process exit. The rejected admission
report is retained. A later metadata check admitted the same 12GiB budget and
only the missing arm was run. The harness now makes at most three metadata
admission attempts, separated by 2 and 5 seconds, retaining each rejection;
it never silently reduces a budget or repeats inference.

See [combined measurements](residency-comparison.json),
[original experiment](residency-12g/normal/summary.json), and
[completed third arm](core-cache-retry/summary.json).

## Decode screen

At the same 12GiB budget, core-cache residency and tile-8/precomputed-GDN
configuration, all six fresh-process outputs match:

| Expert execution | 2K TTFT | Generation | p95 token | GPU dispatches/token |
|---|---:|---:|---:|---:|
| Reference | 225.60 s | 2.636 tokens/s | 434.92 ms | 3247 |
| Direct output | 220.62 s | 2.661 tokens/s | 434.85 ms | 2767 |
| Grouped, limit 1 | 229.18 s | 2.667 tokens/s | 430.38 ms | 2767 |
| Grouped, limit 2 | 227.18 s | 2.702 tokens/s | 427.18 ms | 2512 |
| Grouped, limit 4 | 222.80 s | 2.688 tokens/s | 425.55 ms | 2395 |
| Grouped, limit 8 | 224.97 s | 2.661 tokens/s | 445.75 ms | 2334 |

Direct output removes exactly 480 scatters and 960 allocations per generated
token. Grouped execution further reduces allocations from 3310 to 1870 per
token. Its ready groups are often partial: the limits are not a promise to wait
for that many experts. The ready-group option also applies during prefill.

Group size 2 has the highest generation rate in this one pass, about 2.5%
above the reference. Larger groups reduce dispatch counts further but do not
improve throughput in these observations. This is insufficient evidence for
selecting a production group size. All arms retain 1848 expert slots, finish
near 10.42GiB footprint and show no decompressions or increasing system swap.

See [measurements and allocation counters](decode-comparison.json) and
[complete normal reports](decode-screen/summary.json).

Repeating the grouped-2 serial configuration for the workspace comparison gave
2.642 tokens/s and 225.32s TTFT, illustrating variation comparable to the small
screened decode gain. Its prompt phase contains 220.04s of GPU command intervals,
0.47s of CPU encoding and 169.92GB of application reads. Those intervals include
the command's elapsed GPU lifetime; they are not isolated kernel measurements.
Command lifetimes may overlap, so their sum is not GPU utilization. They suggest
that GPU execution deserves priority in further prompt optimization.
CPU waits and I/O service totals overlap GPU work and must not be added to it.

## Prompt workspace screen

Both arms use grouped-2 expert execution and the same total 12GiB budget.

| Prompt workspace | 2K TTFT | Generation | Expert slots | End footprint |
|---|---:|---:|---:|---:|
| Serial | 225.32 s | 2.642 tokens/s | 1848 | 10.42GiB |
| Double | 189.84 s | 2.565 tokens/s | 1460 | 9.90GiB |

Outputs match. The double path reused 21,559 temporary allocations during the
prompt and retained the same prompt dispatch count. Its two pools reserve about
1GiB before expert-cache admission; the observed physical footprint is lower
than serial because the full reserved capacity was not allocated. GPU command
interval totals fell from 220.04s to 187.90s, but this single-order comparison
cannot distinguish every scheduling, memory-layout or host contribution.

The observed 15.7% prompt improvement trades against a 2.9% generation-throughput
decrease (about 3.0% higher time per generated token). Workspace sizing and the
cost in expert hits need paired normal-request evaluation before promotion.
See [workspace measurements](prefill-comparison.json) and
[normal reports](prefill-screen/normal/summary.json).

## Captured operator selection

A diagnostic 512-token request captured 32 prefill cases containing real
activations and pinned packed weights, totaling 194,682,896 bytes. The capture
hit its case limit; it does not cover every layer, shape or request phase.
Each payload has a size and SHA-256 in the
[capture manifest](operator-capture-manifest.json). Payloads remain under
`.cache/benchmarks/residency-stage/operator-capture/`; the manifest and workload
allow them to be reproduced without committing model fragments here.

Tiles 1, 2, 4 and 8 passed all 640 exact output comparisons over five alternating
repetitions. The selection tool produced 27 candidate shape rules: 18 use tile
8, six use tile 4 and three use tile 2. Uncovered shapes and single-token
operators retain the reference kernel. This establishes isolated operator
evidence only; the policy is not promoted and has no normal-request latency
qualification. See [operator results](captured-operators.json) and
[candidate policy with paired bounds](shape-policy.json).

## Complete cached-token diagnostic

At 2048 prompt tokens, the mixed artifact restored a deep snapshot and executed
the same continuation token five times after an untimed warmup. Every run
matched the original logits, selected routes and all persistent state bytes,
with 480 ready hits and zero checkpoint/prepared reads or ngram misses.

Forward latency was 431.94, 445.07, 472.70, 470.46 and 433.08ms. GPU command
intervals accounted for approximately 345–374ms per token. There were no process
decompressions. This diagnostic used **original kernels**, including original
GDN, and therefore is not a compute floor for the faster kernel candidates.
Priming, snapshots, preload, restore and output hashes are outside those timings.
This is not ordinary generation throughput.

The residency registry held 5,362,515,968 bytes of resident matrices,
571,326,464 bytes of session state and 1,329,070,080 bytes of selected experts.
These are subsets of allocated bytes, not additional memory charges or proof
of guaranteed physical residency. Snapshot bytes are separately admitted.

See [full diagnostic](mixed-cached-core-cache/2048.json).

## Correctness evidence

The current build passes 42 native tests and 6293 assertions with Metal API and
shader validation, with zero skips. The Python tooling passes 42 tests. The
native cases exercise output offsets, partial groups, reversed read readiness,
eviction, cancellation, failed reads, snapshot non-aliasing, state replacement,
residency teardown, workspace reuse and captured-file/shape-policy rejection.

The initial implementation passed full 48-layer short session checks on both
Q4 and mixed artifacts, comparing logits, route identities, every persistent
state buffer, continued sessions and fresh replay. Forced failure/cancellation
invalidated partial state and drained outstanding work. Candidate execution
combined core-cache residency, grouped expert execution, double workspace,
tile-8 affine kernels and precomputed GDN gates.

The five-token candidate logits and all saved state digests also matched the
previous stage's independently verified fixture reports. The independent
implementation was not rerun for this comparison. See
[digest comparison](independent-logit-digests.json),
[Q4 session evidence](q4-short/summary.json),
[mixed session evidence](mixed-short/summary.json),
[native checks](native-tests-final.txt), and [tooling checks](tool-tests.txt).

The current `b3d9e55d` build additionally passes the full 48-layer
[257+129 mixed-artifact case](mixed-microchunks/summary.json), including fresh
replay, failed partial panels and cancellation. This combines the captured shape
policy, precomputed GDN gates, grouped-2 experts, core-cache residency and double
workspaces. It exercises full 128-token microchunks and irregular tails, but
does not cross the long sparse-attention boundary. The current build also passes
the [Q4 short session and sample-free priming checks](q4-short-current/summary.json).

Both artifacts' current-build five-token logits and every saved state buffer
match the previously independently verified fixtures, as recorded in the
[current digest recheck](current-independent-digests.json). This re-executes the
native candidate, not the independent implementation.

## Larger-budget admission

One metadata check admitted 18GiB, but the subsequent comparison could not
retain that budget. After the setup path was updated to use the same bounded
metadata retries as individual requests, all three attempts admitted only about
17.72–17.73GiB. No 18GiB inference workload ran, and no smaller budget is reported
as an 18GiB result. These are observations of changing live memory availability,
not a claim that 18GiB can never fit on this machine. The original and retried
rejections are retained under `residency-18g/` and
[residency-18g-retry/](residency-18g-retry/setup-error.json).

## Promotion status

No configuration is promoted. The completed screens remain below the generation
target and above the initial-prompt target at the admitted 12GiB budget. Long
2053+129, 4096+128 and 7K state qualification, five alternating normal-request
pairs (ten if inconclusive), 7K performance, the 20-minute memory check and the
coding/tool recovery workflow remain distinct promotion requirements. Neither
these short checks nor zero-read diagnostics substitute for them.
