# M1 evidence, September 7, 2026

**The saved full-model numerical fixture passes. Release qualification remains
open.** `release.json` rejects stale build evidence and requires performance,
API, sustained-session, and coding checks in addition to numerical agreement.

| Evidence | Status |
|---|---|
| `logits.json` | All 248320 logits bit-identical to original MLX; relative L2 0, cosine 1 |
| `full-layer-parity.json` | All 48 complete layer outputs bit-identical on the five-token fixture |
| `cache-invariance.json` | Identical full-model logits with chunk=8/cache=32 and chunk=1/cache=64, including forced eviction |
| `operator-parity.json` | All 290 independent CPU operator checks pass; maximum relative L2 0.00643 |
| `native-tests.txt` | 18 tests, 599 assertions pass with Metal API and shader validation |
| `release-tests.txt` | Three rejection-path tests pass: missing assets, stale builds, and diagnostic performance evidence |
| `moe-replay.json` | All 48 router matrices, selections, weights, and expert sums exactly match MLX on identical inputs |
| `attention-replay.json` | All twelve attention kernels exactly match MLX on identical query/key/value inputs |
| `first-attention-layer.json` | First complete attention layer matches the original layer on identical input |
| `performance-startup.json` | Normal 8K-context startup refused by memory admission; no throughput recorded |
| `performance.json` | No qualifying 2K/4K throughput, append, or 7K measurements |
| `inspect.json` | Earlier hardware/memory snapshot; availability changed between runs, so startup records are authoritative for each attempt |
| `storage.json` | Actual M1 Pro / internal SSD, three repetitions of real expert and ngram reads |
| `slotstream.txt` | Published binary rejected by macOS 15.6 because its metallib requires Metal language 4.0 |
| `historical/` | Superseded numerical builds, generation, API smoke, and cache/chunk checks |

The full numerical fixture is the five-token prompt `[760, 6511, 314, 9338, 369]`
("The capital of France is"). All layers and final logits match without changing
weights or expert selection. The release tolerance remains relative logit L2
<= 0.02 and cosine >= 0.9998; it was not widened to accept the earlier mismatch.
These results do not establish longer-context agreement or coding quality.

Four arithmetic corrections resolved this fixture: fixed router matrix
accumulation, the original eight-part expert reduction, BF16 attention
scores/probabilities, and the original recurrent memory sum. Compensated
summation changed one real BF16 midpoint; its minimal input is retained in
`tests/fixtures/qwen-gdn-rounding.json` as a regression case.

`native_logits.f32` and `reference_logits.f32` retain the matching vectors.
Reports identify the native source fingerprint and replay input hashes. Large
layer traces remain at their recorded temporary paths and can be regenerated
from the pinned model using `docs/qwen_engine.md`. All 24 required checkpoint
files are SHA256-verified; missing assets cannot qualify a release.

The exact numerical run used diagnostic layer streaming with 32 expert slots
and a total planned budget of 2.12GiB. It cannot establish normal generation
speed. Memory later fell below the normal 8K configuration's minimum: about
4.91GiB engine allocations/reserve plus 1.5GiB separate system headroom.
Thirteen attempts over a bounded one-minute window were refused. The attempted
128-token/32-output baseline would not itself satisfy performance acceptance.

The normal 2K/4K workloads, retained 128-token appends, 7K context, current full
API sessions, 20-minute soak, and coding recovery workflow remain unqualified.
Storage counters include other processes; short observations cannot establish
a sustained-session memory guarantee.
