# Native MTP draft-head preparation

The original model's trained prediction head is now downloaded, quantized and
prepared as a separate artifact. Existing native Metal kernels correctly read
its packed matrices. This preparation stage did not implement a complete forward
pass or measure acceptance. The subsequent [native forward stage](../2026-09-16-mtp-forward/README.md)
now passes independent numerical checks and measures 4.7904 tokens/s with real
drafting in one clean short screen. Production inference and the target model's
weights are unchanged.

## Follow-up review

The [review](review-01/summary.json) found and fixed two reproducible validation
issues: two same-sized dense matrices could exchange their manifest offsets and
still pass the audit, and changing live Hub download/like counters invalidated
inspection of an otherwise unchanged checkpoint. The audit now requires the
canonical tensor layout, while inspection pins immutable checkpoint metadata
separately from the local API snapshot hash. Missing source locks are rejected.

Raw writes now complete partial writes or fail before publishing a payload.
Download retries also check each saved tensor's source binding. Memory checks
require valid counters, zero compression/decompression and unchanged swap;
missing observations cannot qualify as clean. Verifier helper dependencies are
included in its frozen source identity.

All 12 focused tests pass. The existing artifact passes the stricter audit with
the same manifest and payload hashes. A fresh [native check](native-02/summary.json)
passes the same 25 matrices with clean memory. The native executable is unchanged.
Original reports and source snapshots are preserved; reviewed tools are archived
under `review-01/sources/`. These fixes do not establish draft acceptance or speed.

## Source and preparation

Pinned source: `Qwen/Qwen3.8-Flash-Next` revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`. The new
[mtp-models.lock.json](../../../mtp-models.lock.json) pins all 31 tensor names,
shapes, source byte ranges, shard identities, metadata hashes and tokenizer
checks. The one-layer head has 2,607,150,848 parameters excluding the shared
embedding/output matrices: 5,214,301,696 BF16 bytes, or 4.856GiB.

Only those tensor ranges were downloaded. Tokenizer, tokenizer configuration,
vocabulary and merges match the original checkpoint exactly. Large tokenizer
files use their upstream LFS SHA256; Git-stored files use their Git blob hash.
The failed first inspection retained its metadata and log: it initially compared
an LFS pointer's Git hash with tokenizer contents. The corrected inspection used
the published content hash and verified compatibility.

Range extraction requires exact HTTP 206 ranges and lengths, bounded reads, safe
filenames, pinned source metadata and per-tensor SHA256 receipts. Completed
source tensors are rehashed before conversion and can be reused on download
retry. The receipts do **not** establish the upstream full-shard LFS checksum:
we deliberately did not download unrelated weights to calculate it.

Recipe `mtp-affine-q4-experts-q8-dense-64-v1`:

- All 512 routed experts: affine Q4 with groups of 64, using the native expert
  record layout, including original gate/up splitting and 16KiB slot alignment.
- Sixteen dense projections: affine Q8 with groups of 64 and BF16 scales/biases.
- Thirteen small tensors: original BF16. These include routers, injection
  matrices and normalization weights.
- Shared token embeddings and output projection: reference the existing mixed
  target artifact; no duplicate payload is stored.

This is a new draft-only min/max quantization from original BF16. It is not an
MLX checkpoint conversion or a claim of unchanged draft logits. Its precision
must be assessed by measured proposal acceptance. Main-model quantization and
router behavior remain unchanged.

Normalization weights remain in their original zero-centred representation.
The manifest explicitly requires the future consumer to apply `1 + weight`
once. They must not be passed directly to the existing multiply-only norm
kernel, and must not be shifted twice. The final mixer is the MTP head's own
trained mixer, not the main model's mixer.

## Exact bytes and memory plan

| Component | Bytes |
|---|---:|
| 512 aligned Q4 expert records | 1,417,674,752 |
| Aligned dense/BF16 payload | 98,041,856 |
| Prepared weight total | 1,515,716,608 — 1.412GiB |
| Draft state and workspace allowances | 125,790,208 |
| Planned total incremental allocation | 1,641,506,816 — 1.529GiB |

The [combined plan](memory-plan.json) includes an 8,192-token private attention/
index state, four-token hidden/logit buffers, an append-tail checkpoint, 64MiB
scratch and 16MiB metadata allowance. Combined with the existing target plan and
perfect-verifier checkpoint/logit reservation, it totals 11.653GiB against 12GiB,
leaving 355.56MiB planned headroom and retaining 1,072 target expert slots.

These are allocation calculations, not measured full-draft residency. The target
plan was obtained from a memory-disturbed validation; its timings are not reused.
Additional native allocations must be charged before admission. If the planned
allowances prove insufficient, reduce cache capacity explicitly and compare at
the same total budget; do not silently oversubscribe or change precision.

## Verification

Six focused Python tests cover HTTP range rejection, tensor bounds, duplicate
metadata, packed code ordering, constant/tiny/negative values, irregular row
chunks, corruption detection and precision policy. Compilation used the native
C++23 flags, including warnings as errors and disabled fast math.

Preparation independently decoded every quantized row block and checked its
reconstruction error against the affine rounding bound. Every prepared file and
all 512 records were rehashed; padding, intervals and original BF16 bytes were
checked. These checks establish packing/conversion integrity, not coding quality.
The routed-expert reconstruction RMSE is 0.001772 (relative L2 error 0.09299),
reported only as a weight-reconstruction statistic.

[Native validation](native-01/summary.json) on the Apple M1 Pro then exercised
all 16 quantized dense matrices plus gate/up/down projections from experts 0, 256
and 511. It used 100 single-token and 25 four-token calls with one-hot input columns
0, 31, 63 and the final input column. Every output matched an independent CPU byte
decoder. Metal API and shader validation were enabled. Shared-buffer peak was
96.203MiB; physical-footprint peak 156.314MiB; compression and decompressions were
zero and swap stayed unchanged. This does not test a complete MTP layer, all
512 experts on Metal, real proposal acceptance, or normal-request performance.

Actual artifacts:

- `.cache/qwen-mtp-source-02/`: pinned metadata, selective BF16 payloads and download receipt.
- `.cache/prepared/mtp-q4-q8-v1/`: `dense.bin`, `experts.bin`, manifest and source receipts.
- `.cache/mtp-check-build-01/`: separate native format validator and build provenance.

The sealed native report retains exact source and executable identities.
Copies of new tools and the source lock are under `sources/`. No production
kernel, default build, memory limit or main-model artifact was changed.

## Next boundary

The [expanded packed-Q8 verifier](../2026-09-16-q8-expanded/README.md) now passes
the optimistic gate with two clean fresh candidate runs at 5.2814 / 5.3016
verified tokens/s and full numerical/recovery agreement. Bounded MTP forward and
real acceptance/cost experiments may proceed. The older capacity/cache results
retain their original status; they are not pooled into the new measurement.
The larger target cache still leaves insufficient room for this fully resident
draft plan. Start joint admission with at most 128 draft slots, keeping 1,460
target slots: existing allowances imply about 11.6631GiB combined, subject to
actual allocation verification and measurement of draft reads/forward cost.
Only about 11ms per verified token remains before the 200ms target at perfect
acceptance. Count drafting, rejected proposals and recovery from the first real
screen; do not assume the optimistic throughput transfers to actual generation.

For a promising verifier, implement the dedicated MTP forward path with an
independent numerical reference. Preserve the main model's pre-final-mixer
10,240-wide hidden stream, pair it with the next token embedding, apply the two
pre-FC norms (the hidden norm covers the full wide stream), apply shared
`fc_hidden` across the four branches, and run the head's attention/MoE/mixer.
MTP has private attention/index state and no PLE/ngram lookup. Return both the
collapsed stream for the shared output projection and the wide stream for the
next draft step. Priming must stream bounded target hidden panels; do not retain
all context hidden states outside the admitted budget.

Then validate prompt alignment, per-prefix rejection recovery and target-session
state before measuring actual draft cost and acceptance on coding continuations.
Production session/RNG rollback and non-greedy exact sampling remain separate
work. The two-/four-token perfect-verifier experiment cannot qualify those paths.

Architecture reference: [vLLM's Qwen4Exp MTP implementation](https://docs.vllm.ai/en/latest/api/vllm/models/qwen4_exp/nvidia/mtp/).
The original checkpoint's license continues to apply to its derived weights.
