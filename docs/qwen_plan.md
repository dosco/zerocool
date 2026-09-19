# FreeLLM: quality-preserving inference beyond RAM on a 32GB M1 Pro

Current execution priority: the [200ms/token generation stage](qwen_decode_target_stage.md).
Current experimental work: [exact embedding storage in real MTP](benchmarks/2026-09-18-streamed-mtp/README.md).
All six original-producer full-model numerical pairs now match, including actual
draft proposals, rejected-row logits, EOS and persistent target/draft state.
Some diagnostics contain compression, so clean full qualification remains open.
The first normal storage pair has clean processes at 9.734/9.107GiB physical
peak, but mismatched starting draft-cache state invalidates the comparison.
No speed result is accepted. A separate native producer now uses the existing
fixed-batch scheduler only for draft warm-up; generation keeps its normal
completion-driven scheduler. It builds and passes ten clean real embedding
checks, but full-model admission stops at 7.15GiB available versus 13.5GiB required.
Validate this producer and compare its state with the saved original before new
timing. The evidence query tool now audits storage comparisons and excludes raw
resource-disturbed throughput estimates. All 546 Python tests pass. Keep the
saved 642.15625MiB unused until the separate cache experiment; no production
promotion or historical timing reuse.

Previous broader exploration: [coordinate memory, cache admission and verification windows](benchmarks/2026-09-17-verifier-horizon/README.md).
The broader reassessment now includes exact token-row streaming, an eight-token
verifier, and a separate four-token compute tile within that wider storage window.
Streaming removes 675,446,784 bytes of resident allocation and adds a 2MiB host
allowance: **642.15625MiB less planned memory**, with unchanged quantization and
1460/32 target/draft slots. Real embedding checks cover eviction before GPU
submission, failed reads and destruction. All nine real-model verifier logits and
persistent states match serial replay. Real MTP generation with streamed rows and
production integration still require their own qualification.

Two fresh, separate 64-token perfect-proposal screens reject fixed width eight
at the current cache capacity. The first is 5.6127/4.3174 verified tokens/s for
four/eight. Capping GPU compute tiles at four removes the GPU-time penalty, but
the new pair is still 5.5641/5.4303. These are **verifier ceilings**, not generated
tokens/s, and timings from the two producers are not pooled. Both normal pairs
are memory-clean. Eight-token passes visit about 1990 distinct layer/expert pairs
on this workload, exceeding 1460 slots; their cache hits are zero. Four-token
passes visit about 1237 and retain roughly 31% hits. The next stage tests cache
admission and returning unused prefill workspace to experts, before more draft
or kernel work. Earlier small cache increases and SLRU results remain rejected
for their original workloads and capacities. The small gate/up screen is parked.
The Python suite now passes 525 tests; production defaults remain unpromoted.

Earlier recovery work: [bounded target state recovery and draft-length selection](qwen_mtp_recovery_stage.md).
The [recovery implementation and tooling](benchmarks/2026-09-17-target-recovery/README.md)
now pass clean real capture/replay: eight exact prefix checks, plus cancellation
and delayed-completion checks. Fixture writing now uses bounded uncached I/O.
All five full-model numerical diagnostics now match all logits, tokens, future
proposals and target/draft state: accepted prefixes 1/2/3/4 and immediate EOS.
The candidate performs zero full-target forwards and expert reads during
recovery. Several runs recorded compression, so clean full qualification remains
outstanding. The early 64-token timing control also encountered compression;
it establishes no speed comparison. Strict timing/resource gates are unchanged.
The runner can now reuse individually verified clean correctness runs from
sealed attempts, recompare every complete pair and run only missing checks.
Preliminary timing may reject weak candidates earlier but cannot qualify adoption.
An isolated [fixed-width 1/2/4 candidate](benchmarks/2026-09-17-mtp-widths/README.md)
now passes ten real-model cases with clean memory at 9.788GiB peak: all accepted
prefixes, immediate EOS and an irregular output length. Every logit and committed
target/draft state matches serial-width replay. Twelve synthetic two-/four-row
recovery checks also pass. Width screening retains full-replay while recovery
remains unqualified. A bounded allocation investigation identified CPU owners,
but reducing prepared pipelines from 77 to 39 did not eliminate compression;
that intervention is set aside. Width two failed its fresh 64-token LRU screen
with 10.94% higher latency and is set aside. Two fresh alternating 128-token
LRU pairs favor width one: **3.83–3.86 versus 3.16–3.17 tokens/s**, a **17.77%**
geometric latency reduction. All outputs and final state match, with clean
memory at 9.738GiB peak. The other two 128-token prompts now have two clean fresh
pairs each: width one increases interval-merging latency by 9.27% but reduces
retry/backoff latency by 3.43%. Each workload keeps its own measurements. The
earlier short attempt remains incomplete and no default changes. Next, use
[bounded current-width target profiling](benchmarks/2026-09-17-mtp-widths/next-protocol.md)
to identify exposed verification dependencies; no measured width reaches
5 tokens/s. Five-pair confidence, long-context and sustained qualification remain
outstanding. The Python suite now passes 515 tests.
The [current-width target profiler](benchmarks/2026-09-17-mtp-width-profile/README.md)
now completes clean command and per-dispatch captures for interval merging,
with exact full-request references and 9.7462GiB maximum physical footprint.
Each covers five target calls, 20 verified rows and 16 committed outputs. GPU
execution is the largest command-profile overlap bucket. The counter capture
ranks multi-row routed gate/up above the remaining generic Q8 hyper projections;
the recurrent scan alone is small. The [proposed small operator screen](benchmarks/2026-09-17-mtp-width-profile/next-protocol.md)
tests shared input/bias work across gate and up while preserving arithmetic; it
is parked behind the broader memory/cache/verifier work above.
The LRU retry passed admission but recorded 4.0625MiB compression before its
profile window; it remains resource-blocked and its timing is excluded. No
normal-throughput improvement or production default change follows from profiling.
The current rejection path repeats 14 / 46 / 33 target input rows across the three
128-token coding runs. Retaining small state-update inputs may avoid that repeated
forward work. Measure this separately from the already implemented draft catch-up;
even zero recovery cost would leave all three cases below 5 tokens/s.

The [single-row expert direct-output trial](benchmarks/2026-09-16-mtp-direct-output/README.md)
is now implemented and measured. Real-weight destination/lifetime tests, full-model
forced rejection, and all logits/tokens/state checks pass; 42 focused MTP tests
pass. Both alternating sixteen-token pairs reach **5.04 tokens/s**, with **3.14%**
lower geometric-mean cycle latency and exactly 3,110 fewer scatter dispatches.
The fresh 128-token control/candidate comparisons are **4.4128 / 4.4042** for
interval merging, **3.0916 / 3.1595** for LRU repair and **3.6269 / 3.7164** for
retry/backoff. One is flat and two improve about 2–2.4%; the across-case reduction
is 1.46%, without repeated-pair confidence. All six processes stay memory-clean,
peaking at **9.7273GiB** under the same 12GiB admission. Both arms also match the
earlier serial reference numerically; no historical timing is reused. Keep the
candidate experimental and stop before long-context/release qualification.
The next decision must address verification/recovery work and expert reads per
committed token: those reads remain 628–762MiB/token within identical cache
behavior. The changes below remain distinct experiments, not additive gains.

The [verifier scratch follow-up](benchmarks/2026-09-16-mtp-ngram-init/README.md)
now completes a clean sixteen-token comparison: **5.0355 tokens/s** with bounded
expert temporary reuse versus **4.9727** for the fresh control. All logits, tokens,
recovery boundaries and target/draft state match. Allocations fall from 28,000 to
8,296, but cycle latency improves only **1.2465%**, below the predeclared 2% gate.
Stop this candidate before reverse-order and longer runs; it is not promoted.
Peak physical footprint is 9.6485GiB with zero compression/decompression and
unchanged swap. This short, all-proposals-accepted result does not establish the
5-token/s target on longer coding requests. All 38 focused MTP tests pass.

Both timing arms share a separately tested lazy ngram initialization: reserve
the same 158,275 rows but construct only inserted rows. The real-table fixture
preserves bytes, addresses, hit/miss decisions and FIFO eviction. A fresh full-
model rejection pair also matches the previous eager reference exactly. Earlier
memory/host stops and bounded VM-map diagnostics remain separate; lazy construction
does not guarantee freedom from compression. The timing pair measures scratch
reuse, not the independent speed effect of lazy initialization. Production is
unchanged. Its [direct-output follow-up](benchmarks/2026-09-16-mtp-ngram-init/next-protocol.md)
is now complete as described above, with reference Q4 kernels and final reduction
unchanged. The audited profile's 3,110 eligible down projections out of 4,937
expert scatters motivated the trial; the normal-request results now determine
its limited practical benefit.

The [longer continuation and current-verifier profiling stage](benchmarks/2026-09-16-mtp-continuation/README.md)
now completes exact, memory-clean 128-token comparisons on three coding prompts.
Real MTP versus serial: interval merging **4.4038 / 4.1173 tokens/s**, LRU repair
**3.1987 / 4.0629**, TypeScript retry **3.6897 / 3.9437**. Proposal acceptance is
86.67%, 66.67% and 82.41%. Peak footprint is about 9.77GiB under the same 12GiB
admission. The first suite preserves its thermal stop; the last case uses its own
fresh complete pair. No timing samples are pooled. The independent 22-boundary
fixture, full-target rejection replay and evidence audits pass; 30 focused MTP
tests and seven timeline tests pass. Fixed four-token drafting is not ready for
promotion or expensive 256-token/sustained qualification. Even eliminating all
recovery cost would leave these cases below 5 tokens/s. The repaired current
verifier profile now passes exact outputs/state and all 192 layer passes with
clean memory. It exposes 26,932 target-buffer allocations over sixteen tokens
and zero temporary reuse; the existing pool only applies to single-token calls.
Next screen completion-bound expert-temporary reuse for four-token verification,
preserving two GPU groups, cache capacity, arithmetic and the same 12GiB total
budget. Allocation counts and instrumented idle intervals are hypotheses, not
projected savings. Keep acceptance-aware block selection separate. The earlier
thermal stop and rejected profiler capture remain preserved.

Earlier short result: [real native MTP drafting](benchmarks/2026-09-16-mtp-forward/README.md)
passes independent numerical fixtures and target rejection recovery. Its first
clean 16-token screen reaches **4.7904 tokens/s**, including actual proposal and
recovery costs, versus 3.6393 for the fresh serial control. All 12 proposals are
accepted; final target state matches exactly. This is one short timing pair,
not 5-token/s or long-context qualification. The combined 12GiB plan retains
1,460 target slots and 32 draft slots after two separately preserved
compression-blocked 128-slot attempts. State-only MTP catch-up is now implemented
and passes exact private-state, proposal and rejection/replay checks. Its first
two timing attempts remain memory-blocked and separate. A third fresh pair now
completes cleanly at **4.9713 tokens/s** versus 4.8719 for full catch-up, with
exact state and a directional 2.00% latency reduction. The 5-token/s floor fails,
so no further unchanged pairs are run and no confidence-bounded gain is claimed.
Use the longer, varied coding results above to choose the next experiment.
Even removing all catch-up time from the clean result would leave about
202.64ms/token; that optimization alone cannot meet 5 tokens/s.

The [expanded packed-Q8 verifier](benchmarks/2026-09-16-q8-expanded/README.md)
passes two fresh alternating pairs at **5.2814 / 5.3016 verified tokens/s**, versus
4.6298 / 4.5894 for the unchanged four-token path. All full-model exactness and
rollback checks pass with clean memory. This satisfies the short perfect-proposal
gate and enabled the bounded real MTP work above. Long-context performance
remains unmeasured. Production defaults are unchanged.

The authorized [perfect-draft verifier screen](benchmarks/2026-09-16-perfect-draft/protocol-stable-prime.md)
tests a larger change: amortize execution and weight reads across two or four
known-correct continuation tokens. It includes bounded rollback checkpoints,
all-row logit and persistent-state comparisons, and fresh-process timing at the
existing memory/cache budget. This reopens speculative decoding research only;
there was no complete draft forward path at that stage. The
[follow-up screen](benchmarks/2026-09-16-perfect-draft/README.md) now passes exact
full-vocabulary logits, routes, persistent state and rejected-prefix recovery at
widths 1/2/4. Deterministic benchmark priming fixes differing initial cache hashes;
measured decode retains completion-driven execution. All 18 focused tests pass.
Four of six timing processes completed cleanly: serial 4.076, width 2 at 3.614,
and width 4 at 4.124/4.106 verified tokens/s. The next process failed fixed memory
admission. Preserve its incomplete status; there is no complete paired speedup.
Both measured candidates missed the absolute 5-token/s floor, even with
free, perfect proposals. That gate prevented integration until the later
expanded-Q8 verifier result above; do not repeat this unchanged screen.

The [width-4 capacity screen](benchmarks/2026-09-16-perfect-draft-capacity/README.md)
now tests 1,460 versus 1,072 slots at the same 12GiB, with fresh serial and block
correctness/recovery checks. Its first clean timing round records 3.8610 tokens/s
at 1,072 slots, 4.0650 for serial at 1,460, and 4.6655 for width4 at 1,460.
Larger capacity reduces expert bytes by 21.59% and raises cache hits from zero to
21.59%. The directional throughput increase is 20.84%, but the predeclared
all-runs 5-token/s floor fails. The runner stops early, preserving the incomplete
paired comparison; no reverse round or prior timing is pooled. All 24 focused
tests, native rollback self-test and both evidence audits pass. The initial
memory-blocked attempt remains separate. Production and full-draft integration
remain unchanged.

The [block cache capture](benchmarks/2026-09-16-block-cache/README.md) now completes
two clean traces and reproduces every native cache decision and all 14 cache
snapshots, including real lease lifetimes. All 240 layer passes per process and
all priming work are covered. Seven focused tests and independent audits pass.
SLRU at 1,460 slots increases simulated reads by 0.28–0.34%; CLOCK at 1,536 saves
only 0.13–0.18%, across both recorded orders. Neither meets the predeclared 10%
read-reduction screen. No cache candidate is selected; no latency follows from
these conditional simulations. The earlier memory-blocked attempt is separate.

The [four-token compute profiler](benchmarks/2026-09-16-block-compute/README.md)
now completes both modes cleanly in 54.43 seconds, with exact logits/routes/state,
zero compression and 9.523GiB peak physical memory. Seven tests and the independent
audit pass. Each mode captures all 192 layer passes and 21,747 dispatches. The
first admission-blocked and second compression-disturbed attempts remain separate.
Offline revalidation shows 62.99% of expert calls have one row,
23.15% two, 9.16% three and only 4.70% four. The existing selector already clamps
single-row tiles; measure the actual shape mix instead of assuming uniform width4.

The counter profile ranks three recurring four-token GDN Q8 shapes at a combined
41.17ms per input token before expert reads can begin. These are instrumented
costs, not attainable savings. The [actual-input row-pair screen](benchmarks/2026-09-16-block-gdn/README.md)
now completes cleanly in 4.36 seconds with exact outputs. Its isolated GPU ratio
is 0.90268 (95% interval 0.89093–0.91458), but its 2.94ms/token shape-frequency
projection misses the declared 10ms gate; do not run the expensive verifier for
this option. The source tensor capture remains compression-blocked; the separate
replay reuses only verified tensor bytes and measures fresh small processes.
The [packed-Q8 block screen](benchmarks/2026-09-16-q8-block-packed/README.md)
now completes cleanly in 4.89 seconds. Word loads preserve every output bit and
reduce summed isolated GPU time by 24.49% (paired ratio 0.75507, 95% interval
0.74386–0.76645). Its 7.56ms/token shape-frequency projection remains below the
predeclared 10ms advancement gate. Eight focused tests and the evidence audit
pass. Keep the measured component; do not promote it or repeat this unchanged
GDN-only screen. No new full-request throughput has been measured.
The [expanded screen](benchmarks/2026-09-16-q8-expanded/README.md) now measures
attention, vocabulary output and GDN together: isolated weighted GPU ratio
0.583989, median projection 25.945ms/token. It then completes fresh serial/block
correctness, recovery and two alternating full-verifier timing pairs. Candidate
throughput is 5.2814 / 5.3016 tokens/s, 14.07% / 15.52% above paired controls.
All outputs, routes, persistent state, initial cache, cache hits and expert read
bytes match; all seven verifier processes are clean. Nineteen focused tests,
native self-tests and the independent audits pass. The input capture's startup
compression remains a separate blocked attempt, supplying verified bytes only.
The optimistic gate is passed. Next implement and independently validate the
prepared MTP forward path with an explicit 128-slot draft cache, then measure
draft cost and acceptance before production integration. The joint estimate is
11.6631GiB inside 12GiB; actual allocations and reads still need measurement.
The measured 188.983ms/token verifier leaves only about 11ms/token before the
5-token/s target, so include proposal and rejection costs from the first screen.
Keep CLOCK at
1,460 as the experimental reference. Do not repeat cache
policy/capacity timing for the rejected hypotheses, combine changes, or treat
single-token kernel results as four-token evidence. Preserve all startup work.
The earlier 214.34ms/token capacity result is superseded for this developer
verifier by the fresh packed-Q8 comparison; preserve its original disposition.
The larger verifier plan is 11.1245GiB. Adding the fully resident 1.529GiB MTP
head would exceed 12GiB; later integration needs an explicit bounded draft-cache
plan and measured draft traffic/cost. Do not automatically grow memory or repeat
the stopped candidate's missing rounds.
Advance to a real draft experiment only if this optimistic verifier reaches
5 verified tokens/s in both alternating rounds with exact state and clean memory.
The [MTP preparation stage](benchmarks/2026-09-15-mtp-preparation/README.md) now
extracts the original trained head, verifies tokenizer compatibility, and produces
a separate 1.412GiB Q4/Q8 artifact. All 512 records pass storage checks; 25 real
matrices pass native one-/four-token column checks with clean memory. The combined
allocation plan is 11.653GiB, retaining 1,072 target slots within 12GiB. This is
prepared weight storage and operator validation, not a complete draft forward
path or an acceptance/performance result; the verifier gate remains in force.
Its [follow-up review](benchmarks/2026-09-15-mtp-preparation/review-01/summary.json)
fixes tensor-offset validation, mutable Hub metadata pinning and partial writes.
All 12 focused tests and a fresh 25-matrix native check pass; prepared weights
and the production build remain unchanged.
The [bounded current-build stage](qwen_stage200.md) is implemented and screened;
its [results](benchmarks/2026-09-13-stage200/README.md) retain 1460 slots provisionally,
record unexercised live pressure handling and inconclusive Q3 timing, and preserve
the timed-out 2K diagnostic. No new exact optimization met the evidence gate.
The [submission/residency follow-up](benchmarks/2026-09-14-submission/README.md)
now measures 4.35% lower short-conversation latency across five fresh pairs
with core-plus-expert residency and exact state parity. The append saving is
11.24ms/token, below the declared 20ms stage gate; defaults remain unchanged.
The [capacity follow-up](benchmarks/2026-09-14-cache-residency/README.md) now
shows a small directional gain: 1.77% lower conversation latency across two
clean pairs, but only 14.78/12.65ms saved per decode token. Stop capacity
qualification at the failed stage gate and keep 1072 slots as the experimental
reference. A clean residency-enabled trace finds substantial read-dependent
idle and resident GPU work; typical read queueing and buffer retirement are
small. The previous-token top-two predictor targets only one observed new miss
across both captured phases. Next isolate the resident operators before routing
using real-weight replay before selecting another request-level experiment.
The [resident-operator follow-up](benchmarks/2026-09-14-resident-operators/README.md)
rejects exact Q8 load-ahead after a 3.78-second real-input screen. Its initially
blocked GPU breakdown subsequently completed at the unchanged allocation.
The resulting [packed-Q4 probes](benchmarks/2026-09-14-q4-packed/README.md)
cut isolated expert GPU time by 43–47% with exact outputs. Their roughly
14ms/token projection still misses the individual 20ms gate. The
[combined Q4/cache request stage](qwen_combined_decode_stage.md) now integrates
the decode-only option and passes native operator, full-model state and failure
checks. Its [four-arm request comparison](benchmarks/2026-09-14-combined-q4/README.md)
completed: packed Q4 slows decode at both cache sizes, and the combined change
is 1.98% slower for whole conversations. All memory/identity checks pass, but
performance gates fail. Keep reference Q4 and 1072 slots; no long qualification
for this candidate. The [native replay](benchmarks/2026-09-14-native-q4-replay/README.md)
now confirms that the GPU gain survives native scatter, actual resident
allocations and the all-hit expert coordinator. Older command traces show
72–77% single-expert groups despite the group limit of four. The
[read-arrival follow-up](benchmarks/2026-09-14-q4-read-arrivals/README.md) now
verifies actual device reads and retains a roughly 47% expert GPU gain with
six misses per eight-expert batch, but only 6.6–7.0% lower coordinator wall time.
The first cached-file conditions remain inconclusive and separate. Small groups
and read arrivals alone do not reproduce the full-request regression. The
[shared-expert follow-up](benchmarks/2026-09-15-q4-shared-arrivals/README.md) now
queues the actual resident shared chain before the first routed command and
passes 32 independent CPU cases plus native byte checks. Its combined GPU gain
remains 37.3–38.0%, with 6.9–8.2% lower replay wall time. The
[scratch-lifetime diagnostic](benchmarks/2026-09-15-q4-scratch-lifetime/README.md)
retains shared/routed temporaries across 48 passes: 20.25MiB versus 0.42MiB
with per-batch reuse. Its blocked and inconclusive attempts remain preserved.
The [validation-memory follow-up](benchmarks/2026-09-15-q4-validation-memory/README.md)
now completes four matched validation-off/on/on/off processes with identical
work and no compression. Validation adds about 51MiB of footprint after warmup;
the previous compression is not reproduced. Fresh full validation/timing/trace
then qualifies the forward diagnostic with 7.1–8.3% lower coordinator wall
time. A separate fresh batch control remains inconclusive at group one; no
samples are pooled or discarded. This partial replay covers only 27.2% of
measured full-token scratch. Close this memory attribution attempt and return
to a bounded actual-request profile to locate the remaining runtime difference.
These diagnostics do not qualify normal requests; the original rejection stands.
The [actual-request profiler](benchmarks/2026-09-15-q4-request-context/README.md)
is now implemented with complete short decode-window validation and a conditional
reverse-order trace pair. Its first attempt stopped before inference at memory
admission: 9.421GiB reclaimable on the last check versus 13.5GiB required for
the fixed 12GiB comparison. No timing was collected. Rerun the unchanged
protocol into a new evidence directory when headroom is available. The retry
after Brave closed passed admission and completed the reference conversation,
but reached 104.547MiB cumulative process compression during that work. It
stopped before the packed variant. Its observed 3.709/3.464 tokens/s are
memory-disturbed, unpaired measurements; no speedup or bottleneck conclusion
follows. Preserve both blocked reports and the original performance decisions.
The separate [reference diagnostic](benchmarks/2026-09-15-reference-diagnostic/README.md)
now completes under a declared 512MiB compression-peak allowance for diagnosis
only. It captures all 32 forwards: GPU execution is 126.51/145.44ms per token,
pending-read idle 58.37/61.83ms, and total traced forward time 274.06/297.74ms.
Physical peak is 8.604GiB with 108.75MiB compressed peak and unchanged swap.
The strict clean-memory gate remains false. The largest mixed resident command
class motivates a [complete hyper-block fusion experiment](benchmarks/2026-09-15-reference-diagnostic/next-experiment.md),
with a cheap whole-block screen before integration. No production speed claim,
new packed-Q4 comparison or long-context qualification follows from this trace.
The [complete hyper-fusion screen](benchmarks/2026-09-15-hyper-fusion/README.md)
now passes four native-output and fused-output byte comparisons plus twelve
edge cases, but its full block cycle is 1.10% slower across five pairs. The
frequency projection is −0.316ms/token, far below the 20ms gate. Keep the current
path, record this composition as rejected, and skip native integration and
long qualification. Fewer dispatches reduced encoding time without improving
whole-block wall time; select the next intervention from the dependency budget.
Coalescing all remaining expert reads is rejected: fewer submissions made
requests slower. Defer the small projection
experiment and separate append optimization; retain correctness, memory, and
first-token regression guards. Short diagnostics do not replace 2K/4K acceptance.

Approved September 7, 2026. This is the implementation contract; status and
measured evidence belong in [qwen_engine.md](qwen_engine.md) and the benchmark
reports. Planned capabilities must not be presented as implemented.

## Direction and success criteria

Build one native C++23/Metal engine for Qwen3.8-Flash-Next. Coordinate weight
precision, SSD layout, caching, and execution order to reduce the time spent
waiting for each token's required weights while preserving useful coding
quality. Promote optimizations using complete requests at equal memory budgets.

| Requirement | Target |
|---|---|
| Machine | Actual 32GB M1 Pro and internal SSD |
| Engine memory | At most 22GiB, reduced by system limits |
| Context | 8192 tokens, including output |
| Concurrency | One active conversation |
| Generation | At least 5 tokens/s, aim for 8 |
| Initial 2K prompt | First token within 60 seconds |
| 128-token append to retained 4K history | First token within 10 seconds |
| Sustained use | 20-minute coding session without progressive memory or sustained swap growth |

The quality target is the mixed 4/8-bit artifact. Retain the existing Q4 artifact
as the unchanged numerical and performance control. Publisher-reported lower
quantization error motivates evaluation; it does not establish native coding
quality or M1 performance. [Published comparison](https://huggingface.co/pipenetwork/Qwen3.8-Flash-Next-MLX-4bit#quality)

| Purpose | Repository | Revision |
|---|---|---|
| Q4 control | pipenetwork/Qwen3.8-Flash-Next-MLX-4bit | aa7c790e804bbf9d491ddb109c3d61bc4a555f7c |
| Quality reference | pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit | b2c422f3c643e36f04227a64d61796b44a4b1029 |
| Original source | Qwen/Qwen3.8-Flash-Next | de4b8e4d43b917e7706784d8bb445c9af86a3540 |

Within an artifact, execute every router-selected expert with fixed arithmetic
and reduction order. Cache state and read-completion order cannot alter results.
Different quantizations may produce different routes. Never change precision or
OS memory limits automatically. Changing artifacts requires new session state.

The next exact-arithmetic performance stage is specified in
[qwen_next_stage.md](qwen_next_stage.md). It preserves this plan's artifact and
quality boundaries while measuring prompt and generation latency separately.

## Integrated architecture

### Prepared storage

Create a versioned artifact before lower-bit kernels. Place each expert's packed
projections, scales, and biases in one aligned contiguous record, preserving
source bytes. Record formats, dimensions, offsets, lengths, alignment, and
hashes. Reuse the index design for Q4 and later mixed formats.

Interleave each ngram row's existing 80+10+10 bytes into a 100-byte record without
changing its hash-to-row address. Coalesce duplicate rows and storage pages,
including outstanding reads. Store decoded cached values as BF16: the current
decoder already rounds to BF16, so this introduces no additional loss. Keep the
packed ngram tables unchanged and SSD-backed; their roughly 32GB size alone
prevents full residency.

### Completion pipeline

One inference coordinator owns expert-cache mutation and Metal encoding. Reader
and GPU callbacks publish completions. Use a sliding window of at most 32 leases
and two live command groups, admitted under the byte budget. Compute any ready
experts, scatter to original token/expert positions, and retain the existing
final reduction. Submit ready work before waiting; begin with groups of four
and sweep 1, 2, 4, 8. Release resources after their final GPU use, then admit more
reads without a batch barrier.

Use one bounded scheduler with eight workers initially. Current dependencies
outrank future reads. Reserve half the queue for demand, cap prefetch, and prevent
large ngram lookups from filling the demand queue before routed experts.

This applies PowerInfer's completion-to-task mechanism at expert granularity,
without its model-specific neuron skipping or moved router.
[Implementation](https://github.com/Tiiny-AI/PowerInfer/blob/8bd56d69906c9d2dba4d3bf6899763401e01a9a4/smallthinker/powerinfer/moe_sparse_pipeline/expert_cache.cpp#L119-L128)

### Request schedules

| Schedule | Behavior |
|---|---|
| Generation | Route one token, overlap misses with ready/shared work, reduce |
| Retained-history append | Group appended tokens by expert, retain cache and state; explicitly cover 128-token follow-ups |
| Large prefill | Take a bounded panel through one layer at a time, grouping expert work across the panel |

Start prefill with a 1024-token panel and 128-token attention/recurrent
microchunks. Stream each selected expert once per panel where possible, compute
its rows in bounded microbatches, and never require a complete expert layer in
RAM. Compare panels 256, 512, 1024 against the old chunk-major schedule at equal
total memory. Track absolute positions per layer, commit progress after the
panel, and invalidate partially updated state on failure/cancellation after
draining outstanding users. Keep generation and short append dedicated paths.

### Joint memory and precision planning

Account for actual aligned allocations, metadata, resident weights, recurrent
and attention state, panel activations, expert contributions, both caches,
staging, live GPU resources, and driver reserve. Count shared allocations once.
Use a few expert record-size classes within one byte budget with free-capacity
rebalancing. Start with global CLOCK. Shrink by evicting eligible entries while
retaining useful survivors; reduce panel size before refusing a request when
that makes it fit. Higher-precision resident weights reduce expert-cache space
and must be charged in every comparison.

## Implementation milestones

1. **Measure dependencies.** Replay 48 successive layers with ten selected
   experts each from recorded hit/miss distributions and representative GPU
   work. Measure queue delay, read service, last-required-read latency,
   ready-to-GPU delay, GPU work, and buffer lifetime. Separate ready hits,
   outstanding-read joins, and new misses. Report application bytes separately
   from device counters, which include other processes. Establish normal Q4
   initial, append, and generation baselines; diagnostic trunk streaming cannot
   qualify performance. Budgets are 200ms/token for 5 tokens/s and 125ms for 8.
2. **Improve unchanged Q4.** Implement prepared expert/ngram records, bounded
   I/O priorities, completion-driven submission and release. Measure layout and
   scheduling independently and together. Require unchanged logits and state.
3. **Establish mixed 4/8 reference.** Add checkpoint-specific affine Q8 for
   sensitive non-expert matrices while preserving BF16 tensors and Q4 experts
   and ngrams. Verify tokenizer, architecture, tensor correspondence, and hashes
   before payload reuse. Compare native operators and full inference with an
   independent implementation. Measure actual resident increase/cache decrease.
4. **Reduce prefill rereads.** Implement layer-major panels with the existing
   schedule retained as reference/fallback. Select the fastest measured admitted
   configuration, not merely the largest panel.
5. **Calibrate selective compression.** Gather bounded per-expert activation
   statistics from coding, tools, continuations, and general text. Keep
   calibration, selection, and held-out evaluation disjoint. Implement one first
   lower-bit format: affine Q3, group size 64. Compare Q3 gate/up + Q4 down;
   Q3 routed projections with Q4 restored in sensitive groups; and the unchanged
   mixed 4/8 reference. Uncovered groups remain Q4. Derive new weights from the
   verified original source; Q4-to-Q3 is exploratory only. Gate/up-only Q3 saves
   about 15% of current expert payload, not 25%, after scale/bias overhead.
   Select using quality, latency, and total memory together. Defer Q2.
   Tie each expert's gate/up format, group size, and bitrate so its fused kernel
   remains valid; down precision may differ. Calibration captures exact templated
   model-visible token streams, including generated reasoning, coding, tool calls,
   tool results, continuations and recovery. Keep calibration, candidate selection,
   and final evaluation conversations disjoint. Evaluate actual quantized
   candidates rather than importing EXL3's noise-sensitivity estimates, whose
   optimizer is explicitly untested on sparse models. Measure padded record bytes,
   resulting cache capacity, exposed read waits, decoding computation and complete
   requests together. Tool parser/recovery checks must pass independently.
   EXL3 trellis support is deferred until affine-Q3 results identify a concrete
   need; any later investigation starts with a bounded M1 operator experiment.
6. **Trace-justified cache/prefetch.** Compare CLOCK with one probation/protected
   policy at equal bytes; protected entries remain evictable and prefill gives
   no permanent importance. First run expert prediction in shadow mode. Initial
   live prediction uses previous-token routes one layer ahead, at most two
   predicted experts and one active speculative read. Yield to demand, do not
   displace actively needed records, and always fall back to the true router.
   Prediction affects read timing only. [PowerInfer-2](https://arxiv.org/html/2406.06282v3#S4)
   The offline `query_evidence.py cache` query now provides equal-byte CLOCK,
   probation/protected SLRU and future-aware MIN curves from saved routes, under
   explicit fixed-order, immediate-release assumptions. It reports application
   read bytes, not a native speedup. Existing five-token prefill and cancellation
   fixtures do not establish normal generation locality; do not change the live
   policy from those curves. First obtain a deadline-limited 32-token normal
   continuation with complete routes and explicit phase/session/reset boundaries.
   Extend promising evidence to 256 generated tokens and retained-history append,
   then screen a single policy at equal admitted memory before full qualification.
   `capture_routes.py` now implements the first capture with a 180-second default
   deadline, the mixed artifact and unchanged reference kernels at 12GiB. Native
   committed-route markers distinguish prefill, decode, session changes and aborts;
   no partial forward is admitted to the cache simulation. Instrumented timings
   remain ineligible for performance qualification.
   The [first complete normal capture](benchmarks/2026-09-10-normal-routes/README.md)
   recorded all 32 decode steps in 34 seconds at the fixed budget. At its actual
   1,848 slots, SLRU simulated 10.71% fewer decode expert reads than CLOCK, but no
   latency gain is established. This supports the longer locality capture and
   append before implementing a policy; native CLOCK remains the default.
   The [extended capture](benchmarks/2026-09-10-extended-routes/README.md) completed
   256 decode steps, a 128-token append and 32 further steps in 167 seconds at the
   same budget, verifying 328 tokens of actual state reuse. SLRU simulated 5.72%
   fewer first-request decode reads and 9.46% fewer append reads, but 0.42% more
   reads during generation after the append: 4.95% fewer across the conversation.
   Instrumented native generation was 2.47 tokens/s and append TTFT 24.13 seconds;
   neither proves acceptance. Next implement this one policy behind an experimental
   option, verify lifetimes and exact state, then run a short equal-budget paired
   request screen including append. Keep CLOCK until measured latency supports a
   change; the modest simulated read benefit does not establish a speedup.
   SLRU is now implemented behind `bench`/`inspect --cache-policy slru`; CLOCK
   remains the production default. The [native screen](benchmarks/2026-09-10-slru-screen/README.md)
   passed exact all-layer logits/routes/state and failure/cancellation checks under
   forced eviction. In two alternating short-history pairs, SLRU issued 9.72% fewer
   expert reads, but whole-conversation time was 15.97% lower in one pair and 5.08%
   higher in the other. The predeclared repeatability gate failed; no five-pair or
   long qualification followed. Preserve this inconclusive result in the ledger.
   Use existing memory/dependency evidence to investigate variable initial decode
   latency before tuning policies or rerunning expensive validation. Do not assign
   the observed timing variance to compression without testing that explanation.
   The [bounded startup diagnostic](benchmarks/2026-09-10-decode-startup/README.md)
   now links per-token counters to two alternating off/core residency pairs at
   the same 12GiB allocation. Core reduced the first four decode forwards from
   3.72/4.15s to 1.69/1.71s, with decompressions falling from 478,188/446,245 to
   1/54 and identical output tokens. This supports the startup-memory hypothesis;
   all runs are instrumented and do not qualify normal latency. Next screen
   off/core without diagnostics, including retained-history append, before
   longer qualification or more cache tuning. Keep existing defaults; observed
   core generation remains about 2.57 tokens/s, below the 5 tokens/s target.
   The [normal residency screen](benchmarks/2026-09-10-residency-screen/README.md)
   then passed its predeclared gate: conversation time fell 4.94% and 6.06% in
   two alternating pairs, with identical outputs, 104-token follow-up reuse,
   memory plans and expert-read counts. Diagnostics were disabled. Next use five
   paired normal repetitions with uncertainty estimates before longer
   qualification; keep defaults unchanged. Core still generates about 2.5
   tokens/s, and the short-history follow-up still waits about 24s for its first
   token. This does not meet the product targets.
   The [five-pair confirmation](benchmarks/2026-09-10-residency-paired/README.md)
   failed its fixed gate: primary geometric-mean core/off conversation ratio
   1.0228, with a model-based 95% interval of 0.7707–1.3573. Core won three pairs
   but the last core conversation slowed substantially with few decompressions
   and longer GPU durations. Preserve every run and stop longer residency
   qualification; no default changes. Next add a small resident GPU timing
   reference outside request timing plus available host-condition metadata, to
   distinguish device-speed variation from request-specific waits. Retain the
   existing per-phase read and memory counters, and keep missing data explicit.

Mixed-reference validation and calibration preparation may proceed alongside
the first two milestones. The first deliverable is a dependency-level report
and numerically identical Q4 inference using contiguous records and completions.

## Current exact-attention experiment

The [sparse-attention stage](qwen_sparse_attention_stage.md) specifies CPU-equivalent
GPU selection, wholly masked score-tile skipping, heavily selected expert and
execution-lifetime regression coverage, and isolated paired measurements. This
stage changes no artifact bytes or production defaults. Its numerical and timing
qualification does not substitute for quantization quality or product acceptance.

The next deliverable is [screen-first selector evaluation](qwen_selector_qualification_stage.md):
CPU/full versus GPU/full only at the fixed 12GiB budget. Verify quick operator
and saved-input correctness, then use `triage_selector.py` for five cached 4K
pairs with a 600-second total deadline. Stop weak or inconclusive candidates;
timeouts and resource blocks remain unfinished. A promising result proceeds to
a small normal append screen (`triage_requests.py`, one pair with a 900-second
total deadline), then full 7K/session/recovery
validation and the established paired normal-performance gate. The existing
all-phase qualifier retains its old order and is reserved for survivors. Reuse
only original, revalidated evidence. No default or precision change is implied.

## Current bounded memory experiment

The [memory-balance stage](qwen_memory_balance_stage.md) implements a fixed Q8
GPU boundary reference and one 1848/1460-slot comparison at the same 12GiB
ceiling. Probe-enabled diagnostics/screens remain separate from probe-off
confirmation. Only a short-screen survivor receives five fresh paired runs;
production defaults and long-context qualification remain unchanged.
The [completed bounded stage](benchmarks/2026-09-10-memory-balance/README.md)
passed exact probe-on/off state checks. The smaller cache reduced footprint
about 1GiB but its two conversation ratios were 1.00954 and 1.00598, failing
the short gate. No confirmation ran. Endpoint probes did not reproduce the
earlier late GPU slowdown; its cause remains unresolved.

## Current isolated Q8 experiment

The [bounded packed-Q8 stage](qwen_q8_steady_stage.md) tests the existing
two-row packed kernel against the original schedule with 1848 slots and the
same 12GiB ceiling. It uses fresh captured inputs, short all-layer state parity,
and two alternating probe-off conversations. Historical bundled-kernel results
select the candidate but do not qualify this configuration. A noisy five-pair
operator pilot is retained separately from one 20-pair follow-up; neither is
pooled into subsequent request confirmation. No default or precision changes.
The [completed isolated screen](benchmarks/2026-09-11-q8-steady/README.md)
preserved all-layer state and reduced conversation time by a median 10.0% in
two alternating pairs. Median generation rose from 2.55 to 3.50 tok/s initially
and 2.48 to 3.20 after the append. First-token latency remains largely unchanged.
Next confirm this same configuration with five fresh pairs before long-context
qualification; do not pool the short screen or change production defaults.
The [five-pair result](benchmarks/2026-09-11-q8-confirmation/README.md) now records
an 11.06% complete-conversation reduction, but the full gate is inconclusive:
initial first-token uncertainty has an upper ratio of 1.03437 versus the 1.03
guard. Preserve that decision and the defaults. Next evaluate the explicit
[12GiB/18GiB memory tradeoff](qwen_memory_budget_stage.md) with packed Q8 kept
experimental in both arms; it does not retroactively qualify this result.

## Interfaces and qualification

Use the [offline evidence queries and experiment ledger](qwen_evidence_queries.md)
to choose experiments and retain negative or blocked outcomes. Start with existing
reports and verified paired comparisons. The bounded cache simulation answers
read-volume what-ifs while keeping incomplete trace coverage explicit. Critical-path
ranking, shared runtime event IDs and new lifecycle probes follow only when a
concrete decision cannot be answered from current evidence. The index remains rebuildable;
raw reports and ledger JSON retain their original provenance.

Keep `inspect`, `run`, `bench`, `serve`. Inspect reports artifact/recipe identity,
precision distribution, actual prepared bytes, and allocations. Bench adds
completion tracing, recorded-route replay, per-layer waits, byte-weighted cache
metrics, prefetch usefulness, and actual computation reuse. Run/serve require
an explicit artifact; API model IDs identify the recipe. Offline developer tools
prepare, calibrate, convert, and evaluate; native inference needs no Python/MLX.

Validate reversed read completion, delayed GPU completion, all-hit/all-miss and
mixed loads, duplicates, eviction, resizing, cancellation, corrupted reads,
irregular panels, sparse boundaries, ngram history/EOS, and continuation versus
fresh replay. Include changed tools, edited history, rollback, and compaction.
Use independent decoders/operators for Q3/Q4 experts and Q8 resident matrices.
Use five alternating paired performance repetitions on identical workloads,
plus freely generated coding sessions whose routes may differ.

Keep runtime correctness and quantization quality separate. Compression must
add at most 0.02 nats/token held-out NLL and lose at most two percentage points
coding success relative to mixed 4/8, using paired confidence bounds.
Inconclusive results fail promotion. Required tool-call and recovery cases must
pass separately. Promotion also requires all latency targets, 7K reporting,
and the sustained coding workflow. Bind reports to native build, artifact,
recipe, and evaluation data. Missing model assets never produce a release pass.

## Lessons retained from ds4 and Lily

Use distinct short-appends, ordinary prefill cache references, direct reads into
reusable shared buffers, overlapped hit/miss work, and bounded resource lifetimes.
Avoid serial duplicate read-ahead before worker submission. Use M1-compatible
Metal rather than M5-only TensorOps. Adapt fusion, packed computation, and fewer
intermediate dispatches to this artifact's affine format; IQ2/Q4_K kernels are
not interchangeable. ds4 comparisons on larger Macs justify experiments, not
M1 performance claims. Preserve MIT/Apache notices when reusing code.

Vision, extra models/backends, neuron pruning, training, speculative decoding,
and lossy KV/recurrent/ngram compression remain outside v1.
