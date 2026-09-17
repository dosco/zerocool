# Fixed draft-width candidate

The isolated native candidate accepts explicit `FREELLM_MTP_WIDTH=1`, `2`, or
`4`. Its full-model correctness comparison now passes all ten cases with clean
memory. Two fresh 128-token LRU pairs favor width one by 17.77% in generation
latency. Performance screening remains preliminary; no policy is promoted.
See [the recovery plan](../../qwen_mtp_recovery_stage.md).

## Implementation

`scripts/qwen/build_mtp_widths.py` derives from the target-recovery producer.
All widths retain the same four-row checkpoint, journal and output capacities,
16MiB journal allowance, 1,460 target slots, 32 draft slots and 12GiB joint
admission. A two-row journal copies only its populated rows. Partial rejection
restores the corresponding attention tail and replays the existing recurrent
and convolution operators for the accepted prefix.

The one-token setting keeps the draft state synchronized through the existing
catch-up operation. Its measured cost must include that work; it is not the
older serial harness that stops maintaining draft state. Near the requested
output boundary, the candidate falls back to one-token cycles. Width selection
is fixed for a run; no adaptive policy is implemented.

Both `full-replay` and `state-only` recovery remain explicit. Width screening
retains full-replay while state-only recovery remains unqualified. The
existing saved-fixture format remains restricted to width four. The unsuccessful
[pipeline preparation experiment](../2026-09-17-target-recovery/allocation-followup.md)
is also excluded; all widths use the original pipeline preparation.

## Verification completed

[Native checks](native-check-01/summary.json) passed with Metal API and shader
validation. All **12 cases** match the independent row-at-a-time state update:

- Width two: accepted prefixes 1/2 at absolute offsets 3 and 8,190.
- Width four: accepted prefixes 1/2/3/4 at offsets 3 and 8,188.

The tests exercise the actual journal-copy path at each width and compare all
persistent buffers, including the untouched attention tail. Peak footprint was
**1.253GiB**, with zero recorded compression/decompression and stable system
swap. All GPU buffer owners were released. Inputs are synthetic; these checks
do not establish full-model logits, proposal alignment, coding quality, or
throughput.

The [producer](native-check-01/producer.json) binds the source, compiler inputs
and native binary. The binary SHA256 is
`5724c33c429522234a42afad25a88d12edc872ac214d741bb8554adc5487f256`.

## Remaining gates

The [full-model stage](validation-03/summary.json) covers every accepted prefix
at widths one/two/four, an irregular seven-token output, and immediate EOS at
each width. All ten native runs pass the strict memory and host checks. Every
candidate matches the serial-width reference's full-vocabulary logits and
complete persistent state at each committed boundary. The serial reference
also matches previously established independent logit hashes. Metal API and
shader validation were enabled. Peak physical footprint was 9.788GiB, with zero
compression/decompression and unchanged swap in all ten runs. These results
establish no speed measurement. The current native binary SHA256 is
`b12a79beadaa6ce501d13cf0840094ec6566dd30d24b68bc0133a6b5884a5deb`.

The [first attempt](numerical-01/summary.json) retains its checker failure.
Width two ended with a two-row draft operation, so the final `token_tile`
statistic correctly changed from one to two. The corrected checker derives
the allowed tile from the actual input shape and still rejects every other
kernel-policy change. The completed stage rechecks and reuses only its two
clean native reports; eight remaining runs are fresh. A subsequent JSON key
normalization fix was rechecked over all ten clean reports without new model
inference. The strict final stage binds the current checker; original attempts
remain unchanged. Timing is never reused.

Fresh 64-token LRU screens compare four versus two, then four versus one.
Reject a candidate below 3% first-pair latency reduction. Only survivors earn
a reverse-order pair; both must improve and the geometric reduction must reach
3%. These screens are preliminary, with unchanged resource gates. A survivor
still needs fresh longer coding cases, five paired repetitions, context and
sustained-session qualification. No adaptive policy is selected without a
measured workload crossover.

`screen_mtp_widths.py` provides validation, clean correctness resume and early
screening. The offline `cycles`, `compare` and `next` queries understand width
reports, including all draft catch-up, checkpoint and recovery costs.

## First speed screen

[Width two](screen-width-2-02/summary.json) failed its first 64-token LRU pair:
width four measured **3.3244 tokens/s**, width two **2.9966 tokens/s**. Candidate
latency was **10.94% higher**. Both runs were clean and matched every committed
logit and final state byte. Stop before a reverse pair; there is no confidence
interval from this one pair and no long-context conclusion.

The cycle accounting explains the tradeoff without assigning causality to an
individual kernel. Width two reduced replayed target rows from 22 to 8 and
expert application reads from 717.31 to 664.82MiB per committed token. Draft and
recovery time fell, but target verification rose from 235.21 to 301.76ms per
committed token. The net request became slower. Lower read volume alone did
not select the better schedule.

The earlier [screen setup attempt](screen-width-2-01/summary.json) stopped
before model inference because acceptance-position keys changed type during
JSON serialization. The reader now uses canonical string keys, with regression
coverage and fresh prerequisite validation. It contains no timing sample.

## Width one: fresh longer LRU comparison

The [64-token attempt](screen-width-1-01/summary.json) contains one complete,
clean pair: 3.3219 tokens/s at width four versus 3.9560 at width one. Its reverse
control observed an 8MiB decrease in total system swap. The unchanged-swap rule
therefore rejects that sample even though process compression was zero. The
attempt remains incomplete; its timings are not reused or pooled.

A separate [128-token comparison](long-lru-01/summary.json) completed two fresh
pairs in alternating order, using the same producer and memory capacities:

| Pair | Width four tokens/s | Width one tokens/s | Candidate/control latency |
|---|---:|---:|---:|
| Four, then one | 3.1709 | 3.8643 | 0.820568 |
| One, then four | 3.1568 | 3.8312 | 0.823967 |

The geometric latency ratio is **0.822266**, a **17.77% reduction** on this
workload. Every committed full-vocabulary logit, output token and final target
and draft state matches. All four processes recorded zero compression and
decompression with unchanged system swap; peak physical footprint was
**9.738GiB**, within identical 12GiB admission. The runs use 128 output tokens
after a 198-token prompt, not the required 2K/4K prompt acceptance workloads.
There are only two pairs, so no confidence interval or promotion claim.

Width one includes draft-state maintenance. Target verification still costs
255.27–257.45ms per token versus 246.25–247.43ms at width four. Its gain comes
from avoiding proposal work and rejected-block recovery in this request; it
does not demonstrate a faster verification kernel. Total cycle cost remains
258.78–261.02ms per token, above the 200ms target. Application expert reads fall
from 799,135,200 to 624,974,400 bytes per committed token, while the computation
and committed state remain exact. These counters are not physical SSD traffic.

The fixed-length screen now rejects an EOS-shortened timing sample; immediate
EOS remains valid in the numerical suite. Offline comparison rechecks this
rule and refuses to treat the incomplete short stage as a paired result.

The [other two coding prompts](other-coding-01/summary.json) now complete two
fresh alternating pairs each, with exact outputs/state and clean memory:

| Workload | Width four tokens/s | Width one tokens/s | Geometric candidate/control latency |
|---|---:|---:|---:|
| Interval merging | 4.3393, 4.3394 | 4.0537, 3.8906 | 1.092681: 9.27% worse |
| Retry/backoff | 3.5937, 3.5631 | 3.6630, 3.7485 | 0.965688: 3.43% better |

The prompt lengths are 72 and 93 tokens. All eight runs generate the requested
128 tokens, with peak physical footprint **9.740GiB**, zero compression and
decompression, and unchanged swap. These workloads remain separate from each
other and the longer LRU comparison; no pooled confidence interval is reported.

There is no universally best measured width. Width one regresses on interval
merging, while every measured setting remains below 5 tokens/s. The next
[bounded dependency capture](next-protocol.md) focuses on target verification
under the winning measured settings. A later adaptive policy needs separate
calibration and held-out evaluation; selecting winners from these prompts
alone would not establish its performance.

The current Python suite passes **505 tests**, including real-report query
revalidation, incomplete-stage rejection, numerical-versus-timing EOS handling,
fixed-width accounting and shape-dependent kernel metadata. The ten-case clean
numerical stage was [rechecked under the timing checker](validation-04/summary.json)
without reusing timings. The [sealed review](review-01/summary.json) retains
source snapshots, query answers, test logs, raw evidence hashes and unchanged
production identities.

Build and run the bounded operator check with:

```sh
.cache/qwen-reference-venv/bin/python scripts/qwen/build_mtp_widths.py \
  --output .cache/mtp-widths-build-new
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  .cache/mtp-widths-build-new/probe-mtp-forward \
  --width-recovery-self-test .cache/mtp-widths-native-new.json
```

Full-model runs still require the existing exclusive GPU lease, host/memory
preflight, frozen evidence and correctness prerequisites. The raw native
executable is not a replacement for those guards.
