# Bounded single-token temporary reuse

Status: implemented behind `bench --decode-scratch reuse`; both timing pairs won,
but memory disturbance keeps the result inconclusive. Default remains `none`.
No artifact, expert-selection, arithmetic,
cache capacity or production selector changes.

The [buffer measurement](../2026-09-13-buffer-costs/README.md) found 54–59ms/token
of allocation/retirement CPU intervals, with model compression making that
capture inconclusive. A separate probe of the existing scratch pool saved a
median 51.78ms per synthetic iteration with no observed compression. This is a
reason to screen reuse, not a prediction that inference will save 51ms/token.

The candidate uses the existing coordinator-owned scratch pool during a
single-token forward. Every temporary buffer has a distinct slot within that
forward; reuse begins only after the preceding GPU users finish. The capacity
is at most **128MiB within the existing 512MiB temporary allowance**. Persistent
state, resident weights and expert records bypass the pool. Exact physical-size
reuse and eviction of unused shapes preserve the same byte bound as context
length changes. Multi-token ingestion releases the retained pool before doing
new work. Failed/cancelled forwards drain GPU and I/O users before release;
partially updated session state remains invalid.

The initial option requires resident normal execution, the reference decode
schedule, original GDN, wait-mode expert tails and serial fixed memory. Other
execution combinations remain outside this experiment. `run` and `serve` do
not accept it. No OS memory limits are raised, and no extra expert capacity is
removed to pay for retained temporaries.

`scripts/qwen/screen_decode_scratch.py --output FRESH_DIRECTORY` executes two
alternating normal conversations: 72 prompt tokens / 33 outputs, then a retained
128-token append / 33 outputs. Both arms use mixed 4/8-bit, prepared Q4 records,
12GiB admission, 1848 CLOCK slots, packed Q8 rows 2, SIMD routing, eight readers
and ready groups of four. Only `decode_scratch` changes. Profiling, diagnostic
timers, Metal validation and boundary GPU probes are disabled during timing.
Sources, tools, binaries and artifacts are frozen under the shared GPU lease.
Initial allocation and the release before the append remain in request times.

The validator requires exact prior-control output tokens and unchanged dispatch
counts, all 32 candidate forwards to use the pool, bounded and inactive pool
ownership at phase boundaries, and a reduction in physical allocations. Both
conversations and every decode comparison must improve; median conversation
ratio must be at most 0.99, and the existing request/TTFT/decode median guards
must remain at most 1.03. Observed decode compression or decompression keeps a
timing win inconclusive. All pairs remain in the report.

Only a survivor advances to the all-48-layer logits/routes/state check with
forced cache eviction, fresh replay, cancellation and deliberate single-token
failure. GPU users and retained pool storage must drain on those failure paths.
Request screening is capped at 600 seconds and 150 seconds per process; later
state checks have a separate 360-second total cap. A failed early screen does
not consume long-context or five-pair qualification.

The native suite passed **62 tests / 48795 assertions** under Metal API/shader
validation. Tooling passed **221 tests**, including requested-mode verification,
pool bounds/release, missing memory evidence, exact state and query-axis checks.
No result from this short screen qualifies 5 tokens/s at 2K/4K context or
sustained coding use.

## Result: timing gates passed; memory gate did not

Native build `62d1bd6f` completed the screen in 236.14 seconds. All four
conversations produced identical tokens to the previous control. GPU arithmetic
dispatch counts matched, no phase exceeded two live command groups, and every
candidate decode phase executed all 32 pooled forwards. All phase boundaries
had inactive scratch scopes and drained command groups. Multi-token ingestion
released the previous decode pool as required.

| Pair | Phase | Control ms/token | Reuse ms/token | Reuse tokens/s |
|---|---|---:|---:|---:|
| 1 | Initial generation | 376.88 | 248.30 | 4.03 |
| 1 | Generation after append | 294.67 | 246.48 | 4.06 |
| 2 | Initial generation | 274.26 | 225.64 | 4.43 |
| 2 | Generation after append | 306.04 | 244.46 | 4.09 |

Conversation ratios were **0.90281 and 0.93681**, median **0.91981**. All request,
first-token and decode timing guards passed. Initial TTFT median ratio was
0.99590; append TTFT median ratio was 0.99746, including pool release. These
are two short pairs, with no confidence bound and no claim of repeatable gain.

Physical allocations fell from **3238 to 137.03125 per token** over each full
32-step phase, including the cold pool allocation. The remaining allocations
include 37 state buffers per token and 3201 initial workspace buffers amortized
over 32 steps. Each phase averaged 3100.96875 pool reuses per token. Retained
workspace was **74.39MiB initially / 74.58MiB after append**, within the fixed
128MiB cap and existing 12GiB allocation. The 1848 expert slots were unchanged.

The declared memory gate failed. The measured decode observations were:

| Run | Initial ending compression | Initial decompressions | Append ending compression | Append decompressions |
|---|---:|---:|---:|---:|
| Pair 1 control | 5.61GiB | 762202 | 5.59GiB | 46231 |
| Pair 1 reuse | 5.13GiB | 100168 | 5.14GiB | 22697 |
| Pair 2 reuse | 1.04GiB | 1115 | 1.04GiB | 0 |
| Pair 2 control | 0.38GiB | 0 | 1.46GiB | 35832 |

GPU command duration also varied, from 110.23 to 134.68ms/token. No samples were
discarded, adjusted by GPU duration, or pooled with older benchmarks. The report
status is `memory_disturbed`, with `advance_to_confirmation=false`; full-model
state/failure qualification and five-pair confirmation were therefore not run.
The additional native state/failure checks are implemented for a later survivor,
but normal output agreement is not presented as full persistent-state proof.

[Sealed summary](raw/summary.json), [source-revalidated comparison](comparison.json),
[native tests](native-tests.log), [tooling tests](python-tests.log).
The prior-output binding was added after the full tooling suite; its final
[focused checks](focused-tooling.log) also pass. The normal timing runs include
that binding and all runtime changes.

## Next step

Preserve this candidate for a fresh comparison when memory conditions improve;
do not discard it as a demonstrated regression or adopt it as a confirmed gain.
Post-screen reclaimable memory was about 20.65GiB, versus 19.56–21.17GiB at the
four admissions, so another unchanged retry was not justified. A clean survivor
then receives the existing complete state/failure checks before five fresh
confirmation pairs. The candidate's observed median latency still leaves about
37ms initially and 45ms after append to reach 200ms/token. Those short-request
observations do not establish the remaining gap at 2K/4K context.
