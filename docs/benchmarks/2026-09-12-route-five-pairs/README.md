# Parallel expert selection passes five fresh conversation pairs

Five fresh alternating pairs reduced complete-conversation time by a
geometric mean of **2.00%**. The candidate/control ratio was **0.98002**,
with a paired 95% interval of **0.96644–0.99380**: approximately **0.62–3.36%**
less time. Every pair favored the candidate and all secondary latency guards
passed. This qualifies the selector for later validation, not production
promotion or the original long-context performance targets.

| Pair | Candidate/control conversation ratio | Time reduction |
|---|---:|---:|
| 1 | 0.97178 | 2.82% |
| 2 | 0.99362 | 0.64% |
| 3 | 0.96853 | 3.15% |
| 4 | 0.98960 | 1.04% |
| 5 | 0.97683 | 2.32% |

| Median metric | Serial selector | Parallel selector |
|---|---:|---:|
| Initial conversation request | 23.56s | 23.04s |
| Initial first token | 14.09s | 14.05s |
| Initial generation | 3.38 tokens/s | 3.59 tokens/s |
| Retained-append request | 34.02s | 33.34s |
| Retained-append first token | 23.90s | 23.86s |
| Retained-append generation | 3.18 tokens/s | 3.38 tokens/s |

The initial prompt contains 72 tokens and requests 33 generated tokens.
The follow-up appends 128 tokens and requests another 33, with 104 tokens of
actual computation reuse. The append also ingests the preceding request's
one pending output token. Generation throughput uses wall time for the 32
decode tokens after the first output. These are deliberately short confirmation workloads;
they are not 2K/4K prompt measurements or a sustained coding session.

The conversation geometric mean clears the unchanged 0.99 threshold and its
95% upper bound is below one. All six secondary upper bounds are below 1.03;
the largest is 1.01687 for retained-append first-token latency. Initial and
append decode-time geometric ratios are 0.95245 and 0.95011 respectively.
Intervals use paired log ratios and Student-t with four degrees of freedom.
They assume independent, approximately normal pair log ratios; temporal host
effects may violate that assumption, and marginal intervals are not joint
coverage for every metric. First-token improvements are not established by
these intervals, even though their regression guards pass.

Both arms use the same mixed 4/8-bit artifact, prepared Q4 experts, 12GiB
allocation, 1848 CLOCK slots, 512-token panel, packed-Q8 two-row kernels,
eight I/O workers, ready groups of four, and original execution schedule.
Only `route_selection` changes from `serial` to `simd`. Timing excludes
Metal validation, profiling, capture, and the GPU reference probe. All twenty
requests match generated tokens, reuse, allocation, and non-routing dispatches.

Memory admission passed after the user freed host memory. Across the recorded
boundaries, maximum physical footprint was **10.46GiB** and process compression
and peak compression remained zero. Observed system swap use never increased
between a conversation's start and end; it fell by 8MiB in two conversations.
These are sampled boundaries and system-wide counters, not proof of an
absence of transient paging or long-session memory growth.

The complete retry took **601.24 seconds**, within the 900-second total cap.
Every process stayed within its 150-second cap. No earlier timing pairs were
pooled, discarded, or substituted. The
[earlier memory-blocked attempt](../2026-09-12-route-confirmation/raw/summary.json)
remains unfinished and contributes no samples.

The runner and offline query independently reconstruct the result from hashed
original reports. The sealed prerequisite supplies the captured-input operator
checks and all-48-layer exact logits/routes/state, fresh replay, forced eviction,
cancellation and failure checks. They remain valid for this unchanged native
build; no expensive correctness rerun was needed. The confirmation tooling's
[198 Python checks](../2026-09-12-route-confirmation/python-tests.log) were already
passing on this same source revision.

[Protocol](../../qwen_route_selection_stage.md), [raw result](raw/summary.json),
[source-revalidated comparison](comparison.json), [identity](raw/identity.json),
[retained prerequisite](raw/screen/summary.json).

Native build:
`3a0ea9334c448434ca7a534a623888e94c20350c29d29f7354bb4be2cc334a53`.
Mixed artifact: `b2c422f3c643e36f04227a64d61796b44a4b1029`.
Prepared manifest:
`c4bb4db3220a0de2218128738b2a6ac650a3d65086690c8dc2e8c23d097160da`.

Production defaults stay unchanged. Both timing arms use experimental packed
Q8, so this selector comparison does not resolve its earlier inconclusive
first-token guard. The 5 tokens/s, 2K/4K latency, 7K reporting, and sustained
coding-session targets remain open. Next, replay real weights and activations
for the [narrow BF16 projection candidate](../2026-09-12-route-confirmation/README.md)
before testing a shape-limited implementation on normal requests.
