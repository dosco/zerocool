# Four-token cache capacity: fewer reads, below the throughput gate

Follow-up: [exact block-cache replay](../2026-09-16-block-cache/README.md) now
reproduces live decisions and rejects both bounded cache tweaks. Neither SLRU at
1,460 nor CLOCK at 1,536 earns another native timing experiment.

Increasing the target expert cache from 1,072 to 1,460 slots at the same 12GiB
budget reduced expert reads and improved the first clean four-token timing.
It still missed 5 verified tokens/s. The runner stopped at the predeclared
early boundary rather than spending another three processes on a candidate
that could no longer satisfy the all-runs floor.

The [screen](screen-02/summary.json) and its [independent audit](audit-02.json)
complete successfully as an **early negative screen**, with
`paired_comparison_complete: false` and `capacity_promising: false`. There is no
complete alternating-pair speedup, confidence interval, production promotion or
real drafting/acceptance measurement. This result rejects advancement under the
current gate; it does not reject block verification as a design.

## Clean observations

All three fresh Metal-validated processes passed: serial at 1,072 slots, width4
at 1,072 slots and width4 at 1,460 slots. Complete vocabulary logits, routes,
persistent state, causal-prefix independence, zero-accept rollback and every
partial accepted-prefix recovery match the serial reference. Prime mathematics
matches the older sealed anchor; initial cache hashes match within each capacity.
The explicitly different capacities are allowed different initial cache hashes.

| Order | Slots | Width | Verified tokens/s | Cache hits | Expert MiB/token | GPU ms/token |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1,072 | 4 | 3.8610 | 0% | 813.59 | 136.45 |
| 1 | 1,460 | 1 | 4.0650 | 50.55% | 625.89 | 136.96 |
| 2 | 1,460 | 4 | 4.6655 | 21.59% | 637.92 | 123.25 |

Each timing consumes sixteen fixed continuation tokens after the same 72-token
prompt. Proposals are free and always accepted; no bonus token is credited.
Checkpoint copies, target execution and greedy acceptance are timed. Model load,
prime and evidence hashing are outside continuation timing. All completed
processes have zero observed process compression, unchanged decompressions/swap,
nominal thermal state, AC power and Low Power Mode off.

The single observed capacity comparison is 20.84% higher throughput / 17.24%
lower verifier latency. Width4 at the larger capacity is 14.77% faster than its
same-capacity serial observation. These are directional first-round observations,
not settled performance estimates. The reverse round was intentionally not run,
and prior attempts contribute no samples.

Larger capacity reduces expert reads by 21.59% and turns zero cache hits into
21.59% hits. This supports the earlier cache-thrashing hypothesis. It does not
establish that cache capacity alone explains the wall-time difference: GPU
duration also changes, and this is one process per configuration.

The four larger-cache block rates were 4.3283, 4.8560, 4.5357 and 5.0020 tokens/s.
The final block alone is not an acceptance result. There is no consistent trend
that justifies dropping startup work or replacing the declared sixteen-token
measurement with the fastest block. The complete measured interval takes
214.34ms/token, leaving 14.34ms/token to remove before even free proposals reach
the 200ms target. Real drafting and rejection would add further costs.
Checkpoint copying accounts for approximately 1.01ms/token, so optimizing it
alone cannot close that gap.

## Memory and implementation

The 1,460-slot target plus the shared capacity-four checkpoint/logit reservation
plans 11.1245GiB within the unchanged 12GiB admission. Timing peak physical
footprints were 8.523GiB at 1,072 slots and 9.523–9.552GiB at 1,460 slots.
The extra 388 slots consume 1,074,331,648 bytes. No OS memory limit, production
default, model weight or quantization changed.

The standalone harness now accepts only the two declared capacities; its default
remains 1,072. A separate runner reconstructs exactness, workload counters and
the early-stop decision from sealed native evidence. Resource disturbances cannot
become weak-candidate measurements. A fixed-admission failure is recorded as
resource-blocked rather than a performance rejection. Native power checks still
precede model loading.

Twenty verifier/builder/capacity tests and four host-preflight tests pass, as does
the [native checkpoint self-test](checkpoint-self-test.json). The production
fingerprint remains
`51877f96f015386dd642fd4e63fbc39ff2e75c1156f5ae1bcba764f546963594`;
the production binary SHA256 remains
`05495d0baf993b1f7f1bc4b9365753e88603909ac66db3309a1380ea8c7c519f`.
The separate executable, protocol and tool identities are bound in each report.
Tool snapshots are under `screen-sources/`.

The earlier [attempt](screen-01/summary.json) remains resource-blocked: its serial
validation reached 1.5MiB process compression before any candidate process.
Its [audit](audit-01.json) passes, but it contributes no timings or correctness
qualification to the retry. A [native headroom check](retry-headroom.json) then
showed reclaimable memory increasing from 15.86GiB to 19.29GiB, justifying one
fresh unchanged retry. The successful early screen took 187.18 seconds.

## Next decision

Keep production unchanged and defer the full draft forward path. Do not repeat
the missing rounds or move to a bigger allocation on the basis of this early
result. The useful finding is that multi-token cache behavior is materially
different from serial behavior; next collect the actual block expert-demand
order and replay bounded cache choices offline before choosing another GPU test.

Use the existing trace/simulation tools where possible. Preserve full priming
order and block boundaries; compare CLOCK, the already implemented SLRU policy,
and only capacities that leave explicit room for a bounded draft cache. Existing
serial SLRU results remain negative/inconclusive for their original workload;
any new test needs evidence from this block workload. Mark all replay estimates
as simulated application reads, never expected latency, and validate ordering
and lease assumptions against the measured live hit counts.

The currently planned fully resident 1.529GiB MTP addition would exceed 12GiB
beside the larger target. A 128-expert draft cache could reduce its weight
allocation on paper, but its extra reads, metadata and actual forward cost are
unmeasured. Joint target/draft memory admission must be explicit. Do not silently
reduce fidelity, increase memory, omit proposal cost or count rejected tokens.
