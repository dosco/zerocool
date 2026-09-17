# Lazy ngram initialization and completed verifier scratch screen

The first clean short comparison reaches **5.0355 tokens/s** with bounded expert
temporary reuse, versus **4.9727** for its fresh control. Latency falls only
**1.2465%**, below the predeclared 2% gate. The runner stops after this pair;
no reverse pair, 128-token suite or production promotion follows. This is a
sixteen-token, all-proposals-accepted result, not the project's sustained 5-token/s
qualification. Production sources and binary are unchanged by this stage.

| Sixteen generated tokens after a 72-token prompt | Control | Expert scratch reuse |
|---|---:|---:|
| Generation tokens/s | 4.9727 | 5.0355 |
| Complete cycle ms/token | 201.096 | 198.590 |
| Target verifier ms/token | 189.707 | 187.163 |
| New Metal allocations | 28,000 | 8,296 |
| Scratch reuse events | 0 | 19,704 |
| Kernel dispatches | 22,815 | 22,815 |
| GPU submissions | 4,330 | 4,426 |

The [raw pair](short-01/summary.json) and [independent audit](audit-short-01.json)
preserve all measurements. Both processes have zero recorded compression and
decompression, unchanged swap, AC power, Low Power Mode off and nominal thermals.
Peak physical footprint is **9.6485GiB** within the same 12GiB joint admission,
1,460 target expert slots, 32 draft slots and 8,192-token state capacity.
Generation timing includes real proposals, verification and recovery; prompt
priming is separate. The requested long-context lengths remain unqualified.

## Implementation and exactness

The previous scratch implementation reserves two completion-owned 1MiB pools in
both arms. A separate startup change reserves the original ngram row capacity
without eagerly constructing every row. Rows are constructed on first insertion;
all values are initialized before lookup, and the original FIFO ring runs once
full. The exact 158,275-row capacity, 64MiB budget, map reservation, packed tables,
BF16 values and hash-to-row mapping are unchanged. Reserved bytes remain admitted.

The native [real-table fixture](fixture-01/ngram.json) compares eager/lazy output,
logical slots, addresses and hit/miss counters at 64KiB and 64MiB budgets. It covers
irregular chunks, duplicates, hits, EOS history and repeated FIFO wraparound.
The full cache begins with zero constructed rows and ends with 1,720 rows:
564,160 initialized bytes versus 51,914,200 reserved row bytes. This proves less
initialization; it does not prove a speed benefit or eliminate memory compression.
The existing real-expert lifetime, cancellation, failure and delayed-GPU fixture
also passes under Metal API/shader validation.

The fresh [full-model pair](validation-03/summary.json) passes all logit hashes,
committed tokens, intermediate rejection/recovery boundaries, future proposals
and final target/draft state. Both arms exactly match the earlier eager reference,
including ngram hit/miss counts. Memory and host checks are clean. The
[validation audit](audit-validation-03.json) reconstructs these conclusions.
All **38 focused MTP Python tests** pass, including malformed memory-map and
incomplete evidence rejection. Native model assets were actually exercised.

Both timing arms use lazy initialization. They compare scratch off/on within one
producer; they do not measure lazy versus eager performance. Source-copy developer
tools are `build_mtp_ngram_init.py`, `mtp_ngram_init.hpp`, `probe_mtp_ngram_init.cpp`
and `screen_mtp_ngram_init.py`. The executable is
`.cache/mtp-ngram-init-build-01/probe-mtp-forward`, SHA256
`abe129fa9e1650d125042ad70b69e249c36fe53fbd11e663712a157c42b994c3`.

## Memory investigation and preserved stops

The preceding eager scratch control (`../2026-09-16-mtp-expert-scratch/short-04`)
reached 53.094MiB peak compression despite 16.79GiB initial headroom. Ngram row
initialization was a concrete avoidable allocation, but similar byte counts did
not identify the compressed owner. Lazy `validation-01` and `validation-02` still
hit 29.328/29.609MiB peak compression. Their completed control outputs match the
reference, but the stages remain incomplete/resource-blocked with no candidate.
See [raw reconstruction](audit-lazy-01.json).

Three bounded read-only maps inspect only each diagnostic's owned native child:

| Capture | Boundary | Peak process compression over entire run | Exact reference |
|---|---|---:|---|
| `memory-01` | Draft priming, ordinary mode | 3.922MiB | Yes |
| `memory-validation-01` | Draft priming, Metal validation | 25.125MiB | Yes |
| `memory-recovery-01` | After first forced recovery | Zero | Yes |

At capture time, GPU-tagged allocations show no compressed pages. The maps have
about 9.5GiB resident GPU allocations; the large heap region consistent with the
ngram reserve has only 376,832 resident bytes out of 51,920,896 virtual bytes.
The maps classify one 16KiB page as owned unmapped memory. This category does not
identify a compressed C++ allocation. In the first two processes compression
appeared later; the recovery capture happened to remain clean. The cause of the
intermittent compression is unresolved. These are separate processes, and mapping
inspection can perturb execution. None is a performance sample or a substitute
for ordinary qualification. The [memory audit](audit-memory-01.json) reconstructs
categories from raw maps and excludes vmmap's optional second TOTAL from category
counts, correcting the earlier parser output retained in the sealed recovery run.

After the clean diagnostic, one fresh ordinary validation pair and the short
timing pair both completed cleanly. Earlier failures are unchanged, and their
timings are never pooled with these results. No unrelated process was stopped,
no OS memory setting changed, and no qualification threshold was weakened.

## Decision and next step

The scratch change removes 70.37% of allocations but only 2.507ms/token in this
single pair. It fails advancement; do not spend the long-run budget on it.
Keep the exact implementation isolated for a future measured combination.
Lazy initialization has independent correctness/memory evidence but no isolated
performance comparison or production qualification.

The [next trial](next-protocol.md) removes a redundant scatter copy for experts
with one token row inside a four-token verifier call. It keeps the reference Q4
kernels and output reduction. The prior profile contains 3,110 such calls out of
4,937 expert scatter calls; counts motivate the experiment but do not forecast
latency. Prior packed-Q4 and expert-tail experiments remain rejected.
