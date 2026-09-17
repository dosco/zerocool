# Single-row expert direct-output trial

Implement the previously proposed `../2026-09-16-mtp-ngram-init/next-protocol.md`
without changing its advancement gates. Native C++23/Metal source-copy build;
production and previous producers remain unchanged. Both arms share lazy ngram
initialization, reference Q4 kernels, the existing Q8 policy, 1,460 target slots,
32 draft slots and the same 12GiB admission, including the prior 2MiB reserve.
Bounded expert scratch is off in both arms. Do not combine the earlier small gain.

Enable direct writes only in four-token target decode calls, for individual
expert microbatches containing exactly one row. Use existing `linear_into` with
the original expert contribution offset. Keep multirow experts, priming, shared
experts and single-token recovery on their existing paths. Retain selected-expert
reduction order, the coordinator's two live GPU groups, and completion ownership.
Counters track eligible calls, actual direct writes and avoided scatter bytes.

First run native real-weight fixtures under Metal validation: complete output and
sentinel comparison, all four destination rows, mixed expert row counts, invalid
destinations, reversed reads, cache eviction, all hits, delayed GPU completion,
cancellation and injected read/encoding failures. Require complete buffer release.
Retest the common lazy ngram fixture. Then compare fresh scratch-off control and
direct-output candidate on the eight-token forced-rejection workload, including
all logits, committed tokens, intermediate boundaries, future proposals and final
target/draft state. Both must match the qualified lazy control in `validation-03`.
Validation timing is not performance evidence. Keep existing clean-memory/host
requirements and preserve failed or blocked stages without pooling them.

Run a fresh sixteen-token pair on the same 72-token merge-intervals prompt. Stop
unless candidate cycle latency is at least 2% lower. Only a survivor gets a second
pair in reverse order; that pair must improve, and the two-pair geometric mean
ratio must be <=0.98. Compare exact outputs/state and require the dispatch-count
reduction to equal the recorded direct-write count. Report allocations, copies,
submissions, physical memory and all cycle components. Never turn counts into a
speedup claim. All normal timing must have zero recorded compression/decompression,
unchanged swap, AC power, Low Power Mode off and nominal thermals. Initial
available memory remains >=13.5GiB.

Only then run the three 128-token coding comparisons with fresh off/on processes
and alternating case order. A selected case after a resource stop requires its
own fresh pair. No 2K/4K/7K context, coding-quality, sustained-session or production
qualification follows automatically. The longer cases remain authoritative about
draft acceptance; the short all-accepted trace cannot establish general 5 tokens/s.
