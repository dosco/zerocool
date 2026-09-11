# Larger expert cache reduces reads but loses complete-request time

Two fresh alternating pairs on the actual 32GiB M1 Pro returned
`improvement_not_demonstrated`. The 18GiB/4175-slot conversation was **8.57%**
and **1.17% slower** than its paired 12GiB/1848-slot control: median ratio
**1.04874**. This ends the candidate before five-pair confirmation. Production
defaults remain unchanged.

Both arms explicitly used experimental packed Q8 two-row kernels, identical
mixed-precision weights, the original schedule, and the same short conversation
(72-token prompt, 33 outputs, retained 128-token append, 33 outputs). Timing
excluded profiling, Metal validation and GPU boundary probes. All eight requests
matched generated tokens, exact dispatch counts, and actual 104-token reuse;
fixed allocation categories matched and both requested budgets stayed admitted.
The unchanged native build already passed the all-layer state/eviction checks
and 57 native tests with Metal validation. The budget tooling passed 192 Python
tests. The earlier Q8 first-token confidence guard remains inconclusive.

| Observation | 12GiB control | 18GiB candidate |
|---|---:|---:|
| Initial generation expert misses / 32 tokens | 6533 | 3586 |
| Append generation expert misses / 32 tokens | 7516 | 4357 |
| Initial generation median tokens/s | 3.486 | 3.305 |
| Append generation median tokens/s | 3.277 | 3.713 |
| Initial first-token median seconds | 13.887 | 16.228 |
| Append first-token median seconds | 23.536 | 24.549 |
| Sampled process footprint after generation | 10.42GiB | 16.42GiB |
| Sampled process compressed bytes | 0 | 4.70–5.80GiB |

Reads fell **45.1%** initially and **42.0%** after the append. During the first
candidate process, reported system swap rose from 3.85GiB to 5.83GiB during
initial generation and ended near 5.80GiB. The second candidate did not add
further swap, but still retained roughly 4.7GiB of compressed process memory.
System swap includes other processes; these observations do not identify every
swapped page or prove sustained swap growth. They do show that successful
allocation admission is insufficient evidence of useful resident cache capacity.

The saved-route simulation predicted fewer reads and the actual runs confirmed
that direction. It did not predict latency or model compression. The first
preflight curve rounded to 4174 slots; the corrected curve uses 4175, with both
preserved. Its route source is the older complete 256-step capture, not this
short native run. No simulation result is treated as measured throughput.

[Protocol](../../qwen_memory_budget_stage.md), [raw summary](raw/summary.json),
[source-revalidated comparison](comparison.json), [phase and memory observations](observations.json),
[Python tests](python-tests.log). The whole stage took 243.05s.

Native build: `1ddec378e0a2550eda51c351e5e9315863c15b61773617c457b4d8e2f8df9189`.
Artifact: `b2c422f3c643e36f04227a64d61796b44a4b1029`.

Do not spend more memory on this candidate. Next measure the serial expert
ranking dependency at 12GiB, then test an exact parallel selection operator
before any expensive full-model qualification. The original 5 tok/s, 2K/4K,
7K and sustained coding-session targets remain unmet.
