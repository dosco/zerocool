# Fresh timing after completed numerical validation

The two recovery validations in `recovery-02` completed and their full target
and draft states, forced-rejection recovery and proposal records compare exactly.
The control was clean; the candidate had a 32KiB peak of process compression.
The existing report remains `resource_blocked` and none of its timing is reused.

Prospectively separate numerical qualification from performance cleanliness, as
in the earlier real-weight capture/replay experiments. The new timing stage may
consume only completed, sealed, numerically identical validation outputs from
the exact same native producer, artifacts, memory plan and workload. It does not
convert disturbed validation into a clean memory or latency result.

All normal timing processes are fresh, with Metal validation disabled. The
original strict gates remain: zero observed process compression/decompression,
unchanged swap, AC power, Low Power Mode off and nominal thermals. Keep the same
12GiB plan and 1460/32 target/draft slots. Alternate control/candidate order;
stop if the first candidate is below5 tokens/s; otherwise finish five pairs.
No prior sample is pooled, and no source report is rewritten.
