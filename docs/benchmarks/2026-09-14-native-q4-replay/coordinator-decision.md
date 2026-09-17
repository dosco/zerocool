# All-hit coordinator follow-up

Declare this condition after the first three reports, before its timing. Keep
their original gates, source snapshots and results unchanged.

Resident allocations alone preserve packed-Q4 GPU and wall-time gains. Existing
command traces show predominantly one-expert submissions despite a ready-group
limit of four. Test the remaining coordinator boundary before adding heavier
resident computation or SSD contention.

Use the actual `execute_experts` path, eight real ExpertCache slots, eight I/O
workers and real resident allocations. Prime all records from verified fixture
files before timing and check their bytes. Every measured arm must have exactly
256 ready hits, no misses or joins, no read calls or read bytes, and the same
gate/down/scatter counts and submission counts as the original replay. Keep the
same five alternating pairs, group limits one/four, output guards, 12GiB ceiling
and process/stage deadlines. No per-pass profiling.

Both the fixed replay's `Metal::wait` and the production coordinator already use
completion events. The intervention adds cache leases, ready scans and the
coordinator's own completion/reap loop. Record its waiting separately from
Metal's CPU-wait counter; zero Metal wait must not imply no waiting.

This condition has all experts ready together, so it still does not reproduce
the distribution of completion arrivals or mixed shared/expert command groups.
The scratch scope remains one eight-expert batch. It tests coordinator overhead
with ordinary numerical behavior; it does not qualify normal request speed.
If the gain survives, add real read-arrival behavior as the next condition. No
unchanged full-request rerun or larger-cache change is part of this diagnostic.
