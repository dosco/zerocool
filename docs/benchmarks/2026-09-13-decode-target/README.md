# Generation target measurement is ready; live capture is memory-blocked

The current priority is reducing normal generation to **200ms/token**. The
[updated stage](../../qwen_decode_target_stage.md) defers the small projection
experiment, measures the exposed generation delays, and screens one candidate
with a plausible saving of tens of milliseconds before long qualification.

The [baseline analysis](baseline.json) binds every row to the prior five-pair
candidate's raw report and hash. Initial generation is a median 278.50ms/token;
generation after append is 296.19ms/token. Expert reads are respectively
564.45MB and 649.38MB per token. GPU command durations and coordinator waits
overlap, so their sums cannot identify the critical path.

The new collector pairs an unchanged, uninstrumented request with a 16-token
decode trace using existing command-group and expert-read timestamps. The
analyzer verifies full layer, expert, dispatch and token coverage, partitions
time without double counting, and separately measures submission gaps after
expert GPU completion. It does not label overlap as a causal explanation or
predict a whole-request speedup. Original command groups and native arithmetic
are unchanged. The normal/traced ratio exposes the measurement's overhead.

The first attempt stopped after three bounded metadata admission checks:
`resource_blocked`, `complete: false`, **no inference or timing samples**.
Only about 7GiB was reclaimable; the fixed 12GiB allocation needs 13.5GiB with
the safety margin. The attempt did not reduce its budget or load the model.
The collector subsequently gained stricter source validation and an explicit
17-output validator; the archived attempt predates those tooling changes and
remains unfinished. No native sources changed.

[Blocked attempt](blocked/summary.json), [admission](blocked/normal.admission.json),
[Python verification](python-tests.log). The tests cover interval unions,
duplicate overlap, clipping, missing layers/experts/commands, mismatched
identities, reversed timestamps, ordering independence, short-output validation,
and intervening attention work that must not inflate the boundary gap.

After freeing sufficient host memory, run:

```sh
.cache/qwen-reference-venv/bin/python scripts/qwen/trace_decode_target.py \
  --output .cache/benchmarks/decode-target-20260913-retry1
```

The five-minute cap, 12GiB budget, 1848 slots, exact token checks and source
freezing remain enforced. Use a fresh directory and retain this blocked
attempt. Capture results choose the next optimization; no runtime candidate
has been selected or implemented from this unfinished measurement. The
5 tokens/s target at 2K/4K remains unqualified.
