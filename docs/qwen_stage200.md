# Current-build experiments toward 200ms/token

This stage implements the approved bounded plan: a cache-stability comparison,
event-driven pressure release, a fresh dependency profile, one substantial
exact-weight candidate if the new evidence supports it, and an isolated Q3
expert probe. No experimental option is a production promotion.

Implementation and bounded runs are complete; see the
[measured outcomes and preserved failures](benchmarks/2026-09-13-stage200/README.md).
The exact optimization branch did not obtain sufficient evidence to proceed.

The fixed control is mixed 4/8-bit, prepared Q4 experts, 12GiB, scratch reuse,
packed Q8 rows 2, SIMD routing, CLOCK, eight readers, ready groups of four,
panel 512 and microchunk 128 on the actual 32GiB M1 Pro.

## Bounded entrypoints

Use the pinned reference Python environment and a fresh output directory:

```sh
python scripts/qwen/stage200.py capacity --output FRESH_DIRECTORY
python scripts/qwen/stage200.py pressure --slots 1460 --output FRESH_DIRECTORY
python scripts/qwen/trace_decode_target.py --current-build --slots 1460 --output FRESH_DIRECTORY
python scripts/qwen/trace_decode_target.py --current-build --slots 1460 --prompt-tokens 2048 --output FRESH_DIRECTORY
python scripts/qwen/stage200.py q3 --output FRESH_DIRECTORY
```

Capacity compares 1848/1460 in two alternating pairs (600s total, 150s/process).
If neither is clean, one explicit 1460/1072 follow-up is permitted. Keep 1848 if
both arms are clean; select the smaller clean arm provisionally only within a
5% median conversation-time cost. Compression-disturbed controls remain valid
stability observations, not clean compute-speed evidence.

`--memory-pressure-policy observe|shrink` defaults to observe and preserves the
existing emergency safeguard. The callback only publishes event bits/counters.
At safe boundaries the coordinator drains users before evicting up to 388 slots
on warning, or halving on critical, with a 32-slot floor and no automatic regrowth.
Warnings are rate-limited to one reduction per second; critical can escalate
immediately. State and actual releases are recorded. The separate fixed-capacity
pressure experiment reports `not_exercised` if no real shrink occurred.

Version 2 tracing captures the first 16 decode forwards of initial and retained
append requests. `--profile-decode-only 1` excludes ingestion dispatch records
without adding GPU waits or command boundaries. Truncation cannot pass. Original
trace protocols remain readable. Observed overlap categories are not causal
speedup estimates; investigate opportunities of at least 20ms/token. A survivor
must pass operator/lifetime checks, two normal pairs, all-layer state/recovery,
and five fresh confirmation pairs before product qualification.

## Q3 probe contract

`qwen_q3_probe` is a developer executable; production artifact loading is unchanged.
Capture eight actual one-token inputs and two selected Q4 expert records in layers
0, 16, 32, 47. Separate tuning and held-out prompts use disjoint source token windows.
Capture uses at most 1,460 slots; `--slots 1072` selects the smaller admitted cache.
An input row need not have selected both fixture experts; this is operator coverage.
The capture replays 72 prompt tokens and eight recorded continuation tokens
through the complete model. It is instrumented and teacher-forced, not a free
generation or inference benchmark. Production EOS handling is unchanged.

Each 64-weight Q3 group has 24 little-endian packed code bytes followed by BF16
scale and bias. Quantization uses deterministic min/max, rounded metadata and
nearest-even clamped codes. Constant groups use zero scale. Non-finite or
unrepresentable metadata fail. Q3 gate/up and unchanged Q4 down use the reference
rounding points and reduction structure. Input fixtures derive from existing Q4;
they cannot establish original-to-Q3 model quality.

GPU runs have a 512MiB allocation limit, a 2GiB process peak limit and a 256MiB
evidence limit. The probe
compares identical preallocated output buffers and full gate/up/down chains at
1,2,4,8 rows. Validation is separate from five alternating timing pairs. Independent
CPU codec/operator checks and exact expansion into existing Q4 arithmetic cover
decoding. Advance only with at least 14% padded record savings and a one-token
paired 95% latency-ratio upper bound at most 1.03. Compression-disturbed or missing
memory measurements cannot advance. A failed tuning screen stops before held-out
capture. No quality or request-speed claim
follows from that decision; full original-weight conversion and held-out quality
gates remain a later stage.

## Product acceptance remains unchanged

5 tokens/s over 256 outputs at 2K and 4K prompts; initial 2K TTFT at most 60s;
128-token append to retained 4K TTFT at most 10s; 7K reporting; 20-minute coding
workflow without progressive memory or sustained swap growth. Full-model Q3
would additionally require the existing paired NLL/coding-quality gates.

Raw reports are sealed, source/build/artifact/workload identities are frozen,
GPU work uses the shared lease, and missing or blocked runs remain incomplete.
