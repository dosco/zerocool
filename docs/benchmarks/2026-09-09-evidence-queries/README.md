# Offline evidence queries: first implementation

This is the original implementation snapshot. The [follow-up audit](../2026-09-10-evidence-audit/README.md)
records comparison and importer fixes, 116 passing tests, and unchanged historical
Q8 measurements. Original JSON examples and verification hashes below are retained.

The [query tool](../../qwen_evidence_queries.md) imported **409 distinct JSON/JSONL
documents from 465 paths**, with no import issues in the selected directories.
Identical copies share an ID. The local SQLite index can be rebuilt; the raw
reports and two retrospective ledger entries remain independent of it.

All **105 Python tests pass**, including 15 new query/index/ledger tests. They
exercise mismatched artifacts, workloads, budgets, sampling, builds and profiling;
missing, duplicate or reordered pairs; tampered summaries; changed raw reports;
malformed and oversized imports; trace limits; missing physical measurements;
ledger rebuilding; and refusal to overwrite an unrelated SQLite database.

The native build remains
`c4f983f03f28974dd2c4a935d6d5233f1b28a1f5e7cb4800fe7529a64c058c2b`.
Every original run-03 frozen source, binary, tooling and asset dependency was
rechecked unchanged. No inference workload or new instrumentation was added.
[verification.json](verification.json) binds the code, test log and example
answers. These are queries of historical measurements, not fresh performance
benchmarks.

## What the existing evidence answers

The Q8 rows-of-two experiment illustrates why cached and normal timings stay
separate. These are medians of candidate/control paired ratios:

| Scope | Recorded pairs | Latency ratio | Interpretation |
|---|---:|---:|---|
| Cached full-token replay | 5 | 0.6901; paired 95% interval 0.6758–0.7108 | About 31% less cached decode time |
| Normal 2K request | 2 | 0.96695 | About 3.3% lower request latency; no confidence claim |
| Normal retained append | 2 | 0.87354 | About 12.6% lower request latency; no confidence claim |

Normal first-token medians were essentially unchanged. [cached.json](cached.json)
and [normal.json](normal.json) include original sources, raw-report verification,
the controlled difference, correctness scope and limitations. No query promotes
the candidate or establishes coding quality.

[memory.json](memory.json) shows original continued/fresh state snapshots,
separating memory plans, physical footprint, live GPU groups, retirements and
cache/scratch accounting. Missing boundaries remain missing.

[timeline.json](timeline.json) returns one captured 512-token panel at layer 0,
including eight of its 447 captured expert lifecycles. Its original phase is
`unspecified`. The command trace says it was not truncated, while detailed
dependency capture is limited to 48 passes and 8,192 read records per phase.
Whole-request coverage and classified blocking reasons are not established.

[history.json](history.json) finds both retrospective ledger entries. The selector
attempt remains blocked by memory admission, with no timings and no performance
rejection. [next.json](next.json) asks for matched normal dependency evidence
before ranking cached hotspots; its possible request benefit is null.

Cache-size simulation, cross-build comparison, runtime event IDs and new memory
lifecycle probes remain follow-ups tied to specific unanswered decisions.
