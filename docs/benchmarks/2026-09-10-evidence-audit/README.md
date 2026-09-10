# Evidence-query audit

The follow-up review found and fixed errors in the offline query layer. It did
not change native inference or any original qualification report.

| Reproduced problem | Corrected behavior |
|---|---|
| One raw pair repeated under five pair IDs produced confidence bounds | Every normal observation must bind a distinct full SHA256 raw report |
| Summary rows could relabel a prompt as an append or omit a declared workload | Native workload labels and declared coverage must match |
| A hash prefix could stand in for a frozen raw-report digest | Summary dependencies require complete SHA256 hashes; user query selectors still accept prefixes |
| Contradictory failed/complete reports could be compared | Failed or unfinished evidence cannot establish a successful comparison |
| Adding unknown metadata to a blocked source allowed a positive ledger decision | Benefit/adoption requires completed, nonfailed evidence; rejection requires an explicit terminal result |
| An explicitly requested changed path could silently use an earlier-sorting copy | Check the requested path first and report its failure before falling back to an exact copy |
| JSONL `show` failed on multiple lines | Return compact, verified trace metadata |
| Explicit null memory fields failed; omitted trace records became zero | Preserve null measurements and unknown record coverage |
| Duplicate JSON keys and overflowing numbers were accepted; a growing file bypassed the stat-only bound | Strict parsing and an actual bounded read |
| Incremental import kept projections from an older extractor | Recompute projections on reimport; descriptions verify and extract original content |

New regression cases were run against the pre-fix implementation and reproduced
13 assertion failures across nine tests, plus a separate null-memory error.
The completed suite passes **116 tests**, including **26 query tests**.
[The pre-fix output](regressions-before.log), [null-memory reproduction](null-before.log),
and [passing suite](python-tests.log) preserve that evidence. These use synthetic
fixtures to exercise rejection paths, not fabricated performance measurements.

The live index rebuilt **416 distinct documents from 472 paths** without import
issues. The original Q8 cached and normal comparisons still pass revalidation;
their entire numeric comparison results are unchanged from the original example
answers. The normal experiment remains a two-pair screen with null confidence
bounds. JSONL `show` successfully describes the existing 48-line route trace.
The profile timeline still reports bounded, partial dependency coverage.

[verification.json](verification.json) binds the current code, test outputs and
live queries. Every original run-03 frozen dependency was checked unchanged.
These fixes strengthen evidence handling; they establish no new inference speedup,
coding-quality result or production promotion.
