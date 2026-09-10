# First confirmation attempt: incomplete memory admission

The first five-pair attempt stopped after **76.07 seconds**, with one completed
off conversation and no core conversation. The off run took 72.93 seconds; it
is not a complete pair and is excluded from confirmation estimates.

The next process's metadata inspection found 7,829,700,608 reclaimable bytes,
which could not admit the fixed 12GiB budget. The runner stopped and sealed its
evidence. It did not lower the budget or run the candidate.

A subsequent [metadata check](readmission.json), without inference or an OS limit
change, admitted the same 12GiB plan again. This establishes that available memory
recovered; it does not identify the source of the temporary pressure.

The runner was then changed to use the existing bounded admission helper: at
most three metadata checks, with 2- and 5-second waits, keeping rejected reports.
The helper now also respects the outer deadline. Only metadata is retried;
inference is never repeated by that retry loop. A new full capture starts from
pair zero and does not pool this unpaired control or the earlier short screen.

[Raw summary](raw/summary.json), [source verification](verification.json), and
the copies of the three tooling files preserve the incomplete attempt and the
code versions that preceded the admission change. All 176 Python tests passed
before this attempt; the later admission tests are reported with the new run.
