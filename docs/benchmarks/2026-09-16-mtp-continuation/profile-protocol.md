# Updated verifier dependency capture

Use a separate source-copy binary with the current expanded packed-Q8 target and
real trained MTP proposals. The initial diagnostic is restricted to the existing
72-token interval-merging prompt and four fully accepted four-token blocks.
A changed acceptance path stops capture; it must not silently omit recovery.

Require a sealed completed 16-token normal continuation as a numerical reference.
Its timing is never imported into the diagnostic. Compare every target logit hash,
committed token, final target and draft state, proposal and acceptance decision.
Bind the reference to its exact native producer, input and artifact identities.

Capture all target command groups and 48 expert dependency passes per block.
Drain and clear captures after each call, preserving the existing GPU submission
boundaries. MTP computation remains unprofiled. Reconcile dispatch and submission
counts against GPU counter snapshots immediately around each target call, so
draft dispatches cannot be misattributed to missing target work.

Retain 12GiB, 1,460/32 target/draft slots and the same arithmetic. Admit a separate
512MiB trace allowance inside 12GiB. Limit each block to 20,000 dispatches and
32MiB serialized evidence. No trace may report truncation. Join read timestamps,
ready experts, GPU activity and resource-release times using the existing audited
timeline analysis. Require clean memory, unchanged swap and nominal AC-power
conditions before using the capture to rank another experiment.

These are instrumented overlap observations. They do not establish normal
throughput, causes of waiting or recoverable latency. There is no promotion or
automatic kernel selection. Choose a subsequent change only after actual current
captures and the longer continuation screen support it.
