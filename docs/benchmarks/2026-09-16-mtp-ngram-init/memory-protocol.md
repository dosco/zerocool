# Startup memory attribution, separate from timing qualification

Two lazy full-model checks preserved exact reference output but still observed
compression before decode. Stop unchanged qualification retries. Capture one
read-only macOS VM map of the owned native process after target priming starts
draft priming. Run the same bounded 72-prompt/16-output workload with scratch off
and lazy ngram construction. Map inspection can disturb execution: no request
timing from this diagnostic qualifies performance or changes an earlier result.

Keep the original 12GiB physical-footprint bound, cache counts, 13.5GiB initial
availability, nominal thermal state, AC power, Low Power Mode off and unchanged
swap. Compression is an observed quantity here, capped at 512MiB as an operational
diagnostic limit, following the existing request-memory attribution workflow.
This is not a relaxed clean-memory performance pass. Do not silently convert the
capture into a normal speed test if it happens to be clean. Compare all resulting
tokens, logits and target/draft state with the existing sixteen-token reference.

The optional validation capture instead uses the existing eight-token forced-
rejection workload, with Metal API and shader validation enabled. Compare its
full outputs, intermediate recovery boundaries and target/draft state with the
previously qualified eager control. This attributes the validation path separately;
neither capture is a timing sample, and they are not a paired performance comparison.
After the first validation capture reported zero compression before decode but
25.1MiB after its first forced-rejection cycle, a follow-up may capture the first
`draft_verify` boundary with at least one committed token. Record this phase
choice explicitly. Startup maps cannot attribute compression that arose later.

Only inspect the child process created by this diagnostic. Bound native execution
to 160 seconds, map capture to 10 seconds, and the entire stage to 220 seconds.
Drain or stop that owned process on cancellation/failure. Preserve mapping errors
and missing phase capture as failures. Report allocation categories separately
from exact ownership: a region classification alone may not identify a C++ object.
