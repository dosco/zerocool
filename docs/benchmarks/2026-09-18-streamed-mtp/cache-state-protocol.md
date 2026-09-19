# Numerical cache invariance and controlled timing

Registered after numerical-04 case four stopped at the initial-cache comparison,
before any normal timing run. Preserve that failed attempt and its raw files.

The resident and row-storage processes have identical target-cache state and
identical draft-cache counters after priming. Their draft eviction-state hashes
differ. That hash includes slot order, CLOCK references/hand, pins and readiness.
Draft priming uses completion-driven expert scheduling, so a different read
completion order can change this derived cache state. The completed pair has
identical primed model state, every draft proposal and logit, every verified row,
committed outputs, state boundaries and final model buffers. It is not a timing
comparison.

Numerical comparisons must preserve artifact, producer, workload, cache capacity,
policy and all model-state/arithmetic controls. They record initial cache-state
equality separately; differing derived eviction state does not invalidate exact
logit/state agreement. This checks the required invariance to cache behavior.
It does not permit changed precision, omitted experts or a numerical tolerance.

Normal timing comparisons still require identical initial target and draft cache
hashes, plus every existing resource gate. A cache-state mismatch stops that
comparison; neither its timing nor a selected retry can establish a speed claim.
The six-pair numerical matrix is still required. Diagnostic compression remains
unqualified for clean correctness, and no production promotion follows.
