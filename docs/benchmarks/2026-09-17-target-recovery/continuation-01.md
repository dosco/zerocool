# Continuation using the optimization tools

The existing `compare`, `opportunity`, `next`, `trial`, replay and ledger tools
were used for this continuation. No replacement tooling or production change
was introduced.

The comparison query revalidates the three earlier direct-output coding pairs.
The opportunity query reconstructs the same optimistic recovery-free bounds of
4.7572 / 3.8080 / 4.3419 tokens/s. Those bounds do not establish the performance
of state recovery.

## New attempt

[Trial 02](trial-02/summary.json) built a fresh isolated native candidate. The
[preflight](trial-02/fixtures/capture-recovery-host.json) admitted the full-model
capture at 14.6377GiB available memory, AC power, Low Power Mode off and nominal
thermals. An earlier read-only preflight had found 8.189GiB available.

The native capture completed and produced exact reference states for prefixes
1/2/3/4. The stage is nevertheless **failed and unqualified**: the source guard
detected concurrent edits to `target_recovery_checks.py`; inspection also found
that `trial_recovery.py` had changed. The native capture independently recorded
31,195,136 bytes (29.75MiB) of peak compression. Neither condition is waived.
The original trial, source identity, native output and ledger result remain intact.

## Diagnostic saved-state replay

The already built executable and generated source/object hashes were verified
against that trial's immutable producer. The fixture manifest and 1.075GiB
payload hashes were also checked before replay.

[Diagnostic replay](replay-diagnostic-01/summary.json) used the existing native
replay operation without loading the full model. All eight real-input checks
(prefixes 1/2/3/4, repeated twice) matched every persistent state buffer against
the separately computed full-target references. Geometry, coverage, invalid
prefix, corruption, cancellation, failure, delayed completion and final buffer
release checks passed. Peak physical footprint was 772,639,552 bytes (0.720GiB).

This is diagnostic correctness evidence only. The source capture is unqualified,
and replay itself recorded two decompressions despite zero recorded compressed
bytes. It therefore does **not** satisfy the recipe's clean fixture prerequisite.
No full-model candidate/control correctness pair or timing sample was run.

The diagnostic report is sealed, indexed and recorded as inconclusive in the
experiment ledger. A clean attempt must use stable source files and a fresh
capture; none of these observations may be pooled into its timing evidence.
