# Keep native execution independent of unrelated checkout rebuilds

Registered after fixed-numerical-03 refused to start because the shared native
library, command-line executable and source fingerprint changed during separate
chat/UI work. Do not overwrite those changes or relink the experimental producer.
The original producer executable, generated sources and objects remain intact.

An explicit execution capsule copies that exact executable and its existing
native host probe. Their SHA-256 values must match a sealed earlier validation's
producer and frozen identity. Preserve the original producer metadata and native
build identity. Record the runner's current workspace identity separately. Metal
shader source is embedded in this producer, not loaded from the live checkout.

Before every process, verify capsule provenance, executable hashes, actual Python
runner modules, protocols, workloads and unchanged model-asset receipts and
payload fingerprints. Retain the same GPU lease, disk and memory admission,
thermal/power checks, timeouts, process cleanup and output sealing. The ordinary
current-source build path retains its existing checks.

This changes execution packaging only. It permits resuming the same producer's
numerical cases after rechecking every imported sample; it never reuses timing.
Require the complete six-pair matrix and its independent original-producer
numerical comparison before the existing preliminary storage-cost screen.
No precision, cache capacity, runtime scheduling or production-default change is
authorized by packaging an executable. A changed capsule or payload must fail.
