# First capped selector screen

The 600-second cached 4K screen stopped before inference after three bounded
metadata-only admission attempts. No timing measurement or performance decision
was produced. The native runner reports `resource_blocked`; the temporary outer
launcher reported generic `failed`. Both original reports are copied unchanged.
The reusable triage command now handles `ResourceBlocked` explicitly.

Admission found approximately 2.57GiB available against an 8.70GiB minimum. The
comparison also requires the full, unchanged 12GiB budget. Brave was the largest
memory user in a subsequent process scan; its summed process RSS is not unique
physical memory. No applications were closed and no system limits were changed.

Recovery and capture prerequisites were verified from the original run-03 seals
and source/build/asset identity. The attempt launched only the cached 4K phase,
with Metal API/shader validation disabled and a 600-second outer deadline.

[invocation.json](invocation.json), [native runner summary](cached-run/summary.json)
and [provenance.json](provenance.json) preserve the original outcome and hashes.
This blocked attempt does not reject the selector on performance and does not
qualify any speedup.
