# Lazy construction of ngram cache rows

The scratch-reuse control again hit compression before generation (53.094MiB
peak), with AC/normal thermals and 16.8GiB available at the initial probe.
NgramStore eagerly constructs approximately 49.5MiB of rows and reserves its map
at startup, although the short request uses only a small prefix of the ring.
Similar byte counts motivate a test; they do not identify compressed pages.

Isolate a source-copy change: reserve exactly the original row capacity, construct
one row on each first insertion while the ring fills, and keep the original
FIFO index and eviction once full. Row values remain initialized before use.
Never use allocator rounding to increase capacity. Keep the same advertised
64MiB cache allowance, exact lookup addresses, BF16 values, hit/miss decisions,
target/draft weights, expert slots, context and 12GiB joint admission.

First compare eager and lazy stores using real packed tables. Exercise irregular
chunks, duplicate rows, live hits, EOS history, a small ring with repeated eviction,
and the full reserved capacity with most rows unused. Compare all outputs, logical
slots, replacement index and hit/miss counters. Then run the existing forced-
rejection check with lazy initialization in both scratch-off/on arms and compare
both with the previously qualified eager full-model outputs and cache counters.

Only after those checks, resume the original short scratch off/on timing screen
with lazy initialization shared by both arms. This is a new baseline and a fresh
pair, never a speed comparison with historical eager runs. The original early
stop rules (2% first-pair gain, reverse-order confirmation), clean-memory/power/
thermal gates and longer qualification requirements remain unchanged. No claim
that lazy construction eliminates compression follows from a single clean pair.
Record constructed/reserved bytes at decode boundaries separately from physical
memory. Reserve bytes remain admitted even when not all pages are initialized.
