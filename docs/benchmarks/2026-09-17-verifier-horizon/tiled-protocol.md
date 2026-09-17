# Decouple the storage window and GPU compute tile

The clean 64-token early pair measured 5.6127 verified tokens/s at width four
and 4.3174 at width eight. Exact logits and final state match. Eight-row expert
cache hits are zero, versus 31.23% at four rows. Application expert reads rise
from 560.71 to 655.72MiB per verified token. GPU command execution per token
rises from 98.23 to 151.57ms, while routed-expert command time rises only from
37.00 to 39.65ms. These counters identify a wider compute problem as well as
cache thrashing; they do not isolate its cause or predict recoverable latency.

One bounded correction is warranted before closing wider verification: preserve
the eight-token storage/expert grouping, but cap GPU compute tiles at four.
Run the existing packed four-row Q8 arithmetic in two independent grid slices.
Use existing tile-four generic/Q4 kernels on the larger input set. Keep all
quantization, lane partitioning, scalar operations and per-token reductions.
Leave the 12GiB total, 1460/32 slots, exact streamed embeddings and other controls
unchanged. Do not combine this with a larger cache or an eviction policy change.

Recheck both Q8 geometries/FP32 modes and guards under Metal validation, followed
by all nine real-model row logits and serial state boundaries. Use a separate
source-bound numerical comparison across producers; never compare their timing.
Then measure a fresh four/eight 64-token pair in the new producer. Retain the
same 15%/6.5 verified-token/s early thresholds and strict normal-timing resource
gates. A failed correction stops this horizon. A promising result only supports
an acceptance/cost model, not integration or the 5-generated-token/s claim.
