# Separate numerical diagnosis from an early timing decision

Both resident and streamed attempts completed exact nine-token serial checks
under Metal validation but recorded startup compression. The streamed run also
passed real embedding, eviction-before-submission, failed-read, destruction and
eight-row Q8 operator checks with clean memory. Its resident allocation is
642.15625MiB smaller, yet the process still recorded 33.594MiB compression during
priming. Keep both resource-blocked results unchanged.

Following the existing early-rejection approach, reuse only the sealed streamed
serial **numerical** reference. Run one nine-token width-eight diagnostic using
the same producer and Metal validation. Allow at most 128MiB compression solely
for this diagnostic; keep the 12GiB process ceiling, 13.5GiB admission, AC/thermal
checks and unchanged swap. Compare every logit and persistent state boundary
against serial. Resource-disturbed numerics do not qualify runtime adoption.

If exact, run one fresh normal four/eight pair over the same 64 known tokens,
without validation instrumentation. These timings must satisfy the unchanged
zero-compression/decompression and unchanged-swap gates. Reject eight if latency
does not fall by 15% or its perfect-proposal rate is below 6.5 verified tokens/s.
Do not retry disturbed timings unchanged. A promising result permits acceptance
cost modelling only; full clean qualification and real MTP remain outstanding.

This does not convert any diagnostic timing into performance evidence or turn
a known-continuation ceiling into generation throughput. The purpose is to avoid
spending more qualification time on a verifier that lacks adequate headroom.
