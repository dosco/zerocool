# Proposed experiment: complete hyper-connection fusion

Status: implemented as a standalone probe and rejected by the
[completed screen](../2026-09-15-hyper-fusion/README.md). Outputs match exactly,
but the complete cycle is 1.10% slower and its frequency projection is −0.316ms/token.
No native integration follows. The proposal below records the predeclared
experiment; earlier Q4 request rejection and qualification criteria are unchanged.

## Hypothesis and exact scope

The largest GPU command class includes resident work before routing. Test
whether fusing the layer hyper-connection up-projection, sigmoid and four-stream
mix reduces complete-block time through fewer intermediate writes and dispatches.
The prior Q8 load-ahead and narrow BF16 load4 experiments did not test this
composition. Do not repeat either as this candidate.

The current mixed artifact uses Q8 group-64 matrices of 10240→320 (down) and
320→10240 (up), plus the unchanged BF16 10240→4 injection. The reference path
performs norm, down-projection, activation, up-projection, sigmoid, four-stream
mix and the independent injection path. Keep the whole block in the timing
comparison, including its norm, down, activation and injection work.

For the single-token candidate, compute four up-projection rows belonging to
the same output feature, at row addresses `h*2560+i` for `h=0..3`. Preserve:

- Each original Q8 width-four lane partition, scalar accumulation and SIMD sum.
- The BF16 rounding between the projection and sigmoid and within sigmoid.
- Each BF16 multiplication/addition and the stream order `h=0,1,2,3` in `hc_mix`.
- The injection branch, input/output ownership, cancellation and output shape.

Apply only to the attention and MLP hyper blocks in the 48 layers: 96 blocks
per generated token. Leave the final model mixer and every multi-token path
unchanged. This removes exactly two dispatches and two 10240-element FP32
temporary vectors per block: 192 dispatches/token and 7.5MiB of
logical temporary payload across a full token (9MiB with native allocation rounding
if all buffers are independently retained). Verify actual buffer lifetime
accounting; theoretical eliminated vectors are not measured footprint savings.

## Smallest useful test

1. Verify availability and hashes of four real complete-block fixtures:
   attention/MLP inputs from one GDN layer and one attention layer. Capture
   missing inputs in one bounded developer pass; never synthesize a missing
   real-model fixture or relabel its producer identity.
2. Build the fused kernel only in a standalone developer probe. Use the same
   real weights and inputs for reference/candidate. Compare full block and
   injection outputs byte-for-byte with separate Metal validation. Also check
   BF16 edge cases and that input/neighbor output storage remains untouched.
3. Freeze protocol, payloads and tool sources. Run five alternating pairs of
   complete blocks with validation/profiling off, within a 60-second GPU limit
   and 256MiB shared-buffer bound. Report whole-block CPU/wall and GPU time;
   do not time only the eliminated operations or subtract unchanged work.
4. Use the pair as the statistical unit. Stop if outputs differ, memory is
   disturbed, the upper 95% bound does not establish a block-wall gain, or
   even a declared frequency-weighted projection misses 20ms/token. Projection
   remains an operator-screen heuristic, not predicted request savings.
5. Only a survivor gets native experimental integration, a short normal
   request screen and fresh full-model state/failure validation. A short screen
   precedes expensive long-context qualification. Production promotion still
   needs clean paired complete-request evidence and the original acceptance.

The current command trace cannot isolate the proposed fusion's cost. Older
instrumented pass durations put the up projection and mixing far below the
entire resident class, and those intervals overlap. Therefore a small or zero
gain is plausible; this experiment must be cheap to reject. It is not presented
as a single change that will bridge the remaining 74–98ms/token target distance.
