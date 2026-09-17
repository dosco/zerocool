# Resident operator investigation: Q8 lookahead rejected

Continuation: the previously blocked capture completed in `capture-02`.
See the [resident breakdown and packed-Q4 follow-up](../2026-09-14-q4-packed/README.md)
and its [request audit](../2026-09-14-q4-packed/resident-audit.json). The incomplete
attempt described below is retained; this Q8 screen still fails its gate.

The bounded real-input experiment is complete. Reading two iterations of
packed Q8 weights ahead of their arithmetic preserved outputs, but did not
provide a material gain. Keep the existing kernel. No native runtime,
quantization, cache configuration, or production default changed.

The requested full-model dispatch breakdown stopped at memory admission.
There is no fresh request timing or new per-dispatch attribution from that
attempt. The original 5 tokens/s and long-workload targets remain open.

## Exact Q8 operator screen

[Completed screen](lookahead-02/summary.json), [timing](lookahead-02/timing.json),
and [Metal validation](lookahead-02/validate.json).

One standalone candidate preloads two consecutive iterations of the current
two-row packed-Q8 kernel. Input summation, integer-code dot products, scale/bias
application, outer accumulation, and final SIMD reduction retain their original
order. Metal uses safe arithmetic. The reference shader is read from the
unchanged native source; the probe appends its candidate separately.

Six real single-token projections from GDN layer 0 and attention layer 31 were
replayed from the September 11 captures. Their original producer fingerprint
is retained, not relabeled as the current build. All 18 packed weight/scale/bias
payloads were also rehashed against the currently pinned mixed checkpoint.
Captured activations and weights were checked against their manifest hashes
before GPU submission. BF16-rounded and FP32 outputs are byte exact for all
six cases, with separate Metal API and shader validation.

Five alternating pairs, 32 repeated dispatches per sample, GPU microseconds:

| Projection | Current median | Lookahead median |
|---|---:|---:|
| GDN 2560 → 10240 | 174.63 | 173.34 |
| GDN 2560 → 6144 | 91.55 | 90.02 |
| GDN 6144 → 2560 | 98.88 | 94.16 |
| Attention 2560 → 12288 | 218.33 | 215.84 |
| Attention 2560 → 512 | 11.23 | 11.12 |
| Attention 6144 → 2560 | 98.40 | 97.49 |
| All six matrices in sequence | 770.28 | 775.44 |

The combined sequence's paired geometric ratio is **0.9738**, with a 95%
interval of **0.8358–1.1347**. This does not establish an improvement. All
pairs, including the unusually slow first control sample, remain in the report.

Multiplying isolated savings by matching layer/shape frequencies gives a
median of **0.47ms/token**, far below the predeclared 20ms screening threshold.
That is a rough operator projection, not measured or predicted request latency.
Only six matrices from two layers are present; repeated dispatches reuse
weight caches. The equal-sized K/V projection frequency uses the captured K
matrix as a timing proxy, not as correctness evidence for V. Width-four and
partial-iteration cases were not qualified. No runtime integration or lengthy
full-model qualification follows this result.

The completed stage took **3.78 seconds**, excluding compilation and the failed
setup attempt below. Shared GPU allocations total **107.99MiB**; the final
sampled process footprint was **224.08MiB**. No process compression,
decompression, or within-sample swap change was observed. Physical observations
are sampled, and swap counters are system-wide.

## Preserved incomplete attempts

- [capture-01](capture-01/summary.json): resource blocked before inference.
  The final admission observed **9.05GiB reclaimable** against **10.00GiB
  planned**, within the unchanged 12GiB maximum and 1072-slot configuration.
  Neither normal nor traced inference started. No smaller cache was substituted.
- [lookahead-01](lookahead-01/summary.json): failed before operator execution
  because the selected older capture directory retained its manifest but lacked
  payload files. No timing was produced. The runner now verifies fixture files
  before invoking the GPU. The completed attempt uses an intact earlier capture
  set; no missing activation was synthesized or relabeled.

## Reproduction and verification

Compile the standalone probe:

```sh
clang++ -std=c++23 -O2 -fobjc-arc -framework Foundation -framework Metal \
  scripts/qwen/probe_q8_lookahead.mm -o /tmp/freellm-q8-lookahead
python3 scripts/qwen/screen_q8_lookahead.py --binary /tmp/freellm-q8-lookahead \
  --output FRESH_DIRECTORY
```

The runner owns the GPU lease, verifies the source/artifact identity, enforces
a 128MiB shared-buffer limit, and separates validation from timing. Old fixture
files must remain available. The native engine fingerprint is unchanged:
`81e6f5ced54d2802716bd5dd211e47428da0d39d65d55b8fe7564ce94bd26b79`.

The [read-only audit](verification.json) reconstructs decisions from sealed raw
reports and preserves both incomplete attempts. [Python checks](python-tests.log)
passed all 257 tests. Probe sources are preserved in `screen-sources`; the
failed attempt's earlier runner is in `initial-sources`. Native source for this
fingerprint is also archived in the preceding cache/residency report.

Next run `profile_resident_operators.py --output FRESH_DIRECTORY` when the
unchanged allocation is admitted. It compares a normal conversation with a
four-step per-dispatch diagnostic for both initial and retained-append decode,
rejects missing counters or layer coverage, and keeps instrumentation costs
separate. Use that breakdown to select the next complete resident-block
optimization. Do not repeat Q8 lookahead, the small narrow-projection probe,
or cache enlargement as though they were untested routes to 5 tokens/s.
