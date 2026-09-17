# Resident allocation follow-up

The scatter result remains `no_clear_native_gain` under the predeclared combined
gate: group one's wall-time upper bound is 1.03964, above 1.03. Do not change
that result. Both command geometries retain a clear GPU gain, however, and the
normal group-four path has wall-time ratio 0.67404 [0.65709, 0.69143]. The separate
direct condition also retains GPU and wall-time gains.

These measurements justify the already planned resident-allocation diagnostic:
the native scatter path alone does not reproduce the normal request regression.
Run `resident` once, with its own five pairs and unchanged protocol. Load and
register the real mixed artifact's resident buffers before timing, under the
same 12GiB bound. This is an investigation of the surviving GPU gain, not a
promotion of the inconclusive scatter result or a comparison pooled across
conditions. No SSD traffic or resident computation is introduced into timing.
