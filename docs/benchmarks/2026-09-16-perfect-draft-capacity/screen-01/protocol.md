# Four-token verifier cache capacity — declared before execution

Hypothesis: a four-token block's approximately 1,234-record working set thrashes
the 1,072-slot global CLOCK cache. Compare 1,460 slots at the same 12GiB total
allocation. The earlier serial capacity rejection remains unchanged. No prior
timing samples contribute to this experiment; older reports supply only fixed
tokens, identity and the original serial-prime arithmetic anchor.

Use the existing standalone source-copy verifier, mixed artifact and prepared Q4
records, 8,192-token context, 512-token panel, 128-token chunk, core/cache
residency, deterministic fixed-lease priming and completion-driven decode.
Retain every existing arithmetic, byte-exact logit, route, persistent-state,
rollback, live-resource, memory and host check. Add an explicit 1,072/1,460-slot
argument; production source, executable, weights and defaults do not change.
Every arm reserves the same capacity-four checkpoint and logit workspace.
Charge these against the actual admitted 12GiB plan. No OS or precision changes.

Fresh Metal-validated processes, in order: serial/1,072, width4/1,072,
width4/1,460. Compare complete vocabulary logits and every matching state/route
prefix, causal-prefix independence, zero-accept rollback and all partial-prefix
recoveries against the fresh serial process. Require prime mathematics to match
the sealed earlier serial anchor. Require identical initial cache hashes within
each capacity, including validation and timing runs. Different capacities may
have different cache hashes; this is the only priming comparison exemption.

Then time fresh processes in this order (capacity, width):

1. Round 0: (1,072, 4), (1,460, 1), (1,460, 4).
2. Round 1: (1,460, 4), (1,460, 1), (1,072, 4).

Each consumes the same sixteen continuation inputs after a 72-token prompt.
Proposals are free and always correct, with no bonus token. Checkpoint copies,
forward execution and greedy acceptance are timed. Evidence scans occur at the
same consumed-token boundaries. Require exact outputs/state/routes throughout.
Record cache hits/misses, application bytes, GPU time and wall time separately.

Stop at the first clean width4/1,460 timing below 5 verified tokens/s: the
predeclared all-candidate-runs floor cannot then pass. Mark this a completed
early screen with `paired_comparison_complete: false`, preserving all collected
rows; do not compute or claim a complete paired speedup. Never apply this
negative decision to resource-disturbed or incomplete native processes. Stop
with an incomplete resource-blocked/failed report if admission, host, memory or
correctness fails. A missing measurement is not a weak candidate measurement.

If all six timing processes finish, require both larger-cache width4 runs to
exceed or equal 5 verified tokens/s and improve wall time over both their
same-round small-cache width4 and larger-cache serial controls. This is a
two-pair directional gate with no confidence-bounded promotion. Report every
paired ratio, even if unfavorable. A pass authorizes joint draft memory design
only; a fully resident prepared MTP head no longer fits beside this target cache
at 12GiB. Real drafting cost and acceptance remain unmeasured.

Hold the existing exclusive GPU experiment lease. Cap the stage at 600 seconds,
validation processes at 150 seconds and timing processes at 90 seconds. Perform
the native no-model power/thermal check before every model process. Require
nominal thermal state, Low Power Mode off, stable power source, zero observed
process compression, unchanged decompressions and unchanged swap at native
lifecycle observations. Freeze source, executable, artifact and protocol
identities, preserve raw failures and independently reconstruct the decision.

This short diagnostic does not qualify normal generation, long contexts,
non-greedy sampling, production session/RNG rollback, tool workflows or sustained
coding. Application reads are not observed device traffic. Aggregate cache
counts cannot establish a per-block capacity threshold or critical-path timing.
