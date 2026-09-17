# Exact MTP attention-state catch-up

Hypothesis: the complete draft layer wastes work when consuming target-aligned
hidden states solely to catch up its private cache. Its only persistent state
is attention keys/values/index. Compute the two pre-FC branches, the attention
input mixer, and key/value/index projections, then commit the same positions.
No attention query, attention result, MoE, final mixer or head is needed there.
Autoregressive proposal generation retains the full trained draft.

Keep 1,460 target slots, 32 draft slots, the 12GiB plan, all weights, target
arithmetic, initial priming and greedy workload unchanged. The controlled change
is catch-up only, including grouping its corrected input rows into one bounded
microchunk. Do not mix these samples with the first joint screen.

Require the existing independent full-forward fixtures; exact full versus
state-only private state at four rows and each proper prefix; no expert-cache
activity on the state-only call; forced joint rejection; identical intermediate
target/draft state; identical future proposals, acceptance and final target state.
Run both validation arms with Metal API/shader validation. Timing has validation
disabled and uses fresh processes with alternating order. Every sample must
have clean compression/decompression/swap counters, nominal thermals and AC power.

Run the first 16-token control/candidate pair. If the candidate remains below
5 tokens/s, stop before expensive repetitions and report it as a short-screen
target miss. Otherwise complete five alternating pairs and report a paired log
ratio interval. A promising short result requires its upper 95% bound below one
and every candidate at least 5 tokens/s. Neither outcome qualifies long-context
performance, sustained use, non-greedy sampling or production promotion.
