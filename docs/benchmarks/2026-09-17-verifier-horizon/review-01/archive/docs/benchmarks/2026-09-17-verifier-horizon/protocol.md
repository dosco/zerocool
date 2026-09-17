# Wider verification before another local kernel experiment

The current real-MTP screens reach about 4.34 tokens/s on interval merging;
width one is faster on the LRU prompt. The current clean target profile spends
120.60ms per committed output with a GPU command executing and 110.60ms in the
other overlap categories. Those instrumented intervals are not additive speedup
opportunities. A small gate/up change is parked while we test a larger question:
can more useful token rows share the cost of a complete target pass?

Compare four and eight **known-correct** continuation inputs per pass, with a
serial numerical reference. Extend packed Q8 across the independent token
dimension and use existing tile-eight Q4/generic kernels. Preserve scalar
arithmetic, reduction order, quantization and actual router selections. Extend
the existing direct output path to eligible one-row experts in an eight-row pass.
This is a verifier ceiling, not a larger production draft implementation.

Keep the 12GiB joint admission, 1460/32 target/draft slots, core-cache residency,
reference Q4, packed Q8, lazy ngram initialization, scratch off, two live command
groups and 8192 context. Allocate and prime the actual MTP object in every arm.
Then supply independently recorded greedy tokens without executing the draft.
Reserve the same eight-row target checkpoint and enlarged logit allowance in all
arms, within the original 128MiB checkpoint reserve and 12GiB total. Allocate the
existing four-row recovery journal but do not capture or execute it. Draft
checkpoint capacity stays four rows because draft state is never advanced here.

First check nine real-model inputs with Metal validation, at widths one, four and
eight. Compare every vocabulary logit hash with the existing clean reference and
all committed persistent-state boundaries with fresh serial replay. This covers
an eight-row block and a one-row tail. It does not validate eight-row rejection.

Then run a fresh four/eight pair over 64 known continuation inputs. Stop if the
eight-row pass fails to lower latency by at least 15% or reaches less than 6.5
verified tokens/s. A survivor gets one fresh reverse pair. These thresholds are
early rejection screens, not sufficient margins for real drafting: acceptance
and every draft/recovery cost must subsequently support 5 generated tokens/s.
Preserve missing or disturbed runs; no unchanged timing retries. Require AC,
Low Power Mode off, nominal thermal state, 13.5GiB available at full-model
admission, zero process compression/decompression and unchanged system swap.

If the ceiling fails, do not build a larger MTP proposer on this execution path.
If it passes, estimate required accepted tokens per block, then measure real
eight-row proposal yield before integrating recovery or starting qualification.
Keep a separate quality-calibrated expert compression path available if byte and
compute costs remain too high. Existing Q3 operator evidence and source-weight
requirements still apply; no automatic precision switch or Q4-to-Q3 promotion.

Other cheap findings from the broader reassessment: actual pinned input
embeddings occupy 675,430,400 bytes (644.14MiB), so streaming them cannot be
described as freeing several GiB. An exploratory causal suffix-copy replay over
the three existing short outputs found few long exact matches; it does not yet
justify replacing the trained MTP head. Input-grounded edit workloads remain a
distinct opportunity, motivated by [Prompt Lookup Decoding](https://github.com/apoorvumang/prompt-lookup-decoding)
and [SuffixDecoding](https://arxiv.org/abs/2411.04975). No external speed claim
transfers to this laptop or this model.

Raw reports identify the exact input, artifact, native base, generated producer
and resource observations. They never expose the diagnostic as ordinary
`tokens_per_second`. Production and the earlier sealed results remain unchanged.
