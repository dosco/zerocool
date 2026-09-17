# Longer real MTP continuations

Keep the existing mixed target, prepared trained MTP head, exact target arithmetic,
12GiB admission, 1,460 target slots, 32 draft slots and 8,192-token state capacity.
The new harness extends only developer experiments. Production remains unchanged.

First finish the pending state-only recovery comparison with the original
producer when at least 13.5GiB is currently available. Preserve any resource stop;
do not reduce the fixed cache or memory criteria to obtain a result.

The continuation harness uses the same trained forward and state-only catch-up.
Bound prompt ingestion to 128 target rows and 16 draft rows. Process each hidden
panel before reuse, including the exact next-token alignment across panels. Keep
only the final target hidden row for the next proposal. Prompts are capped at
512 tokens for this first stage; 2K/4K/7K requests remain a later qualification.

Validate the unchanged draft equations with the independent four-row reference,
native input bounds and rollback tests. An eight-token forced-rejection replay
compares full versus state-only catch-up, including proposals after recovery.
Each normal continuation is also compared with a fresh serial target process:
every committed token, full-vocabulary logit hash and final target state must agree.

Measure three natural coding prompts (interval merging, an LRU cache and async
retry/backoff), initially 128 tokens each. They use the verified tokenizer and
the same explicit non-thinking chat wrapper as the short screen. Alternate arm
order across cases. EOS stops both arms; early EOS is reported and cannot count
as completing the requested length. No supplied continuation is used to propose
tokens in normal timing. Record proposed/accepted counts, rejected blocks, draft,
verification, recovery and full cycle time. Stop resource-disturbed measurements.

The measured count remains committed target inputs, matching the short-screen
convention. It includes the initially known next token and its final forward
pass. Priming/first-token latency is separate. All checkpoint, drafting,
verification and rejection recovery work is timed. Full-vocabulary evidence
hashing and periodic report writes occur outside cycle timing; report the wall
interval including them separately. No normal API latency claim follows.

One comparison per distinct workload is screening evidence, not paired confidence
or coding-quality qualification. Five repetitions, 2K/4K prompts, 7K reporting and
the sustained tool workflow remain required before promotion. Profile the updated
target path only after these observations identify a useful next experiment;
older instrumented costs are not presumed to describe the new kernels.
