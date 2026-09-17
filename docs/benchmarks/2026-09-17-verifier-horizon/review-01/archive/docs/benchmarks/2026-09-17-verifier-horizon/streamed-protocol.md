# Explicit memory intervention: stream exact token embeddings

The resident horizon attempt passed its bounded eight-row Q8 checks and all
nine serial target logit checks. It failed resource admission for evidence after
the process recorded 30,212,096 compressed bytes and 280 decompressions during
startup/priming. The host had 18.66GiB available at admission and about 9.13GiB
available after priming. Repeating that run unchanged is not the next experiment.

The pinned mixed input-embedding table has 635,699,200 weight bytes and two
19,865,600-byte metadata tables: 675,430,400 bytes in total. The allocator charges
675,446,784 bytes after per-buffer alignment. Each selected row requires only
2,720 original bytes. Preserve the original Q8 codes, BF16 scales and BF16 biases
in a shared target/draft FIFO cache of 256 rows. Admit a fixed 2MiB host allowance.
Read a missing row's three ranges before publishing it; duplicate token lookups
hit the same cache. This first implementation uses the existing uncached files;
it does not create another large prepared artifact.

Gather selected packed rows into bounded per-call Metal buffers and call the
existing embedding kernel. The GPU buffers own their copies through completion;
subsequent CPU cache eviction cannot overwrite them. Preserve exact BF16 output
and fourfold target replication. Reject IDs and unsupported layouts before reads.
The experiment caps embedding batches at 128, matching this verifier's target
priming and MTP paths. No general production panel integration is claimed.

Reduce resident admission by the removed aligned table minus the 2MiB allowance.
Keep 1460 target slots and 32 draft slots. The freed memory remains headroom;
this stage does not conflate streaming with larger expert capacity. Both four-
and eight-row timing arms use this same storage policy, start from identical
priming, and must retain the original 12GiB total and resource gates.

Before a full model, compare against the complete resident table using real
checkpoint bytes. Cover repeated IDs, vocabulary endpoints, 128-row batches,
FIFO eviction, target replication, invalid IDs and destruction. Then rerun the
same nine-token width 1/4/8 numerical checks and the predeclared horizon screen.
An earlier compressed run remains compressed; it supplies no timing comparison.
No memory or speed improvement is established by allocation arithmetic alone.
