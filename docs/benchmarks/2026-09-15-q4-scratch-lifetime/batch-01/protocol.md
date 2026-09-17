# Shared and routed scratch lifetime over 48 passes

Declared before native timing. The preceding shared-arrival replay preserved
the packed-Q4 GPU gain with actual shared work and verified SSD reads. Test
whether retaining its temporary buffers across 48 passes suppresses that gain.
Production inference, arithmetic, weights, model state, and defaults remain
unchanged. This is a developer diagnostic, not a full model forward pass.

Run two separately sealed conditions, forward first and fresh batch second.
Within each condition run five alternating reference/packed pairs at group caps
one and four. Every arm, warmup, and post-timing exact check uses six cycles of
the existing eight fixture batches: 48 shared chains and 384 routed experts.
Do not pool scopes or reuse previous timings. Packed/reference pairs within
each scope are the statistical units; cross-scope absolute timing differences
are contextual rather than a paired causal estimate.

Batch scope begins and ends scratch for every pass. Forward scope begins once
before the first shared chain and ends after the last routed work completes.
Both scopes drain GPU and I/O users between passes. File preparation, shared
output-view release, and fixed expert-cache reuse take place at those drained
boundaries and must leave scratch ownership and allocation unchanged.

All 27 temporary requests per pass have 16KiB allocation charges: three shared
outputs and three outputs per each of eight routed experts. Batch scope retains
27 buffers /442,368 bytes; forward scope retains 1,296 buffers /21,233,664 bytes.
Both use the same 128MiB arena ceiling. Warm the complete 48-pass scope before
every arm. Require zero new measured Metal allocations and 1,296 scratch reuses
per arm, identical allocation bytes before/after each arm, and a drained,
inactive arena at arm boundaries. Release the arena before final reporting.

The real short-request arena measures 3,201 buffers /78,004,224 bytes in the
source-sealed combined-Q4 request. This test covers only 40.5% of its buffer
count and 27.2% of its bytes: it omits two selected experts per layer and the
surrounding attention, recurrence, residual, routing, embedding, and logits
temporaries. It repeats four shared-layer and eight expert fixtures; it does
not replay 48 different layers, real token evolution, full state, or full cache
capacity. A retained gain cannot rule out the complete scratch environment.

Keep actual Q8 shared gate/up, Q8 down, and BF16 gate before the existing expert
coordinator, with no added submit or wait. Preserve the preceding independent
CPU reference, selected tensor hashes, layer/row mapping, native output hashes,
and destination canaries. Check all 32 CPU cases on process setup, every batch
in check/trace/warm/post-timing, and the final measured batch in timing. Require
byte-identical native shared and routed outputs in both Q4 variants and scopes.
The prior native full-state proof remains required; it does not qualify a
production packed-Q4 change.

Use eight fixed expert slots, two ready hits and six misses per pass, eight I/O
workers, and at most two live GPU groups. Explicitly invalidate only the eight
selected expert file ranges after drain and before priming. Each arm therefore
has 96 ready hits, 288 new misses, zero loading joins, 384 invalidations,
796,262,400 demand-read bytes, and 265,420,800 preparation-read bytes. Every
timing arm must observe systemwide device reads within 90–110% of the total
1,061,683,200 application bytes. Counters include other processes and do not
establish NAND-cold storage. Exclude file preparation from coordinator wall
time; report it and whole-arm elapsed time separately.

Use separate check, timing, and trace processes. Check enables Metal validation.
Timing disables validation/profiling and records no per-pass scratch snapshots.
Trace captures all 48 passes per arm with native command profiles, dependency
events, and three scratch snapshots per pass: before preparation, after
preparation, and after work. All snapshots are outside measured coordinator
windows and must have zero live GPU groups. Infer the used buffer prefix from
allocated minus unused retained bytes divided by the verified 16KiB charge;
this is not a new production cursor API. Validate continuity between snapshots,
unchanged preparation, and forward-prefix growth by 27 buffers per pass.
Preserve existing shared-prefix command and routed resource-release checks.
Do not compare trace timings with normal timings or split mixed command GPU
duration into separate shared and routed costs.

Use the existing exclusive experiment lease, 12GiB process ceiling, memory and
host gates, source/build/artifact seals, and disk admission. Bound each native
process to 60 seconds and each condition to 180 seconds. Stop on failed
correctness, identity, resource, or storage gates. Preserve incomplete or
disturbed reports; they cannot pass. No global purge or OS limit changes.

Retain the diagnostic criterion: GPU-duration paired upper 95% bound below one
and coordinator-wall upper bound at most 1.03 at both group caps, with all other
gates satisfied. If the gain survives, record that this partial lifetime change
does not reproduce the normal-request regression and identify the smallest
remaining contextual difference before another experiment. If it disappears,
inspect the affected commands and allocation boundaries before a scheduling
change. Neither outcome qualifies production defaults, 5 tokens/s, 2K/4K
latency, 7K reporting, or sustained coding.
