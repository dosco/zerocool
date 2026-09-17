# Independent operator replay from verified tensor bytes

The original capture/screen remains resource-blocked: its capture matched every
logit, route and state boundary, but process compression invalidated its timings
and stopped the stage before either operator process. Do not change that status
or combine its memory/performance samples with a later process.

This separate experiment uses only the saved tensor bytes. Revalidate full-model
numerical identity, build/artifact identity, all fixture payload hashes and the
nine packed-weight/scale/bias ranges against the pinned checkpoint before use.
The earlier clean command/counter profile remains the basis for choosing these
three matrices. No source-capture timing is a premise of the operator test.

Run only the existing small operator binary: fresh Metal validation, then five
alternating pairs of 32 dispatches on the unchanged real four-token inputs.
Keep the prior candidate, warmup, paired-confidence and 10ms projection criteria.
Maximum Metal buffers are 256MiB; native physical/compression/decompression/swap
and host power/thermal observations belong to these fresh operator processes.
Use native preflight and the exclusive GPU lease. Each process is bounded to
60 seconds and this stage to 180. Any resource disturbance stops the replay.

This does not relax the original capture or whole-verifier performance gates.
It avoids reloading the full model solely to recreate already exact input bytes.
A passing isolated result still requires fresh clean full-verifier correctness,
recovery and normal timing; no production promotion or real draft claim follows.
