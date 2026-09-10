# Metal submission cleanup

Run 02 stopped when a later model instance could not obtain memory. A bounded
reproduction found that command submission retained physical buffers under Metal
API and shader validation even after C++ owners and allocation counters had
returned to zero. Plain C++ callers supplied no outer autorelease pool.

`Metal::submit()` now drains autoreleased Objective-C objects at submission.
The pending command group still owns its command buffer and all allocations
until GPU completion is reaped. This changes resource lifetime, not inference
arithmetic, routing or reduction order.

The existing synchronous drain uses `waitUntilCompleted`, whose contract includes
completion handlers as well as GPU execution. This supports reaping ownership
after that wait; `Metal::wait` also explicitly waits on the completion event.
See [Apple's API contract](https://developer.apple.com/documentation/metal/mtlcommandbuffer/waituntilcompleted%28%29?language=objc).

The regression copies between two 64MiB buffers, completes GPU work, releases
both buffers and checks logical allocation/residency counters. After two warmup
cycles it measures twelve further cycles, alternating `finish()` with explicit
`submit()`/`wait()`. Physical footprint may grow by at most 256MiB.

Before the fix the test failed: footprint grew from 295,290,880 bytes to
1,906,938,944 bytes despite zero logical allocations. It passes after the fix.
A separate twelve-cycle diagnostic reached about 1.6GB without cleanup and stayed
near 161MB when only submission received an autorelease scope. These are small
reproduction measurements, not full-model performance results.

Verification with both Metal validators enabled:

- Targeted regression: 1 test / 43 assertions passed.
- Complete native suite: 51 tests / 6,931 assertions passed, none skipped.
- Python suite: 84 tests passed, including preservation of nested resource-blocked
  errors without treating an ordinary child failure as a resource failure.
- Build and whitespace checks passed.

Native source fingerprint:
`c4f983f03f28974dd2c4a935d6d5233f1b28a1f5e7cb4800fe7529a64c058c2b`.
Verification logs are copied here with their SHA-256 inventory. The full-model
rerun uses fresh evidence in
`/repo/.cache/benchmarks/selector-qualification/2026-09-09-run-03`.
It must independently prove recovery, state equivalence and performance; this
regression alone cannot establish that every source of full-model memory growth
has been removed.

The fresh [full-model boundary comparison](../run-03/qualify/boundary/summary.json)
has since passed exact logits, routes and state across all 48 layers. Both paths
passed all nine checks, including the model-recreation, failure and cancellation
checks that previously stopped at memory admission. Longer-context qualification
and normal performance measurements are still pending.
