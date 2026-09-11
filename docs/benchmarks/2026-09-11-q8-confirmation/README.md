# Q8 speed benefit repeats; first-token guard remains inconclusive

Five fresh alternating pairs reduced complete-conversation time in every pair.
The geometric-mean candidate/control ratio was **0.88936**, a **11.06% reduction**,
with a paired log-ratio Student-t 95% interval of **0.87554–0.90340**. Both initial
and retained-append generation improved consistently. The full predeclared gate
nevertheless returned `improvement_not_demonstrated`: the initial first-token
upper bound was **1.03437**, just above the allowed **1.03**. Defaults stay unchanged.

The five complete-conversation ratios were 0.86967, 0.89657, 0.89329, 0.89468,
and 0.89286. The first control's initial first-token wait was 15.514s versus
13.881s for its candidate. The other four control waits were 13.848–14.003s
and their candidates 13.798–13.869s. This extra variation widens the interval
even though the initial first-token geometric-mean ratio favors the candidate,
at 0.97459. All samples remain included. No outlier was removed, gate relaxed,
or screen sample pooled. This does not establish a first-token regression;
it fails to exclude more than 3% regression under this interval model.

Initial generation's geometric-mean time ratio was **0.73022** (95% interval
0.71412–0.74669); append generation's was **0.75531** (0.74738–0.76333).
Median generation was approximately **2.57 → 3.48 tok/s** initially and
**2.45 → 3.25 tok/s** after the append. Both request-time bounds passed. Append
first-token behavior also passed. Five pairs assume independent, approximately
normal log-ratios; these are marginal intervals, not simultaneous coverage.

The complete experiment took **631.27s**, within its 900-second deadline. Each
arm was a fresh process using the same 72+33, retained-128+33 conversation on the
actual 32GiB M1 Pro. Both used the 12GiB ceiling, 1848 CLOCK slots, original
schedule and kernel defaults, except the candidate's packed Q8 two-row path.
No profiling, validation, boundary probes, precision change, or memory resize
occurred during timing. Generated tokens, actual 104-token computation reuse,
normalized dispatch counts, and every admitted allocation category matched.

The unchanged native build passed **57 tests / 45,704 assertions** under Metal
validation; **190 Python tests** passed. Prior current-build all-layer exact
state/failure checks and fresh captured-operator comparisons were revalidated
from original hashes. The offline query reconstructs both the prerequisite and
all ten fresh reports, including the failed secondary guard.

This remains an experimental configuration with demonstrated short-workload
speed benefit and unresolved first-token non-inferiority. It is not production
promotion, 5 tok/s acceptance, long-context qualification, or a coding-quality
claim. Record and commit the evidence, then evaluate the separately declared
12GiB/18GiB memory tradeoff using explicit experimental packed-Q8 settings in
both arms. That new experiment does not retroactively pass this gate.

Native build: `1ddec378e0a2550eda51c351e5e9315863c15b61773617c457b4d8e2f8df9189`.
Artifact: `b2c422f3c643e36f04227a64d61796b44a4b1029`.

[Protocol](../../qwen_q8_steady_stage.md), [raw summary](raw/summary.json),
[reconstructed comparison](comparison.json), [native tests](native-tests.log),
[Python tests](python-tests.log). Raw files and nested evidence inventories are
preserved byte-for-byte. Earlier operator payloads remain local as specified in
the copied prerequisite archive.
