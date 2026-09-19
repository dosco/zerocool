# Documentation index

## Start here

| Document | What it covers |
|---|---|
| [qwen_plan.md](qwen_plan.md) | The plan of record: goals, architecture, milestones and success criteria |
| [qwen_engine.md](qwen_engine.md) | What the engine actually implements, and its qualification limits |
| [qwen_usability.md](qwen_usability.md) | Terminal chat controls, the HTTP API contract, and usability gates |
| [qwen_evidence_queries.md](qwen_evidence_queries.md) | How to query the recorded benchmark evidence |

## Stage reports

Each stage document states a hypothesis, the experiment that screened it, and
what the result does and does not establish. A stage is benchmark-only until
paired normal-request evidence qualifies it.

| Stage | Subject |
|---|---|
| [qwen_next_stage.md](qwen_next_stage.md) | Exact-arithmetic kernels: packed weight reuse, precomputed recurrent gates |
| [qwen_memory_compute_stage.md](qwen_memory_compute_stage.md) | Workspace reclamation, lazy cache regrowth, exact output blocking |
| [qwen_residency_stage.md](qwen_residency_stage.md) | Bounded residency, cached-token replay, direct and grouped decode |
| [qwen_decode_compute_stage.md](qwen_decode_compute_stage.md) | GPU-pass diagnostics and exact packed Q8 decode |
| [qwen_sparse_attention_stage.md](qwen_sparse_attention_stage.md) | Sparse block selection and masked score tiles |
| [qwen_selector_qualification_stage.md](qwen_selector_qualification_stage.md) | Qualifying a shape-selection rule from measurements |
| [qwen_memory_balance_stage.md](qwen_memory_balance_stage.md) | Balancing cache, workspace and state under the budget |
| [qwen_memory_budget_stage.md](qwen_memory_budget_stage.md) | Explicit budgets and the compression tradeoff |
| [qwen_q8_steady_stage.md](qwen_q8_steady_stage.md) | Steady-state Q8 decode behaviour |
| [qwen_route_selection_stage.md](qwen_route_selection_stage.md) | Router selection on GPU versus CPU |
| [qwen_stage200.md](qwen_stage200.md) | Calibrated affine-Q3 expert screening |
| [qwen_decode_target_stage.md](qwen_decode_target_stage.md) | The 200ms-per-token generation target |
| [qwen_combined_decode_stage.md](qwen_combined_decode_stage.md) | Combining the qualified decode changes |
| [qwen_mtp_recovery_stage.md](qwen_mtp_recovery_stage.md) | Multi-token prediction drafting and target recovery |

## Measurement evidence

`benchmarks/` holds one dated directory per screen, each with a `README.md`
stating the setup, the raw data and the decision. `experiments/` holds
hash-named experiment records queried by `scripts/qwen/query_evidence.py`.
Raw captures are large; read the stage report first and open the raw data only
when you need to reproduce a number.

Headline results so far, all with stated limits in their reports:

- Full-model agreement with the MLX reference is exact for both artifacts on the
  saved five-token fixture, across storage layouts, cache sizes, token batching
  and short session continuations.
- A normal mixed 2K run at a 12GiB budget measured 470.6s to first token and
  1.99 tokens/s without swap growth. That single run misses the latency targets.
- Two alternating normal-request screens at 12GiB improved generation from 2.68
  to 3.53 tokens/s after a 2K prompt, and from 2.64 to 3.21 after a 128-token
  append to retained 4K history, with identical generated tokens. These are
  64-output-token screens and remain benchmark-only.
- An earlier 48-layer check over a 257-token prompt and a 129-token append
  preserved retained state and reduced expert read bytes by 59.4%. Equal-memory
  request timing remains unqualified.

Nothing above qualifies a runtime default. The gates that remain open are listed
in [qwen_engine.md](qwen_engine.md) and enforced by
`scripts/qwen/release_check.py`.

## Licenses

`licenses/` holds the license text for the model and for each third-party
component, referenced from [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
