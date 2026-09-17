"""Fixed scope of the six-case four-token packed-Q8 experiment."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'docs/benchmarks/2026-09-16-q8-expanded'
KERNEL = 'q8_expanded_t4_w8'
VARIANT = dict(candidate=KERNEL, control='q8_mm_t4', reference='q8_mm',
    token_tile=4, output_rows=1, lane_width=8, weight_load_bits=32)
# Output head first so the new widest case is screened before the familiar GDN.
CASES = [
    dict(name='head', stage='logits', layer=-1, K=2560, N=248320, frequency=1, tensor='lm_head'),
    dict(name='attention_q', stage='attention', layer=3, K=2560, N=12288, frequency=12, tensor='model.layers.3.self_attn.q_proj'),
    dict(name='attention_o', stage='attention', layer=3, K=6144, N=2560, frequency=12, tensor='model.layers.3.self_attn.o_proj'),
    dict(name='gdn_qkv', stage='gdn', layer=0, K=2560, N=10240, frequency=36, tensor='model.layers.0.linear_attn.in_proj_qkv'),
    dict(name='gdn_z', stage='gdn', layer=0, K=2560, N=6144, frequency=36, tensor='model.layers.0.linear_attn.in_proj_z'),
    dict(name='gdn_out', stage='gdn', layer=0, K=6144, N=2560, frequency=36, tensor='model.layers.0.linear_attn.out_proj')]
CRITERIA = dict(cases=CASES, variant=VARIANT, pairs=5, repeats=32,
    minimum_projection_ms_per_token=10, paired_ratio_upper_below=1,
    operator_gpu_bytes=1024**3, operator_physical_bytes=2*1024**3,
    input_capture_bytes=1024**2, engine_bytes=12*1024**3, expert_slots=1460,
    capture_seconds=150, operator_seconds=90, stage_seconds=420,
    source_timing_reused=False, production_promoted=False)
LIMITATIONS = [
    'The operator screen uses one four-token input block, layer 0 GDN, layer 3 attention and the output head; other inputs and contexts need full-model validation.',
    'Frequency-weighted savings and paired intervals describe repeated isolated kernels, not complete requests. Pairs in one process may be temporally dependent.',
    'All six cases are remeasured together. No earlier GDN timing or memory samples are pooled.',
    'The input capture reports exact numerical identity separately from its resource status; no capture timing qualifies speed.',
    'A passing screen authorizes a fresh perfect-proposal verifier comparison, not production promotion or a real speculative throughput claim.']
