#!/usr/bin/env python3
"""Recheck historical lease traces and explicit larger-cache allocation arithmetic."""
import argparse
from pathlib import Path

from block_cache_replay import decode, simulate, SLOT_BYTES
from cache_residency import require
from qualification_evidence import save, sha, verify_seal
from trial_recovery import read

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'docs/benchmarks/2026-09-16-block-cache/capture-02'
MEMORY = ROOT/'docs/benchmarks/2026-09-17-verifier-horizon/tiled-early-01/pair-0-width-8.json'


def budgets(admission):
    target = admission['target']
    require(target['limit_bytes'] == 12*1024**3 and target['expert_slots'] == 1460 and
        target['expert_bytes'] == 1460*SLOT_BYTES and target['scratch_bytes'] == 512*1024**2 and
        admission['checkpoint_rows'] == 8 and admission['host_checkpoint_logits_bytes'] == 128*1024**2+4*8*248320*4+1024**2 and
        admission['combined_bytes'] == target['planned_bytes']+sum(admission[k] for k in
            ('draft_bytes', 'host_checkpoint_logits_bytes', 'expert_scratch_reserve_bytes', 'target_recovery_reserve_bytes')),
        'Changed base memory geometry')
    fixed8 = admission['combined_bytes']-target['expert_bytes']
    result = []
    for width, fixed in ((4, fixed8-4*4*248320*4), (8, fixed8)):
        for scratch in (512, 128):
            adjusted = fixed-(512-scratch)*1024**2
            result.append(dict(width=width, scratch_mib=scratch, fixed_bytes=adjusted,
                maximum_slots=(12*1024**3-adjusted)//SLOT_BYTES,
                bytes_at_2048=adjusted+2048*SLOT_BYTES,
                workspace_policy_implemented=scratch == 512, larger_cache_qualified=False))
    return result


def run(output):
    output = Path(output).resolve()
    require(not output.exists(), 'Analysis output already exists')
    verify_seal(SOURCE, sha(SOURCE/'evidence-files.json'))
    verify_seal(MEMORY.parent, sha(MEMORY.parent/'evidence-files.json'))
    work, identity = read(SOURCE/'workload.json'), read(SOURCE/'identity.json')
    rows = []
    for capacity in (1072, 1460):
        raw = SOURCE/f'capture-{capacity}.json'
        trace_path = SOURCE/f'capture-{capacity}.cache.jsonl'
        trace = decode(trace_path.read_bytes(), read(raw), work, identity['build'])
        curves = [simulate(trace, n, 'clock') for n in (1460, 1536, 1909, 2048)]
        baseline = curves[0]['decode_misses']
        rows.append(dict(captured_capacity=capacity, raw=dict(path=str(raw), sha256=sha(raw)),
            trace=dict(path=str(trace_path), sha256=sha(trace_path)),
            native_counts_exact=trace['native_counts_exact'], native_slot_state_exact=trace['native_slot_state_exact'],
            snapshots_checked=trace['snapshots_verified'],
            curves=[dict(c, relative_miss_reduction=1-c['decode_misses']/baseline) for c in curves]))
    report = dict(kind='streamed_mtp_cache_budget_analysis_v1', complete=True, status='offline_analysis_only',
        performance_measurement=False, production_promoted=False, source_build=identity['build'],
        source=dict(path=str(SOURCE), seal_sha256=sha(SOURCE/'evidence-files.json')),
        memory_source=dict(path=str(MEMORY), sha256=sha(MEMORY)),
        analysis_sources={str(p): sha(p) for p in (Path(__file__).resolve(), ROOT/'scripts/qwen/block_cache_replay.py',
                                                 ROOT/'scripts/qwen/cache_simulation.py')},
        captured_width=4, captured_decode_tokens=16, replays=rows, memory_arithmetic=budgets(read(MEMORY)['admission']),
        limitations=['Historical four-token perfect-proposal traces, not current eight-token or real-MTP measurements.',
            'Counterfactuals hold captured request and lease order fixed; changed I/O and GPU timing are not predicted.',
            'These larger capacities were not in the original candidate screen and do not reverse its decisions.',
            'A 128MiB generation workspace and cache growth are allocation hypotheses; runtime lifetimes remain unqualified.',
            'Application-read savings are not latency or device-traffic predictions.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
