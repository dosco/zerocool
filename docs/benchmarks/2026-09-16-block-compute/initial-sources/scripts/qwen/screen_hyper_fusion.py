#!/usr/bin/env python3
"""Bounded complete hyper-block fusion screen; operator evidence cannot promote production."""
import argparse
import math
from pathlib import Path
import statistics

from cache_residency import require
from capture_routes import load
from qualification_evidence import save, sha, verify_seal
from screen_residency import paired_log_interval
from stage200 import Experiment

ROOT = Path(__file__).resolve().parents[2]
FREQUENCIES = [36, 36, 12, 12]
CYCLE = [0, 1, 0, 1, 0, 1, 2, 3]
PROTOCOL = ROOT/'docs/benchmarks/2026-09-15-hyper-fusion/protocol.md'


def analyze(raw):
    require(raw.get('kind') == 'hyper_fusion_probe_v1' and raw.get('complete') is True and
            raw.get('validation') is False and raw.get('device') == 'Apple M1 Pro', 'Incomplete M1 timing')
    require(raw.get('dispatch_repeats') == 32 and raw.get('reference_dispatches_per_block') == 8 and
            raw.get('candidate_dispatches_per_block') == 6 and raw.get('edge_cases') == 12 and
            raw.get('neighbor_guards_intact') is True and raw.get('inputs_unchanged') is True,
            'Missing complete-block or correctness coverage')
    require(type(raw.get('max_shared_buffer_bytes')) is int and 0 < raw['max_shared_buffer_bytes'] <= 256*1024**2 and
            type(raw.get('gpu_stage_elapsed_seconds')) in (int, float) and
            0 < raw['gpu_stage_elapsed_seconds'] <= 30, 'Resource bounds exceeded')
    require(len(raw.get('cases', [])) == 4 and all(c.get('exact') is True and c.get('native_reference_exact') is True
            for c in raw['cases']), 'Missing real native-output comparisons')
    require([(c['origin']['layer'], c['origin']['stage']) for c in raw['cases']] ==
            [(0, 'attention_input'), (0, 'mlp_input'), (3, 'attention_input'), (3, 'mlp_input')],
            'Fixture frequency assignment differs')
    require([(p['pair'], p['case']) for p in raw['pairs']] == [(p, c) for p in range(5) for c in range(5)],
            'Missing or duplicate paired timing')
    observations = [raw[k] for k in ('memory_before', 'timing_memory_before', 'timing_memory_after', 'memory_after')]
    results = []; projection = []; gpu_projection = []
    for case in range(5):
        values = {False: [], True: []}; gpu = {False: [], True: []}; encoding = {False: [], True: []}
        for pair in (p for p in raw['pairs'] if p['case'] == case):
            require([a['candidate'] for a in pair['arms']] == [bool(pair['pair'] % 2), not bool(pair['pair'] % 2)],
                    'Alternating order differs')
            for arm in pair['arms']:
                require(type(arm['candidate']) is bool, 'Arm identity must be boolean')
                require(set(arm['sample']) == {'wall_us', 'gpu_us', 'encode_us'}, 'Sample fields differ')
                require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in arm['sample'].values()),
                        'Invalid timing')
                require(all(arm['sample'][k] <= arm['sample']['wall_us']*1.02+1 for k in ('gpu_us', 'encode_us')),
                        'Component time exceeds complete wall time')
                values[arm['candidate']].append(arm['sample']['wall_us'])
                gpu[arm['candidate']].append(arm['sample']['gpu_us'])
                encoding[arm['candidate']].append(arm['sample']['encode_us'])
                observations += [arm['memory_before'], arm['memory_after']]
        ratios = [b/a for a, b in zip(values[False], values[True])]
        savings = [(a-b)/1000 for a, b in zip(values[False], values[True])]
        results.append(dict(case=case, wall_us=values, gpu_us=gpu, encode_us=encoding,
            wall_confidence_95=paired_log_interval(ratios), wall_savings_ms=savings,
            median_wall_saving_ms=statistics.median(savings)))
        if case == 4:
            # One cycle has each GDN fixture3x and attention fixture1x; twelve cycles represent96 blocks.
            projection = [v*12 for v in savings]
            gpu_projection = [(a-b)*12/1000 for a, b in zip(gpu[False], gpu[True])]
    fields = ('physical_footprint_bytes', 'physical_footprint_peak_bytes', 'compressed_bytes',
              'compressed_peak_bytes', 'decompressions', 'system_swap_used_bytes')
    require(all(all(type(o.get(k)) is int and o[k] >= 0 for k in fields) for o in observations),
            'Missing memory observations')
    require(all(0 < o['physical_footprint_bytes'] <= o['physical_footprint_peak_bytes'] <= 12*1024**3 and
                o['compressed_bytes'] <= o['compressed_peak_bytes'] for o in observations), 'Impossible or excessive memory footprint')
    clean = (all(o['compressed_bytes'] == o['compressed_peak_bytes'] == 0 for o in observations) and
             len({o['decompressions'] for o in observations}) == 1 and
             len({o['system_swap_used_bytes'] for o in observations}) == 1)
    material = statistics.median(projection) >= 20 and results[-1]['wall_confidence_95']['high'] < 1
    return dict(status='memory_disturbed' if not clean else 'worth_request_screen' if material else 'insufficient_benefit',
        clean_memory=clean, cases=results, frequency_weights=FREQUENCIES, cycle=CYCLE,
        frequency_projection_ms_per_token=projection, median_frequency_projection_ms_per_token=statistics.median(projection),
        gpu_frequency_projection_ms_per_token=gpu_projection,
        median_gpu_frequency_projection_ms_per_token=statistics.median(gpu_projection),
        max_physical_footprint_bytes=max(o['physical_footprint_peak_bytes'] for o in observations),
        advance_to_request_screen=clean and material, normal_request_latency_qualified=False, production_promoted=False,
        limitations=['Four captured blocks, not all96 blocks or full-model state; fixture native producer identity is retained.',
            'Repeated resident weights and preallocated scratch differ from live requests, SSD streaming and native buffer lifetimes.',
            'Frequency projection is a screening heuristic, not predicted request savings. Five pairs in one process may be correlated.',
            'Memory counters are sampled at boundaries plus process lifetime peaks; device swap includes other processes.',
            'No footprint saving is inferred from eliminated logical intermediates; both probe arms coexist.'])


def validate_pair(a, b):
    require(a.get('kind') == b.get('kind') == 'hyper_fusion_probe_v1' and
            a.get('complete') is True and a.get('validation') is True and a.get('edge_cases') == 12 and
            a.get('inputs_unchanged') is True and a.get('neighbor_guards_intact') is True and
            a.get('device') == b.get('device') == 'Apple M1 Pro' and a.get('pairs') == [] and
            type(a.get('gpu_stage_elapsed_seconds')) in (int, float) and
            0 < a['gpu_stage_elapsed_seconds'] <= 30 and
            a['gpu_stage_elapsed_seconds'] + b['gpu_stage_elapsed_seconds'] <= 60 and
            type(a.get('max_shared_buffer_bytes')) is int and 0 < a['max_shared_buffer_bytes'] <= 256*1024**2,
            'Missing bounded validation')
    for key in ('cases', 'reference_shader_sha256', 'candidate_shader_sha256', 'fixture_manifest_sha256',
                'dispatch_repeats', 'reference_dispatches_per_block', 'candidate_dispatches_per_block'):
        require(a[key] == b[key], 'Validation and timing identity differs')


def run(out, binary, fixtures):
    exp = Experiment(out, 'hyper_fusion_screen_v1', [], [], 180)
    with exp:
        from prepare_hyper_fixtures import verify_prepared
        exp.report['verified_fixtures'] = verify_prepared(fixtures, exp.model)
        paths = [binary, ROOT/'scripts/qwen/probe_hyper_fused.metal',
                 ROOT/'kernels/metal/qwen.metal', ROOT/'scripts/qwen/probe_hyper_fused.mm', PROTOCOL,
                 *[p for p in fixtures.iterdir() if p.is_file()]]
        for p in paths: exp.frozen['files'][str(p.resolve())] = sha(p)
        save(exp.out/'identity.json', exp.frozen)
        exp.report['source_inputs'] = {k:dict(path=str(p.resolve()), sha256=sha(p)) for k,p in (
            ('reference_shader_sha256', ROOT/'kernels/metal/qwen.metal'),
            ('candidate_shader_sha256', ROOT/'scripts/qwen/probe_hyper_fused.metal'),
            ('fixture_manifest_sha256', fixtures/'manifest.json'))}
        exp.persist()
        exp.guard.check_resources(initial=True)
        for mode in ('validate', 'timing'):
            exp.command([binary, ROOT/'kernels/metal/qwen.metal', ROOT/'scripts/qwen/probe_hyper_fused.metal',
                         fixtures, exp.out/(mode+'.json'), mode], mode, limit=60, validation=mode == 'validate')
        a, b = (load(exp.out/(mode+'.json')) for mode in ('validate', 'timing'))
        validate_pair(a, b)
        for key, entry in exp.report['source_inputs'].items():
            require(b[key] == entry['sha256'] == exp.frozen['files'][entry['path']], 'Probe source differs from frozen input')
        exp.report.update(analyze(b))
    return exp.report


def audit(directory):
    verify_seal(directory, sha(directory/'evidence-files.json'))
    raw = load(directory/'timing.json'); saved = load(directory/'summary.json')
    require(saved.get('complete') is True and saved.get('kind') == 'hyper_fusion_screen_v1', 'Stage is unfinished or wrong kind')
    recomputed = analyze(raw)
    for k, v in recomputed.items():
        # JSON serializes boolean map keys; normalize before comparing.
        import json
        require(json.loads(json.dumps(v)) == saved[k], 'Saved decision differs: '+k)
    a = load(directory/'validate.json')
    validate_pair(a, raw)
    frozen = load(directory/'identity.json')['files']
    for key, entry in saved['source_inputs'].items():
        require(a[key] == raw[key] == entry['sha256'] == frozen[entry['path']], 'Source differs from frozen identity')
    return dict(verified=True, decision=recomputed['status'], normal_request_latency_qualified=False,
                production_promoted=False, evidence_sha256=sha(directory/'evidence-files.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True);p.add_argument('--binary', type=Path)
    p.add_argument('--fixtures', type=Path);p.add_argument('--audit', action='store_true')
    args = p.parse_args()
    if args.audit:
        import json
        print(json.dumps(audit(args.output), indent=2))
    else:
        require(args.binary and args.fixtures, 'Binary and fixtures required')
        raise SystemExit(0 if run(args.output, args.binary.resolve(), args.fixtures.resolve())['complete'] else 2)
