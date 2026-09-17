#!/usr/bin/env python3
"""Bounded exact-Q4 expert screen over real records and saved activations."""
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
FIXTURES = ROOT/'docs/benchmarks/2026-09-13-stage200/q3-02/tuning'
PROFILE = ROOT/'docs/benchmarks/2026-09-14-resident-operators/capture-02'


def verify_records(prepared):
    manifest = load(FIXTURES/'manifest.json')
    require(manifest.get('complete') is True and manifest.get('origin') == 'existing-Q4-experts',
            'Requires complete original Q4 capture')
    index = load(prepared/'manifest.json')
    verified = []
    for layer in (0, 16, 32, 47):
        item = manifest['layers'][str(layer)]
        files = [item['inputs'], *[e['record'] for e in item['experts']]]
        for entry in files:
            path = FIXTURES/entry['file']
            require(path.is_file() and path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
                    'Missing or changed fixture payload')
        source = index['experts'][layer]
        require(source['layer'] == layer and source['stride'] == 2768896 and source['length'] == 2764800,
                'Prepared layout differs')
        with (prepared/source['file']).open('rb') as stream:
            for expert in item['experts']:
                require(0 <= expert['expert'] < 512, 'Expert out of range')
                stream.seek(source['offset']+expert['expert']*source['stride'])
                import hashlib
                data = stream.read(source['length'])
                digest = hashlib.sha256(data).hexdigest()
                require(len(data) == source['length'] and digest == expert['record']['sha256'],
                        'Real fixture differs from pinned prepared Q4 record')
                verified.append(dict(layer=layer, expert=expert['expert'], sha256=digest))
    return verified


def analyze(raw):
    stream = raw.get('stream_bytes_before_each_group', 0)
    require(stream in (0, 128*1024**2) and raw.get('stream_time_included', False) is False, 'Changed streaming experiment')
    require(raw.get('complete') is True and raw.get('validation') is False and raw.get('exact') is True and
            raw.get('device') == 'Apple M1 Pro' and raw.get('cycles') == 4 and
            raw.get('expert_input_cases') == 64 and raw.get('output_modes') == ['BF16', 'FP32'] and
            0 < raw.get('max_shared_buffer_bytes', 0) <= (384 if stream else 64)*1024**2, 'Incomplete bounded Q4 operator screen')
    require([(p['pair'], p['group']) for p in raw['pairs']] == [(p, g) for p in range(5) for g in (1, 4)],
            'Missing alternating expert group pairs')
    clean = True; results = []
    for group in (1, 4):
        paired = {v: dict(gpu=[], wall=[], savings=[]) for v in (1, 2)}
        for row in [p for p in raw['pairs'] if p['group'] == group]:
            require([a['variant'] for a in row['arms']] == ([2, 0, 1] if row['pair']%2 else [0, 2, 1]),
                    'Changed primary paired order')
            values = {}
            for arm in row['arms']:
                sample = arm['sample']
                for key in ('gpu_us_per_expert', 'wall_us_per_expert'):
                    v = sample.get(key)
                    require(type(v) in (float, int) and math.isfinite(v) and v > 0, 'Invalid timing')
                a, b = (arm['memory_'+k] for k in ('before', 'after'))
                clean &= (a.get('compressed_bytes') == b.get('compressed_bytes') == 0 and
                    type(a.get('decompressions')) is int and a['decompressions'] == b.get('decompressions') and
                    type(a.get('system_swap_used_bytes')) is int and b.get('system_swap_used_bytes') == a['system_swap_used_bytes'])
                values[arm['variant']] = sample
            for variant in (1, 2):
                a, b = values[0], values[variant]
                paired[variant]['gpu'].append(b['gpu_us_per_expert']/a['gpu_us_per_expert'])
                paired[variant]['wall'].append(b['wall_us_per_expert']/a['wall_us_per_expert'])
                paired[variant]['savings'].append((a['gpu_us_per_expert']-b['gpu_us_per_expert'])*480/1000)
        results.append(dict(group=group, variants=[dict(variant=v,
            gpu_confidence_95=paired_log_interval(paired[v]['gpu']), wall_confidence_95=paired_log_interval(paired[v]['wall']),
            isolated_frequency_projection_ms=paired[v]['savings'],
            median_isolated_frequency_projection_ms=statistics.median(paired[v]['savings'])) for v in (1, 2)]))
    primary = [r['variants'][1] for r in results]
    material = all(p['median_isolated_frequency_projection_ms'] >= 20 and
                   p['gpu_confidence_95']['high'] < 1 and p['wall_confidence_95']['high'] <= 1.03 for p in primary)
    return dict(status='memory_disturbed' if not clean else 'worth_request_screen' if material else 'insufficient_benefit',
        clean_memory=clean, results=results, advance_to_request_screen=clean and material,
        normal_request_latency_qualified=False, production_promoted=False,
        limitations=['Eight original Q4 experts and eight real inputs per layer; no quantization change.',
            'Single-token expert shapes only. Stored inputs need not have selected every fixture expert.',
            'Existing paired-gate/row-two kernels are a diagnostic third arm, not the predeclared primary candidate.',
            'Isolated frequency projection uses 480 experts/token and omits SSD/dependency overlap; it is not request latency.',
            'Five pairs in one process use resident records and correlated inputs; full-model state and timing remain unqualified.'])


def run(out, binary, stream=False):
    verify_seal(PROFILE, sha(PROFILE/'evidence-files.json'))
    profile = load(PROFILE/'summary.json')
    require(profile.get('complete') is True, 'Requires completed dependency diagnostic')
    exp = Experiment(out, 'q4_packed_operator_screen_v1', [], [], 240)
    with exp:
        require(exp.report['identity'] == profile['identity'], 'Changed profile/native identity')
        paths = [binary, ROOT/'scripts/qwen/probe_q4_packed.mm', ROOT/'scripts/qwen/probe_q4_packed.metal',
                 PROFILE/'summary.json', *FIXTURES.iterdir()]
        for path in paths:
            if path.is_file(): exp.frozen['files'][str(path.resolve())] = sha(path)
        save(exp.out/'identity.json', exp.frozen)
        exp.report['verified_records'] = verify_records(exp.prepared); exp.persist()
        for mode in ('validate', 'timing'):
            exp.command([binary, ROOT/'kernels/metal/qwen.metal', ROOT/'scripts/qwen/probe_q4_packed.metal',
                         FIXTURES, exp.out/(mode+'.json'), mode+('-stream' if stream else '')], mode, limit=90, validation=mode == 'validate')
        validation, timing = (load(exp.out/(name+'.json')) for name in ('validate', 'timing'))
        require(validation.get('complete') is True and validation.get('validation') is True and
                validation.get('exact') is True and validation.get('expert_input_cases') == 64 and
                all(validation[k] == timing[k] for k in
                    ('reference_shader_sha256', 'candidate_shader_sha256', 'fixture_manifest', 'output_modes')),
                'Missing or differing validation evidence')
        require(timing.get('stream_bytes_before_each_group', 0) == (128*1024**2 if stream else 0), 'Missing streaming setup')
        exp.report.update(analyze(timing), stream_bytes_before_each_group=128*1024**2 if stream else 0,
            stream_limitation='A fixed 128MiB device copy precedes each measured expert group. Its time is excluded. '
                              'This perturbs locality; it is not a cache flush guarantee or a normal request.')
    return exp.report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--stream', action='store_true')
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output, args.binary.resolve(), args.stream)['complete'] else 2)
