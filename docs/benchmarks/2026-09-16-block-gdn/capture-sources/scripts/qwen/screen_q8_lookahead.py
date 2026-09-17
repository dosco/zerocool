#!/usr/bin/env python3
"""Screen one exact load-ahead Q8 kernel using existing verified real inputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

from cache_residency import require
from capture_routes import load
from qualification_evidence import save, sha
from screen_residency import paired_log_interval
from stage200 import Experiment
from verify_checkpoint import fingerprint

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = [ROOT/'.cache/benchmarks/q8-steady-20-20260911/capture-0-gdn',
            ROOT/'.cache/benchmarks/q8-steady-20-20260911/capture-31-attention']
NAMES = ['0.linear_attn.in_proj_qkv', '0.linear_attn.in_proj_z', '0.linear_attn.out_proj',
         '31.self_attn.q_proj', '31.self_attn.k_proj', '31.self_attn.o_proj']
SHAPES = [(2560, 10240), (2560, 6144), (6144, 2560), (2560, 12288), (2560, 512), (6144, 2560)]
# Shape-frequency projection only; k and v have the same geometry but only k is captured.
CALLS = [36, 36, 36, 12, 24, 12]


def verify_weights(model, frozen):
    mapping = load(model/'model.safetensors.index.json')['weight_map']
    verified = []
    for directory, names in zip(FIXTURES, (NAMES[:3], NAMES[3:])):
        manifest = load(directory/'manifest.json')
        require(manifest['artifact_revision'] == frozen['artifact_revision'] and len(manifest['cases']) == 3,
                'Fixture artifact identity differs')
        for case, name in zip(manifest['cases'], names):
            for entry in case['tensors'].values():
                path = directory/entry['file']
                require(path.is_file() and path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
                        'Missing or changed captured fixture payload: '+str(path))
            for key, suffix in (('w', 'weight'), ('s', 'scales'), ('b', 'biases')):
                tensor = 'language_model.model.layers.'+name+'.'+suffix
                path = model/mapping[tensor]
                require(frozen['assets'][str(path.resolve())] == fingerprint(path), 'Stale source receipt')
                with path.open('rb') as stream:
                    size = struct.unpack('<Q', stream.read(8))[0]
                    require(0 < size < 32*1024**2, 'Invalid safetensors header')
                    header = json.loads(stream.read(size))
                    low, high = header[tensor]['data_offsets']
                    require(0 <= low < high <= path.stat().st_size-size-8, 'Invalid tensor range')
                    entry = case['tensors'][key]
                    require(high-low == entry['bytes'], 'Source tensor length differs')
                    stream.seek(8+size+low)
                    digest = hashlib.sha256()
                    remaining = high-low
                    while remaining:
                        chunk = stream.read(min(1024**2, remaining))
                        require(bool(chunk), 'Short source tensor read')
                        digest.update(chunk); remaining -= len(chunk)
                require(digest.hexdigest() == entry['sha256'], 'Fixture weight differs from pinned source')
                verified.append(dict(tensor=tensor, bytes=entry['bytes'], sha256=entry['sha256']))
    return verified


def analyze(raw):
    require(raw.get('complete') is True and raw.get('validation') is False and
            raw.get('device') == 'Apple M1 Pro' and raw.get('dispatch_repeats') == 32 and
            0 < raw.get('max_shared_buffer_bytes', 0) <= 128*1024**2, 'Incomplete bounded M1 timing')
    require([(c['K'], c['N']) for c in raw['cases']] == SHAPES and
            all(c.get('exact') is True and c['width'] == 8 for c in raw['cases']), 'Changed fixture coverage')
    require([(p['pair'], p['case']) for p in raw['pairs']] ==
            [(p, c) for p in range(5) for c in range(7)], 'Missing or duplicate paired timing')
    results = []; clean = True
    weighted = [0.0]*5
    for case in range(7):
        ratios = []; savings = []
        for p in (v for v in raw['pairs'] if v['case'] == case):
            require([a['candidate'] for a in p['arms']] == [bool(p['pair']%2), not bool(p['pair']%2)],
                    'Changed alternating order')
            values = {}
            for arm in p['arms']:
                for value in arm['sample'].values():
                    require(type(value) in (int, float) and math.isfinite(value) and value > 0, 'Invalid timing')
                a, b = (arm['memory_'+k] for k in ('before', 'after'))
                clean &= (a.get('compressed_bytes') == b.get('compressed_bytes') == 0 and
                    type(a.get('decompressions')) is int and a['decompressions'] == b.get('decompressions') and
                    type(a.get('system_swap_used_bytes')) is int and b.get('system_swap_used_bytes') == a['system_swap_used_bytes'])
                values[arm['candidate']] = arm['sample']['gpu_us']
            ratios.append(values[True]/values[False]); savings.append((values[False]-values[True])/1000)
            if case < 6: weighted[p['pair']] += savings[-1]*CALLS[case]
        results.append(dict(case=case, confidence_95=paired_log_interval(ratios),
                            savings_ms=savings, median_saving_ms=statistics.median(savings)))
    material = statistics.median(weighted) >= 20 and results[-1]['confidence_95']['high'] < 1
    return dict(status='memory_disturbed' if not clean else 'worth_request_screen' if material else 'insufficient_benefit',
        clean_memory=clean, cases=results, shape_frequency_projection_ms=weighted,
        median_shape_frequency_projection_ms=statistics.median(weighted), advance_to_request_screen=clean and material,
        normal_request_latency_qualified=False, production_promoted=False,
        limitations=['Old real activations, with all packed weights rehashed against the pinned mixed artifact.',
            'Only six matrices from two layers, width 8, even row counts. Width 4 and partial iterations are not qualified.',
            'Shape frequencies project isolated GPU savings, not request savings. Repeated weights change cache behavior.',
            'Five pairs in one process are correlated; intervals are screening evidence only.'])


def run(out, binary):
    exp = Experiment(out, 'q8_lookahead_screen_v1', [], [], 240)
    with exp:
        paths = [binary, Path(__file__).with_suffix('.py'), ROOT/'scripts/qwen/probe_q8_lookahead.mm',
                 *[p for d in FIXTURES for p in d.iterdir() if p.is_file()]]
        for path in paths: exp.frozen['files'][str(path.resolve())] = sha(path)
        save(exp.out/'identity.json', exp.frozen)
        exp.report['verified_weight_tensors'] = verify_weights(exp.model, exp.frozen)
        exp.persist()
        for mode in ('validate', 'timing'):
            exp.command([binary, ROOT/'kernels/metal/qwen.metal', *FIXTURES, exp.out/(mode+'.json'), mode],
                        mode, limit=90, validation=mode == 'validate')
        validation, timing = (load(exp.out/(name+'.json')) for name in ('validate', 'timing'))
        require(validation.get('complete') is True and validation.get('validation') is True and
                validation['cases'] == timing['cases'] and validation['shader_sha256'] == timing['shader_sha256'],
                'Validation and timing differ')
        exp.report.update(analyze(timing))
    return exp.report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output, args.binary.resolve())['complete'] else 2)
