#!/usr/bin/env python3
"""Capture actual four-token GDN inputs and screen one existing exact Q8 kernel."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import struct
import subprocess

import block_compute_profile as profile
import build_block_gdn as builder
import perfect_draft as verifier
from benchmark_host import build_probe, preflight, observe as observe_host
from cache_residency import require
from capture_block_profile import ANCHOR, admission_failure
from capture_routes import load
from combined_q4 import freeze
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
from screen_residency import paired_log_interval
from stage200 import Experiment
from verify_stage200 import verify_sources

ROOT = builder.ROOT
BASE = ROOT/'docs/benchmarks/2026-09-16-block-gdn'
SOURCE = ROOT/'docs/benchmarks/2026-09-16-block-compute/capture-03'
PROTOCOL = BASE/'protocol.md'
SHAPES = [(2560, 10240), (2560, 6144), (6144, 2560)]
NAMES = ['in_proj_qkv', 'in_proj_z', 'out_proj']
CRITERIA = dict(width=4, expert_slots=1460, memory_bytes=verifier.BUDGET,
    candidate_output_rows=2, control_output_rows=1, token_tile=4, pairs=5, repeats=32,
    minimum_projection_ms_per_token=10, paired_ratio_upper_below=1, stage_seconds=300,
    production_promoted=False, normal_request_latency_qualified=False)
LIMITATIONS = [
    'One layer and first four-token block provide real inputs for three recurring GDN shapes; other layers and contexts remain untested.',
    'Shape-frequency projections assume 36 calls of each shape per four-token block. They are isolated GPU opportunities, not predicted request savings.',
    'Repeated weights and five pairs within one process differ from inference. A promising screen still needs full-model state/recovery and normal verifier timing.',
    'Fixture capture adds waits and file writes. Capture timing is not a performance measurement; production kernels and defaults are unchanged.']


def capture_observation(output, frozen, work):
    raw = load(output/'capture.json')
    return profile.exact_observation(raw, frozen, work, sha(output/'workload.json'), load(ANCHOR), 'commands')


def verify_weights(directory, model, frozen):
    manifest = load(directory/'manifest.json')
    require(manifest['kind'] == 'captured_affine_operators' and manifest['build_fingerprint'] == frozen['build'] and
        manifest['artifact_revision'] == frozen['artifact_revision'] and len(manifest['cases']) == 3, 'Invalid fixture identity')
    mapping = load(model/'model.safetensors.index.json')['weight_map']; result = []
    for case, shape, name in zip(manifest['cases'], SHAPES, NAMES):
        require(case['matrix'] == dict(K=shape[0], N=shape[1], rows=4, group=64, bits=8, fused=False, gathered=False) and
            case['phase'] == 'decode' and case['context']['stage'] == 'gdn' and case['context']['layer'] == 0 and
            case['context']['offset'] == 72, 'Changed real input scope')
        require(set(case['tensors']) == {'w', 's', 'b', 'x'}, 'Unexpected fixture tensors')
        for entry in case['tensors'].values():
            file = entry['file']; require(Path(file).name == file and bool(file), 'Unconfined tensor path')
            p = directory/file
            require(p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256'], 'Changed fixture bytes')
        for key, suffix in (('w', 'weight'), ('s', 'scales'), ('b', 'biases')):
            tensor = f'language_model.model.layers.0.linear_attn.{name}.{suffix}'
            path = model/mapping[tensor]
            with path.open('rb') as f:
                length = struct.unpack('<Q', f.read(8))[0]; require(0 < length < 32*1024**2, 'Invalid tensor header')
                header = json.loads(f.read(length)); low, high = header[tensor]['data_offsets']
                entry = case['tensors'][key]
                require(0 <= low < high <= path.stat().st_size-length-8 and high-low == entry['bytes'], 'Invalid tensor bounds')
                f.seek(8+length+low); digest = hashlib.sha256(); remaining = high-low
                while remaining:
                    chunk = f.read(min(1024**2, remaining)); require(bool(chunk), 'Truncated source tensor')
                    digest.update(chunk); remaining -= len(chunk)
            require(digest.hexdigest() == entry['sha256'], 'Captured weight does not match pinned original')
            result.append(dict(tensor=tensor, bytes=entry['bytes'], sha256=entry['sha256']))
    return result


def analyze(raw, validation, frozen, manifest):
    require(raw.get('kind') == 'block_gdn_operator_v1' and raw.get('complete') is True and
        raw.get('validation') is validation and raw.get('build') == frozen['build'] and
        raw.get('artifact_revision') == frozen['artifact_revision'] and raw.get('source') == manifest and
        raw.get('device') == 'Apple M1 Pro' and 0 < raw.get('peak_gpu_bytes', 0) <= 256*1024**2 and
        raw.get('production_promoted') is False and raw.get('normal_request_latency_qualified') is False,
        'Incomplete or changed operator screen')
    hosts = [raw['host_before'], raw['host_after']]
    clean_host = all(h['thermal_state'] == 0 and h['low_power_mode'] is False and h['power_source'] == 'AC Power' for h in hosts)
    require(hosts[0]['monotonic_ns'] <= hosts[1]['monotonic_ns'], 'Reversed host clock')
    require([(c['K'], c['N']) for c in raw['cases']] == SHAPES, 'Missing operator shape')
    memory = [raw['memory_before'], raw['memory_after_destroy']]; expected_pairs = 1 if validation else 5
    weighted = [[0., 0.] for _ in range(expected_pairs)]; cases = []
    for case in raw['cases']:
        require(case['tokens'] == 4 and [p['pair'] for p in case['pairs']] == list(range(expected_pairs)) and
            [a['candidate'] for a in case['warmup']] == [False, True], 'Changed paired coverage')
        ratios = []
        for pair in case['pairs']:
            require([a['candidate'] for a in pair['arms']] == [bool(pair['pair']%2), not bool(pair['pair']%2)], 'Changed alternating order')
            times = {}
            for arm in pair['arms']:
                require(type(arm['candidate']) is bool and arm['repeats'] == (1 if validation else 32), 'Changed arm or repeats')
                times[arm['candidate']] = arm['gpu_ns']/arm['repeats']
                weighted[pair['pair']][int(arm['candidate'])] += times[arm['candidate']]*36/4e6
            ratios.append(times[True]/times[False])
        for arm in [*case['warmup'], *[a for p in case['pairs'] for a in p['arms']]]:
            require(arm['exact'] is True and arm['output_sha256'] == case['reference_sha256'] and
                type(arm['gpu_ns']) is int and arm['gpu_ns'] > 0 and type(arm['wall_ns']) is int and arm['wall_ns'] > 0,
                'Invalid timing or changed output')
            memory += [arm['memory_before'], arm['memory_after']]
        require(all(a['repeats'] == 1 for a in case['warmup']), 'Changed warmup')
        cases.append(dict(K=case['K'], N=case['N'], paired_ratios=ratios))
    fields = ('physical_footprint_bytes', 'physical_footprint_peak_bytes', 'compressed_bytes', 'compressed_peak_bytes',
        'decompressions', 'system_swap_used_bytes')
    require(all(type(v.get(k)) is int and v[k] >= 0 for v in memory for k in fields) and
        all(0 < v['physical_footprint_bytes'] <= v['physical_footprint_peak_bytes'] <= 2*1024**3 for v in memory),
        'Missing or excessive operator memory')
    clean = (all(v['compressed_bytes'] == v['compressed_peak_bytes'] == 0 for v in memory) and
        len({v['decompressions'] for v in memory}) == len({v['system_swap_used_bytes'] for v in memory}) == 1)
    result = dict(exact=True, clean_memory=clean, clean_host=clean_host, cases=cases,
        peak_physical_bytes=max(v['physical_footprint_peak_bytes'] for v in memory), production_promoted=False,
        normal_request_latency_qualified=False)
    if validation: return result
    ratios = [b/a for a, b in weighted]; savings = [a-b for a, b in weighted]; confidence = paired_log_interval(ratios)
    promising = clean and clean_host and confidence['high'] < 1 and statistics.median(savings) >= 10
    return dict(result, status='worth_verifier_screen' if promising else 'insufficient_isolated_benefit',
        advance_to_verifier_screen=promising, shape_frequency_projection_ms=savings,
        median_projection_ms_per_token=statistics.median(savings), weighted_ratios=ratios, confidence_95=confidence,
        limitations=LIMITATIONS)


def run(output, directory):
    work, tokens = verifier.source_input()
    exp = Experiment(output, 'block_gdn_screen_v1', [verifier.configuration(4, expert_slots=1460)], work, 300)
    with exp:
        verify_seal(SOURCE, sha(SOURCE/'evidence-files.json')); require(load(SOURCE/'summary.json')['complete'] is True, 'Incomplete profile source')
        proof = builder.verify(directory, exp.frozen['build']); cfg = builder.settings(directory)
        host = build_probe(ROOT/'.cache/benchmark-host'/hashlib.sha256(str(exp.out).encode()).hexdigest())
        save(exp.out/'producer.json', proof['producer']); save(exp.out/'host-producer.json', host['producer'])
        shutil.copyfile(PROTOCOL, exp.out/'protocol.md')
        exp.report.update(criteria=CRITERIA, limitations=LIMITATIONS, token_source=tokens, build_directory=str(directory),
            profile_source_seal_sha256=sha(SOURCE/'evidence-files.json'), host_preflight=[])
        freeze(exp, [*proof['files'], *host['files'], PROTOCOL, exp.out/'protocol.md', exp.out/'producer.json',
            exp.out/'host-producer.json', ANCHOR, SOURCE/'summary.json', SOURCE/'evidence-files.json'])
        exp.persist(); exp.guard.check_resources(initial=True); preflight(exp, host, 'capture')
        try: exp.command([cfg['binary'], exp.model, exp.prepared, exp.out/'workload.json', exp.out/'capture.json', '4', 'timing', '1460'], 'capture', limit=120)
        except subprocess.CalledProcessError:
            if (exp.out/'capture.json').exists() and admission_failure(load(exp.out/'capture.json')):
                raise ResourceBlocked('Fixed fixture capture memory admission failed') from None
            raise
        exact = capture_observation(exp.out, exp.frozen, work); exp.report['capture'] = exact; exp.persist()
        if not exact['clean_memory'] or not exact['clean_host']: raise ResourceBlocked('Fixture capture memory or host disturbed')
        fixtures = exp.out/'gdn-fixtures'; exp.report['verified_weights'] = verify_weights(fixtures, exp.model, exp.frozen)
        freeze(exp, list(fixtures.iterdir())); exp.persist()
        for mode in ('validate', 'timing'):
            preflight(exp, host, mode)
            exp.command([cfg['operator_binary'], fixtures/'manifest.json', exp.out/(mode+'.json'), mode], mode, limit=60, validation=mode == 'validate')
            result = analyze(load(exp.out/(mode+'.json')), mode == 'validate', exp.frozen, load(fixtures/'manifest.json'))
            exp.report[mode] = result; exp.persist()
            if not result['clean_memory'] or not result['clean_host']: raise ResourceBlocked('Operator screen memory or host disturbed')
        exp.report.update(status=result['status'], advance_to_verifier_screen=result['advance_to_verifier_screen'])
    return exp.report


def audit(output):
    seal = sha(output/'evidence-files.json'); verify_seal(output, seal)
    saved, frozen = load(output/'summary.json'), load(output/'identity.json')
    require(saved['kind'] == 'block_gdn_screen_v1' and saved['criteria'] == CRITERIA and saved['limitations'] == LIMITATIONS and
        saved['identity'] == {k: frozen[k] for k in saved['identity']}, 'Changed screen identity')
    provenance = verify_sources(output.parent, frozen)
    producer = load(output/'producer.json')
    if saved['complete']:
        proof = builder.verify(Path(saved['build_directory']), frozen['build'])
        require(proof['producer'] == producer and all(frozen['files'][str(p)] == sha(p) for p in proof['files']), 'Changed producer')
    else:
        # A failed capture must stay auditable after its builder is corrected.
        # verify_sources above checks changed tooling against frozen snapshots;
        # compiled/generated outputs and all non-tool inputs must still match.
        require(producer['complete'] is True and producer['base_native_fingerprint'] == frozen['build'], 'Invalid historical producer')
        for group in ('original_inputs', 'generated', 'binaries', 'objects'):
            for name, digest in producer[group].items():
                require(frozen['files'][name] == digest, 'Unfrozen historical producer')
                if Path(name).relative_to(ROOT).parts[0] != 'scripts':
                    require(sha(name) == digest, 'Changed historical producer payload')
    verify_seal(SOURCE, saved['profile_source_seal_sha256'])
    require(sha(output/'protocol.md') == frozen['files'][str(PROTOCOL)], 'Changed protocol')
    work, tokens = verifier.source_input(); require(work == saved['workload'] == load(output/'workload.json') and tokens == saved['token_source'], 'Changed inputs')
    for i, check in enumerate(saved['host_preflight']):
        path = output/(['capture', 'validate', 'timing'][i]+'-host.json')
        require(check == dict(source=path.name, sha256=sha(path), observation=observe_host(load(path), frozen['build'])), 'Changed host observation')
    if 'capture' in saved: require(saved['capture'] == capture_observation(output, frozen, work), 'Changed capture exactness')
    if 'verified_weights' in saved:
        require(saved['verified_weights'] == verify_weights(output/'gdn-fixtures', ROOT/'.cache/qwen-mixed-reference', frozen), 'Changed weights')
    for mode in ('validate', 'timing'):
        if mode in saved: require(saved[mode] == analyze(load(output/(mode+'.json')), mode == 'validate', frozen, load(output/'gdn-fixtures/manifest.json')), 'Changed operator result')
    if saved['complete']:
        require(len(saved['host_preflight']) == 3 and all(saved[k]['clean_memory'] and saved[k]['clean_host'] for k in ('capture','validate','timing')) and
            saved['status'] == saved['timing']['status'] and saved['advance_to_verifier_screen'] == saved['timing']['advance_to_verifier_screen'], 'Incomplete screen')
    else: require(saved['status'] in ('failed','resource_blocked','interrupted','time_budget_exhausted'), 'Invalid incomplete status')
    return dict(kind='block_gdn_audit_v1', complete=True, audit_passed=True, recorded_complete=saved['complete'],
        recorded_status=saved['status'], source_seal_sha256=seal, source_provenance=provenance, production_promoted=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['run','verify'])
    p.add_argument('--output', type=Path, required=True); p.add_argument('--build-directory', type=Path); p.add_argument('--source', type=Path)
    a = p.parse_args()
    if a.action == 'run':
        if a.build_directory is None: p.error('--build-directory required')
        raise SystemExit(0 if run(a.output, a.build_directory.resolve())['complete'] else 2)
    else:
        if a.source is None: p.error('--source required')
        save(a.output, audit(a.source.resolve()))
