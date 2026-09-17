#!/usr/bin/env python3
"""Isolate one shared-expert prelude in the verified physical-read diagnostic."""
import argparse
import math
from pathlib import Path
import shutil
import struct

from cache_residency import require
from capture_routes import load
from combined_q4 import freeze, shader_origin, state_proof
from native_q4_replay import valid_sha256
import q4_read_arrivals as arrivals
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
from screen_q4_packed import FIXTURES, verify_records
from stage200 import Experiment
from verify_stage200 import verify_sources

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/'docs/benchmarks/2026-09-15-q4-shared-arrivals/protocol.md'
REFERENCE = PROTOCOL.parent/'reference'
CONDITIONS = ('control', 'shared')
CRITERIA = dict(arrivals.CRITERIA, hits_per_batch=2,
    cache_preparation=arrivals.INVALIDATE_FILES, shared_chains_per_batch=1,
    shared_layer_order=arrivals.SHARED_LAYERS,
    shared_gpu_scope='shared-and-routed-command-intervals')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def shared_proof(raw, manifest, gate_values=None):
    from shared_expert_reference import validate_manifest
    validate_manifest(manifest)
    shared = raw.get('shared_prelude', {})
    require(shared.get('enabled') is True and shared.get('reference_manifest') == manifest and
            valid_sha256(shared.get('reference_manifest_sha256')) and
            valid_sha256(shared.get('expected_native_sha256')) and
            shared.get('cpu_reference_passed') is True and shared.get('shared_outputs_exact') is True and
            shared.get('layer_order') == arrivals.SHARED_LAYERS and
            shared.get('gpu_scope') == CRITERIA['shared_gpu_scope'],
            'Missing independent shared-reference or native exact-output evidence')
    checks = shared.get('reference_checks')
    require(isinstance(checks, list) and [(c.get('layer'), c.get('row')) for c in checks] ==
            [(layer, row) for layer in (0, 16, 32, 47) for row in range(8)],
            'Missing, duplicate or reordered shared reference cases')
    limits = manifest['tolerance']
    for check in checks:
        for name in ('activation', 'shared'):
            measured = check.get(name, {})
            require(finite(measured.get('relative_l2')) and
                    0 <= measured['relative_l2'] <= limits['vector_relative_l2_max'] and
                    finite(measured.get('cosine')) and
                    limits['vector_cosine_min'] <= measured['cosine'] <= 1+1e-12,
                    'Shared CPU vector reference exceeds its predeclared tolerance')
        measured = check.get('gate', {})
        require(all(finite(measured.get(k)) for k in ('expected', 'absolute_error', 'limit')),
                'Missing scalar reference value or error')
        expected_limit = limits['scalar_absolute_max']+abs(measured['expected'])*limits['scalar_relative_max']
        require(math.isclose(measured['limit'], expected_limit, rel_tol=1e-12, abs_tol=1e-15) and
                0 <= measured['absolute_error'] <= expected_limit,
                'Shared gate exceeds or changed its predeclared tolerance')
        if gate_values is not None:
            require(measured['expected'] == gate_values[check['layer']][check['row']],
                    'Reported gate oracle differs from its verified CPU fixture')
    return dict(reference_manifest_sha256=shared['reference_manifest_sha256'],
                expected_native_sha256=shared['expected_native_sha256'],
                verified_cases=len(checks), independent_cpu_reference_passed=True,
                shared_outputs_exact=True)


def analyze(raw, condition, frozen=None, expected_records=None, reference_manifest=None, gate_values=None):
    require(condition in CONDITIONS and raw.get('hits_per_batch') == 2 and
            raw.get('cache_preparation') == arrivals.INVALIDATE_FILES,
            'Changed shared-prelude condition, hit population or cache preparation')
    shared = condition == 'shared'
    if shared:
        require(reference_manifest is not None, 'Shared diagnostic has no verified independent reference')
        proof = shared_proof(raw, reference_manifest, gate_values)
    else:
        require('shared_prelude' not in raw, 'Fresh control unexpectedly includes shared work')
    result = arrivals.analyze(raw, frozen, expected_records, shared=shared)
    result['condition'] = condition
    if shared: result['shared_reference'] = proof
    result['limitations'].append('The control and shared conditions are separate processes. Their absolute timings are contextual, not a paired estimate of shared-work cost.')
    return result


def reference_identity(manifest, frozen):
    expected = {'generator_sha256': ROOT/'scripts/qwen/shared_expert_reference.py',
                'bf16_reference_sha256': ROOT/'scripts/qwen/reference_numpy.py',
                'artifact_lock_sha256': ROOT/'mixed-models.lock.json',
                'fixture_manifest_sha256': FIXTURES/'manifest.json'}
    require(all(frozen['files'].get(str(path.resolve())) == manifest[key] for key, path in expected.items()),
            'Shared CPU reference producer, rounding source, artifact lock or inputs differ from the frozen experiment')


def reference_files(directory):
    require(directory.is_dir(), 'Missing shared CPU reference directory')
    files = sorted(p for p in directory.iterdir() if p.is_file())
    require(files and all(not p.is_symlink() for p in files), 'Invalid shared CPU reference files')
    return files


def reference_inputs(directory):
    from shared_expert_reference import verify_reference
    manifest = verify_reference(directory)
    require(manifest['fixture_manifest_sha256'] == sha(FIXTURES/'manifest.json'),
            'Shared CPU reference uses different original Q4 token inputs')
    original = load(FIXTURES/'manifest.json')
    gates = {}
    for layer in (0, 16, 32, 47):
        item = manifest['layers'][str(layer)]
        require(item['inputs'] == original['layers'][str(layer)]['inputs'] and
                item['offsets'] == original['layers'][str(layer)]['offsets'],
                'Shared CPU reference has changed token rows or inputs')
        data = (directory/item['gate']['file']).read_bytes()
        require(len(data) == 8*4, 'Shared scalar reference has wrong shape')
        gates[layer] = struct.unpack('<8f', data)
    return manifest, gates


def run(args):
    exp = Experiment(args.output, 'q4_shared_arrivals_diagnostic_v1', [], [], CRITERIA['stage_seconds'])
    with exp:
        exp.report.update(condition=args.condition, hits_per_batch=2, criteria=CRITERIA,
                          shader_origin=shader_origin(), cache_preparation=arrivals.INVALIDATE_FILES)
        shutil.copyfile(PROTOCOL, exp.out/'protocol.md')
        exp.report['protocol_sha256'] = sha(PROTOCOL)
        manifest, gates = reference_inputs(args.shared_fixtures)
        exp.report['shared_reference_source'] = dict(path=str(args.shared_fixtures.resolve()),
            manifest_sha256=sha(args.shared_fixtures/'manifest.json'), manifest=manifest)
        exp.report['correctness'] = state_proof(args.state, exp.frozen)
        exp.report['control'] = arrivals.control_proof(args.control, exp.frozen)
        for key in ('state', 'control'):
            path = getattr(args, key)
            exp.report[key+'_source'] = dict(path=str(path.resolve()), sha256=sha(path/'evidence-files.json'))
        freeze(exp, [args.binary, PROTOCOL, exp.out/'protocol.md',
            ROOT/'scripts/qwen/probe_q4_packed.metal', ROOT/'kernels/metal/qwen.metal',
            *reference_files(args.shared_fixtures),
            *[p for d in (args.state, args.control, FIXTURES) for p in d.iterdir() if p.is_file()]])
        reference_identity(manifest, exp.frozen)
        exp.report['verified_records'] = verify_records(exp.prepared); exp.persist()
        exp.guard.check_resources(initial=True)
        for mode in ('check', 'timing', 'trace'):
            extra = [args.shared_fixtures] if args.condition == 'shared' else []
            exp.command([args.binary, FIXTURES, exp.out/(mode+'.json'), 'arrivals', exp.model, exp.prepared,
                         '2', mode, 'invalidate', *extra], mode, CRITERIA['process_seconds'], validation=mode == 'check')
            raw = load(exp.out/(mode+'.json'))
            require(raw.get('mode') == mode, 'Native shared replay returned wrong mode')
            if args.condition == 'shared':
                require(raw['shared_prelude']['reference_manifest_sha256'] == sha(args.shared_fixtures/'manifest.json'),
                        'Native shared replay used another CPU fixture')
            exp.report[mode] = analyze(raw, args.condition, exp.frozen, exp.report['verified_records'], manifest, gates)
            exp.persist()
            if not all(exp.report[mode][k] for k in ('clean_memory', 'clean_host')):
                raise ResourceBlocked('Shared replay has disturbed or unavailable memory/host observations')
        if args.condition == 'shared':
            require(len({exp.report[m]['shared_reference']['expected_native_sha256'] for m in ('check', 'timing', 'trace')}) == 1,
                    'Native shared outputs differ between independent mode processes')
        exp.report.update(status=exp.report['timing']['status'], normal_request_latency_qualified=False, production_promoted=False)
    return exp.report


def verify(directory):
    directory = directory.resolve(); seal_digest = sha(directory/'evidence-files.json')
    verify_seal(directory, seal_digest)
    saved, frozen = load(directory/'summary.json'), load(directory/'identity.json')
    require(saved.get('kind') == 'q4_shared_arrivals_diagnostic_v1' and saved.get('condition') in CONDITIONS and
            saved.get('hits_per_batch') == 2 and saved.get('cache_preparation') == arrivals.INVALIDATE_FILES and
            saved.get('criteria') == CRITERIA and saved.get('configurations') == [] and saved.get('workload') == [] and
            saved['identity'] == {k: frozen[k] for k in saved['identity']}, 'Changed shared experiment identity or design')
    sources = dict(frozen, files={p: h for p, h in frozen['files'].items() if Path(p).is_relative_to(frozen['root'])})
    provenance = verify_sources(directory.parent, sources)
    require(sha(directory/'protocol.md') == saved['protocol_sha256'] == frozen['files'][str(PROTOCOL.resolve())],
            'Changed declared shared arrival protocol')
    require(saved.get('shader_origin') == shader_origin(), 'Changed shared integrated shader origin')
    for key, checker in (('state', state_proof), ('control', arrivals.control_proof)):
        if key+'_source' not in saved: continue
        source = saved[key+'_source']; path = Path(source['path'])
        require(sha(path/'evidence-files.json') == source['sha256'] and
                checker(path, frozen) == saved['correctness' if key == 'state' else 'control'], 'Changed prior '+key+' evidence')
    source = saved['shared_reference_source']; fixture_path = Path(source['path'])
    manifest, gates = reference_inputs(fixture_path)
    reference_identity(manifest, frozen)
    require(manifest == source['manifest'] and sha(fixture_path/'manifest.json') == source['manifest_sha256'],
            'Changed shared reference source')
    for file in reference_files(fixture_path):
        require(frozen['files'].get(str(file.resolve())) == sha(file), 'Changed frozen shared reference file')
    results = {}
    for mode in ('check', 'timing', 'trace'):
        if mode not in saved: continue
        raw = load(directory/(mode+'.json'))
        require(raw.get('mode') == mode, 'Wrong captured shared mode')
        if saved['condition'] == 'shared':
            require(raw['shared_prelude']['reference_manifest_sha256'] == source['manifest_sha256'], 'Wrong captured shared reference')
        results[mode] = analyze(raw, saved['condition'], frozen, saved['verified_records'], manifest, gates)
        require(saved[mode] == results[mode], 'Changed shared '+mode+' analysis')
    if saved['complete'] is True:
        require(set(results) == {'check', 'timing', 'trace'} and 'state_source' in saved and 'control_source' in saved and
                saved['status'] == results['timing']['status'] and
                all(r[k] for r in results.values() for k in ('clean_memory', 'clean_host')), 'Incomplete or disturbed completed shared stage')
        if saved['condition'] == 'shared':
            require(len({r['shared_reference']['expected_native_sha256'] for r in results.values()}) == 1,
                    'Changed shared outputs across mode processes')
    else:
        require(saved['status'] in ('failed', 'resource_blocked', 'interrupted', 'time_budget_exhausted'),
                'Unfinished shared stage lacks a terminal disposition')
    return dict(kind='q4_shared_arrivals_audit_v1', complete=True, audit_passed=True,
        source_seal_sha256=seal_digest, source_provenance=provenance, condition=saved['condition'],
        recorded_complete=saved['complete'], recomputed_status=saved['status'], modes=results,
        normal_request_latency_qualified=False, production_promoted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='mode', required=True)
    execute = sub.add_parser('run'); execute.add_argument('--output', type=Path, required=True)
    execute.add_argument('--condition', choices=CONDITIONS, required=True)
    execute.add_argument('--shared-fixtures', type=Path, default=REFERENCE)
    execute.add_argument('--binary', type=Path, default=ROOT/'build/qwen/qwen_q4_check')
    execute.add_argument('--state', type=Path, default=arrivals.STATE)
    execute.add_argument('--control', type=Path, default=arrivals.CONTROL)
    audit = sub.add_parser('verify'); audit.add_argument('directory', type=Path)
    audit.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'verify': save(args.output, verify(args.directory))
    else:
        for field in ('binary', 'state', 'control', 'shared_fixtures'): setattr(args, field, getattr(args, field).resolve())
        raise SystemExit(0 if run(args)['complete'] else 2)
