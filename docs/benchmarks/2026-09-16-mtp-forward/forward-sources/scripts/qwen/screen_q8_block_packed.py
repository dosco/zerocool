#!/usr/bin/env python3
"""Screen one packed-Q8 block kernel on verified real inputs, before model timing."""
import argparse
import hashlib
from pathlib import Path
import re
import shutil

import build_q8_block_packed as builder
import perfect_draft as verifier
import replay_block_gdn as fixtures
import screen_block_gdn as screen
from benchmark_host import build_probe, preflight, observe as observe_host
from cache_residency import require
from capture_routes import load
from combined_q4 import freeze
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
from stage200 import Experiment
from verify_stage200 import verify_sources

ROOT = builder.ROOT
BASE = ROOT/'docs/benchmarks/2026-09-16-q8-block-packed'
PROTOCOL = BASE/'protocol.md'
CRITERIA = dict(screen.CRITERIA, candidate_output_rows=1, weight_load_bits=32, stage_seconds=180)
LIMITATIONS = [*screen.LIMITATIONS[:3],
    'Only verified tensor bytes and numerical identity are reused from the memory-disturbed capture; every timing and memory observation is fresh.',
    'The packed candidate has one output row and four tokens. It is not combined with the prior output-row-pair candidate.']


def analyze(raw, validation, frozen, manifest):
    require(raw.get('kind') == 'q8_block_packed_operator_v1' and raw.get('variant') == builder.VARIANT,
        'Changed packed-Q8 operator identity')
    for case in raw['cases']:
        require(re.fullmatch('[0-9a-f]{64}', case['reference_sha256']) is not None, 'Invalid reference hash')
        for arm in [*case['warmup'], *[a for p in case['pairs'] for a in p['arms']]]:
            kernel = builder.VARIANT['candidate' if arm['candidate'] else 'control']
            require(arm['kernel_dispatches'] == {kernel: arm['repeats']}, 'Wrong native kernel dispatches')
    # Reuse the established shape, exactness, alternating-pair, native-memory and
    # materiality checks. The raw report keeps its distinct experiment identity.
    result = screen.analyze(dict(raw, kind='block_gdn_operator_v1'), validation, frozen, manifest)
    if not validation: result['limitations'] = LIMITATIONS
    return result


def references(raw):
    return [(c['K'], c['N'], c['reference_sha256']) for c in raw['cases']]


def run(output, directory):
    work, tokens = verifier.source_input(); exp = Experiment(output, 'q8_block_packed_screen_v1', [], work, 180)
    with exp:
        source = fixtures.source_proof(exp.frozen, work)
        proof = builder.verify(directory, exp.frozen['build']); cfg = builder.settings(directory)
        host = build_probe(ROOT/'.cache/benchmark-host'/hashlib.sha256(str(exp.out).encode()).hexdigest())
        save(exp.out/'producer.json', proof['producer']); save(exp.out/'host-producer.json', host['producer'])
        shutil.copyfile(PROTOCOL, exp.out/'protocol.md'); tensor_dir = fixtures.SOURCE/'gdn-fixtures'
        exp.report.update(criteria=CRITERIA, limitations=LIMITATIONS, variant=builder.VARIANT,
            source_proof=source, token_source=tokens, host_preflight=[], build_directory=str(directory),
            source_timing_reused=False, native_model_loaded=False)
        freeze(exp, [*proof['files'], *host['files'], *tensor_dir.iterdir(), PROTOCOL, exp.out/'protocol.md',
            exp.out/'producer.json', exp.out/'host-producer.json', fixtures.SOURCE/'summary.json', fixtures.SOURCE/'evidence-files.json'])
        exp.persist(); exp.guard.check_resources(initial=True); expected = None
        for mode in ('validate','timing'):
            preflight(exp, host, mode)
            exp.command([cfg['binary'], tensor_dir/'manifest.json', exp.out/(mode+'.json'), mode],
                mode, limit=60, validation=mode == 'validate')
            raw = load(exp.out/(mode+'.json'))
            result = analyze(raw, mode == 'validate', exp.frozen, load(tensor_dir/'manifest.json'))
            if expected is None: expected = references(raw)
            else: require(references(raw) == expected, 'Cross-process reference outputs changed')
            exp.report[mode] = result; exp.persist()
            if not result['clean_memory'] or not result['clean_host']: raise ResourceBlocked('Packed-Q8 operator memory or host disturbance')
        exp.report.update(status=result['status'], advance_to_verifier_screen=result['advance_to_verifier_screen'])
    return exp.report


def audit(output):
    seal = sha(output/'evidence-files.json'); verify_seal(output, seal)
    saved, frozen = load(output/'summary.json'), load(output/'identity.json')
    require(saved['kind'] == 'q8_block_packed_screen_v1' and saved['criteria'] == CRITERIA and
        saved['limitations'] == LIMITATIONS and saved['variant'] == builder.VARIANT and saved['source_timing_reused'] is False and
        saved['native_model_loaded'] is False and saved['identity'] == {k: frozen[k] for k in saved['identity']}, 'Changed packed screen scope')
    provenance = verify_sources(output.parent, frozen); work, tokens = verifier.source_input()
    require(work == saved['workload'] == load(output/'workload.json') and tokens == saved['token_source'], 'Changed inputs')
    require(saved['source_proof'] == fixtures.source_proof(frozen, work), 'Changed tensor proof')
    proof = builder.verify(Path(saved['build_directory']), frozen['build'])
    require(proof['producer'] == load(output/'producer.json') and all(frozen['files'][str(p)] == sha(p) for p in proof['files']) and
        sha(output/'protocol.md') == frozen['files'][str(PROTOCOL)], 'Changed producer or protocol')
    host = load(output/'host-producer.json')
    require(host['complete'] is True and host['base_native_fingerprint'] == frozen['build'] and
        host['binary_sha256'] == sha(host['binary']) == frozen['files'][host['binary']], 'Changed host producer')
    for i, check in enumerate(saved['host_preflight']):
        path = output/(['validate','timing'][i]+'-host.json')
        require(check == dict(source=path.name, sha256=sha(path), observation=observe_host(load(path), frozen['build'])), 'Changed host check')
    for mode in ('validate','timing'):
        if mode in saved:
            require(saved[mode] == analyze(load(output/(mode+'.json')), mode == 'validate', frozen,
                load(fixtures.SOURCE/'gdn-fixtures/manifest.json')), 'Changed analysis')
    if saved['complete']:
        require(len(saved['host_preflight']) == 2 and all(saved[k]['clean_memory'] and saved[k]['clean_host'] for k in ('validate','timing')) and
            saved['status'] == saved['timing']['status'] and saved['advance_to_verifier_screen'] == saved['timing']['advance_to_verifier_screen'] and
            references(load(output/'validate.json')) == references(load(output/'timing.json')), 'Incomplete clean screen')
    else: require(saved['status'] in ('failed','resource_blocked','interrupted','time_budget_exhausted'), 'Invalid incomplete screen')
    return dict(kind='q8_block_packed_audit_v1', complete=True, audit_passed=True, recorded_complete=saved['complete'],
        recorded_status=saved['status'], source_seal_sha256=seal, source_provenance=provenance, production_promoted=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['run','verify'])
    p.add_argument('--output', type=Path, required=True); p.add_argument('--source', type=Path); p.add_argument('--build-directory', type=Path)
    a = p.parse_args()
    if a.action == 'run':
        if a.build_directory is None: p.error('--build-directory required')
        raise SystemExit(0 if run(a.output, a.build_directory.resolve())['complete'] else 2)
    if a.source is None: p.error('--source required')
    save(a.output, audit(a.source.resolve()))
