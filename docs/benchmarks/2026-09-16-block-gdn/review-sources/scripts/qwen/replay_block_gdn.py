#!/usr/bin/env python3
"""Independent small operator screen using exact tensors, never capture timing."""
import argparse
import hashlib
from pathlib import Path
import shutil

import build_block_gdn as builder
import perfect_draft as verifier
import screen_block_gdn as screen
from benchmark_host import build_probe, preflight, observe as observe_host
from cache_residency import require
from capture_routes import load
from combined_q4 import freeze
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
from stage200 import Experiment
from verify_stage200 import verify_sources

ROOT = builder.ROOT
SOURCE = screen.BASE/'screen-03'
PROTOCOL = screen.BASE/'protocol-replay.md'


def source_proof(frozen, work):
    seal = sha(SOURCE/'evidence-files.json'); verify_seal(SOURCE, seal)
    saved, original = load(SOURCE/'summary.json'), load(SOURCE/'identity.json')
    require(saved['kind'] == 'block_gdn_screen_v1' and saved['complete'] is False and
        saved['status'] == 'resource_blocked' and 'validate' not in saved and 'timing' not in saved,
        'Changed fixture-only source disposition')
    require(all(original[k] == frozen[k] for k in ('build','artifact_revision','prepared_manifest_sha256')) and
        work == saved['workload'] == load(SOURCE/'workload.json'), 'Changed fixture source identity')
    exact = screen.capture_observation(SOURCE, original, work)
    require(exact == saved['capture'] and exact['exact_logits_state_routes'] is True and
        exact['exact_initial_cache'] is True, 'Source numerical correctness does not reproduce')
    require(saved['build_directory'] == str(ROOT/'.cache/block-gdn-build-04'), 'Changed fixture producer')
    proof = builder.verify(Path(saved['build_directory']), frozen['build'])
    require(proof['producer'] == load(SOURCE/'producer.json'), 'Changed source producer')
    weights = screen.verify_weights(SOURCE/'gdn-fixtures', ROOT/'.cache/qwen-mixed-reference', frozen)
    return dict(path=str(SOURCE), source_seal_sha256=seal, recorded_complete=False, recorded_status='resource_blocked',
        exact_logits_state_routes=True, capture_clean_memory=exact['clean_memory'], verified_weights=weights,
        use='verified tensor bytes and numerical identity only; no capture timing or memory samples reused')


def run(output):
    work, tokens = verifier.source_input()
    exp = Experiment(output, 'block_gdn_replay_v1', [], work, 180)
    with exp:
        source = source_proof(exp.frozen, work); directory = Path(load(SOURCE/'summary.json')['build_directory'])
        proof = builder.verify(directory, exp.frozen['build']); cfg = builder.settings(directory)
        host = build_probe(ROOT/'.cache/benchmark-host'/hashlib.sha256(str(exp.out).encode()).hexdigest())
        save(exp.out/'producer.json', proof['producer']); save(exp.out/'host-producer.json', host['producer'])
        shutil.copyfile(PROTOCOL, exp.out/'protocol.md'); fixtures = SOURCE/'gdn-fixtures'
        exp.report.update(criteria=screen.CRITERIA, limitations=screen.LIMITATIONS,
            source_proof=source, token_source=tokens, host_preflight=[], build_directory=str(directory),
            source_timing_reused=False, native_model_loaded=False)
        freeze(exp, [*proof['files'], *host['files'], *fixtures.iterdir(), PROTOCOL, exp.out/'protocol.md',
            exp.out/'producer.json', exp.out/'host-producer.json', SOURCE/'summary.json', SOURCE/'evidence-files.json'])
        exp.persist(); exp.guard.check_resources(initial=True)
        for mode in ('validate','timing'):
            preflight(exp, host, mode)
            exp.command([cfg['operator_binary'], fixtures/'manifest.json', exp.out/(mode+'.json'), mode],
                mode, limit=60, validation=mode == 'validate')
            result = screen.analyze(load(exp.out/(mode+'.json')), mode == 'validate', exp.frozen, load(fixtures/'manifest.json'))
            exp.report[mode] = result; exp.persist()
            if not result['clean_memory'] or not result['clean_host']: raise ResourceBlocked('Independent operator memory or host disturbance')
        exp.report.update(status=result['status'], advance_to_verifier_screen=result['advance_to_verifier_screen'])
    return exp.report


def audit(output):
    seal = sha(output/'evidence-files.json'); verify_seal(output, seal)
    saved, frozen = load(output/'summary.json'), load(output/'identity.json')
    require(saved['kind'] == 'block_gdn_replay_v1' and saved['criteria'] == screen.CRITERIA and
        saved['limitations'] == screen.LIMITATIONS and saved['source_timing_reused'] is False and
        saved['native_model_loaded'] is False and saved['identity'] == {k: frozen[k] for k in saved['identity']}, 'Changed replay scope')
    provenance = verify_sources(output.parent, frozen); work, tokens = verifier.source_input()
    require(work == saved['workload'] == load(output/'workload.json') and tokens == saved['token_source'], 'Changed replay inputs')
    require(saved['source_proof'] == source_proof(frozen, work), 'Changed tensor proof')
    proof = builder.verify(Path(saved['build_directory']), frozen['build'])
    require(proof['producer'] == load(output/'producer.json') and
        all(frozen['files'][str(p)] == sha(p) for p in proof['files']) and
        sha(output/'protocol.md') == frozen['files'][str(PROTOCOL)], 'Changed producer or protocol')
    host = load(output/'host-producer.json')
    require(host['complete'] is True and host['base_native_fingerprint'] == frozen['build'] and
        host['binary_sha256'] == sha(host['binary']) == frozen['files'][host['binary']], 'Changed host producer')
    for i, check in enumerate(saved['host_preflight']):
        path = output/(['validate','timing'][i]+'-host.json')
        require(check == dict(source=path.name, sha256=sha(path), observation=observe_host(load(path), frozen['build'])), 'Changed host check')
    for mode in ('validate','timing'):
        if mode in saved:
            require(saved[mode] == screen.analyze(load(output/(mode+'.json')), mode == 'validate', frozen,
                load(SOURCE/'gdn-fixtures/manifest.json')), 'Changed replay analysis')
    if saved['complete']:
        require(len(saved['host_preflight']) == 2 and all(saved[k]['clean_memory'] and saved[k]['clean_host'] for k in ('validate','timing')) and
            saved['status'] == saved['timing']['status'] and saved['advance_to_verifier_screen'] == saved['timing']['advance_to_verifier_screen'], 'Incomplete clean replay')
    else: require(saved['status'] in ('failed','resource_blocked','interrupted','time_budget_exhausted'), 'Invalid incomplete replay')
    return dict(kind='block_gdn_replay_audit_v1', complete=True, audit_passed=True, recorded_complete=saved['complete'],
        recorded_status=saved['status'], source_seal_sha256=seal, source_provenance=provenance, production_promoted=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['run','verify'])
    p.add_argument('--output', type=Path, required=True); p.add_argument('--source', type=Path)
    a = p.parse_args()
    if a.action == 'run': raise SystemExit(0 if run(a.output)['complete'] else 2)
    if a.source is None: p.error('--source required')
    save(a.output, audit(a.source.resolve()))
