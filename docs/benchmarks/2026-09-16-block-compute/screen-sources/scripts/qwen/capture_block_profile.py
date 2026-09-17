#!/usr/bin/env python3
"""Capture complete command and dispatch profiles for the fixed four-token verifier."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess

import block_compute_profile as analysis
import build_block_profile as builder
import perfect_draft as verifier
from benchmark_host import build_probe, preflight, observe as observe_host
from cache_residency import require
from capture_routes import load
from combined_q4 import freeze
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
from stage200 import Experiment
from verify_stage200 import verify_sources

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'docs/benchmarks/2026-09-16-block-compute'
PROTOCOL = BASE/'protocol.md'
SOURCE = ROOT/'docs/benchmarks/2026-09-16-perfect-draft-capacity/screen-02'
ANCHOR = SOURCE/'pair-0-slots-1460-width-4.json'
CRITERIA = dict(modes=list(builder.MODES), expert_slots=1460, width=4, memory_bytes=verifier.BUDGET,
    context=8192, prompt_tokens=72, continuation_tokens=16, profile_workspace_bytes=builder.WORKSPACE,
    process_seconds=150, stage_seconds=360, observed_compression_allowed=False,
    performance_measurement=False, production_promoted=False)


def observation(output, stem, frozen, work):
    path = output/(stem+'.json'); raw = load(path)
    exact = analysis.exact_observation(raw, frozen, work, sha(output/'workload.json'), load(ANCHOR), stem)
    profiles = []
    for i, block in enumerate(raw['blocks']):
        name = f'{stem}-block-{4*i}.profile.json'
        require(block['profile_file'] == name and block['profile_sha256'] == sha(output/name), 'Changed block profile')
        require((output/name).stat().st_size <= 32*1024**2, 'Oversized profile file')
        profiles.append(load(output/name))
    result = analysis.analyze(raw, profiles, frozen['build'], frozen['artifact_revision'], stem)
    return dict(mode=stem, source=path.name, sha256=sha(path), exact=exact, analysis=result)


def run(output, binaries):
    work, token_source = verifier.source_input()
    exp = Experiment(output, 'block_compute_capture_v1', [verifier.configuration(4, expert_slots=1460)], work, 360)
    with exp:
        exp.report.update(criteria=CRITERIA, limitations=analysis.LIMITATIONS, captures=[], token_source=token_source,
            binaries={k: str(v) for k, v in binaries.items()}, performance_measurement=False,
            production_promoted=False, host_preflight=[])
        verify_seal(SOURCE, sha(SOURCE/'evidence-files.json'))
        exp.report['reference_source'] = dict(path=str(ANCHOR), sha256=sha(ANCHOR), source_seal_sha256=sha(SOURCE/'evidence-files.json'))
        files = [ANCHOR, SOURCE/'evidence-files.json', PROTOCOL]
        for mode in builder.MODES:
            proof = builder.verify(binaries[mode], exp.frozen['build'], mode)
            save(exp.out/(mode+'-producer.json'), proof['producer']); files += proof['files']
        host = build_probe(ROOT/'.cache/benchmark-host'/hashlib.sha256(str(exp.out).encode()).hexdigest())
        save(exp.out/'host-producer.json', host['producer']); files += host['files']
        shutil.copyfile(PROTOCOL, exp.out/'protocol.md')
        files += [exp.out/'protocol.md', exp.out/'host-producer.json',
            *[exp.out/(mode+'-producer.json') for mode in builder.MODES],
            *[p for p in verifier.SOURCE.iterdir() if p.is_file()]]
        freeze(exp, files); exp.persist(); exp.guard.check_resources(initial=True)
        for mode in builder.MODES:
            preflight(exp, host, mode); path = exp.out/(mode+'.json')
            try:
                exp.command([binaries[mode], exp.model, exp.prepared, exp.out/'workload.json', path,
                    '4', 'timing', '1460'], mode, limit=150)
            except subprocess.CalledProcessError:
                if path.is_file() and load(path).get('error') in ('fixed memory admission failed', 'checkpoint workspace not admitted'):
                    raise ResourceBlocked('Fixed profile memory admission failed') from None
                raise
            result = observation(exp.out, mode, exp.frozen, work)
            exp.report['captures'].append(result); exp.persist()
            if not result['exact']['clean_memory'] or not result['exact']['clean_host']:
                raise ResourceBlocked('Block profile memory or host is disturbed')
        exp.report.update(status='captured', comparison=analysis.compare(*[c['analysis'] for c in exp.report['captures']]))
    return exp.report


def audit(output):
    seal = sha(output/'evidence-files.json'); verify_seal(output, seal)
    saved, frozen = load(output/'summary.json'), load(output/'identity.json')
    require(saved['kind'] == 'block_compute_capture_v1' and saved['criteria'] == CRITERIA and
        saved['limitations'] == analysis.LIMITATIONS and saved['performance_measurement'] is False and
        saved['production_promoted'] is False and saved['identity'] == {k: frozen[k] for k in saved['identity']} and
        saved['configurations'] == [verifier.configuration(4, expert_slots=1460)], 'Changed capture protocol')
    provenance = verify_sources(output.parent, frozen)
    work, token_source = verifier.source_input()
    require(work == saved['workload'] == load(output/'workload.json') and token_source == saved['token_source'] and
        sha(output/'protocol.md') == frozen['files'][str(PROTOCOL)] and
        saved['reference_source'] == dict(path=str(ANCHOR), sha256=sha(ANCHOR), source_seal_sha256=sha(SOURCE/'evidence-files.json')) and
        frozen['files'][str(ANCHOR)] == sha(ANCHOR), 'Changed workload or reference')
    verify_seal(SOURCE, saved['reference_source']['source_seal_sha256'])
    for mode in builder.MODES:
        proof = builder.verify(Path(saved['binaries'][mode]), frozen['build'], mode)
        require(proof['producer'] == load(output/(mode+'-producer.json')) and
            all(frozen['files'][str(p)] == sha(p) for p in proof['files']), 'Changed producer')
    host = load(output/'host-producer.json')
    require(host['complete'] is True and host['base_native_fingerprint'] == frozen['build'] and
        host['binary_sha256'] == sha(host['binary']) == frozen['files'][host['binary']] and
        all(frozen['files'][p] == h for p, h in host['files'].items()), 'Changed host producer')
    checks, captures = saved['host_preflight'], saved['captures']
    require(len(captures) <= len(checks) <= min(len(captures)+1, 2), 'Missing host checks or extra work')
    for i, check in enumerate(checks):
        path = output/(builder.MODES[i]+'-host.json'); observed = observe_host(load(path), frozen['build'])
        require(check == dict(source=path.name, sha256=sha(path), observation=observed), 'Changed host preflight')
        require(observed['clean_host'] or (i == len(checks)-1 and saved['status'] == 'resource_blocked' and
            not (output/(builder.MODES[i]+'.json')).exists()), 'Work continued after blocked host')
    for i, capture in enumerate(captures):
        require(capture == observation(output, builder.MODES[i], frozen, work), 'Changed profile analysis')
        if not capture['exact']['clean_memory'] or not capture['exact']['clean_host']:
            require(i == len(captures)-1 and len(checks) == len(captures) and saved['status'] == 'resource_blocked',
                'Work continued after disturbed profile')
    if saved['complete']:
        require(len(captures) == len(checks) == 2 and saved['status'] == 'captured' and
            saved['comparison'] == analysis.compare(*[c['analysis'] for c in captures]), 'Incomplete profile comparison')
    else:
        require(saved['status'] in ('failed', 'resource_blocked', 'interrupted', 'time_budget_exhausted'), 'Invalid incomplete status')
    return dict(kind='block_compute_audit_v1', complete=True, audit_passed=True, recorded_complete=saved['complete'],
        recorded_status=saved['status'], source_seal_sha256=seal, source_provenance=provenance,
        performance_measurement=False, production_promoted=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['run', 'verify'])
    p.add_argument('--output', type=Path, required=True); p.add_argument('--commands', type=Path)
    p.add_argument('--dispatch', type=Path); p.add_argument('--source', type=Path)
    a = p.parse_args()
    if a.action == 'run':
        if a.commands is None or a.dispatch is None: p.error('Both profile binaries are required')
        raise SystemExit(0 if run(a.output, {mode: getattr(a, mode).resolve() for mode in builder.MODES})['complete'] else 2)
    else:
        if a.source is None: p.error('--source required')
        save(a.output, audit(a.source.resolve()))
