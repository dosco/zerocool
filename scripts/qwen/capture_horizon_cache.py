#!/usr/bin/env python3
"""Capture on/off target-cache controls; simulate reads only after exact replay."""
import argparse
import hashlib
from pathlib import Path

import build_horizon_cache_trace as builder
import horizon_cache_replay as replay
from benchmark_host import build_probe
from cache_residency import require
from combined_q4 import freeze
from perfect_draft import configuration
from prepare_mtp import verify as verify_artifact
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
from screen_mtp_continuation import PREPARED, host_check
from screen_verifier_horizon import SOURCE, prefix, read, reference
from stage200 import Experiment
from target_recovery_checks import replay_resources, resource_failure

ROOT = builder.ROOT
PROTOCOL = ROOT/'docs/benchmarks/2026-09-19-horizon-cache/protocol.md'
LIMITATIONS = [
    'One known 64-token coding continuation; no real proposal or rejection costs are measured.',
    'Tracing changes execution timing. No captured throughput is used as a speed measurement.',
    'Capacity simulations preserve the captured acquire/pin/release order, not changed completion timing.',
    '1909 and 2048 slots have not been admitted or executed by this capture.',
    'Memory savings and simulated read reductions do not establish five generated tokens per second.']


def fixtures(exp, cfg, host):
    for name, args in [('embedding', ['--streamed-embedding-test', exp.model]),
                       ('kernel', ['--horizon-kernel-test'])]:
        host_check(exp, host, name, full=False)
        path = exp.out/(name+'.json')
        exp.command([cfg['binary'], *args, path], name, limit=60, validation=True)
        raw = read(path)
        require(raw['complete'] is True and raw['performance_measurement'] is False, 'Incomplete fixture')
        if name == 'embedding':
            require(raw['kind'] == 'streamed_embedding_test_v1' and
                raw['owner_released'] is raw['invalid_request_atomic'] is raw['failed_read_atomic'] is True and
                raw['evictions_before_gpu_submission'] == 2 and raw['peak_gpu_bytes'] <= 1024**3 and
                len(raw['cases']) == 10 and all(x['exact'] is True for x in raw['cases']) and
                raw['cache']['evictions'] > 0 and raw['cache']['hits'] > 0, 'Incomplete embedding coverage')
        else:
            require(raw['kind'] == 'verifier_horizon_kernel_test_v1' and raw['peak_gpu_bytes'] <= 128*1024**2 and
                len(raw['cases']) == 4 and {(x['K'], x['N'], x['float_output']) for x in raw['cases']} ==
                {(k, n, f) for k, n in ((2560, 6144), (6144, 2560)) for f in (False, True)} and
                all(x['exact_reference'] is x['guards_intact'] is True and x['rows'] == 8 for x in raw['cases']),
                'Incomplete kernel coverage')
        result = replay_resources(raw)
        exp.report[name] = dict(result, source=path.name, sha256=sha(path)); exp.persist()
        if not result['clean_memory'] or not result['clean_host']:
            raise ResourceBlocked(resource_failure(raw, name))


def run(output, directory):
    work = prefix(reference(), 64)
    configs = [dict(configuration(4, expert_slots=1460), token_tile=w, embedding_storage='rows',
                    trace_workspace_bytes=replay.WORKSPACE) for w in (4, 8)]
    exp = Experiment(output, 'horizon_cache_capture_v1', configs, work, 900)
    with exp:
        cfg, proof = builder.verify(directory)
        require(proof['base_native_fingerprint'] == exp.frozen['build'], 'Native base changed')
        save(exp.out/'producer.json', proof)
        save(exp.out/'draft-audit.json', verify_artifact(PREPARED))
        host = build_probe(ROOT/'.cache/benchmark-host'/hashlib.sha256(str(exp.out).encode()).hexdigest())
        save(exp.out/'host-producer.json', host['producer'])
        (exp.out/'protocol.md').write_bytes(PROTOCOL.read_bytes())
        freeze(exp, [*builder.inputs(cfg), *builder.generated(cfg['output']), *cfg['objects'], cfg['binary'],
            *host['files'], PREPARED/'manifest.json', PREPARED/'dense.bin', PREPARED/'experts.bin', PROTOCOL,
            exp.out/'protocol.md', exp.out/'producer.json', exp.out/'host-producer.json',
            Path(work['reference']['path']), SOURCE/'evidence-files.json'])
        exp.env.update(ZEROCOOL_Q8_EXPANDED='packed', ZEROCOOL_MTP_NGRAM_INIT='lazy', ZEROCOOL_MTP_EXPERT_SCRATCH='off',
                       ZEROCOOL_MTP_DIRECT_OUTPUT='on', ZEROCOOL_TARGET_RECOVERY='full-replay')
        exp.report.update(samples=[], comparisons=[], simulations=[], performance_measurement=False,
            perfect_proposals_only=True, limitations=LIMITATIONS, protocol_sha256=sha(PROTOCOL))
        exp.guard.check_resources(initial=True)
        fixtures(exp, cfg, host)
        for width in (4, 8):
            pair = []
            for mode in ('off', 'on'):
                stem = f'width-{width}-{mode}'; path = exp.out/(stem+'.json')
                host_check(exp, host, stem)
                exp.env.update(ZEROCOOL_VERIFIER_HORIZON=str(width), ZEROCOOL_HORIZON_CACHE_TRACE=mode)
                exp.command([cfg['binary'], exp.model, PREPARED, exp.out/'workload.json', path, 'fast-timing'],
                            stem, limit=180)
                raw = read(path)
                result = replay.observe(raw, work, sha(exp.out/'workload.json'), proof['binary_sha256'], width, mode == 'on')
                sample = dict(width=width, mode=mode, source=path.name, sha256=sha(path), observation=result)
                exp.report['samples'].append(sample); exp.persist()
                if not result['clean_memory'] or not result['clean_host']:
                    raise ResourceBlocked(resource_failure(raw, stem))
                pair.append(raw)
            exp.report['comparisons'].append(replay.compare_modes(*pair)); exp.persist()
            trace_path = exp.out/f'width-{width}-on.cache.jsonl'
            traced = replay.decode(trace_path.read_bytes(), pair[1], work, exp.frozen['build'])
            exp.report['simulations'].append(dict(width=width, source=trace_path.name, sha256=sha(trace_path),
                                                  **replay.summarize(traced))); exp.persist()
        exp.report.update(status='cache_reads_simulated', advance_to_real_draft=False)
    return exp.report


def audit(source):
    """Rebuild every completed observation from sealed raw evidence, without Metal."""
    source = Path(source).resolve(); seal = sha(source/'evidence-files.json'); verify_seal(source, seal)
    summary = read(source/'summary.json'); work = read(source/'workload.json'); proof = read(source/'producer.json')
    require(summary['kind'] == 'horizon_cache_capture_v1' and summary['complete'] is True and
        summary['status'] == 'cache_reads_simulated' and summary['performance_measurement'] is False and
        summary['limitations'] == LIMITATIONS and summary['production_promoted'] is False and
        summary['normal_request_latency_qualified'] is False and
        summary['protocol_sha256'] == sha(source/'protocol.md'), 'Incomplete or changed capture')
    samples = []; comparisons = []; simulations = []
    for width in (4, 8):
        pair = []
        for mode in ('off', 'on'):
            path = source/f'width-{width}-{mode}.json'; raw = read(path)
            result = replay.observe(raw, work, sha(source/'workload.json'), proof['binary_sha256'], width, mode == 'on')
            require(result['clean_memory'] and result['clean_host'], 'Disturbed capture')
            samples.append(dict(width=width, mode=mode, source=path.name, sha256=sha(path), observation=result)); pair.append(raw)
        comparisons.append(replay.compare_modes(*pair))
        path = source/f'width-{width}-on.cache.jsonl'
        traced = replay.decode(path.read_bytes(), pair[1], work, proof['base_native_fingerprint'])
        simulations.append(dict(width=width, source=path.name, sha256=sha(path), **replay.summarize(traced)))
    require((samples, comparisons, simulations) == (summary['samples'], summary['comparisons'], summary['simulations']),
            'Summary differs from raw capture')
    return dict(kind='horizon_cache_audit_v1', complete=True, audit_passed=True, source=str(source),
                source_seal_sha256=seal, performance_measurement=False, production_promoted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'audit'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build', type=Path); parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    if args.action == 'run':
        if args.build is None: parser.error('--build required')
        raise SystemExit(0 if run(args.output, args.build)['complete'] else 2)
    else:
        if args.source is None: parser.error('--source required')
        result = audit(args.source)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save(args.output, result)
