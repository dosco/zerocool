#!/usr/bin/env python3
"""Fixed-budget verifier capacity screen; no production speculation claim."""
import argparse
import copy
import hashlib
from pathlib import Path
import shutil
import subprocess

import perfect_draft as verifier
from benchmark_host import build_probe, preflight, observe as observe_host
from build_perfect_draft import verify_producer
from capture_routes import load
from combined_q4 import freeze
from native_q4_replay import uint
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
from stage200 import Experiment
from verify_stage200 import verify_sources

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'docs/benchmarks/2026-09-16-perfect-draft-capacity'
PROTOCOL = BASE/'protocol.md'
VALIDATION = [(1072, 1), (1072, 4), (1460, 4)]
ORDER = [(0, 1072, 4), (0, 1460, 1), (0, 1460, 4),
         (1, 1460, 4), (1, 1460, 1), (1, 1072, 4)]
CRITERIA = dict(memory_bytes=verifier.BUDGET, validation=VALIDATION, timing_order=ORDER,
    validation_tokens=4, timing_tokens=16, floor_tokens_per_second=5,
    stop_first_clean_candidate_below_floor=True, pairs=2, stage_seconds=600,
    validation_seconds=150, timing_seconds=90, initial_cache_match='within capacity',
    actual_draft_measured=False, prime_expert_schedule='fixed-lease-batches-32')
LIMITATIONS = verifier.LIMITATIONS + [
    'Only capacity and verifier width vary. Initial cache identity must match within a capacity, but different capacities intentionally retain different initial caches.',
    'A declared early stop completes only the absolute-floor screen; it cannot establish a complete paired speedup. Resource-disturbed and unfinished observations cannot trigger that decision.',
    'A fully resident prepared MTP head plus 1460 target slots exceeds this 12GiB plan. Passing the optimistic screen still requires joint draft memory design and real acceptance/cost measurement.',
    'Application bytes and aggregate cache/GPU counters do not establish physical device traffic, per-block working sets or a dependency critical path.']
require = verifier.require


def configuration(slots, width):
    require((slots, width) in VALIDATION or (slots, width) == (1460, 1), 'Unsupported capacity arm')
    return verifier.configuration(width, expert_slots=slots)


def stages():
    return [('validate', None, s, w, f'validate-{s}-width-{w}') for s, w in VALIDATION] + [
        ('timing', p, s, w, f'pair-{p}-slots-{s}-width-{w}') for p, s, w in ORDER]


def counters(raw):
    before, after = raw['before'], raw['after']
    out = {}
    for group, fields in [('expert_cache', ('hits', 'misses', 'application_read_bytes')),
                          ('metal', ('gpu_command_ns', 'dispatches'))]:
        for field in fields:
            a, b = before[group].get(field), after[group].get(field)
            require(uint(a) and uint(b) and b >= a, 'Missing or reset work counter')
            out[field] = b-a
    require(out['hits']+out['misses'] > 0, 'No observed expert work')
    out['cache_hit_fraction'] = out['hits']/(out['hits']+out['misses'])
    out['expert_bytes_per_token'] = out['application_read_bytes']/raw['verified_tokens']
    out['gpu_ns_per_token'] = out['gpu_command_ns']/raw['verified_tokens']
    return out


def compare_math(reference, candidate):
    """Relax only the explicitly varied cache identity; compare real model state."""
    verifier.check_prime_reference(candidate['prime'], reference['prime'])
    adjusted = copy.deepcopy(candidate)
    adjusted['prime']['cache_state'] = reference['prime']['cache_state']
    return verifier.compare(reference, adjusted)


def check_progress(observed, slots, validated, measured, primes, anchor):
    verifier.check_prime_reference(observed['prime'], anchor)
    if slots in primes:
        require(observed['prime'] == primes[slots], 'Starting cache differs within capacity')
    else:
        primes[slots] = observed['prime']
    if observed['validation']:
        if validated:
            compare_math(validated[0]['observation'], observed)
    else:
        require(len(validated) == len(VALIDATION), 'Timing ran without complete validation')
        serial = validated[0]['observation']
        require(observed['row_logits_sha256'][:4] == serial['row_logits_sha256'] and
                observed['endpoints'][0] == serial['endpoints'][-1],
                'Timed prefix differs from validated serial state or outputs')
        if measured:
            first = measured[0]['observation']
            require(observed['row_logits_sha256'] == first['row_logits_sha256'] and
                    observed['endpoints'] == first['endpoints'],
                    'Capacity or width changed timed logits, state or routes')
        require(all(observed[k] == serial[k] for k in ('snapshot_allocated_bytes', 'host_logits_bound_bytes')),
                'Verifier workspace reservation changed')


def decide(rows):
    require([(r['pair'], r['slots'], r['observation']['width']) for r in rows] == ORDER[:len(rows)] and
            len(rows) <= len(ORDER), 'Missing, duplicated or reordered capacity timings')
    for row in rows:
        o = row['observation']
        require(not o['validation'] and o['verified_tokens'] == 16 and
                o['clean_memory'] and o['clean_host'], 'Disturbed or invalid timing cannot decide capacity')
        require(uint(o['decode_wall_ns']) and o['decode_wall_ns'] > 0 and
                o['verified_tokens_per_second'] == 16e9/o['decode_wall_ns'], 'Inconsistent throughput')
    weak = [i for i, r in enumerate(rows) if r['slots'] == 1460 and r['observation']['width'] == 4 and
            r['observation']['verified_tokens_per_second'] < 5]
    common = dict(paired_comparison_complete=False, capacity_promising=False,
                  confidence_95=None, **verifier.FLAGS)
    if weak:
        require(weak[0] == len(rows)-1, 'Work continued after declared early-stop boundary')
        return dict(common, status='verifier_capacity_below_floor', early_stop=True,
                    stop_source=rows[-1]['source'], remaining_timing_processes=len(ORDER)-len(rows))
    if len(rows) < len(ORDER):
        return dict(common, status='running', early_stop=False)
    by = {(r['pair'], r['slots'], r['observation']['width']): r['observation'] for r in rows}
    ratios = [by[p,1460,4]['decode_wall_ns']/by[p,1072,4]['decode_wall_ns'] for p in (0,1)]
    serial_ratios = [by[p,1460,4]['decode_wall_ns']/by[p,1460,1]['decode_wall_ns'] for p in (0,1)]
    promising = all(r < 1 for r in ratios+serial_ratios)
    return dict(common, status='verifier_capacity_promising' if promising else 'verifier_capacity_no_advantage',
                paired_comparison_complete=True, capacity_promising=promising, early_stop=False,
                capacity_wall_ratios=ratios, larger_cache_block_to_serial_wall_ratios=serial_ratios)


def run(output, binary):
    work, source = verifier.source_input(); verifier.valid_work(work)
    configs = [configuration(s,w) for s,w in [(1072,1),(1072,4),(1460,1),(1460,4)]]
    exp = Experiment(output, 'perfect_draft_capacity_v1', configs, work, 600)
    with exp:
        require(all(source[k] == exp.frozen[k] for k in ('artifact_revision','prepared_manifest_sha256')),
                'Changed fixed-token artifact')
        exp.report.update(criteria=CRITERIA, limitations=LIMITATIONS, token_source=source,
                          validation=[], host_preflight=[], binary=str(binary), **verifier.FLAGS)
        proof = verify_producer(binary, exp.frozen['build'])
        anchor_path = verifier.PRIME_REFERENCE
        verify_seal(anchor_path.parent, sha(anchor_path.parent/'evidence-files.json'))
        anchor = load(anchor_path)
        require(anchor['complete'] is True and anchor['validation'] is True and anchor['width'] == 1 and
            anchor['input_sha256'] == sha(exp.out/'workload.json') and
            anchor['before']['metal']['build_fingerprint'] == exp.frozen['build'] and
            anchor['artifact_revision'] == exp.frozen['artifact_revision'] and
            anchor['prepared_manifest_sha256'] == exp.frozen['prepared_manifest_sha256'], 'Invalid prime anchor')
        exp.report['prime_reference'] = dict(path=str(anchor_path), sha256=sha(anchor_path))
        save(exp.out/'producer.json', proof['producer'])
        exp.report['producer_sha256'] = sha(exp.out/'producer.json')
        shutil.copyfile(PROTOCOL, exp.out/'protocol.md'); exp.report['protocol_sha256'] = sha(PROTOCOL)
        host = build_probe(ROOT/'.cache/benchmark-host'/hashlib.sha256(str(exp.out).encode()).hexdigest())
        save(exp.out/'host-producer.json', host['producer'])
        exp.report['host_producer_sha256'] = sha(exp.out/'host-producer.json')
        freeze(exp, [*proof['files'], *host['files'], anchor_path, PROTOCOL, exp.out/'protocol.md',
                     exp.out/'producer.json', exp.out/'host-producer.json',
                     *[p for p in verifier.SOURCE.iterdir() if p.is_file()]])
        exp.persist(); exp.guard.check_resources(initial=True)
        primes = {}
        for mode, pair, slots, width, stem in stages():
            preflight(exp, host, stem)
            raw_path = exp.out/(stem+'.json')
            try:
                exp.command([binary, exp.model, exp.prepared, exp.out/'workload.json', raw_path,
                             str(width), mode, str(slots)], stem,
                            limit=150 if mode == 'validate' else 90, validation=mode == 'validate')
            except subprocess.CalledProcessError:
                if raw_path.is_file() and load(raw_path).get('error') == 'fixed memory admission failed':
                    raise ResourceBlocked('Fixed 12GiB/capacity admission failed; see '+str(raw_path)) from None
                raise
            raw = load(raw_path)
            observed = verifier.observe(raw, exp.frozen, work, width, mode == 'validate',
                                        sha(exp.out/'workload.json'), expert_slots=slots)
            check_progress(observed, slots, exp.report['validation'], exp.report['measurements'], primes, anchor['prime'])
            item = dict(source=raw_path.name, sha256=sha(raw_path), slots=slots, observation=observed)
            if mode == 'validate':
                exp.report['validation'].append(item)
            else:
                item.update(pair=pair, counters=counters(raw)); exp.report['measurements'].append(item)
            exp.persist()
            if not observed['clean_memory'] or not observed['clean_host']:
                raise ResourceBlocked('Verifier capacity observation disturbed by memory or host conditions')
            if mode == 'timing':
                decision = decide(exp.report['measurements']); exp.report.update(decision); exp.persist()
                if decision['early_stop']: break
    return exp.report


def audit(output):
    seal_hash = sha(output/'evidence-files.json'); verify_seal(output, seal_hash)
    saved, frozen = load(output/'summary.json'), load(output/'identity.json')
    require(saved['kind'] == 'perfect_draft_capacity_v1' and
            saved['criteria'] == verifier.load_json(CRITERIA) and saved['limitations'] == LIMITATIONS and
            saved['identity'] == {k:frozen[k] for k in saved['identity']} and
            saved['configurations'] == [configuration(s,w) for s,w in [(1072,1),(1072,4),(1460,1),(1460,4)]] and
            all(saved[k] is False for k in verifier.FLAGS), 'Changed capacity protocol or scope')
    provenance = verify_sources(output.parent, frozen)
    proof = verify_producer(Path(saved['binary']), frozen['build'])
    require(saved['producer_sha256'] == sha(output/'producer.json') and
            load(output/'producer.json') == proof['producer'] and
            all(frozen['files'][str(Path(p).resolve())] == sha(p) for p in proof['files']), 'Changed producer')
    work, source = verifier.source_input()
    require(saved['token_source'] == source and saved['workload'] == work == load(output/'workload.json') and
            saved['protocol_sha256'] == sha(output/'protocol.md') == frozen['files'][str(PROTOCOL)],
            'Changed workload or declared protocol')
    anchor_path = verifier.PRIME_REFERENCE
    require(saved['prime_reference'] == dict(path=str(anchor_path), sha256=sha(anchor_path)) and
            frozen['files'][str(anchor_path)] == sha(anchor_path), 'Changed prime anchor')
    anchor = load(anchor_path)['prime']
    host = load(output/'host-producer.json')
    require(saved['host_producer_sha256'] == sha(output/'host-producer.json') and
            host['complete'] is True and host['base_native_fingerprint'] == frozen['build'] and
            host['binary_sha256'] == sha(host['binary']) == frozen['files'][host['binary']] and
            all(frozen['files'][p] == h for p,h in host['files'].items()), 'Changed host producer')
    schedule = stages(); checks = saved['host_preflight']
    count = len(saved['validation'])+len(saved['measurements'])
    require(len(saved['validation']) <= 3 and len(saved['measurements']) <= 6 and
            count <= len(checks) <= min(count+1,len(schedule)), 'Missing host checks or extra processes')
    for i, item in enumerate(checks):
        stem = schedule[i][-1]; path = output/(stem+'-host.json'); result = observe_host(load(path), frozen['build'])
        require(item == dict(source=path.name, sha256=sha(path), observation=result), 'Changed host observation')
        require(result['clean_host'] or (i == len(checks)-1 and saved['status'] == 'resource_blocked' and
                not (output/(stem+'.json')).exists()), 'Work ran after blocked host')
    validated, measured, primes = [], [], {}
    for i, item in enumerate(saved['validation']+saved['measurements']):
        mode, pair, slots, width, stem = schedule[i]; path = output/(stem+'.json')
        require(item['source'] == path.name and item['sha256'] == sha(path) and item['slots'] == slots,
                'Changed source, order or capacity')
        raw = load(path)
        result = verifier.observe(raw, frozen, work, width, mode == 'validate', sha(output/'workload.json'), expert_slots=slots)
        check_progress(result, slots, validated, measured, primes, anchor)
        require(item['observation'] == result, 'Changed derived observation')
        if mode == 'validate': validated.append(item)
        else:
            require(item['pair'] == pair and item['counters'] == counters(raw), 'Changed counters or pair')
            measured.append(item)
        if not result['clean_memory'] or not result['clean_host']:
            require(i == count-1 and saved['status'] == 'resource_blocked' and len(checks) == count,
                    'Work ran after disturbed observation')
    if saved['complete']:
        require(len(validated) == 3 and len(checks) == count and
                all(r['observation']['clean_memory'] and r['observation']['clean_host'] for r in validated),
                'Completed screen lacks clean validation or has extra work')
        decision = decide(measured)
        require(decision['status'] != 'running' and all(saved.get(k) == v for k,v in decision.items()),
                'Saved decision differs from evidence')
        require(all(not (output/(row[-1]+'.json')).exists() for row in schedule[count:]),
                'Native work exists after early stop')
    else:
        require(saved['status'] in ('resource_blocked','failed','time_budget_exhausted','interrupted'),
                'Incomplete stage lacks terminal disposition')
    return dict(kind='perfect_draft_capacity_audit_v1', complete=True, audit_passed=True,
                recorded_complete=saved['complete'], status=saved['status'], source_seal_sha256=seal_hash,
                source_provenance=provenance, **verifier.FLAGS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run','verify'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    if args.action == 'verify':
        if args.source is None: parser.error('--source is required')
        save(args.output, audit(args.source.resolve()))
    else:
        if args.binary is None: parser.error('--binary is required')
        raise SystemExit(0 if run(args.output,args.binary.resolve())['complete'] else 2)
