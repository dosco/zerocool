#!/usr/bin/env python3
"""Replay saved cache demand, then screen one capacity change under residency."""
import argparse
import math
from pathlib import Path
import tempfile

from cache_simulation import cache_curve, SLOT_BYTES
from capture_routes import load
from confirm_expert_residency import validate_requests, validate_state
from evidence_index import Index
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
from screen_cache import validate_request
from screen_coalesced import decide as stage_decision
from stage200 import Experiment, ORDER, clean_memory, configuration
from verify_stage200 import verify_sources

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / 'docs/benchmarks/2026-09-14-submission'
TRACES = [ROOT / f'docs/benchmarks/{name}/raw' for name in
          ('2026-09-10-normal-routes', '2026-09-10-extended-routes')]
SLOTS = (1072, 1460)
CRITERIA = dict(pairs=2, median_conversation_ratio_max=.99,
                each_conversation_ratio_below=1, other_median_ratio_max=1.03,
                minimum_median_decode_saving_ms=20, every_decode_saving_positive=True,
                observed_compression_allowed=False, observed_swap_growth_allowed=False)


def require(value, message):
    if not value:
        raise ValueError(message)


def configs():
    return [dict(configuration(slots, name=name), residency='core-cache',
                 decode_submission='immediate')
            for name, slots in zip(('control', 'candidate'), SLOTS)]


def curve_difference(curve):
    require(curve.get('status') == 'simulated' and
            curve['coverage'].get('whole_request_covered') is True and
            curve['coverage'].get('cold_starts') == 1,
            'Requires a complete unfiltered conversation with a single cold start')
    require([c['slots'] for c in curve['curves']] == list(SLOTS), 'Wrong slot capacities')
    phases = []
    small, large = [next(p for p in c['policies'] if p['policy'] == 'clock')
                    for c in curve['curves']]
    require(len(small['request_phases']) == len(large['request_phases']), 'Different phase coverage')
    for a, b in zip(small['request_phases'], large['request_phases']):
        require(all(a[k] == b[k] for k in ('request_id', 'phase', 'demands')), 'Different demands')
        phases.append(dict(request_id=a['request_id'], phase=a['phase'], demands=a['demands'],
            control_misses=a['misses'], candidate_misses=b['misses'],
            fewer_read_fraction=1-b['misses']/a['misses'] if a['misses'] else None,
            avoided_application_bytes=a['application_miss_bytes']-b['application_miss_bytes']))
    return dict(added_slot_bytes=(SLOTS[1]-SLOTS[0])*SLOT_BYTES, phases=phases,
                predicted_latency_saving_ms=None, predicted_tokens_per_second=None)


def analyze(out):
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for source in TRACES:
        verify_seal(source, sha(source/'evidence-files.json'))
        require(load(source/'summary.json').get('complete') is True, 'Incomplete route capture')
        with tempfile.TemporaryDirectory() as temp:
            index = Index(Path(temp)/'index.sqlite')
            try:
                path = source/'routes.jsonl'
                index.import_paths([path])
                curve = cache_curve(index, str(path),
                    [math.ceil(n*SLOT_BYTES/1024**2) for n in SLOTS], per_layer=True)
            finally:
                index.close()
        name = source.parent.name+'.json'
        save(out/name, curve)
        results.append(dict(source=str(source), source_seal_sha256=sha(source/'evidence-files.json'),
                            curve=name, curve_sha256=sha(out/name), **curve_difference(curve)))
    summary = dict(kind='cache_residency_demand_analysis_v1', complete=True, status='simulated',
        capacities=list(SLOTS), results=results, normal_request_latency_qualified=False,
        production_promoted=False, limitations=[
            'Historical route builds differ from the current runtime. This measures saved demands, not new inference.',
            'Fixed ascending demand order and immediate release omit native hit-first admission and GPU leases.',
            'Read counts are application bytes. Device traffic, residency, compression and latency are not simulated.',
            'Short and extended captures share a prompt; they are not independent workloads.',
            'The append follows 32 or 256 generation steps, not the required 4K history.'])
    save(out/'summary.json', summary)
    return summary


def memory_observation(raw):
    states = [r['phases'][phase][k]['process'] for r in raw['runs']
              for phase in ('ingest', 'decode') for k in ('before', 'after')]
    swaps = [s.get('system_swap_used_bytes') for s in states]
    valid = bool(swaps) and all(type(n) is int and n >= 0 for n in swaps)
    growth = max((b-a for a, b in zip(swaps, swaps[1:])), default=0) if valid else None
    clean = clean_memory(raw)
    return dict(clean_memory=clean, swap_observations_bytes=swaps,
                maximum_observed_swap_increase_bytes=max(0, growth) if valid else None,
                memory_screen_passed=clean and valid and growth <= 0)


def observe(raw, frozen, config, work, expected):
    result = validate_request(raw, frozen, config, work, expected, capacity_axis=True)
    for row in raw['runs']:
        for phase in row['phases'].values():
            for key in ('before', 'after'):
                state = phase[key]
                require(state['memory_pressure']['policy'] == 'observe', 'Changed pressure policy')
                require(state['metal']['live_command_groups'] == 0 and
                        state['metal']['peak_command_groups'] <= 2, 'Live or unbounded GPU users')
                residency = state['metal']['residency']
                # Slots are allocated on first use, so use measured live bytes at each boundary.
                require(residency['mode'] == 'core-cache' and residency['pending_retirements'] == 0,
                        'Missing residency or pending retirements')
        final = row['after']
        require(final['metal']['residency']['bytes_by_class'].get('expert') ==
                final['memory_plan']['expert_bytes'], 'Expert cache is not fully enrolled')
        a, b = (row['phases']['decode'][k]['metal'] for k in ('before', 'after'))
        counts = {k: b['kernel_dispatches'].get(k, 0)-a['kernel_dispatches'].get(k, 0)
                  for k in b['kernel_dispatches']}
        require(expected.setdefault('dispatches_'+row['name'], counts) == counts,
                'Arithmetic dispatch counts changed')
    return result


def decide(rows):
    require([(r['pair'], r['configuration']) for r in rows] == list(ORDER), 'Incomplete paired screen')
    result = stage_decision(rows)
    memory_ok = all(r.get('memory_screen_passed') is True for r in rows)
    passed = result['advance_to_confirmation'] and memory_ok
    # A two-pair directional result is not a confidence-based speed claim.
    directional = (all(r < 1 for r in result['conversation_ratios']) and
                   all(v > 0 for phase in result['decode_savings_ms_per_token'] for v in phase))
    return dict(result, status='memory_disturbed' if not memory_ok else 'promising' if passed
                else 'directional_gain_below_gate' if directional else 'no_clear_screen_gain',
                advance_to_confirmation=passed, stage_target_met=passed,
                request_gain_qualified=False, memory_screen_passed=memory_ok)


def run(out):
    prior = validate_requests(PRIOR/'residency-confirm-01', 5)
    state = PRIOR/'residency-state-01'
    verify_seal(state, sha(state/'evidence-files.json'))
    work = prior['workload']
    exp = Experiment(out, 'cache_residency_screen_v1', configs(), work, 600)
    with exp:
        require(exp.report['identity'] == prior['identity'], 'Changed build/artifact/budget since residency proof')
        proof = validate_state([load(state/f'state-{arm}.json') for arm in ('control', 'candidate')],
                               exp.frozen['build'])
        exp.report.update(criteria=CRITERIA, prior_state_proof=proof,
            prior_state_scope='Same build forced eviction at 32 slots; no fresh state qualification at 1460 slots.')
        for directory in (PRIOR/'residency-confirm-01', state):
            for path in directory.iterdir():
                if path.is_file(): exp.frozen['files'][str(path.resolve())] = sha(path)
        save(exp.out/'identity.json', exp.frozen)
        exp.persist()
        exp.guard.check_resources(initial=True)
        expected = {r['name']: r['output_token_ids'] for r in
                    load(PRIOR/'residency-confirm-01/pair-0-candidate.json')['runs']}
        for pair, arm in ORDER:
            config = configs()[arm == 'candidate']
            stem = f'pair-{pair}-{arm}'
            raw = exp.bench(config, stem)
            requests = observe(raw, exp.frozen, config, work, expected)
            memory = memory_observation(raw)
            exp.report['measurements'].append(dict(pair=pair, configuration=arm, requests=requests,
                source=stem+'.json', sha256=sha(exp.out/(stem+'.json')), **memory))
            exp.persist()
            if not memory['memory_screen_passed']:
                raise ResourceBlocked('Observed compression, decompression, unavailable swap counter or swap growth; stop comparison')
        exp.report.update(decide(exp.report['measurements']))
    return exp.report


def verify(out):
    verify_seal(out, sha(out/'evidence-files.json'))
    summary, frozen = load(out/'summary.json'), load(out/'identity.json')
    require(summary['configurations'] == configs() and summary.get('criteria') == CRITERIA,
            'Changed capacity experiment configuration or criteria')
    provenance = verify_sources(out.parent, frozen)
    state = PRIOR/'residency-state-01'
    verify_seal(state, sha(state/'evidence-files.json'))
    proof = validate_state([load(state/f'state-{arm}.json') for arm in ('control', 'candidate')], frozen['build'])
    require(summary.get('prior_state_proof') == proof, 'Changed prior state proof')
    expected = {r['name']: r['output_token_ids'] for r in
                load(PRIOR/'residency-confirm-01/pair-0-candidate.json')['runs']}
    seen = set()
    require([(m['pair'], m['configuration']) for m in summary['measurements']] ==
            list(ORDER)[:len(summary['measurements'])], 'Changed or repeated screen order')
    for row in summary['measurements']:
        path = out/row['source']
        require(sha(path) == row['sha256'] and row['sha256'] not in seen, 'Changed or reused report')
        seen.add(row['sha256'])
        raw = load(path)
        require(observe(raw, frozen, configs()[row['configuration'] == 'candidate'],
                        summary['workload'], expected) == row['requests'], 'Changed measurements')
        require(all(row.get(k) == v for k, v in memory_observation(raw).items()), 'Changed memory observations')
    if summary['complete']:
        decision = decide(summary['measurements'])
        require(all(summary.get(k) == v for k, v in decision.items()), 'Changed decision')
    else:
        require(summary['status'] in ('resource_blocked', 'failed', 'interrupted', 'time_budget_exhausted'),
                'Unfinished screen has no terminal status')
        decision = dict(status=summary['status'], advance_to_confirmation=False)
    return dict(kind='cache_residency_audit_v1', complete=True, audit_passed=True,
        source_seal_sha256=sha(out/'evidence-files.json'), source_provenance=provenance,
        recorded_complete=summary['complete'], decision=decision,
        normal_request_latency_qualified=False, production_promoted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('analyze', 'screen', 'verify'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    if args.mode == 'verify':
        if args.source is None:
            parser.error('--source is required for verify')
        save(args.output, verify(args.source))
    else:
        result = analyze(args.output) if args.mode == 'analyze' else run(args.output)
        raise SystemExit(0 if result['complete'] else 2)
