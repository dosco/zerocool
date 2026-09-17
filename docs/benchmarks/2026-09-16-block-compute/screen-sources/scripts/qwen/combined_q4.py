#!/usr/bin/env python3
"""Exact native Q4 qualification and a bounded kernel/cache factorial screen."""
import argparse
import math
from pathlib import Path
import statistics

from cache_residency import configs as cache_configs, memory_observation, require
from capture_routes import load
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
from qualify_exact_sessions import check_configuration
from screen_cache import CHECKS, correctness_case, validate_request
from screen_decode_scratch import SCRATCH_CHECKS, check_state_workspace
from screen_q4_packed import FIXTURES, verify_records
from selector_qualification import check_machine
from stage200 import Experiment, SOURCE
from verify_stage200 import verify_sources

ROOT = Path(__file__).resolve().parents[2]
ORDER = [(0, a) for a in 'ABCD'] + [(1, a) for a in 'DCBA']
CRITERIA = dict(pairs=2, minimum_decode_saving_ms=20, median_conversation_ratio_max=.99,
    each_conversation_ratio_below=1, secondary_ratio_max=1.03,
    kernel_decode_improves_at_both_capacities=True, process_seconds=150, stage_seconds=900)
PACKED = ('q4_gate_up_packed_r2', 'q4_down_packed_r2')


def configs():
    small, large = cache_configs()
    return [dict(c, name=arm, q4_decode=q4) for arm, c, q4 in
            [('A', small, 'reference'), ('B', small, 'packed-r2'),
             ('C', large, 'reference'), ('D', large, 'packed-r2')]]


def cases():
    return [dict(correctness_case('clock'), kernel_policy='candidate', q8_decode_rows=2,
                 route_selection='simd', decode_scratch='reuse', decode_submission='immediate',
                 residency='core-cache', q4_decode=q4, exercise_decode=True)
            for q4 in ('reference', 'packed-r2')]


def shader_origin():
    probe = ROOT/'scripts/qwen/probe_q4_packed.metal'
    text = probe.read_text(); body = text[text.index('kernel void'):]
    body = body.replace('probe_q4_gate_r2', PACKED[0]).replace('probe_q4_down_r2', PACKED[1])
    require((ROOT/'kernels/metal/qwen.metal').read_text().endswith(body), 'Integrated arithmetic differs from probe')
    origin = load(ROOT/'docs/benchmarks/2026-09-14-combined-q4/integration-origin.json')
    require(sha(probe) == origin['probe_shader_sha256'], 'Pre-integration probe changed')
    return dict(probe_sha256=sha(probe), arithmetic_copied_verbatim_except_entry_names=True)


def validate_operators(raw, frozen, records):
    require(raw.get('complete') is True and raw.get('validation') is True and
            raw.get('kind') == 'native_q4_exact_check_v1', 'Missing native operator validation')
    m = raw['metal']
    require(m['build_fingerprint'] == frozen['build'] and m['device'] == frozen['device'] and
            m['live_command_groups'] == 0 and m['peak_buffer_bytes'] <= 128*1024**2,
            'Operator build/device/buffer bounds differ')
    expected = {(r['layer'], r['expert'], i) for r in records for i in range(8)}
    rows = raw['cases']
    require(len(rows) == 64 and {(r['layer'], r['expert'], r['row']) for r in rows} == expected,
            'Incomplete or duplicate native operator coverage')
    require(all(r[k] is True for r in rows for k in
        ('exact_gate', 'exact_bf16_down', 'exact_fp32_down', 'destination_preserved')), 'Operator mismatch')
    fixture = raw['fixture']
    require(fixture['artifact_revision'] == frozen['artifact_revision'] and fixture['origin'] == 'existing-Q4-experts',
            'Wrong operator artifact')
    require([dict(layer=int(layer), expert=e['expert'], sha256=e['record']['sha256'])
             for layer, item in fixture['layers'].items() for e in item['experts']] == records,
            'Operator records differ from verified prepared bytes')
    require(m['kernel_dispatches'].get(PACKED[0]) == 64 and m['kernel_dispatches'].get(PACKED[1]) == 128,
            'Candidate was not executed')
    return dict(exact=True, real_expert_input_cases=64, native_dispatch=True,
                bf16_and_fp32_outputs=True, nonzero_destination_offsets=True)


def validate_state(reports, frozen):
    require(len(reports) == 2, 'Missing state arm')
    for raw, case in zip(reports, cases()):
        require(raw.get('case') == case and raw.get('passed') is True and raw.get('full_model') is True and
                raw.get('layers') == 48 and len(raw['runs']) == 1, 'Incomplete or changed state case')
        checks = raw['checks']; names = CHECKS | SCRATCH_CHECKS
        require(len(checks) == len(names) and {c['name'] for c in checks} == names and
                all(c['passed'] is True for c in checks), 'Missing state/failure/cancellation checks')
        run = raw['runs'][0]
        require(len(run['stages']) == 3 and all(len(s['layers']) == len(s['routes']) == 48 for s in run['stages']),
                'Missing full-layer state and routes')
        for key in ('continued_statistics', 'after_fresh'):
            state = run[key]; check_configuration(state, case); check_machine(state, frozen)
            check_state_workspace(state, case, key)
            require(state['memory_plan']['expert_slots'] == 32 and state['memory_plan']['panel_tokens'] == 0 and
                    state['expert_cache']['evictions'] > 0 and not state['diagnostic_stream_trunk'] and
                    state['metal']['live_command_groups'] == 0, 'Missing forced eviction or drain')
            counts = state['metal']['kernel_dispatches']
            # Only the two continuation forwards select the candidate; fresh replay is prefill.
            for name in PACKED:
                require(counts.get(name, 0) == (960 if case['q4_decode'] == 'packed-r2' else 0),
                        'State proof did not exercise exactly two complete decode forwards')
    require(reports[0]['runs'][0]['stages'] == reports[1]['runs'][0]['stages'],
            'Packed decode changed logits, routes or persistent state')
    return dict(exact_logits_routes_state=True, all_48_layers=True, continued_equals_fresh=True,
                forced_eviction=True, cancellation_failure_checked=True, decode_dispatch_checked=True,
                independent_model_reference=False)


def observe(raw, frozen, config, work, expected):
    observations = validate_request(raw, frozen, config, work, expected, capacity_axis=True)
    for row in raw['runs']:
        for state in [row['before'], row['after'], *[p[k] for p in row['phases'].values() for k in ('before', 'after')]]:
            m = state['metal']; residency = m['residency']
            require(m['live_command_groups'] == 0 and m['peak_command_groups'] <= 2 and
                    residency['mode'] == 'core-cache' and residency['pending_retirements'] == 0,
                    'Unbounded users or missing core-cache residency')
        final = row['after']
        require(final['metal']['residency']['bytes_by_class'].get('expert') == final['memory_plan']['expert_bytes'],
                'Cache not fully enrolled')
        a, b = (row['phases']['decode'][k]['metal']['kernel_dispatches'] for k in ('before', 'after'))
        delta = {k: b.get(k, 0)-a.get(k, 0) for k in set(a) | set(b)}
        require(all(v >= 0 for v in delta.values()), 'Dispatch counter reset')
        for name, ref in zip(PACKED, ('q4_gate_up', 'q4_mm')):
            value = delta.pop(name, 0)
            require(value == (32*480 if config['q4_decode'] == 'packed-r2' else 0),
                    'Q4 candidate did not execute the requested complete decode work')
            delta[ref] = delta.get(ref, 0)+value
        normalized = {k: v for k, v in delta.items() if v}
        require(expected.setdefault('dispatches_'+row['name'], normalized) == normalized,
                'Dispatches changed beyond the intended Q4 substitution')
    return observations


def decide(rows):
    require([(r['pair'], r['configuration']) for r in rows] == ORDER, 'Missing or reordered factorial measurements')
    by = {(r['pair'], r['configuration']): r for r in rows}
    require(all(len(r['requests']) == 2 for r in rows), 'Missing conversation phase')
    for row in rows:
        for r in row['requests']:
            require(all(type(r[k]) in (int, float) and math.isfinite(r[k]) and r[k] > 0
                for k in ('request_ms', 'time_to_first_token_ms', 'decode_wall_ms')), 'Invalid timing')
    contrasts = {}
    for numerator, denominator in [('B','A'), ('D','C'), ('C','A'), ('D','B'), ('D','A')]:
        name = numerator+'/'+denominator; phases = []
        totals = [sum(r['request_ms'] for r in by[p, numerator]['requests']) /
                  sum(r['request_ms'] for r in by[p, denominator]['requests']) for p in range(2)]
        for phase in range(2):
            metrics = {}
            for key in ('request_ms', 'time_to_first_token_ms', 'decode_wall_ms'):
                ratios = [by[p, numerator]['requests'][phase][key]/by[p, denominator]['requests'][phase][key] for p in range(2)]
                metrics[key] = dict(ratios=ratios, median=statistics.median(ratios))
            savings = [(by[p, denominator]['requests'][phase]['decode_wall_ms']-
                        by[p, numerator]['requests'][phase]['decode_wall_ms'])/32 for p in range(2)]
            phases.append(dict(phase=phase, metrics=metrics, decode_savings_ms_per_token=savings,
                               median_decode_saving_ms=statistics.median(savings)))
        contrasts[name] = dict(phases=phases, conversation_ratios=totals, median_conversation_ratio=statistics.median(totals))
    primary = contrasts['D/A']
    clean = all(r.get('memory_screen_passed') is True for r in rows)
    material = all(p['median_decode_saving_ms'] >= CRITERIA['minimum_decode_saving_ms'] for p in primary['phases'])
    request_gain = all(r < 1 for r in primary['conversation_ratios']) and primary['median_conversation_ratio'] <= .99
    secondary = all(m['median'] <= 1.03 for p in primary['phases'] for m in p['metrics'].values())
    kernel = all(p['metrics']['decode_wall_ms']['median'] < 1 for name in ('B/A', 'D/C') for p in contrasts[name]['phases'])
    passed = clean and material and request_gain and secondary and kernel
    return dict(status='memory_disturbed' if not clean else 'promising' if passed else 'screen_gate_not_met',
        advance_to_confirmation=passed, stage_target_met=passed, contrasts=contrasts,
        gates=dict(clean_memory=clean, material_decode_gain=material, conversation_gain=request_gain,
                   secondary_latencies=secondary, kernel_gain_at_both_capacities=kernel),
        confidence_95=None, prior_pairs_pooled=False, normal_request_latency_qualified=False, production_promoted=False)


def freeze(exp, paths):
    for path in paths:
        exp.frozen['files'][str(path.resolve())] = sha(path)
    save(exp.out/'identity.json', exp.frozen)


def state_proof(directory, frozen):
    verify_seal(directory, sha(directory/'evidence-files.json'))
    saved = load(directory/'summary.json')
    require(saved.get('complete') is True and saved['status'] == 'state_qualified' and
            saved['identity'] == {k: frozen[k] for k in saved['identity']}, 'Missing compatible completed state proof')
    proof = validate_state([load(directory/f'state-{a}.json') for a in 'AB'], frozen)
    require(saved['correctness'] == proof, 'Changed state proof')
    operators = validate_operators(load(directory/'operators.json'), frozen, saved['verified_records'])
    require(saved['operators'] == operators, 'Changed operator proof')
    return proof


def run(args):
    work = load(SOURCE/'workload.json')
    require([len(w['tokens']) for w in work] == [72, 128] and all(w['max_tokens'] == 33 for w in work), 'Changed screen workload')
    exp = Experiment(args.output, 'combined_q4_'+args.mode+'_v1', configs(), work, 480 if args.mode == 'state' else 900)
    with exp:
        exp.report.update(criteria=CRITERIA, shader_origin=shader_origin())
        freeze(exp, [ROOT/'build/qwen/qwen_q4_check', ROOT/'scripts/qwen/probe_q4_packed.metal'])
        exp.guard.check_resources(initial=True)
        if args.mode == 'state':
            exp.report['verified_records'] = verify_records(exp.prepared)
            freeze(exp, list(FIXTURES.iterdir()))
            exp.command([ROOT/'build/qwen/test_qwen'], 'native-tests', 120, True)
            exp.command([ROOT/'build/qwen/qwen_q4_check', FIXTURES, exp.out/'operators.json'], 'operators', 90, True)
            exp.report['operators'] = validate_operators(load(exp.out/'operators.json'), exp.frozen, exp.report['verified_records'])
            reports = []
            for arm, case in zip('AB', cases()):
                path = exp.out/f'{arm}.case.json'; save(path, case); freeze(exp, [path])
                exp.command([ROOT/'build/qwen/qwen_panel_check', exp.model, exp.prepared, path,
                             exp.out/f'state-{arm}.json'], 'state-'+arm, 180, True)
                reports.append(load(exp.out/f'state-{arm}.json'))
            exp.report.update(status='state_qualified', correctness=validate_state(reports, exp.frozen))
        else:
            require(args.state is not None, 'State proof required')
            exp.report['correctness'] = state_proof(args.state, exp.frozen)
            exp.report['state_source'] = dict(path=str(args.state.resolve()), sha256=sha(args.state/'evidence-files.json'))
            freeze(exp, [p for p in args.state.iterdir() if p.is_file()])
            expected = {}
            for pair, arm in ORDER:
                config = next(c for c in configs() if c['name'] == arm); stem = f'pair-{pair}-{arm}'
                raw = exp.bench(config, stem, limit=150)
                memory = memory_observation(raw)
                exp.report['measurements'].append(dict(pair=pair, configuration=arm,
                    requests=observe(raw, exp.frozen, config, work, expected), **memory,
                    source=stem+'.json', sha256=sha(exp.out/(stem+'.json'))))
                exp.persist()
                if not memory['memory_screen_passed']:
                    raise ResourceBlocked('Compression, decompression, unavailable memory counters or swap growth: stop all arms')
            exp.report.update(decide(exp.report['measurements']))
    return exp.report


def verify(directory):
    verify_seal(directory, sha(directory/'evidence-files.json'))
    saved, frozen = load(directory/'summary.json'), load(directory/'identity.json')
    require(saved['configurations'] == configs() and saved.get('criteria') == CRITERIA, 'Changed experiment design')
    provenance = verify_sources(directory.parent, frozen)
    require(saved['identity'] == {k: frozen[k] for k in saved['identity']}, 'Changed identity')
    require(saved['shader_origin'] == shader_origin(), 'Changed shader integration')
    if saved['kind'] == 'combined_q4_state_v1':
        decision = state_proof(directory, frozen) if saved['complete'] else dict(status=saved['status'])
    else:
        require(saved['kind'] == 'combined_q4_screen_v1', 'Unknown experiment')
        if 'state_source' in saved:
            source = Path(saved['state_source']['path']); require(sha(source/'evidence-files.json') == saved['state_source']['sha256'], 'State evidence changed')
            require(state_proof(source, frozen) == saved['correctness'], 'State proof changed')
        expected = {}; seen = set(); rows = saved['measurements']
        require([(r['pair'],r['configuration']) for r in rows] == ORDER[:len(rows)], 'Wrong unfinished order')
        for row in rows:
            path = directory/row['source']; require(path.parent == directory and sha(path) == row['sha256'] and row['sha256'] not in seen, 'Changed or reused raw request')
            seen.add(row['sha256']); raw = load(path); config = next(c for c in configs() if c['name'] == row['configuration'])
            require(observe(raw, frozen, config, saved['workload'], expected) == row['requests'], 'Changed measurements')
            require(all(row.get(k) == v for k,v in memory_observation(raw).items()), 'Changed memory evidence')
        decision = decide(rows) if saved['complete'] else dict(status=saved['status'], advance_to_confirmation=False)
        require(all(saved.get(k) == v for k,v in decision.items() if saved['complete'] or k == 'status'), 'Changed decision')
    if not saved['complete']:
        require(saved['status'] in ('failed', 'resource_blocked', 'interrupted', 'time_budget_exhausted'), 'Unfinished report cannot pass')
    return dict(kind='combined_q4_audit_v1', complete=True, audit_passed=True,
        source_seal_sha256=sha(directory/'evidence-files.json'), recorded_complete=saved['complete'],
        source_provenance=provenance, decision=decision, production_promoted=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=('state','screen','verify'))
    p.add_argument('--output', type=Path, required=True); p.add_argument('--state', type=Path); p.add_argument('--source', type=Path)
    args = p.parse_args()
    if args.mode == 'verify':
        if args.source is None: p.error('--source required')
        save(args.output, verify(args.source.resolve()))
    else: raise SystemExit(0 if run(args)['complete'] else 2)
