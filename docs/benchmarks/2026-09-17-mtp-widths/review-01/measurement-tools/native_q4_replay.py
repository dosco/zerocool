#!/usr/bin/env python3
"""Bounded native expert replay diagnostics; no request or production promotion."""
import argparse
import math
from pathlib import Path
import statistics

from cache_residency import require
from capture_routes import load
from combined_q4 import PACKED, freeze, shader_origin, state_proof, validate_operators
from qualification_evidence import confined, save, sha, verify_seal
from screen_q4_packed import FIXTURES, verify_records
from screen_residency import paired_log_interval
from stage200 import Experiment
from verify_stage200 import verify_sources

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT/'docs/benchmarks/2026-09-14-combined-q4/state-01'
CONDITIONS = ('direct', 'scatter', 'resident', 'coordinator')
BUDGET = 12*1024**3
ORDER = [(pair, group) for pair in range(5) for group in (1, 4)]
COUNTERS = ('cpu_encode_ns', 'cpu_gpu_wait_ns', 'gpu_command_ns', 'submissions',
            'allocation_count', 'scratch_reuses')
POSITIONS = [9, 1, 7, 3, 5, 0, 8, 4]
FIXED_KERNELS = dict(route_selection='simd', attention_score_tiles='full', policy='candidate',
    token_tile=1, affine_rows=1, q8_decode_rows=2, gate_pair=False, gdn='original', gdn_rows=4,
    gdn_block=8, shape_table=None, operator_capture='',
    capture_filter=dict(phase='', operator='', layer=-2), profile=False, profile_decode_only=False,
    counter_profile=False, automatic_rules_promoted=False)
CRITERIA = dict(pairs=5, groups=[1, 4], cycles=4, cases_per_cycle=64,
    max_live_groups=2, budget_bytes=BUDGET, gpu_upper_95_below=1,
    wall_upper_95_max=1.03, process_seconds=60, stage_seconds=120)


def uint(value):
    return type(value) is int and value >= 0


def valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def fixture_records(manifest):
    require(manifest.get('complete') is True and manifest.get('origin') == 'existing-Q4-experts' and
            set(manifest.get('layers', {})) == {'0', '16', '32', '47'}, 'Incomplete original Q4 fixtures')
    records = []
    for layer in (0, 16, 32, 47):
        item = manifest['layers'][str(layer)]
        require(item.get('offsets') == list(range(72, 80)) and len(item.get('experts', [])) == 2,
                'Changed expert/input coverage')
        require(len({e['expert'] for e in item['experts']}) == 2 and
                all(uint(e['expert']) and e['expert'] < 512 for e in item['experts']), 'Duplicate or invalid experts')
        for entry, expected_bytes in [(item['inputs'], 8*2560*4),
                                      *[(e['record'], 2764800) for e in item['experts']]]:
            digest = entry.get('sha256')
            require(entry.get('bytes') == expected_bytes and valid_sha256(digest) and
                    Path(entry.get('file', '')).name == entry.get('file') and entry['file'] not in ('', '.', '..'),
                    'Invalid fixture byte identity')
        records.extend(dict(layer=layer, expert=e['expert'], sha256=e['record']['sha256']) for e in item['experts'])
    return records


def metal_state(state, build, resident_bytes):
    require(state.get('build_fingerprint') == build and state.get('device') == 'Apple M1 Pro' and
            state.get('physical_bytes') == 32*1024**3, 'Changed native build or machine')
    require(state.get('live_command_groups') == 0 and uint(state.get('peak_command_groups')) and
            state['peak_command_groups'] <= 2 and state.get('active_scratch_slot') == -1,
            'Outstanding GPU users or active scratch at a timing boundary')
    require(all(uint(state.get(k)) for k in ('live_buffer_bytes', 'peak_buffer_bytes')) and
            0 < state['live_buffer_bytes'] <= state['peak_buffer_bytes'] <= BUDGET,
            'Native allocation exceeds the fixed budget')
    residency = state.get('residency', {}); classes = residency.get('bytes_by_class', {})
    require(residency.get('mode') == 'core-cache' and residency.get('pending_retirements') == 0 and
            isinstance(classes, dict) and all(uint(v) for v in classes.values()) and
            set(classes) <= {'resident', 'state', 'expert'} and
            residency.get('registered_bytes') == sum(classes.values()) and
            classes.get('resident', 0) == resident_bytes and
            uint(residency.get('set_overhead_bytes')) and residency['set_overhead_bytes'] <= 64*1024**2,
            'Changed resident allocation or incomplete residency retirement')
    require(all(uint(state.get(k)) for k in COUNTERS) and
            isinstance(state.get('kernel_dispatches'), dict) and
            all(uint(v) for v in state['kernel_dispatches'].values()), 'Missing native counters')


def observations_clean(arms):
    """Missing observations remain unknown and cannot produce a clean decision."""
    memory = True; host = True; last_memory = None; last_host = None
    for arm in arms:
        for suffix in ('before', 'after'):
            m = arm.get('memory_'+suffix, {}); h = arm.get('host_'+suffix, {})
            memory &= (m.get('compressed_bytes') == 0 and uint(m.get('decompressions')) and
                       uint(m.get('system_swap_used_bytes')) and
                       uint(m.get('physical_footprint_bytes')) and m['physical_footprint_bytes'] > 0 and
                       uint(m.get('physical_footprint_peak_bytes')) and
                       m['physical_footprint_bytes'] <= m['physical_footprint_peak_bytes'] <= BUDGET)
            if last_memory is not None:
                memory &= (m.get('decompressions') == last_memory.get('decompressions') and
                           m.get('system_swap_used_bytes') == last_memory.get('system_swap_used_bytes'))
            host &= (h.get('thermal_state') == 0 and h.get('low_power_mode') is False and
                     isinstance(h.get('power_source'), str) and bool(h['power_source']) and
                     uint(h.get('monotonic_ns')) and h['monotonic_ns'] > 0)
            if last_host is not None:
                host &= (h.get('power_source') == last_host.get('power_source') and
                         uint(h.get('monotonic_ns')) and uint(last_host.get('monotonic_ns')) and
                         h['monotonic_ns'] >= last_host['monotonic_ns'])
            last_memory, last_host = m, h
    return dict(clean_memory=bool(memory), clean_host=bool(host))


def metric(pairs, key):
    reference = [a[key] for a, _ in pairs]; candidate = [b[key] for _, b in pairs]
    ratios = [b/a if a > 0 and b > 0 else None for a, b in zip(reference, candidate)]
    available = all(r is not None for r in ratios)
    return dict(reference_ns=reference, packed_ns=candidate,
        reference_median_ns=statistics.median(reference), packed_median_ns=statistics.median(candidate),
        ratios=ratios, confidence_95=paired_log_interval(ratios) if available else None,
        ratio_limitation=None if available else 'A zero wait duration makes its multiplicative ratio undefined; no ratio is imputed.')


def analyze(raw, frozen=None):
    condition = raw.get('condition')
    require(raw.get('kind') == 'native_q4_replay_v1' and condition in CONDITIONS and
            raw.get('complete') is True and raw.get('validation') is False and raw.get('exact') is True and
            raw.get('cycles') == 4 and raw.get('cases_per_cycle') == 64 and
            raw.get('max_live_groups') == 2 and raw.get('scratch_scope') == 'eight-expert-batch' and
            raw.get('budget_bytes') == BUDGET, 'Incomplete or changed bounded native replay')
    require(raw.get('positions') == POSITIONS and all(type(p) is int for p in raw['positions']) and
            valid_sha256(raw.get('output_sha256')) and raw.get('timed_final_outputs_checked') is True and
            raw.get('untouched_destinations_checked') is True,
            'Missing timed-output, untouched-destination or position identity checks')
    resident = raw.get('resident_bytes')
    require(uint(resident) and resident == raw.get('expected_resident_bytes') and
            (0 < resident < BUDGET if condition in ('resident', 'coordinator') else resident == 0), 'Wrong resident intervention')
    if condition == 'coordinator':
        require(raw.get('execution_schedule') == 'coordinator-all-hit' and raw.get('cache_slots') == 8 and
                raw.get('io_workers') == 8, 'Changed all-hit coordinator geometry')
    records = fixture_records(raw['fixture_manifest'])
    build = raw['initial_metal'].get('build_fingerprint')
    require(isinstance(build, str) and len(build) == 64, 'Missing native build')
    if frozen is not None:
        require(build == frozen['build'] and raw['fixture_manifest'].get('artifact_revision') ==
                frozen['artifact_revision'] and frozen['budget_bytes'] == BUDGET,
                'Replay differs from frozen artifact, native build or memory budget')
    for name in ('initial_metal', 'final_metal'):
        metal_state(raw[name], build, resident)
    require([(p['pair'], p['group']) for p in raw['pairs']] == ORDER,
            'Missing or reordered five-pair/group coverage')
    all_arms = []; samples = {1: [], 4: []}
    for pair in raw['pairs']:
        expected_order = ['packed-r2', 'reference'] if pair['pair'] % 2 else ['reference', 'packed-r2']
        require([a['variant'] for a in pair['arms']] == expected_order, 'Changed alternating native replay order')
        by_variant = {}
        for arm in pair['arms']:
            sample = arm['sample']; before, after = arm['metal_before'], arm['metal_after']
            for state in (before, after):
                metal_state(state, build, resident)
                require(state.get('kernels') == dict(FIXED_KERNELS, q4_decode=arm['variant']),
                        'Changed native kernel selectors, instrumentation or capture configuration')
            require(sample.get('expert_executions') == 256 and
                    sample.get('submissions') == 256//pair['group'], 'Changed executed work or command grouping')
            for key in ('wall_ns', *COUNTERS):
                value = sample.get(key)
                require(uint(value) and (value > 0 if key in ('wall_ns', 'cpu_encode_ns', 'gpu_command_ns') else True),
                        'Invalid or missing native timing/counter')
                if key != 'wall_ns':
                    require(after[key] >= before[key] and value == after[key]-before[key],
                            'Reported native timing/counter differs from its snapshots')
            counts = {key: after['kernel_dispatches'].get(key, 0)-before['kernel_dispatches'].get(key, 0)
                      for key in set(before['kernel_dispatches']) | set(after['kernel_dispatches'])}
            require(all(v >= 0 for v in counts.values()), 'Native dispatch counter reset')
            counts = {key: value for key, value in counts.items() if value}
            names = PACKED if arm['variant'] == 'packed-r2' else ('q4_gate_up', 'q4_mm')
            expected = dict.fromkeys(names, 256)
            if condition != 'direct': expected['scatter_experts'] = 256
            require(counts == expected == sample.get('kernel_dispatches'), 'Changed dispatch path or incomplete expert work')
            require(sample['scratch_reuses'] > 0, 'Scratch reuse was not exercised')
            if condition == 'coordinator':
                require(uint(sample.get('coordinator_wait_ns')) and sample.get('ready_hits') == 256 and
                        all(type(sample.get(k)) is int and sample[k] == 0 for k in
                            ('new_misses', 'loading_joins', 'read_calls', 'read_bytes', 'allocation_count')),
                        'Coordinator replay has missing counters, reads, allocations or non-ready expert work')
            all_arms.append(arm); by_variant[arm['variant']] = sample
        samples[pair['group']].append((by_variant['reference'], by_variant['packed-r2']))
    clean = observations_clean(all_arms)
    metric_keys = ('gpu_command_ns', 'wall_ns', 'cpu_encode_ns', 'cpu_gpu_wait_ns')
    if condition == 'coordinator': metric_keys += ('coordinator_wait_ns',)
    results = [dict(group=group, metrics={key: metric(samples[group], key) for key in metric_keys}) for group in (1, 4)]
    gain = all(row['metrics']['gpu_command_ns']['confidence_95']['high'] < 1 and
               row['metrics']['wall_ns']['confidence_95']['high'] <= 1.03 for row in results)
    clean_all = all(clean.values())
    result = dict(status='disturbed' if not clean_all else 'diagnostic_gain' if gain else 'no_clear_native_gain',
        condition=condition, **clean, criteria=CRITERIA, results=results,
        records=records, output_sha256=raw['output_sha256'], positions=POSITIONS,
        timed_final_outputs_checked=True, untouched_destinations_checked=True,
        normal_request_latency_qualified=False, production_promoted=False,
        next_condition_automatically_admitted=False,
        limitations=['One process, eight original Q4 experts and 64 saved input/expert cases repeated four times per arm.',
            'Five pair intervals describe correlated replay samples; they are not full-request or sustained decode confidence.',
            'GPU command time, CPU encoding and host waiting can overlap and must not be added as exclusive token costs.',
            'Zero host wait time is retained; undefined wait ratios are not replaced with a synthetic value.',
            'No SSD reads or other model operators occur in timing; resident condition loads real weights before timing.',
            'Direct output is a synthetic bridge using fewer temporaries, not an admitted production scratch configuration.'
                if condition == 'direct' else 'Scatter uses the normal expert contribution path; routing, reads and final token reduction are absent.',
            'Native operator output bytes are checked outside timing; complete-model state comes from the separately bound prior proof.',
            'Memory and host state are sampled at boundaries, so brief events between observations can be missed.',
            'No tokens-per-second projection, runtime selection, sample pooling or promotion follows from this diagnostic.'])
    if condition == 'coordinator':
        result.update(execution_schedule='coordinator-all-hit', cache_slots=8, io_workers=8,
                      all_hit_expert_work=True, timed_read_calls=0, timed_read_bytes=0)
        result['limitations'].append('The actual expert coordinator sees eight already-ready cache entries. '
            'It performs no timed reads; completion geometry under real misses remains unmeasured.')
    return result


def validate_replay(timing, operators, frozen, records):
    require(timing.get('fixture_manifest') == operators.get('fixture'), 'Replay and validation use different fixtures')
    require(fixture_records(timing['fixture_manifest']) == records, 'Changed verified prepared-record identity')
    return analyze(timing, frozen)


def run(args):
    require(args.condition in CONDITIONS, 'An explicit replay condition is required')
    exp = Experiment(args.output, 'native_q4_replay_diagnostic_v1', [], [], CRITERIA['stage_seconds'])
    with exp:
        exp.report.update(condition=args.condition, criteria=CRITERIA, shader_origin=shader_origin())
        exp.report['correctness'] = state_proof(args.state, exp.frozen)
        exp.report['state_source'] = dict(path=str(args.state.resolve()), sha256=sha(args.state/'evidence-files.json'))
        paths = [args.binary, ROOT/'scripts/qwen/probe_q4_packed.metal',
                 ROOT/'kernels/metal/qwen.metal',
                 ROOT/'docs/benchmarks/2026-09-14-combined-q4/integration-origin.json',
                 *[p for p in args.state.iterdir() if p.is_file()],
                 *[p for p in FIXTURES.iterdir() if p.is_file()]]
        freeze(exp, paths)
        exp.report['verified_records'] = verify_records(exp.prepared); exp.persist()
        exp.guard.check_resources(initial=True)
        exp.command([args.binary, FIXTURES, exp.out/'operators.json'], 'operators',
                    CRITERIA['process_seconds'], validation=True)
        operators = load(exp.out/'operators.json')
        exp.report['operators'] = validate_operators(operators, exp.frozen, exp.report['verified_records']); exp.persist()
        command = [args.binary, FIXTURES, exp.out/'timing.json', 'replay', args.condition]
        if args.condition in ('resident', 'coordinator'): command.append(exp.model)
        exp.command(command, 'timing', CRITERIA['process_seconds'])
        timing = load(exp.out/'timing.json')
        require(timing.get('condition') == args.condition, 'Wrong explicit condition was executed')
        exp.report.update(validate_replay(timing, operators, exp.frozen, exp.report['verified_records']))
    return exp.report


def verify(directory):
    """Reconstruct a single sealed condition without executing a native binary."""
    directory = directory.resolve(); digest = sha(directory/'evidence-files.json')
    verify_seal(directory, digest)
    saved, frozen = load(directory/'summary.json'), load(directory/'identity.json')
    require(saved.get('kind') == 'native_q4_replay_diagnostic_v1' and saved.get('condition') in CONDITIONS and
            saved.get('criteria') == CRITERIA and saved.get('configurations') == [] and saved.get('workload') == [] and
            saved['identity'] == {k: frozen[k] for k in saved['identity']}, 'Changed replay experiment identity or design')
    sources = dict(frozen, files={p: h for p, h in frozen['files'].items() if Path(p).is_relative_to(frozen['root'])})
    provenance = verify_sources(directory.parent, sources)
    if 'shader_origin' in saved:
        require(saved['shader_origin'] == shader_origin(), 'Changed shader origin')
    if 'state_source' in saved:
        state = Path(saved['state_source']['path'])
        require(sha(state/'evidence-files.json') == saved['state_source']['sha256'] and
                state_proof(state, frozen) == saved['correctness'], 'Changed prior complete-model state proof')
    result = dict(kind='native_q4_replay_audit_v1', complete=True, audit_passed=True,
        source_seal_sha256=digest, source_provenance=provenance, condition=saved['condition'],
        recorded_complete=saved['complete'], recorded_status=saved['status'],
        normal_request_latency_qualified=False, production_promoted=False)
    if saved['complete'] is not True:
        require(saved['status'] in ('failed', 'resource_blocked', 'interrupted', 'time_budget_exhausted'),
                'Unfinished replay lacks a terminal disposition')
        return dict(result, recomputed_status=saved['status'], result_qualified=False)
    require('state_source' in saved and 'shader_origin' in saved, 'Completed replay is missing prior correctness evidence')
    operators, timing = (load(confined(directory, name+'.json')) for name in ('operators', 'timing'))
    require(validate_operators(operators, frozen, saved['verified_records']) == saved['operators'],
            'Changed native operator validation')
    decision = validate_replay(timing, operators, frozen, saved['verified_records'])
    require(all(saved.get(k) == v for k, v in decision.items()), 'Changed diagnostic decision')
    return dict(result, recomputed_status=decision['status'], results=decision['results'],
        clean_memory=decision['clean_memory'], clean_host=decision['clean_host'], result_qualified=False,
        limitations=['Reconstructs sealed raw evidence and source identities; no new GPU execution or prepared payload rehash.',
                    'Each condition is audited separately; diagnostic_gain does not qualify normal request latency.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    execute = sub.add_parser('run'); execute.add_argument('--output', type=Path, required=True)
    execute.add_argument('--binary', type=Path, default=ROOT/'build/qwen/qwen_q4_check')
    execute.add_argument('--state', type=Path, default=STATE)
    execute.add_argument('--condition', choices=CONDITIONS, required=True)
    audit = sub.add_parser('verify'); audit.add_argument('directory', type=Path)
    audit.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'verify': save(args.output, verify(args.directory))
    else:
        args.binary = args.binary.resolve(); args.state = args.state.resolve()
        raise SystemExit(0 if run(args)['complete'] else 2)
