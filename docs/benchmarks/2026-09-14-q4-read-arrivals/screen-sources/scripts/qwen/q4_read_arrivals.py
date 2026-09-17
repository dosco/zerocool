#!/usr/bin/env python3
"""Prepared-expert arrival diagnostics with independent timing and event capture."""
import argparse
from collections import Counter
from pathlib import Path
import shutil

from cache_residency import require
from capture_routes import load
from combined_q4 import freeze, shader_origin, state_proof
from native_q4_replay import (BUDGET, COUNTERS, FIXED_KERNELS, PACKED, POSITIONS,
    fixture_records, metal_state, metric, observations_clean, uint, valid_sha256)
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
from screen_q4_packed import FIXTURES, verify_records
from stage200 import Experiment
from verify_stage200 import verify_sources

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT/'docs/benchmarks/2026-09-14-combined-q4/state-01'
CONTROL = ROOT/'docs/benchmarks/2026-09-14-native-q4-replay/coordinator-01'
PROTOCOL = ROOT/'docs/benchmarks/2026-09-14-q4-read-arrivals/protocol.md'
OUTPUT_SHA = '05b73dca9c175709ede2bf65096b68e9193d32277a3d95f97d142adfe7728d45'
EXPERT_BYTES = 2764800
CRITERIA = dict(timing_pairs=5, groups=[1, 4], timing_cycles=4, capture_cycles=1,
    cases_per_cycle=64, cache_slots=8, io_workers=8, max_live_groups=2,
    budget_bytes=BUDGET, gpu_upper_95_below=1, wall_upper_95_max=1.03,
    process_seconds=60, stage_seconds=180)


def disk_observation(arm):
    """Device counters cover preparation and other processes, not only timed reads."""
    snapshots = []
    for suffix in ('before', 'after'):
        raw = arm.get('disk_'+suffix)
        if not isinstance(raw, dict) or not isinstance(raw.get('devices'), list) or not raw['devices']:
            return dict(available=False, read_bytes=None, write_bytes=None, reason='Missing device counters')
        devices = raw['devices']
        if any(not all(uint(d.get(k)) for k in ('registry_id', 'read_bytes', 'write_bytes')) for d in devices):
            return dict(available=False, read_bytes=None, write_bytes=None, reason='Invalid device counters')
        if len({d['registry_id'] for d in devices}) != len(devices):
            return dict(available=False, read_bytes=None, write_bytes=None, reason='Duplicate device IDs')
        snapshots.append({d['registry_id']: d for d in devices})
    before, after = snapshots
    if before.keys() != after.keys():
        return dict(available=False, read_bytes=None, write_bytes=None, reason='Device inventory changed')
    if any(after[d][k] < before[d][k] for d in before for k in ('read_bytes', 'write_bytes')):
        return dict(available=False, read_bytes=None, write_bytes=None, reason='Device counter reset')
    return dict(available=True, **{k: sum(after[d][k]-before[d][k] for d in before)
                                  for k in ('read_bytes', 'write_bytes')}, reason=None)


def trace_batches(sample, group, hits, records):
    batches = sample.get('batches')
    require(isinstance(batches, list) and [(b['cycle'], b['batch']) for b in batches] == [(0, b) for b in range(8)],
            'Missing, duplicate or reordered captured batches')
    selected = {(r['layer'], r['expert']): i for i, r in enumerate(records)}
    occupancy = []; commands = []; waits = 0; gpu = 0; overlapping_pairs = 0
    readiness = {name: dict(ready_to_submission_ns=[], read_complete_to_submission_ns=[])
                 for name in ('ready_hit', 'new_miss')}
    previous_release = 0
    for batch in batches:
        hit_indices = [(batch['batch']+j) % 8 for j in range(hits)]
        require(batch.get('hit_indices') == hit_indices, 'Changed rotating hit preparation')
        detail = batch['timing']; events = detail.get('records')
        require(isinstance(events, list) and len(events) == 8 and
                {(r['layer'], r['expert']) for r in events} == set(selected), 'Incomplete selected-expert capture')
        require(detail.get('ready_hits') == hits and detail.get('new_misses') == 8-hits and
                detail.get('loading_joins') == 0 and detail.get('peak_leases') == 8 and
                uint(detail.get('peak_gpu_groups')) and 0 < detail['peak_gpu_groups'] <= 2 and
                uint(detail.get('duration_ns')) and detail['duration_ns'] > 0,
                'Changed detailed coordinator work or resource bounds')
        by_submit = {}; queue = 0; service = 0; ready_encode = 0; ready_gpu = 0
        for record in events:
            index = selected[record['layer'], record['expert']]
            require(record.get('position') == POSITIONS[index] and record.get('acquisition') ==
                    ('ready_hit' if index in hit_indices else 'new_miss'), 'Wrong expert destination or hit classification')
            fields = ('admitted_ns', 'read_queued_ns', 'read_started_ns', 'read_completed_ns',
                      'encoded_ns', 'submitted_ns', 'gpu_start_ns', 'gpu_end_ns', 'released_ns')
            require(all(uint(record.get(k)) and record[k] > 0 for k in fields), 'Missing event timestamps')
            admitted, queued, started, completed, encoded, submitted, start, end, released = (record[k] for k in fields)
            require(queued <= started <= completed <= encoded and
                    previous_release <= admitted <= encoded <= submitted <= start <= end <= released,
                    'Reversed read, encoding, GPU or ownership timestamps')
            if index in hit_indices:
                require(completed <= admitted, 'Prepared hit was not ready at admission')
            else:
                queue += started-queued; service += completed-started
            ready = max(admitted, completed)
            ready_encode += encoded-ready; ready_gpu += start-ready
            readiness[record['acquisition']]['ready_to_submission_ns'].append(submitted-ready)
            readiness[record['acquisition']]['read_complete_to_submission_ns'].append(submitted-completed)
            by_submit.setdefault(submitted, []).append(record)
        require(detail.get('read_queue_sum_ns') == queue and detail.get('read_service_sum_ns') == service and
                detail.get('ready_to_encode_sum_ns') == ready_encode and detail.get('ready_to_gpu_sum_ns') == ready_gpu,
                'Detailed read or readiness sums differ from individual events')
        batch_gpu = 0; batch_commands = []
        for submitted, members in sorted(by_submit.items()):
            require(0 < len(members) <= group, 'Captured command exceeds the ready-group cap')
            bounds = {(r['gpu_start_ns'], r['gpu_end_ns']) for r in members}
            require(len(bounds) == 1, 'One command has conflicting GPU timestamps')
            start, end = bounds.pop()
            command = dict(batch=batch['batch'], submitted_ns=submitted, gpu_start_ns=start,
                           gpu_end_ns=end, experts=len(members))
            overlapping_pairs += sum(max(start, prior['gpu_start_ns']) < min(end, prior['gpu_end_ns'])
                                     for prior in batch_commands)
            batch_commands.append(command); commands.append(command)
            occupancy.append(len(members)); batch_gpu += end-start
        require(uint(detail.get('gpu_execution_sum_ns')) and
                abs(detail['gpu_execution_sum_ns']-batch_gpu) <= len(by_submit) and
                uint(detail.get('coordinator_wait_ns')), 'Invalid command or coordinator wait durations')
        gpu += detail['gpu_execution_sum_ns']; waits += detail['coordinator_wait_ns']
        previous_release = max(r['released_ns'] for r in events)
        require(previous_release-min(r['admitted_ns'] for r in events) <= detail['duration_ns'],
                'Expert lifetimes exceed their captured batch duration')
    require(len(commands) == sample['submissions'] and gpu == sample['gpu_command_ns'] and
            waits == sample['coordinator_wait_ns'], 'Captured commands or waits do not cover the full sample')
    return dict(captured_batches=8, captured_experts=64, captured_commands=len(commands),
        occupancy_histogram={str(k): v for k, v in sorted(Counter(occupancy).items())}, commands=commands,
        readiness_by_acquisition=readiness, gpu_interval_overlap_pairs=overlapping_pairs,
        coordinator_wait_ns=waits, gpu_command_ns=gpu)


def analyze(raw, frozen=None, expected_records=None, expected_output=OUTPUT_SHA):
    mode = raw.get('mode'); hits = raw.get('hits_per_batch')
    cycles = 4 if mode == 'timing' else 1; pairs = 5 if mode == 'timing' else 1
    require(raw.get('kind') == 'native_q4_arrivals_v1' and mode in ('check', 'timing', 'trace') and
            hits in (2, 8) and raw.get('complete') is True and raw.get('exact') is True and
            raw.get('validation') is (mode == 'check') and raw.get('cycles') == cycles and
            raw.get('budget_bytes') == BUDGET and raw.get('cache_slots') == 8 and raw.get('io_workers') == 8 and
            raw.get('max_live_groups') == 2 and raw.get('scratch_scope') == 'eight-expert-batch' and
            raw.get('disk_counter_scope') == 'entire-arm-including-preparation-and-other-processes' and
            raw.get('wall_scope') == 'sum-of-coordinator-windows-excludes-preparation' and
            raw.get('timed_final_outputs_checked') is True and raw.get('untouched_destinations_checked') is True and
            valid_sha256(raw.get('output_sha256')) and raw['output_sha256'] == expected_output,
            'Incomplete or changed native arrival experiment')
    resident = raw.get('resident_bytes')
    require(uint(resident) and 0 < resident < BUDGET and resident == raw.get('expected_resident_bytes'),
            'Missing or changed real resident allocation')
    records = fixture_records(raw['fixture_manifest'])
    if expected_records is not None: require(records == expected_records, 'Changed prepared expert fixtures')
    build = raw['initial_metal'].get('build_fingerprint')
    require(valid_sha256(build) and valid_sha256(raw.get('prepared_manifest_sha256')), 'Missing native or prepared identity')
    if frozen is not None:
        require(build == frozen['build'] and raw['prepared_manifest_sha256'] == frozen['prepared_manifest_sha256'] and
                raw['fixture_manifest'].get('artifact_revision') == frozen['artifact_revision'] and
                frozen['budget_bytes'] == BUDGET, 'Arrival replay differs from frozen model, layout or build')
    for field in ('initial_metal', 'final_metal'): metal_state(raw[field], build, resident)
    require([(p['pair'], p['group']) for p in raw['pairs']] ==
            [(p, g) for p in range(pairs) for g in (1, 4)], 'Missing or reordered arrival pairs')
    arms = []; devices = []; captures = []; samples = {1: [], 4: []}
    batches = cycles*8; executions = cycles*64
    for pair in raw['pairs']:
        require([a['variant'] for a in pair['arms']] ==
                (['packed-r2', 'reference'] if pair['pair'] % 2 else ['reference', 'packed-r2']), 'Changed paired arm order')
        by = {}
        for arm in pair['arms']:
            sample = arm['sample']; before, after = arm['metal_before'], arm['metal_after']
            for state in (before, after):
                metal_state(state, build, resident)
                # Validation mode is a Metal layer setting, not an instrumented kernel configuration.
                require(state.get('kernels') == dict(FIXED_KERNELS, q4_decode=arm['variant']),
                        'Changed selectors or enabled Metal profiling/capture')
            for key in ('wall_ns', *COUNTERS, 'coordinator_wait_ns', 'preparation_ns'):
                require(uint(sample.get(key)) and (sample[key] > 0 if key in
                        ('wall_ns', 'cpu_encode_ns', 'gpu_command_ns', 'preparation_ns') else True),
                        'Missing or invalid arrival timing/counters')
                if key in COUNTERS:
                    require(after[key] >= before[key] and sample[key] == after[key]-before[key],
                            'Arrival sample differs from native counter snapshots')
            require(sample['expert_executions'] == executions and sample['ready_hits'] == batches*hits and
                    sample['new_misses'] == batches*(8-hits) and sample['loading_joins'] == 0 and
                    sample.get('read_bytes') == batches*(8-hits)*EXPERT_BYTES and
                    sample.get('expert_load_calls') == batches*(8-hits) and
                    sample.get('preparation_read_bytes') == batches*hits*EXPERT_BYTES and
                    sample.get('preparation_load_calls') == batches*hits and sample['allocation_count'] == 0,
                    'Wrong prepared hit/miss population, source bytes or allocation count')
            require(uint(sample.get('arm_elapsed_ns')) and
                    sample['arm_elapsed_ns'] >= sample['wall_ns']+sample['preparation_ns'],
                    'Total arm time cannot exclude the preparation window')
            expected_submissions = executions//pair['group']
            require(expected_submissions <= sample['submissions'] <= executions and
                    (sample['submissions'] == expected_submissions if hits == 8 or pair['group'] == 1 else True),
                    'Command count exceeds its cap or all-hit geometry')
            delta = {k: after['kernel_dispatches'].get(k, 0)-before['kernel_dispatches'].get(k, 0)
                     for k in set(before['kernel_dispatches']) | set(after['kernel_dispatches'])}
            require(all(v >= 0 for v in delta.values()), 'Dispatch counters reset')
            delta = {k: v for k, v in delta.items() if v}
            names = PACKED if arm['variant'] == 'packed-r2' else ('q4_gate_up', 'q4_mm')
            require(delta == sample['kernel_dispatches'] == dict.fromkeys((*names, 'scatter_experts'), executions),
                    'Missing experts or changed arithmetic/scatter path')
            require(sample['scratch_reuses'] > 0, 'Native scratch reuse was not exercised')
            label = dict(pair=pair['pair'], group=pair['group'], variant=arm['variant'])
            if mode == 'trace': captures.append(dict(label, **trace_batches(sample, pair['group'], hits, records)))
            else: require(not sample.get('batches'), 'Timing or check arm unexpectedly includes detailed event capture')
            devices.append(dict(label, **disk_observation(arm)))
            arms.append(arm); by[arm['variant']] = sample
        samples[pair['group']].append((by['reference'], by['packed-r2']))
    clean = observations_clean(arms)
    results = []
    if mode == 'timing':
        results = [dict(group=group, metrics={k: metric(samples[group], k) for k in
                   ('gpu_command_ns', 'wall_ns', 'cpu_encode_ns', 'cpu_gpu_wait_ns', 'coordinator_wait_ns', 'preparation_ns')})
                   for group in (1, 4)]
        gain = all(r['metrics']['gpu_command_ns']['confidence_95']['high'] < 1 and
                   r['metrics']['wall_ns']['confidence_95']['high'] <= 1.03 for r in results)
        status = 'diagnostic_gain' if gain else 'no_clear_arrival_gain'
    else: status = 'checked' if mode == 'check' else 'captured'
    return dict(mode=mode, hits_per_batch=hits, status=status if all(clean.values()) else 'disturbed', **clean,
        criteria=CRITERIA, results=results, captures=captures, device_observations=devices,
        device_observations_complete=all(d['available'] for d in devices),
        coverage=dict(pairs=pairs, groups=[1, 4], cycles_per_arm=cycles, expert_executions_per_arm=executions,
                      detailed_batches=len(captures)*8, detailed_experts=len(captures)*64),
        output_sha256=raw['output_sha256'], wall_scope=raw['wall_scope'], disk_counter_scope=raw['disk_counter_scope'],
        normal_request_latency_qualified=False, production_promoted=False,
        limitations=['Eight fixed original Q4 experts; the rotating hit population is declared, not inferred from normal requests.',
            'Wall time sums coordinator/encoding/drain intervals and excludes explicit cache preparation between batches.',
            'Preparation bytes and logical expert-load calls are separate from timed demand bytes and calls.',
            'Internal-device counters cover preparation and unrelated processes; missing observations stay unavailable.',
            'Detailed events come from a separate one-pair capture. They are not pooled with timing or used as exclusive wait costs.',
            'Whole-command GPU timestamp intervals may overlap; their sums are not exclusive elapsed time.',
            'Hit read-complete-to-submission intervals include time before batch admission; ready-to-submission starts at admission or read completion, whichever is later.',
            'Ready-group size is a cap. Mixed misses may submit smaller groups; captured occupancy is measured from actual submissions.',
            'Resident allocation is present, but other model operators, routing and final token reduction are absent.',
            'Boundary memory/host observations can miss brief events. Repeated fixtures and within-process pairs are correlated.',
            'No normal-request throughput projection, complete-request qualification or production selection follows.'])


def control_proof(path, frozen):
    verify_seal(path, sha(path/'evidence-files.json'))
    saved = load(path/'summary.json')
    require(saved.get('complete') is True and saved.get('condition') == 'coordinator' and
            saved['identity'] == {k: frozen[k] for k in saved['identity']} and saved.get('output_sha256') == OUTPUT_SHA,
            'Missing compatible completed all-hit control')
    from native_q4_replay import analyze as replay_analyze
    result = replay_analyze(load(path/'timing.json'), frozen)
    require(all(saved.get(k) == v for k, v in result.items()), 'Changed prior all-hit control')
    return dict(output_sha256=OUTPUT_SHA, build=saved['identity']['build'])


def run(args):
    exp = Experiment(args.output, 'q4_read_arrivals_diagnostic_v1', [], [], CRITERIA['stage_seconds'])
    with exp:
        exp.report.update(hits_per_batch=args.hits, criteria=CRITERIA, shader_origin=shader_origin())
        shutil.copyfile(PROTOCOL, exp.out/'protocol.md')
        exp.report['protocol_sha256'] = sha(PROTOCOL)
        exp.report['correctness'] = state_proof(args.state, exp.frozen)
        exp.report['control'] = control_proof(args.control, exp.frozen)
        for key in ('state', 'control'):
            path = getattr(args, key)
            exp.report[key+'_source'] = dict(path=str(path.resolve()), sha256=sha(path/'evidence-files.json'))
        freeze(exp, [args.binary, PROTOCOL, exp.out/'protocol.md',
            ROOT/'scripts/qwen/probe_q4_packed.metal', ROOT/'kernels/metal/qwen.metal',
            *[p for d in (args.state, args.control, FIXTURES) for p in d.iterdir() if p.is_file()]])
        exp.report['verified_records'] = verify_records(exp.prepared); exp.persist()
        exp.guard.check_resources(initial=True)
        for mode in ('check', 'timing', 'trace'):
            exp.command([args.binary, FIXTURES, exp.out/(mode+'.json'), 'arrivals', exp.model, exp.prepared,
                         str(args.hits), mode], mode, CRITERIA['process_seconds'], validation=mode == 'check')
            raw = load(exp.out/(mode+'.json'))
            require(raw.get('mode') == mode and raw.get('hits_per_batch') == args.hits, 'Wrong requested mode or hit population')
            exp.report[mode] = analyze(raw, exp.frozen, exp.report['verified_records']); exp.persist()
            if not all(exp.report[mode][k] for k in ('clean_memory', 'clean_host')):
                raise ResourceBlocked('Arrival replay has disturbed or unavailable memory/host observations')
        exp.report.update(status=exp.report['timing']['status'], normal_request_latency_qualified=False, production_promoted=False)
    return exp.report


def verify(directory):
    directory = directory.resolve(); seal_digest = sha(directory/'evidence-files.json')
    verify_seal(directory, seal_digest)
    saved, frozen = load(directory/'summary.json'), load(directory/'identity.json')
    require(saved.get('kind') == 'q4_read_arrivals_diagnostic_v1' and saved.get('hits_per_batch') in (2, 8) and
            saved.get('criteria') == CRITERIA and saved.get('configurations') == [] and saved.get('workload') == [] and
            saved['identity'] == {k: frozen[k] for k in saved['identity']}, 'Changed arrival experiment identity or design')
    sources = dict(frozen, files={p: h for p, h in frozen['files'].items() if Path(p).is_relative_to(frozen['root'])})
    provenance = verify_sources(directory.parent, sources)
    if 'protocol_sha256' in saved:
        require(sha(directory/'protocol.md') == saved['protocol_sha256'] ==
                frozen['files'][str(PROTOCOL.resolve())], 'Changed declared arrival protocol')
    if 'shader_origin' in saved: require(saved['shader_origin'] == shader_origin(), 'Changed integrated shader origin')
    for key, checker in (('state', state_proof), ('control', control_proof)):
        if key+'_source' not in saved: continue
        source = saved[key+'_source']; path = Path(source['path'])
        require(sha(path/'evidence-files.json') == source['sha256'] and
                checker(path, frozen) == saved['correctness' if key == 'state' else 'control'], 'Changed prior '+key+' evidence')
    results = {}
    for mode in ('check', 'timing', 'trace'):
        if mode not in saved: continue
        raw = load(directory/(mode+'.json'))
        require(raw.get('mode') == mode and raw.get('hits_per_batch') == saved['hits_per_batch'], 'Wrong captured mode/hit count')
        results[mode] = analyze(raw, frozen, saved['verified_records'])
        require(saved[mode] == results[mode], 'Changed arrival '+mode+' analysis')
    if saved['complete'] is True:
        require(set(results) == {'check', 'timing', 'trace'} and 'state_source' in saved and 'control_source' in saved and
                'protocol_sha256' in saved and
                saved['status'] == results['timing']['status'] and
                all(r[k] for r in results.values() for k in ('clean_memory', 'clean_host')), 'Incomplete or disturbed completed stage')
    else:
        require(saved['status'] in ('failed', 'resource_blocked', 'interrupted', 'time_budget_exhausted'),
                'Unfinished arrivals stage lacks a terminal disposition')
    return dict(kind='q4_read_arrivals_audit_v1', complete=True, audit_passed=True,
        source_seal_sha256=seal_digest, source_provenance=provenance, hits_per_batch=saved['hits_per_batch'],
        recorded_complete=saved['complete'], recomputed_status=saved['status'], modes=results,
        normal_request_latency_qualified=False, production_promoted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='mode', required=True)
    execute = sub.add_parser('run'); execute.add_argument('--output', type=Path, required=True)
    execute.add_argument('--hits', type=int, choices=(8, 2), required=True)
    execute.add_argument('--binary', type=Path, default=ROOT/'build/qwen/qwen_q4_check')
    execute.add_argument('--state', type=Path, default=STATE); execute.add_argument('--control', type=Path, default=CONTROL)
    audit = sub.add_parser('verify'); audit.add_argument('directory', type=Path); audit.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'verify': save(args.output, verify(args.directory))
    else:
        for field in ('binary', 'state', 'control'): setattr(args, field, getattr(args, field).resolve())
        raise SystemExit(0 if run(args)['complete'] else 2)
