#!/usr/bin/env python3
"""Prepared-expert arrival diagnostics with independent timing and event capture."""
import argparse
from collections import Counter
import math
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
STORAGE_PROTOCOL = PROTOCOL.with_name('storage-protocol.md')
RETAIN_CACHE = 'retain-system-cache'
INVALIDATE_FILES = 'invalidate-selected-file-ranges'
STORAGE_CRITERIA = dict(scope='each-timing-arm', minimum_device_to_application_ratio=0.9,
                        maximum_device_to_application_ratio=1.1)
OUTPUT_SHA = '05b73dca9c175709ede2bf65096b68e9193d32277a3d95f97d142adfe7728d45'
EXPERT_BYTES = 2764800
SHARED_KERNELS = ('q8_gate_up', 'q8_mv_packed_r2_w4', 'plain_mm')
SHARED_LAYERS = [0, 16, 32, 47, 0, 16, 32, 47]
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


def shared_trace(batch, commands, variant):
    """Join existing command profiles to leased experts without inventing kernel costs."""
    require(batch.get('shared_layer') == SHARED_LAYERS[batch['batch']], 'Changed rotating shared layer')
    profile = batch.get('command_profile', {})
    require(profile.get('truncated') is False and profile.get('coverage') == 'all-dispatches' and
            profile.get('timing_kind') == 'existing command groups; mixed stages are not isolated kernel costs' and
            profile.get('normal_request_latency_qualified') is False,
            'Missing, truncated or per-dispatch shared command profile')
    groups = profile.get('command_groups')
    require(isinstance(groups, list) and [g.get('submitted_ns') for g in groups] ==
            [c['submitted_ns'] for c in commands], 'Shared profile has missing, reordered or extra commands')
    names = PACKED if variant == 'packed-r2' else ('q4_gate_up', 'q4_mm')
    shared_submit = None; previous_encode = 0
    for index, (group, command) in enumerate(zip(groups, commands)):
        require(all(type(group.get(k)) in (int, float) and math.isfinite(group[k]) and group[k] > 0
                    for k in ('gpu_start_seconds', 'gpu_end_seconds')),
                'Missing shared GPU command timestamps')
        for key, native in (('gpu_start_seconds', 'gpu_start_ns'), ('gpu_end_seconds', 'gpu_end_ns')):
            require(abs(int(group[key]*1e9)-command[native]) <= 2, 'Shared command does not join expert GPU interval')
        operations = group.get('operations')
        require(isinstance(operations, list), 'Missing shared command operations')
        shared = operations[:3] if index == 0 else []
        routed = operations[3:] if index == 0 else operations
        if index == 0:
            require([op.get('kernel') for op in shared] == list(SHARED_KERNELS) and
                    all(op.get('stage') == 'shared_expert' and op.get('layer') == batch['shared_layer'] for op in shared),
                    'Shared chain is absent or was reordered after routed work')
            shared_submit = command['submitted_ns']
        require(len(routed) == 3*command['experts'] and
                [op.get('kernel') for op in routed] == list((*names, 'scatter_experts'))*command['experts'] and
                all(op.get('stage') == 'routed_expert' for op in routed),
                'Shared command contains missing, extra or changed routed arithmetic')
        event_layers = Counter(r['layer'] for r in batch['timing']['records'] if r['submitted_ns'] == command['submitted_ns'])
        require(Counter(op.get('layer') for op in routed) == Counter({k: v*3 for k, v in event_layers.items()}),
                'Routed profile layers disagree with expert ownership')
        for op in operations:
            require(op.get('tokens') == 1 and op.get('offset') == 72+batch['batch'] and
                    uint(op.get('encoded_at_ns')) and uint(op.get('encode_ns')) and
                    previous_encode <= op['encoded_at_ns'] <= op['encoded_at_ns']+op['encode_ns'] <= command['submitted_ns'] and
                    not any(k in op for k in ('counter_index', 'gpu_pass_ns', 'gpu_begin_ticks', 'gpu_end_ticks')),
                    'Changed shared workload, reversed encoding times or per-dispatch timing')
            previous_encode = op['encoded_at_ns']+op['encode_ns']
    return dict(batch=batch['batch'], shared_layer=batch['shared_layer'], submitted_ns=shared_submit,
                shared_operations=3, routed_experts=8, command_count=len(commands),
                pure_routed_gpu_ns=None, gpu_scope='shared-and-routed-command-intervals')


def trace_batches(sample, group, hits, records, shared=False, variant=None, cycles=1):
    batches = sample.get('batches')
    require(isinstance(batches, list) and [(b['cycle'], b['batch']) for b in batches] == [(c, b) for c in range(cycles) for b in range(8)],
            'Missing, duplicate or reordered captured batches')
    selected = {(r['layer'], r['expert']): i for i, r in enumerate(records)}
    occupancy = []; commands = []; waits = 0; gpu = 0; overlapping_pairs = 0; shared_commands = []
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
            if cycles > 1: command['cycle'] = batch['cycle']
            overlapping_pairs += sum(max(start, prior['gpu_start_ns']) < min(end, prior['gpu_end_ns'])
                                     for prior in batch_commands)
            batch_commands.append(command); commands.append(command)
            occupancy.append(len(members)); batch_gpu += end-start
        if shared:
            joined = shared_trace(batch, batch_commands, variant)
            if cycles > 1: joined['cycle'] = batch['cycle']
            shared_commands.append(joined)
        else:
            require('command_profile' not in batch and 'shared_layer' not in batch,
                    'Control contains unsupported shared capture fields')
        require(uint(detail.get('gpu_execution_sum_ns')) and
                abs(detail['gpu_execution_sum_ns']-batch_gpu) <= len(by_submit) and
                uint(detail.get('coordinator_wait_ns')), 'Invalid command or coordinator wait durations')
        gpu += detail['gpu_execution_sum_ns']; waits += detail['coordinator_wait_ns']
        previous_release = max(r['released_ns'] for r in events)
        require(previous_release-min(r['admitted_ns'] for r in events) <= detail['duration_ns'],
                'Expert lifetimes exceed their captured batch duration')
    require(len(commands) == sample['submissions'] and gpu == sample['gpu_command_ns'] and
            waits == sample['coordinator_wait_ns'], 'Captured commands or waits do not cover the full sample')
    result = dict(captured_batches=cycles*8, captured_experts=cycles*64, captured_commands=len(commands),
        occupancy_histogram={str(k): v for k, v in sorted(Counter(occupancy).items())}, commands=commands,
        readiness_by_acquisition=readiness, gpu_interval_overlap_pairs=overlapping_pairs,
        coordinator_wait_ns=waits, gpu_command_ns=gpu)
    if shared: result['shared_command_joins'] = shared_commands
    return result


def analyze(raw, frozen=None, expected_records=None, expected_output=OUTPUT_SHA, shared=False, lifetime=None):
    mode = raw.get('mode'); hits = raw.get('hits_per_batch')
    require((raw.get('shared_prelude') is not None) is shared and
            (not shared or raw['shared_prelude'].get('enabled') is True),
            'Shared prelude requires its explicit experiment analyzer')
    require(lifetime in (None, 'batch', 'forward') and
            (raw.get('scratch_lifetime') is not None) is (lifetime is not None) and
            (lifetime is None or shared), 'Scratch lifetime requires its explicit experiment analyzer')
    preparation = raw.get('cache_preparation', RETAIN_CACHE)
    require(preparation in (RETAIN_CACHE, INVALIDATE_FILES), 'Unknown file-cache preparation condition')
    invalidate = preparation == INVALIDATE_FILES
    cycles = 6 if lifetime is not None else 4 if mode == 'timing' else 1
    pairs = 5 if mode == 'timing' else 1
    scratch_scope = '48-pass-arena' if lifetime == 'forward' else 'eight-expert-batch'
    require(raw.get('kind') == 'native_q4_arrivals_v1' and mode in ('check', 'timing', 'trace') and
            hits in (2, 8) and raw.get('complete') is True and raw.get('exact') is True and
            raw.get('validation') is (mode == 'check') and raw.get('cycles') == cycles and
            raw.get('budget_bytes') == BUDGET and raw.get('cache_slots') == 8 and raw.get('io_workers') == 8 and
            raw.get('max_live_groups') == 2 and raw.get('scratch_scope') == scratch_scope and
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
    arms = []; devices = []; captures = []; storage = []; samples = {1: [], 4: []}
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
                require(state.get('kernels') == dict(FIXED_KERNELS, q4_decode=arm['variant'],
                        profile=shared and mode == 'trace'),
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
            if invalidate:
                require(uint(sample.get('invalidation_ns')) and
                        0 < sample['invalidation_ns'] <= sample['preparation_ns'] and
                        uint(sample.get('invalidation_calls')) and sample['invalidation_calls'] == 8*batches,
                        'Missing selected-file invalidation coverage or invalid preparation timing')
            else:
                require(sample.get('invalidation_ns', 0) == 0 and sample.get('invalidation_calls', 0) == 0,
                        'Retained-cache control unexpectedly invalidated file ranges')
            expected_submissions = executions//pair['group']
            require(expected_submissions <= sample['submissions'] <= executions and
                    (sample['submissions'] == expected_submissions if hits == 8 or pair['group'] == 1 else True),
                    'Command count exceeds its cap or all-hit geometry')
            delta = {k: after['kernel_dispatches'].get(k, 0)-before['kernel_dispatches'].get(k, 0)
                     for k in set(before['kernel_dispatches']) | set(after['kernel_dispatches'])}
            require(all(v >= 0 for v in delta.values()), 'Dispatch counters reset')
            delta = {k: v for k, v in delta.items() if v}
            names = PACKED if arm['variant'] == 'packed-r2' else ('q4_gate_up', 'q4_mm')
            expected_dispatches = dict.fromkeys((*names, 'scatter_experts'), executions)
            if shared:
                expected_dispatches.update(dict.fromkeys(SHARED_KERNELS, batches))
                require(sample.get('shared_chains') == batches and sample['scratch_reuses'] == executions*3+batches*3,
                        'Missing shared chains or changed shared scratch coverage')
            else:
                require('shared_chains' not in sample, 'Control contains unsupported shared work')
            require(delta == sample['kernel_dispatches'] == expected_dispatches,
                    'Missing experts or changed arithmetic/scatter path')
            require(sample['scratch_reuses'] > 0, 'Native scratch reuse was not exercised')
            label = dict(pair=pair['pair'], group=pair['group'], variant=arm['variant'])
            if mode == 'trace': captures.append(dict(label, **trace_batches(sample, pair['group'], hits, records, shared, arm['variant'], cycles)))
            else: require(not sample.get('batches'), 'Timing or check arm unexpectedly includes detailed event capture')
            device = disk_observation(arm)
            devices.append(dict(label, **device))
            if invalidate:
                application_bytes = sample['read_bytes']+sample['preparation_read_bytes']
                ratio = device['read_bytes']/application_bytes if device['available'] else None
                storage.append(dict(label, application_read_bytes=application_bytes,
                    observed_device_read_bytes=device['read_bytes'], device_to_application_ratio=ratio,
                    within_declared_range=ratio is not None and 0.9 <= ratio <= 1.1,
                    invalidation_ns=sample['invalidation_ns'], invalidation_calls=sample['invalidation_calls']))
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
    result = dict(mode=mode, hits_per_batch=hits, status=status if all(clean.values()) else 'disturbed', **clean,
        criteria=CRITERIA if lifetime is None else dict(CRITERIA, timing_cycles=6, capture_cycles=6),
        results=results, captures=captures, device_observations=devices,
        device_observations_complete=all(d['available'] for d in devices),
        coverage=dict(pairs=pairs, groups=[1, 4], cycles_per_arm=cycles, expert_executions_per_arm=executions,
                      detailed_batches=len(captures)*batches, detailed_experts=len(captures)*executions),
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
    if shared:
        result['gpu_scope'] = 'shared-and-routed-command-intervals'
        result['coverage']['shared_chains_per_arm'] = batches
        result['limitations'] = [text.replace(
            'Resident allocation is present, but other model operators, routing and final token reduction are absent.',
            'One rotating shared-expert chain precedes eight routed experts per batch. Routing, other model operators and final token reduction are absent.')
            for text in result['limitations']]
        result['limitations'].append('GPU duration includes shared and routed work in mixed command groups; it is not pure routed-expert time. Shared command profiles are captured only in the separate trace process.')
    if invalidate:
        exercised = all(s['within_declared_range'] for s in storage) if mode == 'timing' else None
        result.update(cache_preparation=preparation, native_performance_status=status if mode == 'timing' else None,
            storage_criteria=STORAGE_CRITERIA, storage_observations=storage, device_reads_exercised=exercised)
        if mode == 'timing' and not exercised and all(clean.values()): result['status'] = 'unqualified_storage'
        result['limitations'] += [
            'Selected file ranges are invalidated before every batch preparation; invalidation time is included in preparation and excluded from coordinator wall time.',
            'Storage coverage requires every timing arm to observe device read bytes within 90–110% of preparation plus timed application bytes. Aggregate coverage cannot replace a failed arm.',
            'Device counters are systemwide. Matching byte coverage supports exercised device reads, but does not isolate this process or establish physical NAND traffic.']
    return result


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
    preparation = INVALIDATE_FILES if args.invalidate_files else RETAIN_CACHE
    protocol = STORAGE_PROTOCOL if args.invalidate_files else PROTOCOL
    with exp:
        exp.report.update(hits_per_batch=args.hits, criteria=CRITERIA, shader_origin=shader_origin())
        if args.invalidate_files: exp.report['cache_preparation'] = preparation
        shutil.copyfile(protocol, exp.out/'protocol.md')
        exp.report['protocol_sha256'] = sha(protocol)
        exp.report['correctness'] = state_proof(args.state, exp.frozen)
        exp.report['control'] = control_proof(args.control, exp.frozen)
        for key in ('state', 'control'):
            path = getattr(args, key)
            exp.report[key+'_source'] = dict(path=str(path.resolve()), sha256=sha(path/'evidence-files.json'))
        freeze(exp, [args.binary, protocol, exp.out/'protocol.md',
            ROOT/'scripts/qwen/probe_q4_packed.metal', ROOT/'kernels/metal/qwen.metal',
            *[p for d in (args.state, args.control, FIXTURES) for p in d.iterdir() if p.is_file()]])
        exp.report['verified_records'] = verify_records(exp.prepared); exp.persist()
        exp.guard.check_resources(initial=True)
        for mode in ('check', 'timing', 'trace'):
            exp.command([args.binary, FIXTURES, exp.out/(mode+'.json'), 'arrivals', exp.model, exp.prepared,
                         str(args.hits), mode]+(['invalidate'] if args.invalidate_files else []),
                        mode, CRITERIA['process_seconds'], validation=mode == 'check')
            raw = load(exp.out/(mode+'.json'))
            require(raw.get('mode') == mode and raw.get('hits_per_batch') == args.hits and
                    raw.get('cache_preparation', RETAIN_CACHE) == preparation,
                    'Wrong requested mode, hit population or file-cache preparation')
            exp.report[mode] = analyze(raw, exp.frozen, exp.report['verified_records']); exp.persist()
            if not all(exp.report[mode][k] for k in ('clean_memory', 'clean_host')):
                raise ResourceBlocked('Arrival replay has disturbed or unavailable memory/host observations')
        exp.report.update(status=exp.report['timing']['status'], normal_request_latency_qualified=False, production_promoted=False)
    return exp.report


def verify(directory):
    directory = directory.resolve(); seal_digest = sha(directory/'evidence-files.json')
    verify_seal(directory, seal_digest)
    saved, frozen = load(directory/'summary.json'), load(directory/'identity.json')
    preparation = saved.get('cache_preparation', RETAIN_CACHE)
    require(preparation in (RETAIN_CACHE, INVALIDATE_FILES), 'Unknown recorded file-cache preparation')
    protocol = STORAGE_PROTOCOL if preparation == INVALIDATE_FILES else PROTOCOL
    require(saved.get('kind') == 'q4_read_arrivals_diagnostic_v1' and saved.get('hits_per_batch') in (2, 8) and
            saved.get('criteria') == CRITERIA and saved.get('configurations') == [] and saved.get('workload') == [] and
            saved['identity'] == {k: frozen[k] for k in saved['identity']}, 'Changed arrival experiment identity or design')
    sources = dict(frozen, files={p: h for p, h in frozen['files'].items() if Path(p).is_relative_to(frozen['root'])})
    provenance = verify_sources(directory.parent, sources)
    if 'protocol_sha256' in saved:
        require(sha(directory/'protocol.md') == saved['protocol_sha256'] ==
                frozen['files'][str(protocol.resolve())], 'Changed declared arrival protocol')
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
        require(raw.get('mode') == mode and raw.get('hits_per_batch') == saved['hits_per_batch'] and
                raw.get('cache_preparation', RETAIN_CACHE) == preparation,
                'Wrong captured mode, hit count or file-cache preparation')
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
    execute.add_argument('--invalidate-files', action='store_true',
                         help='Invalidate selected file ranges before each batch and require device-read coverage')
    execute.add_argument('--binary', type=Path, default=ROOT/'build/qwen/qwen_q4_check')
    execute.add_argument('--state', type=Path, default=STATE); execute.add_argument('--control', type=Path, default=CONTROL)
    audit = sub.add_parser('verify'); audit.add_argument('directory', type=Path); audit.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'verify': save(args.output, verify(args.directory))
    else:
        for field in ('binary', 'state', 'control'): setattr(args, field, getattr(args, field).resolve())
        raise SystemExit(0 if run(args)['complete'] else 2)
