#!/usr/bin/env python3
"""Attribute validation memory with fixed native work; never qualify inference."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from cache_residency import require
from capture_routes import load
from combined_q4 import freeze, shader_origin, state_proof
from native_q4_replay import (BUDGET, COUNTERS, FIXED_KERNELS, fixture_records,
                             metal_state, observations_clean, uint)
import q4_read_arrivals as arrivals
from q4_scratch_lifetime import lifetime_proof
from q4_shared_arrivals import (REFERENCE, reference_files, reference_identity,
                               reference_inputs, shared_proof)
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
from screen_q4_packed import FIXTURES, verify_records
from stage200 import Experiment
from verify_stage200 import verify_sources

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/'docs/benchmarks/2026-09-15-q4-validation-memory/protocol.md'
ORDER = ['off', 'on', 'on', 'off']
PHASES = ['startup', 'pipelines_ready', 'resident_ready', 'fixtures_ready',
          'shared_reference_ready', 'routed_reference_ready', 'expert_pool_ready',
          'post_warmup'] + ['pass_complete']*48 + ['scratch_released']
DISPATCHES = dict(q8_gate_up=48, q8_mv_packed_r2_w4=48, plain_mm=48,
                  q4_gate_up=384, q4_mm=384, scatter_experts=384)
CRITERIA = dict(order=ORDER, process_seconds=60, stage_seconds=180, scope='forward',
    warm_passes=48, observed_passes=48, variant='reference', group=1,
    profiling=False, counters=False, budget_bytes=BUDGET,
    compression_is_observation=True, hard_host_swap_budget_guards=True)
FLAGS = dict(performance_comparison=False, original_lifetime_stage_qualified=False,
             normal_request_latency_qualified=False, production_promoted=False)
LIMITATIONS = [
    'Compression is recorded as a dirty resource observation in this attribution-only experiment; it never becomes a clean-memory or performance pass.',
    'Four fresh processes in off/on/on/off order provide two observations per condition, not a statistical causal estimate or identification of compressed pages.',
    'Boundary observations and cumulative peaks cannot locate transient events within a phase. Device allocation bytes are not driver physical memory.',
    'The repeated shared/routed fixtures retain 20.25MiB, only 27.2% of measured full-token scratch; they do not execute a complete model token.',
    'Previously blocked or inconclusive lifetime reports retain their original decisions.']


def check_phase(phase, previous, first):
    require(phase.get('phase') in PHASES and uint(phase.get('at_ns')) and
            phase['at_ns'] > (previous['at_ns'] if previous else 0),
            'Missing or reordered memory phase')
    memory, host = phase.get('memory', {}), phase.get('host', {})
    for key in ('physical_footprint_bytes', 'physical_footprint_peak_bytes',
                'compressed_bytes', 'compressed_peak_bytes', 'decompressions',
                'system_swap_used_bytes'):
        require(uint(memory.get(key)), 'Missing process memory gauge: '+key)
    require(0 < memory['physical_footprint_bytes'] <= memory['physical_footprint_peak_bytes'] <= BUDGET and
            memory['compressed_bytes'] <= memory['compressed_peak_bytes'],
            'Process footprint exceeds hard bound or invalid compression peak')
    require(memory['system_swap_used_bytes'] == first['memory']['system_swap_used_bytes'] and
            host.get('thermal_state') == 0 and host.get('low_power_mode') is False and
            bool(host.get('power_source')) and host['power_source'] == first['host'].get('power_source'),
            'Hard host or swap guard failed')
    if previous:
        require(all(memory[key] >= previous['memory'][key] for key in
                    ('physical_footprint_peak_bytes', 'compressed_peak_bytes', 'decompressions')),
                'Cumulative process counters moved backwards')
    state, counters = phase.get('metal'), phase.get('memory_counters')
    if phase['phase'] == 'startup':
        require(state is None and counters is None, 'Startup must precede Metal creation')
        return
    require(isinstance(state, dict) and isinstance(counters, dict) and
            state.get('live_command_groups') == counters.get('live_command_groups') == 0 and
            state.get('peak_command_groups', 0) <= 2 and
            counters.get('encoded_buffer_references') == 0 and
            0 <= state.get('live_buffer_bytes', -1) <= state.get('peak_buffer_bytes', -1) <= BUDGET and
            counters.get('live_buffer_bytes') == state['live_buffer_bytes'] and
            counters.get('scratch_bytes') == sum(p['allocated_bytes'] for p in state['scratch_pools']) and
            counters.get('active_scratch_slot') == state['active_scratch_slot'] and
            uint(counters.get('device_allocated_bytes')) and
            state['kernels'].get('profile') is False and state['kernels'].get('counter_profile') is False,
            'Memory phase has outstanding users, changed instrumentation or inconsistent allocation gauges')


def analyze(raw, validation, frozen, records, manifest, gates):
    require(validation in ('off', 'on') and raw.get('kind') == 'native_q4_scratch_memory_v1' and
            raw.get('mode') == 'memory-'+validation and raw.get('complete') is True and
            raw.get('validation') is (validation == 'on') and
            raw.get('profiling') is False and raw.get('counters') is False and
            all(raw.get(k) is False for k in FLAGS) and raw.get('exact') is True and
            raw.get('untouched_destinations_checked') is True,
            'Incomplete memory process or changed instrumentation/scope')
    require(raw.get('hits_per_batch') == 2 and raw.get('cycles') == 6 and
            raw.get('cache_slots') == raw.get('io_workers') == 8 and
            raw.get('max_live_groups') == 2 and raw.get('budget_bytes') == BUDGET and
            raw.get('cache_preparation') == arrivals.INVALIDATE_FILES and
            raw.get('prepared_manifest_sha256') == frozen['prepared_manifest_sha256'] and
            raw.get('output_sha256') == arrivals.OUTPUT_SHA and
            raw['fixture_manifest'].get('artifact_revision') == frozen['artifact_revision'] and
            fixture_records(raw['fixture_manifest']) == records,
            'Changed actual work, artifact, prepared records or routed bytes')
    resident = raw.get('resident_bytes')
    require(uint(resident) and resident == raw.get('expected_resident_bytes') and resident > 0,
            'Invalid resident allocation')
    proof = shared_proof(raw, manifest, gates)
    require([(p['pair'], p['group']) for p in raw['pairs']] == [(0, 1)] and
            len(raw['pairs'][0]['arms']) == 1, 'Memory process must have one reference/group1 execution')
    arm = raw['pairs'][0]['arms'][0]
    require(arm['variant'] == 'reference', 'Candidate timing is outside memory attribution')
    before, after, sample = arm['metal_before'], arm['metal_after'], arm['sample']
    for state in (raw['initial_metal'], before, after, raw['final_metal']):
        metal_state(state, frozen['build'], resident)
        require(state['kernels'] == dict(FIXED_KERNELS, q4_decode='reference'),
                'Changed memory replay arithmetic or profiling')
    counts = dict(expert_executions=384, shared_chains=48, ready_hits=96,
        new_misses=288, loading_joins=0, read_bytes=796262400, expert_load_calls=288,
        preparation_read_bytes=265420800, preparation_load_calls=96,
        invalidation_calls=384, allocation_count=0, scratch_reuses=1296, submissions=384)
    require(all(sample.get(k) == v for k, v in counts.items()) and sample.get('batches') == [] and
            sample.get('kernel_dispatches') == DISPATCHES,
            'Missing fixed work, explicit reads, or zero-allocation warm coverage')
    for key in COUNTERS:
        require(uint(sample.get(key)) and after[key]-before[key] == sample[key],
                'Sample counters differ from native arm boundaries')
    delta = {k: after['kernel_dispatches'].get(k, 0)-before['kernel_dispatches'].get(k, 0)
             for k in set(after['kernel_dispatches']) | set(before['kernel_dispatches'])}
    require({k: v for k, v in delta.items() if v} == DISPATCHES, 'Changed native dispatch population')
    lifetime = lifetime_proof(raw, 'forward')
    phases = raw.get('memory_phases', [])
    require([p.get('phase') for p in phases] == PHASES and
            [p.get('pass_index') for p in phases if p.get('phase') == 'pass_complete'] == list(range(48)),
            'Missing, duplicated or reordered lifecycle coverage')
    signature = []
    for index, phase in enumerate(phases):
        check_phase(phase, phases[index-1] if index else None, phases[0])
        state = phase['metal']
        if state is not None:
            require(state['build_fingerprint'] == frozen['build'], 'Changed native phase source identity')
            signature.append(dict(phase=phase['phase'], pass_index=phase.get('pass_index'),
                live_buffer_bytes=state['live_buffer_bytes'], allocation_count=state['allocation_count'],
                scratch_reuses=state['scratch_reuses'], active_scratch_slot=state['active_scratch_slot'],
                scratch=[{k: p[k] for k in ('capacity_bytes', 'allocated_bytes', 'unused_retained_bytes',
                                           'allocation_count', 'reuses')} for p in state['scratch_pools']],
                kernels=state['kernels'], kernel_dispatches=state['kernel_dispatches']))
        if phase['phase'] == 'post_warmup':
            require(state == before, 'Warmup phase differs from observed arm boundary')
        if phase['phase'] == 'pass_complete':
            n = phase['pass_index']+1; pool = state['scratch_pools'][0]; start_pool = before['scratch_pools'][0]
            require(pool['allocated_bytes'] == 21233664 and pool['unused_retained_bytes'] == (48-n)*442368 and
                    pool['reuses']-start_pool['reuses'] == n*27 and
                    pool['allocation_count'] == start_pool['allocation_count'] and
                    state['scratch_reuses']-before['scratch_reuses'] == n*27 and
                    state['active_scratch_slot'] == (0 if n < 48 else -1) and
                    state['allocation_count'] == before['allocation_count'] and
                    state['live_buffer_bytes'] == before['live_buffer_bytes'],
                    'Per-pass scratch ownership or reuse changed')
    require(phases[-2]['metal'] == after and phases[-1]['metal'] == raw['final_metal'] and
            after['live_buffer_bytes']-raw['final_metal']['live_buffer_bytes'] == 21233664 and
            all(p['allocated_bytes'] == p['capacity_bytes'] == 0 for p in raw['final_metal']['scratch_pools']),
            'Missing final observed boundary or complete scratch release')
    observations = [dict(memory_before=p['memory'], memory_after=p['memory'],
                         host_before=p['host'], host_after=p['host']) for p in phases]
    clean = observations_clean(observations)
    clean['clean_memory'] &= all(p['memory']['compressed_peak_bytes'] == 0 for p in phases)
    first_dirty = next((dict(index=i, phase=p['phase'], pass_index=p.get('pass_index'))
        for i,p in enumerate(phases) if p['memory']['compressed_peak_bytes'] or
        p['memory']['compressed_bytes'] or p['memory']['decompressions'] != phases[0]['memory']['decompressions']), None)
    device = arrivals.disk_observation(arm)
    ratio = device['read_bytes']/1061683200 if device['available'] else None
    require(ratio is not None and 0.9 <= ratio <= 1.1, 'Actual storage coverage missing or disturbed')
    return dict(status='memory_clean' if all(clean.values()) else 'memory_dirty', **clean,
        validation=validation, first_dirty_phase=first_dirty, hard_guards_passed=True,
        output_sha256=raw['output_sha256'], shared_reference=proof, lifetime=lifetime,
        native_work_signature=signature, observed_device_to_application_ratio=ratio,
        peak_physical_bytes=max(p['memory']['physical_footprint_peak_bytes'] for p in phases),
        peak_compressed_bytes=max(p['memory']['compressed_peak_bytes'] for p in phases),
        phases=[dict(phase=p['phase'], pass_index=p.get('pass_index'), memory=p['memory'],
                     memory_counters=p['memory_counters']) for p in phases],
        limitations=LIMITATIONS, **FLAGS)


def decision(results):
    require([r['validation'] for r in results] == ORDER and
            all(r['hard_guards_passed'] for r in results), 'Incomplete attribution sequence')
    require(all(r['native_work_signature'] == results[0]['native_work_signature'] and
                r['output_sha256'] == results[0]['output_sha256'] and
                r['shared_reference'] == results[0]['shared_reference'] for r in results),
            'Validation conditions changed native work or arithmetic')
    off_clean = all(results[i]['clean_memory'] and results[i]['clean_host'] for i in (0,3))
    on_dirty = all(not results[i]['clean_memory'] for i in (1,2))
    intervals = [results[i]['first_dirty_phase'] for i in (1,2)]
    associated = off_clean and on_dirty and intervals[0] == intervals[1]
    return dict(status='validation_memory_association' if associated else 'memory_captured',
        validation_phase_association=associated,
        first_dirty_validation_phase=intervals[0] if associated else None,
        release_processes_clean=off_clean, timing_observation_admissible=off_clean,
        compressed_pages_identified=False, statistical_causal_estimate=False, **FLAGS)


def prepare(exp, args):
    exp.report.update(criteria=CRITERIA, limitations=LIMITATIONS, shader_origin=shader_origin(), **FLAGS)
    shutil.copyfile(PROTOCOL, exp.out/'protocol.md'); exp.report['protocol_sha256'] = sha(PROTOCOL)
    manifest,gates = reference_inputs(args.shared_fixtures)
    exp.report['shared_reference_source'] = dict(path=str(args.shared_fixtures),
        manifest_sha256=sha(args.shared_fixtures/'manifest.json'), manifest=manifest)
    for key,checker in (('state',state_proof),('control',arrivals.control_proof)):
        path=getattr(args,key); exp.report[key] = checker(path,exp.frozen)
        exp.report[key+'_source'] = dict(path=str(path),seal_sha256=sha(path/'evidence-files.json'))
    freeze(exp,[args.binary,PROTOCOL,exp.out/'protocol.md',
        *reference_files(args.shared_fixtures),
        *[p for d in (args.state,args.control,FIXTURES) for p in d.iterdir() if p.is_file()]])
    reference_identity(manifest, exp.frozen)
    exp.report['verified_records']=verify_records(exp.prepared);exp.persist()
    return manifest,gates


def run(args):
    exp=Experiment(args.output,'q4_validation_memory_v1',[],[],180)
    with exp:
        manifest,gates=prepare(exp,args);exp.guard.check_resources(initial=True)
        for index,validation in enumerate(ORDER):
            stem=f'{index:02d}-{validation}'
            try:
                exp.command([args.binary,FIXTURES,exp.out/(stem+'.json'),'arrivals',exp.model,exp.prepared,
                    '2','memory-'+validation,'invalidate',args.shared_fixtures,'scratch-forward'],
                    stem,60,validation=validation=='on')
            except subprocess.CalledProcessError as error:
                log=(exp.out/(stem+'.log')).read_text(errors='replace')
                if 'memory attribution guard at ' in log:
                    raise ResourceBlocked(log.strip().splitlines()[-1]) from error
                raise
            raw=load(exp.out/(stem+'.json'))
            require(raw['shared_prelude']['reference_manifest_sha256']==sha(args.shared_fixtures/'manifest.json'),
                    'Native CPU fixture hash differs')
            result=analyze(raw,validation,exp.frozen,exp.report['verified_records'],manifest,gates)
            exp.report['measurements'].append(dict(source=stem+'.json',sha256=sha(exp.out/(stem+'.json')),
                                                  result=result));exp.persist()
        exp.report.update(decision([m['result'] for m in exp.report['measurements']]))
    return exp.report


def verify(directory):
    digest=sha(directory/'evidence-files.json');verify_seal(directory,digest)
    saved,frozen=load(directory/'summary.json'),load(directory/'identity.json')
    require(saved.get('kind')=='q4_validation_memory_v1' and saved.get('criteria')==CRITERIA and
            saved.get('limitations')==LIMITATIONS and saved.get('configurations')==saved.get('workload')==[] and
            saved['identity']=={k:frozen[k] for k in saved['identity']} and
            all(saved.get(k) is False for k in FLAGS), 'Changed attribution scope or identity')
    provenance=verify_sources(directory.parent,dict(frozen,files={
        p:h for p,h in frozen['files'].items() if Path(p).is_relative_to(frozen['root'])}))
    require(sha(directory/'protocol.md')==saved['protocol_sha256']==frozen['files'][str(PROTOCOL)],
            'Changed declared protocol')
    require(saved['shader_origin']==shader_origin(), 'Changed arithmetic origin')
    for key,checker in (('state',state_proof),('control',arrivals.control_proof)):
        source=saved[key+'_source'];path=Path(source['path'])
        require(sha(path/'evidence-files.json')==source['seal_sha256'] and
                checker(path,frozen)==saved[key], 'Changed prior correctness/control evidence')
    source=saved['shared_reference_source'];path=Path(source['path'])
    manifest,gates=reference_inputs(path);reference_identity(manifest,frozen)
    require(manifest==source['manifest'] and sha(path/'manifest.json')==source['manifest_sha256'],
            'Changed shared CPU source')
    for file in reference_files(path):
        require(sha(file)==frozen['files'].get(str(file.resolve())), 'Changed CPU fixture')
    measurements=saved['measurements']; require(len(measurements)<=4, 'Extra condition observations')
    for i,item in enumerate(measurements):
        require(item['source']==f'{i:02d}-{ORDER[i]}.json', 'Changed condition order')
        file=directory/item['source'];require(sha(file)==item['sha256'], 'Changed raw phase report')
        raw=load(file)
        require(raw['shared_prelude']['reference_manifest_sha256']==source['manifest_sha256'],
                'Changed native CPU fixture hash')
        require(item['result']==analyze(raw,ORDER[i],frozen,saved['verified_records'],manifest,gates),
                'Changed phase analysis')
        sidecar=file.with_suffix('.progress.jsonl')
        require([json.loads(line) for line in sidecar.read_text().splitlines()]==raw['memory_phases'],
                'Flushed lifecycle evidence differs from final report')
    if saved['complete']:
        answer=decision([m['result'] for m in measurements])
        require(all(saved.get(k)==v for k,v in answer.items()), 'Changed attribution decision')
    else:
        require(saved['status'] in ('failed','resource_blocked','interrupted','time_budget_exhausted'),
                'Incomplete attribution lacks a terminal disposition')
    return dict(kind='q4_validation_memory_audit_v1',complete=True,audit_passed=True,
        recorded_complete=saved['complete'],status=saved['status'],source_seal_sha256=digest,
        source_provenance=provenance,measurements=measurements,**FLAGS)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    execute=sub.add_parser('run');execute.add_argument('--output',type=Path,required=True)
    execute.add_argument('--binary',type=Path,default=ROOT/'build/qwen/qwen_q4_check')
    execute.add_argument('--shared-fixtures',type=Path,default=REFERENCE)
    execute.add_argument('--state',type=Path,default=arrivals.STATE)
    execute.add_argument('--control',type=Path,default=arrivals.CONTROL)
    audit=sub.add_parser('verify');audit.add_argument('directory',type=Path)
    audit.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='verify':save(args.output,verify(args.directory.resolve()))
    else:
        for field in ('binary','shared_fixtures','state','control'):
            setattr(args,field,getattr(args,field).resolve())
        raise SystemExit(0 if run(args)['complete'] else 2)
