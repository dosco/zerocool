#!/usr/bin/env python3
"""Screen bounded single-token temporary reuse on two fresh normal request pairs."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

from benchmark_exact import config_args, inspect_admission
from capacity_experiment import decide as timing_decision, order
from capture_routes import load
from qualification_evidence import EvidenceGuard, ResourceBlocked, identity, save, seal, sha
from qualify_exact_sessions import check_configuration
from screen_cache import CHECKS, correctness_case, validate_request
from screen_route_selection import configs as router_configs
from selector_qualification import check_machine

ROOT=Path(__file__).resolve().parents[2]
KIND='decode_scratch_screen_v1'
SCRATCH_CHECKS={'failed_decode_scratch_invalidates_state','failed_decode_scratch_drains_gpu',
                'cancelled_decode_scratch_invalidates_state','cancelled_decode_scratch_drains_gpu'}


def configs():
    return [dict(router_configs()[1],name=name,expert_tail='wait',decode_scratch=mode)
            for name,mode in (('control','none'),('candidate','reuse'))]


def state_cases():
    return [dict(correctness_case('clock'),kernel_policy='candidate',q8_decode_rows=2,
                 route_selection='simd',expert_tail='wait',decode_scratch=c['decode_scratch']) for c in configs()]


def observations(raw,evidence,config,workload,expected):
    rows=validate_request(raw,evidence,config,workload,expected)
    reuse=config['decode_scratch']=='reuse'
    for row,summary in zip(raw['runs'],rows):
        for state in (row['before'],row['after'],*[v[k] for v in row['phases'].values() for k in ('before','after')]):
            m=state['metal']
            if (state.get('expert_tail_pending') is not False or m['live_command_groups']!=0 or
                m['peak_command_groups']>2 or m.get('active_scratch_slot')!=-1):
                raise ValueError('Unreleased or unbounded GPU work')
            pools=m['scratch_pools']
            if (len(pools)!=2 or pools[1]['allocated_bytes']!=0 or
                pools[0]['allocated_bytes']>(128*1024**2 if reuse else 0)):
                raise ValueError('Unbounded decode workspace')
        phase=row['phases']['decode'];a,b=phase['before'],phase['after']
        passes=b['decode_scratch_passes']-a['decode_scratch_passes']
        allocations=b['metal']['allocation_count']-a['metal']['allocation_count']
        reuses=b['metal']['scratch_reuses']-a['metal']['scratch_reuses']
        if passes!=(32 if reuse else 0):raise ValueError('Requested scratch schedule did not execute')
        if reuse and (reuses<32*1600 or not 0<allocations<32*1600):raise ValueError('Physical allocation reduction absent')
        if not reuse and (reuses or allocations!=32*3238):raise ValueError('Changed allocation control')
        # Both fixed workloads finish multi-token ingestion before decode.
        if row['phases']['ingest']['after']['metal']['scratch_pools'][0]['allocated_bytes']!=0:
            raise ValueError('Decode workspace retained through multi-token ingestion')
        counts={k:b['metal']['kernel_dispatches'].get(k,0)-a['metal']['kernel_dispatches'].get(k,0)
                for k in b['metal']['kernel_dispatches']}
        if expected.setdefault('dispatches_'+row['name'],counts)!=counts:raise ValueError('Arithmetic dispatch counts changed')
        summary['allocations_per_token']=allocations/32
        summary['scratch_reuses_per_token']=reuses/32
        summary['decode_pool_bytes']=b['metal']['scratch_pools'][0]['allocated_bytes']
        summary['decode_memory_disturbance']=any(s['process'].get('compressed_bytes')!=0 for s in (a,b)) or (
            type(a['process'].get('decompressions')) is not int or a['process']['decompressions']!=b['process'].get('decompressions'))
    return rows


def check_state_workspace(state,case,stage):
    if stage not in ('continued_statistics','after_fresh'):raise ValueError('Unknown state checkpoint')
    if state['metal'].get('active_scratch_slot')!=-1:raise ValueError('Scratch scope still active')
    pools=state['metal']['scratch_pools']
    sizes=[p.get('allocated_bytes') for p in pools]
    if (len(sizes)!=2 or any(type(n) is not int or n<0 for n in sizes) or
        sizes[1]!=0 or sizes[0]>128*1024**2):raise ValueError('Unbounded state workspace')
    # feed() uses panel_tokens or chunk_tokens; its last chunk can have one
    # token even when the entire fresh history has many tokens. That forward
    # legitimately retains decode scratch after earlier multi-token releases.
    limit=state['memory_plan']['panel_tokens'] or state['chunk_tokens']
    if type(limit) is not int or limit<=0:raise ValueError('Invalid state input limit')
    total=sum(len(case[k]) for k in ('prefix','append','continuation'))
    if not total:raise ValueError('Missing state replay history')
    last=1 if stage=='continued_statistics' else (total-1)%limit+1
    retained=case['decode_scratch']=='reuse' and last==1
    if bool(sizes[0])!=retained:raise ValueError('Workspace does not match the last replay chunk')


def correctness(reports,evidence):
    if len(reports)!=2:raise ValueError('Missing correctness arm')
    for raw,case in zip(reports,state_cases()):
        checks=CHECKS|SCRATCH_CHECKS if case['decode_scratch']=='reuse' else CHECKS
        if (raw.get('case')!=case or raw.get('passed') is not True or raw.get('layers')!=48 or
            raw.get('full_model') is not True or len(raw.get('runs',[]))!=1 or
            {c['name'] for c in raw.get('checks',[])}!=checks or len(raw['checks'])!=len(checks) or
            any(c['passed'] is not True for c in raw['checks'])):raise ValueError('Incomplete state/failure checks')
        stages=raw['runs'][0].get('stages',[])
        if len(stages)!=3 or any(len(s.get('layers',[]))!=48 or len(s.get('routes',[]))!=48 for s in stages):
            raise ValueError('Incomplete all-layer logits/routes/state snapshots')
        for key in ('continued_statistics','after_fresh'):
            state=raw['runs'][0][key];check_machine(state,evidence);check_configuration(state,case)
            if (state['memory_plan']['expert_slots']!=32 or state['expert_cache']['evictions']<=0 or
                state.get('expert_tail_pending') is not False or state['metal']['live_command_groups']!=0 or
                state.get('diagnostic_stream_trunk') is not False):
                raise ValueError('Missing forced eviction or drain')
            if (state['decode_scratch_passes']>0)!=(case['decode_scratch']=='reuse'):raise ValueError('Missing scratch execution')
            if state['memory_plan']['panel_tokens']!=case['panels'][0]:raise ValueError('Changed state replay panel')
            check_state_workspace(state,case,key)
    if reports[0]['runs'][0]['stages']!=reports[1]['runs'][0]['stages']:
        raise ValueError('Scratch reuse changed logits, routes or persistent state')
    return dict(exact_logits_routes_state=True,all_48_layers=True,continued_equals_fresh=True,
                forced_eviction=True,decode_scratch_cancellation_failure=True,independent_model_reference=False)


def decide(rows):
    result=timing_decision(rows,2)
    indexed={(r['pair'],r['configuration']):r['requests'] for r in rows}
    savings=[[ (indexed[p,'control'][i]['decode_wall_ms']-indexed[p,'candidate'][i]['decode_wall_ms'])/32
               for p in range(2)] for i in range(2)]
    # Decode is the priority. Keep the established complete-request/TTFT guards.
    clean=all(r.get('decode_memory_disturbance') is False for row in rows for r in row['requests'])
    timing_pass=result['advance_to_confirmation'] and all(v>0 for phase in savings for v in phase)
    promising=timing_pass and clean
    result.update(status='promising' if promising else ('memory_disturbed' if timing_pass else 'improvement_not_demonstrated'),
        clean_decode_memory=clean,
        advance_to_confirmation=promising,decode_savings_ms_per_token=savings,
        median_initial_saving_ms=statistics.median(savings[0]),target_ms_per_token=200)
    return result


def revalidate(summary,resolve):
    if (summary.get('kind')!=KIND or summary.get('complete') is not True or summary.get('configurations')!=configs() or
        summary.get('gpu_reference_mode')!='off' or summary.get('metal_validation') is not False):
        raise ValueError('Changed or incomplete scratch protocol')
    prior=resolve(summary['prior_output_control']['sha256'])
    rows=summary['measurements'];expected={r['name']:r['output_token_ids'] for r in prior['runs']};seen=set()
    if [(r['pair'],r['configuration']) for r in rows]!=order(2):raise ValueError('Missing alternating pairs')
    for row in rows:
        if row['sha256'] in seen:raise ValueError('Repeated raw report')
        seen.add(row['sha256']);c=next(c for c in configs() if c['name']==row['configuration'])
        if observations(resolve(row['sha256']),summary['identity'],c,summary['workload'],expected)!=row['requests']:
            raise ValueError('Changed request summary')
    decision=decide(rows)
    if any(summary.get(k)!=v for k,v in decision.items()):raise ValueError('Changed timing decision')
    if decision['advance_to_confirmation']:
        proof=correctness([resolve(r['sha256']) for r in summary['correctness_sources']],summary['identity'])
        if summary.get('correctness')!=proof:raise ValueError('Changed correctness summary')
    elif summary.get('correctness_sources') or summary.get('correctness') is not None:
        raise ValueError('Unplanned long validation for a failed screen')
    return decision


def run(out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();deadline=start+600
    report=dict(kind=KIND,complete=False,status='running',phase='prepare',configurations=configs(),measurements=[],
        correctness_sources=[],gpu_reference_mode='off',metal_validation=False,
        normal_request_latency_qualified=False,production_promoted=False,
        limits_seconds=dict(requests=600,correctness=360),started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    def remaining():
        left=deadline-time.monotonic()
        if left<=0:raise subprocess.TimeoutExpired('decode-scratch-screen',0)
        return left
    def source(p):return dict(source=str(p.relative_to(out)),sha256=sha(p))
    try:
        work=load(ROOT/'docs/benchmarks/2026-09-12-route-five-pairs/raw/workload.json');save(out/'workload.json',work)
        prior=out/'prior-output-control.json'
        shutil.copyfile(ROOT/'docs/benchmarks/2026-09-13-buffer-costs/raw/normal.json',prior)
        report['prior_output_control']=source(prior)
        for name,case in zip(('control','candidate'),state_cases()):save(out/(name+'.case.json'),case)
        model,prepared=ROOT/'.cache/qwen-mixed-reference',ROOT/'.cache/prepared/q4-records-v1'
        frozen=identity(ROOT,configs(),model,prepared,out/'workload.json')
        frozen['files'].update({str(p.resolve()):sha(p) for p in out.glob('*.case.json')})
        frozen['files'][str(prior)]=sha(prior)
        save(out/'identity.json',frozen);guard=EvidenceGuard(frozen,out)
        report.update(workload=work,identity={k:frozen[k] for k in ('build','artifact_revision','prepared_manifest_sha256','budget_bytes','device','physical_bytes')})
        env=dict(os.environ)
        for k in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION'):env.pop(k,None)
        def execute(command,stem,limit,validation=False):
            report['phase']=stem.name;save(out/'summary.json',report);print(stem.name,flush=True)
            with stem.with_suffix('.log').open('w') as log:
                guard.run(command,stdout=log,timeout=min(limit,remaining()),
                    env=dict(env,MTL_DEBUG_LAYER='1',MTL_SHADER_VALIDATION='1') if validation else env)
        with (ROOT/'.cache/qwen-qualification.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ResourceBlocked('Another workload owns the GPU lease')
            guard.check_resources(initial=True);expected={r['name']:r['output_token_ids'] for r in load(prior)['runs']}
            for pair,name in order(2):
                c=next(c for c in configs() if c['name']==name);stem=out/f'pair-{pair}-{name}'
                common=['--model',model,'--artifact','mixed-4_8bit','--prepared',prepared,'--memory-gb','12','--context','8192',*config_args(c)]
                with stem.with_suffix('.admission.log').open('w') as log:
                    p=inspect_admission(ROOT/'build/qwen/bin/freellm',common,stem,12*1024**3,512,log,guard,remaining)
                if load(p)['current_admission']['expert_slots']!=1848:raise ResourceBlocked('Expected expert capacity not admitted')
                execute([ROOT/'build/qwen/bin/freellm','bench',*common,'--workload-file',out/'workload.json','--repetitions','1',
                    '--temperature','0','--seed','0','--gpu-reference','off','--bench-progress',stem.with_suffix('.progress.jsonl'),
                    '--json',stem.with_suffix('.json')],stem,150)
                report['measurements'].append(dict(pair=pair,configuration=name,
                    requests=observations(load(stem.with_suffix('.json')),frozen,c,work,expected),**source(stem.with_suffix('.json'))))
                save(out/'summary.json',report)
            report.update(decide(report['measurements']))
            if report['advance_to_confirmation']:
                deadline=time.monotonic()+360;raws=[]
                for c in configs():
                    stem=out/('state-'+c['name'])
                    execute([ROOT/'build/qwen/qwen_panel_check',model,prepared,out/(c['name']+'.case.json'),stem.with_suffix('.json')],stem,180,True)
                    raws.append(load(stem.with_suffix('.json')));report['correctness_sources'].append(source(stem.with_suffix('.json')))
                report['correctness']=correctness(raws,frozen)
            report.update(complete=True,phase='finished')
            originals={r['sha256']:out/r['source'] for r in report['measurements']+report['correctness_sources']}
            originals[sha(prior)]=prior
            revalidate(report,lambda h:load(originals[h]))
    except ResourceBlocked as e:report.update(status='resource_blocked',error=str(e))
    except subprocess.TimeoutExpired:report.update(status='time_budget_exhausted')
    except KeyboardInterrupt:report.update(status='interrupted')
    except Exception as e:report.update(status='failed',error=str(e))
    finally:
        if report['status'] in ('failed','interrupted','resource_blocked','time_budget_exhausted'):
            report.update(complete=False,advance_to_confirmation=False,candidate_for_later_qualification=False)
        report.update(elapsed_seconds=time.monotonic()-start,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(out/'summary.json',report);seal(out);print(json.dumps({k:report.get(k) for k in ('status','complete','error','median_initial_saving_ms')}),flush=True)
    return 0 if report['complete'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(p.parse_args().output))
