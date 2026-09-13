#!/usr/bin/env python3
"""Bounded cache-policy correctness gate and two alternating normal-request pairs."""
import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

from benchmark_exact import configurations, config_args, fixed_sampling
from capture_routes import load
from qualification_evidence import EvidenceGuard, ResourceBlocked, identity, save, sha, seal, verify_seal
from qualify_exact_sessions import check_configuration
from selector_qualification import check_machine

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'docs/benchmarks/2026-09-10-extended-routes/raw'
POLICIES=('clock','slru')
ORDER=((0,'clock'),(0,'slru'),(1,'slru'),(1,'clock'))
CHECKS={'panel_0_rejects_other_artifact_state','panel_0_continued_state','panel_0_fresh_replay',
        'failed_panel_invalidates_state','failed_panel_drains_gpu','failed_panel_refuses_reuse',
        'cancelled_panel_invalidates_partial_state','cancelled_panel_drains_gpu'}


def configs():
    return [dict(configurations()[0],name=p,cache_policy=p) for p in POLICIES]


def correctness_case(policy):
    return dict(context=256,chunk=2,memory_gib=12,layers=48,expert_slots=32,panels=[0],
        diagnostic_stream_trunk=False,artifact='mixed-4_8bit',kernel_policy='reference',cache_policy=policy,
        prefix=[760,369,264,791,13],append=[383,374],continuation=[760,369])


def validate_correctness(reports,evidence):
    for policy,r in zip(POLICIES,reports):
        if (r.get('passed') is not True or r.get('case')!=correctness_case(policy) or
            r.get('full_model') is not True or r.get('layers')!=48 or len(r.get('runs',[]))!=1 or
            len(r.get('checks',[]))!=len(CHECKS) or {c['name'] for c in r['checks']}!=CHECKS or
            any(c.get('passed') is not True for c in r['checks'])):
            raise ValueError('Missing real-model state, cancellation or recovery checks')
        run=r['runs'][0]
        if run['panel']!=0 or len(run['stages'])!=3 or any(len(s['layers'])!=48 or len(s['routes'])!=48 for s in run['stages']):
            raise ValueError('Incomplete all-layer state and route proof')
        for key in ('continued_statistics','after_fresh'):
            s=run[key];check_machine(s,evidence)
            check_configuration(s,dict(configurations()[0],cache_policy=policy,chunk=2,panel=0))
            if (s['memory_plan']['expert_slots']!=32 or s['diagnostic_stream_trunk'] or
                s['expert_cache']['policy']!=policy or s['expert_cache']['evictions']<=0):
                raise ValueError('Correctness gate requires forced eviction with the requested policy')
    if len(reports)!=2 or reports[0]['runs'][0]['stages']!=reports[1]['runs'][0]['stages']:
        raise ValueError('Cache policy changed logits, routes or persistent state')
    return dict(passed=True,all_48_layers=True,exact_logits_routes_state=True,
                continued_equals_fresh=True,cancellation_and_failure_checked=True,expert_slots=32,
                independent_model_reference=False)


def validate_request(raw,evidence,config,workload,expected,*,instrumented=False,gpu_reference='off',capacity_axis=False,memory_budget_bytes=12*1024**3,output_tokens=33):
    if type(output_tokens) is not int or not 2<=output_tokens<=8192:
        raise ValueError('Invalid expected output length')
    if any(task.get('max_tokens')!=output_tokens for task in workload):
        raise ValueError('Workload differs from expected output length')
    if raw.get('gpu_reference_mode','off')!=gpu_reference or (gpu_reference=='off' and raw.get('gpu_references')):
        raise ValueError('Changed GPU reference instrumentation')
    if (raw.get('complete') is not True or raw.get('workloads')!=workload or len(workload) not in (1,2) or len(raw.get('runs',[]))!=len(workload) or
        raw.get('model_revision')!=evidence['artifact_revision'] or not fixed_sampling(raw.get('sampling'))):
        raise ValueError('Missing complete identical normal workload')
    history=[];observations=[]
    plan=raw['runs'][0]['before']['memory_plan']
    if capacity_axis:
        fixed={k:v for k,v in plan.items() if k not in ('expert_slots','expert_bytes','planned_bytes')}
        if expected.setdefault('fixed_memory_plan',fixed)!=fixed: raise ValueError('Changed fixed allocation')
        if (plan['expert_slots']!=config['expert_slots'] or plan['expert_bytes']!=config['expert_slots']*2768896 or
            plan['planned_bytes']!=sum(plan[k] for k in ('resident_bytes','session_bytes','scratch_bytes',
                'panel_scratch_bytes','ngram_bytes','reserve_bytes','expert_bytes','pipeline_scratch_bytes','snapshot_bytes','runtime_control_bytes'))):
            raise ValueError('Invalid explicit expert capacity accounting')
        if expected.setdefault('memory_plan_'+config['name'],plan)!=plan: raise ValueError('Capacity arm changed allocation')
    elif expected.setdefault('memory_plan',plan)!=plan: raise ValueError('Unequal admitted memory allocation')
    for i,(row,task) in enumerate(zip(raw['runs'],workload)):
        prompt=history+task['tokens'] if i else task['tokens']
        reused=len(history)-1 if i else 0
        outputs=row.get('output_token_ids')
        if (row.get('name')!=task['name'] or row.get('repetition')!=0 or row.get('profiling_enabled') is not instrumented or
            row.get('gpu_reference_mode','off')!=gpu_reference or
            row.get('runtime_cache_state')!=('retained' if i else 'empty_at_process_start') or
            row.get('prompt_tokens')!=len(prompt) or row.get('reused_tokens')!=reused or
            row.get('pending_tokens_ingested')!=int(i>0) or row.get('prefill_tokens')!=len(prompt)-reused or
            row.get('finish_reason')!='length' or row.get('output_tokens')!=output_tokens or
            not isinstance(outputs,list) or len(outputs)!=output_tokens or
            any(type(t) is not int or not 0<=t<248320 for t in outputs)):
            raise ValueError('Changed output length, instrumentation, workload or actual session reuse')
        if not instrumented and 'decode_diagnostics' in row: raise ValueError('Decode diagnostics cannot qualify normal timing')
        if set(row.get('phases',{}))!={'ingest','decode'}: raise ValueError('Missing phase measurements')
        if expected.setdefault(task['name'],outputs)!=outputs: raise ValueError('Policies produced different tokens')
        for state in [row['before'],row['after'],*[s[k] for s in row['phases'].values() for k in ('before','after')]]:
            check_machine(state,evidence,budget_bytes=memory_budget_bytes);check_configuration(state,config)
            if (state['memory_plan']!=plan or state['diagnostic_stream_trunk'] or
                state['phase_memory']['pressure_resizes']!=0 or not state['completion_pipeline'] or
                state['expert_cache']['policy']!=config['cache_policy']):
                raise ValueError('Changed schedule, memory or cache policy during request')
        timings={k:row.get(k) for k in ('request_ms','time_to_first_token_ms','decode_wall_ms')}
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in timings.values()):
            raise ValueError('Missing positive finite request timing')
        latency=row.get('token_latency_ms')
        if not isinstance(latency,list) or len(latency)!=output_tokens-1 or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in latency):
            raise ValueError('Missing complete decode timings')
        phases=[]
        for phase,states in row['phases'].items():
            a,b=states['before'],states['after']
            phases.append(dict(phase=phase,expert_hits=b['expert_cache']['hits']-a['expert_cache']['hits'],
                expert_misses=b['expert_cache']['misses']-a['expert_cache']['misses'],
                expert_read_bytes=b['expert_cache']['application_read_bytes']-a['expert_cache']['application_read_bytes']))
        observations.append(dict(name=task['name'],**timings,tokens_per_second=(output_tokens-1)*1000/row['decode_wall_ms'],
            output_token_ids=outputs,reused_tokens=reused,phases=phases,
            memory_before=row['before']['process'],memory_after=row['after']['process']))
        history=prompt+outputs
    return observations


def decide(measurements):
    if [(m['pair'],m['policy']) for m in measurements]!=list(ORDER): raise ValueError('Requires two alternating pairs')
    if any(len(m['requests'])!=2 for m in measurements): raise ValueError('Missing measured request')
    pairs={(m['pair'],m['policy']):m for m in measurements};ratios=[]
    for request in range(2):
        for metric in ('request_ms','time_to_first_token_ms','decode_wall_ms'):
            values=[]
            for pair in range(2):
                a,b=[pairs[pair,p]['requests'][request][metric] for p in POLICIES]
                if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in (a,b)): raise ValueError('Invalid timings')
                values.append(b/a)
            ratios.append(dict(request=request,metric=metric,ratios=values,median=statistics.median(values)))
    totals=[sum(r['request_ms'] for r in pairs[i,'slru']['requests'])/
            sum(r['request_ms'] for r in pairs[i,'clock']['requests']) for i in range(2)]
    promising=(all(r<1 for r in totals) and statistics.median(totals)<=.99 and all(r['median']<=1.03 for r in ratios))
    return dict(status='promising' if promising else 'improvement_not_demonstrated',
        conversation_ratios=totals,median_conversation_ratio=statistics.median(totals),ratios=ratios,
        confidence_95=None,advance_to_five_pairs=promising,normal_request_latency_qualified=False,production_promoted=False)


def run(args):
    if not 0<args.time_limit<=900: raise ValueError('Deadline must be 1..900 seconds')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    report=dict(kind='cache_policy_short_screen_v1',status='running',complete=False,phase='preparation',
        time_limit_seconds=args.time_limit,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),measurements=[],
        criterion=dict(pairs=2,order=ORDER,median_conversation_ratio_max=.99,each_conversation_ratio_below=1,other_median_ratio_max=1.03),
        instrumentation=dict(timing_profile=False,timing_metal_validation=False,correctness_metal_validation=True),
        normal_request_latency_qualified=False,production_promoted=False)
    def remaining():
        seconds=args.time_limit-(time.monotonic()-started)
        if seconds<=0: raise subprocess.TimeoutExpired('cache-screen',args.time_limit)
        return seconds
    def phase(name):
        report['phase']=name;save(output/'summary.json',report);print(name,flush=True)
    try:
        verify_seal(SOURCE,sha(SOURCE/'evidence-files.json'))
        source=load(SOURCE/'summary.json')
        if source.get('status')!='captured' or source.get('complete') is not True or not source['result']['target_steps_met']:
            raise ValueError('Requires completed extended locality capture')
        workload=load(SOURCE/'workload.json');workload[0].update(name='coding_routes_32',max_tokens=33)
        save(output/'workload.json',workload)
        for p in POLICIES: save(output/f'{p}-case.json',correctness_case(p))
        model,prepared=ROOT/'.cache/qwen-mixed-reference',ROOT/'.cache/prepared/q4-records-v1'
        evidence=identity(ROOT,configs(),model,prepared,output/'workload.json')
        if source['artifact_revision']!=evidence['artifact_revision']: raise ValueError('Locality source uses another artifact')
        evidence['files'].update({str(p.resolve()):sha(p) for p in [SOURCE/'summary.json',SOURCE/'workload.json',SOURCE/'evidence-files.json',*[output/f'{p}-case.json' for p in POLICIES]]})
        save(output/'identity.json',evidence);guard=EvidenceGuard(evidence,output)
        report.update(build=evidence['build'],artifact_revision=evidence['artifact_revision'],budget_bytes=evidence['budget_bytes'],configurations=configs())
        env=dict(os.environ)
        for key in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION'): env.pop(key,None)
        with (ROOT/'.cache/qwen-qualification.lock').open('a') as lock:
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: raise ResourceBlocked('Another workload owns the GPU lease')
            guard.check_resources(initial=True)
            correct=[]
            for p in POLICIES:
                phase('correctness_'+p)
                with (output/f'{p}-correctness.log').open('w') as log:
                    guard.run([ROOT/'build/qwen/qwen_panel_check',model,prepared,output/f'{p}-case.json',output/f'{p}-correctness.json'],
                        stdout=log,timeout=min(180,remaining()),env=dict(env,MTL_DEBUG_LAYER='1',MTL_SHADER_VALIDATION='1'))
                correct.append(load(output/f'{p}-correctness.json'))
            report['correctness']=validate_correctness(correct,evidence)
            expected={}
            for pair,p in ORDER:
                config=next(c for c in configs() if c['name']==p)
                stem=output/f'pair-{pair}-{p}';phase(f'timing_pair_{pair}_{p}')
                common=['--model',model,'--artifact','mixed-4_8bit','--prepared',prepared,'--memory-gb','12','--context','8192',*config_args(config)]
                with stem.with_suffix('.log').open('w') as log:
                    guard.run([ROOT/'build/qwen/bin/freellm','inspect',*common,'--json',stem.with_suffix('.admission.json')],stdout=log,timeout=remaining(),env=env)
                    admission=load(stem.with_suffix('.admission.json'))
                    if admission['current_admission']['limit_bytes']!=evidence['budget_bytes'] or admission['current_admission']['panel_tokens']!=512 or admission['cache_policy']!=p:
                        raise ResourceBlocked('Requested cache policy budget/panel is not admitted')
                    guard.run([ROOT/'build/qwen/bin/freellm','bench',*common,'--workload-file',output/'workload.json','--repetitions','1',
                        '--temperature','0','--seed','0','--json',stem.with_suffix('.json')],stdout=log,timeout=min(150,remaining()),env=env)
                raw=load(stem.with_suffix('.json'))
                requests=validate_request(raw,evidence,config,workload,expected)
                report['measurements'].append(dict(pair=pair,policy=p,requests=requests,source=stem.with_suffix('.json').name,sha256=sha(stem.with_suffix('.json'))))
                save(output/'summary.json',report)
            report.update(decide(report['measurements']),complete=True,phase='finished',memory_plan=expected['memory_plan'])
    except ResourceBlocked as error: report.update(status='resource_blocked',error=str(error))
    except subprocess.TimeoutExpired: report.update(status='time_budget_exhausted',error='Deadline reached; incomplete pairs cannot establish improvement.')
    except KeyboardInterrupt: report.update(status='interrupted',error='Interrupted; no performance conclusion.')
    except Exception as error: report.update(status='failed',error=str(error))
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(output/'summary.json',report);report['evidence_seal']=seal(output)
        print(json.dumps(report,indent=2),flush=True)
    return 0 if report['complete'] else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--time-limit',type=int,default=600)
    raise SystemExit(run(parser.parse_args()))
