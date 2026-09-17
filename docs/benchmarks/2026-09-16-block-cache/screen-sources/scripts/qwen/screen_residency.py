#!/usr/bin/env python3
"""Bounded off/core residency screen using normal initial and retained-append requests."""
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

from benchmark_exact import configurations, config_args, inspect_admission
from capture_routes import load
from qualification_evidence import EvidenceGuard, ResourceBlocked, identity, save, seal, sha, verify_seal
from screen_cache import validate_request

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'docs/benchmarks/2026-09-10-slru-screen/raw'
DIAGNOSTIC=ROOT/'docs/benchmarks/2026-09-10-decode-startup/raw'
SCREEN=ROOT/'docs/benchmarks/2026-09-10-residency-screen/raw'
ORDER=((0,'off'),(0,'core'),(1,'core'),(1,'off'))


def pair_order(count):
    if type(count) is not int or count not in (2,5):raise ValueError('Requires two screening or five confirmation pairs')
    return tuple((i,mode) for i in range(count) for mode in (('off','core') if i%2==0 else ('core','off')))


def paired_log_interval(ratios):
    if len(ratios)!=5 or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in ratios):
        raise ValueError('Requires exactly five positive finite paired ratios')
    logs=[math.log(v) for v in ratios];mean=statistics.mean(logs)
    # Two-sided Student-t 0.975 quantile, four degrees of freedom.
    half_width=2.7764451051977987*statistics.stdev(logs)/math.sqrt(5)
    return dict(geometric_mean=math.exp(mean),low=math.exp(mean-half_width),high=math.exp(mean+half_width),pairs=5,
        method='paired log-ratio Student-t interval, two-sided 95%, df=4',
        assumptions='Independent approximately normal pair log-ratios; temporal host effects may violate this. Marginal intervals are not simultaneous coverage.')


def configs():
    return [dict(configurations()[0],name=mode,residency=mode,cache_policy='clock') for mode in ('off','core')]


def check_residency(raw,mode):
    for row in raw['runs']:
        states=[row['before'],row['after'],*[v[k] for v in row['phases'].values() for k in ('before','after')]]
        for state in states:
            reg=state['metal']['residency'];classes=reg['bytes_by_class'];plan=state['memory_plan']
            if (reg['mode']!=mode or reg['set_overhead_bytes']>64*1024**2 or reg['pending_retirements'] or
                any(type(v) is not int or v<0 for v in classes.values()) or
                reg['registered_bytes']!=sum(classes.values()) or set(classes)-{'resident','state'}):
                raise ValueError('Residency enrollment or retirement differs from the core-only experiment')
            if mode=='off':
                if reg['registered_bytes'] or reg['allocations']:raise ValueError('Off mode enrolled allocations')
            elif (classes.get('resident')!=plan['resident_bytes'] or
                  not plan['runtime_control_bytes']<=classes.get('state',0)<=plan['session_bytes']+plan['runtime_control_bytes']):
                raise ValueError('Core matrices/state were not enrolled as planned')
            kernels=state['metal']['kernels']
            if kernels.get('profile') or kernels.get('counter_profile'):raise ValueError('GPU profiling cannot qualify timing')


def decide(measurements,pair_count=2):
    if [(m['pair'],m['residency']) for m in measurements]!=list(pair_order(pair_count)):raise ValueError('Requires the declared alternating off/core pairs')
    if any(len(m['requests'])!=2 for m in measurements):raise ValueError('Missing initial or append request')
    pairs={(m['pair'],m['residency']):m for m in measurements};ratios=[]
    for request in range(2):
        for metric in ('request_ms','time_to_first_token_ms','decode_wall_ms'):
            values=[]
            for pair in range(pair_count):
                a,b=[pairs[pair,mode]['requests'][request][metric] for mode in ('off','core')]
                if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in (a,b)):raise ValueError('Invalid request timing')
                values.append(b/a)
            row=dict(request=request,metric=metric,ratios=values,median=statistics.median(values))
            if pair_count==5:row['confidence_95']=paired_log_interval(values)
            ratios.append(row)
    totals=[sum(r['request_ms'] for r in pairs[i,'core']['requests'])/
            sum(r['request_ms'] for r in pairs[i,'off']['requests']) for i in range(pair_count)]
    if pair_count==5:
        interval=paired_log_interval(totals)
        promising=(interval['geometric_mean']<=.99 and interval['high']<1 and
                   all(r['confidence_95']['high']<=1.03 for r in ratios))
        return dict(status='promising' if promising else 'improvement_not_demonstrated',conversation_ratios=totals,
            median_conversation_ratio=statistics.median(totals),ratios=ratios,confidence_95=interval,
            advance_to_five_pairs=False,advance_to_long_validation=promising,
            normal_request_latency_qualified=False,production_promoted=False)
    promising=all(v<1 for v in totals) and statistics.median(totals)<=.99 and all(r['median']<=1.03 for r in ratios)
    return dict(status='promising' if promising else 'improvement_not_demonstrated',conversation_ratios=totals,
        median_conversation_ratio=statistics.median(totals),ratios=ratios,confidence_95=None,
        advance_to_five_pairs=promising,normal_request_latency_qualified=False,production_promoted=False)


def validate_screen(evidence,workload,expected):
    summary=load(SCREEN/'summary.json')
    if (summary.get('complete') is not True or summary.get('status')!='promising' or summary.get('build')!=evidence['build'] or
        summary.get('artifact_revision')!=evidence['artifact_revision'] or summary.get('configurations')!=configs() or
        load(SCREEN/'workload.json')!=workload):raise ValueError('Requires completed matching short residency screen')
    for m in summary['measurements']:
        path=SCREEN/m['source']
        if sha(path)!=m['sha256']:raise ValueError('Changed short-screen request')
        raw=load(path);config=next(c for c in configs() if c['name']==m['residency'])
        if validate_request(raw,evidence,config,workload,expected)!=m['requests']:raise ValueError('Changed short-screen observations')
        check_residency(raw,m['residency'])
    result=decide(summary['measurements'])
    if not result['advance_to_five_pairs'] or any(summary.get(k)!=v for k,v in result.items()):raise ValueError('Short screen no longer passes')


def run(args):
    pair_count=getattr(args,'pairs',2);order=pair_order(pair_count)
    if not 0<args.time_limit<=(1200 if pair_count==5 else 600):raise ValueError('Deadline exceeds the declared mode limit')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    report=dict(kind='residency_paired_confirmation_v1' if pair_count==5 else 'residency_short_screen_v1',status='running',complete=False,phase='preparation',measurements=[],
        time_limit_seconds=args.time_limit,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        criterion=(dict(pairs=5,order=order,conversation_geometric_mean_max=.99,conversation_upper_95_below=1,other_upper_95_max=1.03,
                        interval_method='paired log-ratio Student-t, two-sided 95%, df=4',prior_pairs_pooled=False,
                        inference_unit='one off/core pair of complete conversations',early_success_stopping=False) if pair_count==5 else
                   dict(pairs=2,order=order,median_conversation_ratio_max=.99,each_conversation_ratio_below=1,other_median_ratio_max=1.03)),
        instrumentation=dict(decode_diagnostics=False,profile=False,metal_validation=False),
        normal_request_latency_qualified=False,production_promoted=False)
    def remaining():
        value=args.time_limit-(time.monotonic()-started)
        if value<=0:raise subprocess.TimeoutExpired('residency-screen',args.time_limit)
        return value
    def phase(name):
        report['phase']=name;save(out/'summary.json',report);print(name,flush=True)
    try:
        for directory in (SOURCE,DIAGNOSTIC,*((SCREEN,) if pair_count==5 else ())):
            verify_seal(directory,sha(directory/'evidence-files.json'))
            if load(directory/'summary.json').get('complete') is not True:raise ValueError('Incomplete prerequisite evidence')
        workload=load(SOURCE/'workload.json');save(out/'workload.json',workload)
        model,prepared=ROOT/'.cache/qwen-mixed-reference',ROOT/'.cache/prepared/q4-records-v1'
        evidence=identity(ROOT,configs(),model,prepared,out/'workload.json')
        if load(DIAGNOSTIC/'summary.json')['build']!=evidence['build']:raise ValueError('Native build changed since the diagnostic')
        for directory,names in ((SOURCE,('summary.json','workload.json','evidence-files.json','pair-0-clock.json')),
                                (DIAGNOSTIC,('summary.json','evidence-files.json'))):
            if load(directory/'summary.json')['artifact_revision']!=evidence['artifact_revision']:raise ValueError('Changed source artifact')
            for name in names:evidence['files'][str((directory/name).resolve())]=sha(directory/name)
        expected={r['name']:r['output_token_ids'] for r in load(SOURCE/'pair-0-clock.json')['runs']}
        if pair_count==5:
            validate_screen(evidence,workload,expected)
            # Bind all prerequisite raw evidence, including its admission and logs.
            for path in SCREEN.iterdir():
                if path.is_file():evidence['files'][str(path.resolve())]=sha(path)
        save(out/'identity.json',evidence);guard=EvidenceGuard(evidence,out)
        report.update(build=evidence['build'],artifact_revision=evidence['artifact_revision'],budget_bytes=evidence['budget_bytes'],configurations=configs())
        env=dict(os.environ)
        for key in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION'):env.pop(key,None)
        with (ROOT/'.cache/qwen-qualification.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ResourceBlocked('Another workload owns the GPU lease')
            guard.check_resources(initial=True)
            for pair,mode in order:
                config=next(c for c in configs() if c['name']==mode);stem=out/f'pair-{pair}-{mode}'
                phase(f'timing_pair_{pair}_{mode}')
                common=['--model',model,'--artifact','mixed-4_8bit','--prepared',prepared,'--memory-gb','12','--context','8192',*config_args(config)]
                with stem.with_suffix('.log').open('w') as log:
                    admission_path=inspect_admission(ROOT/'build/qwen/bin/freellm',common,stem,evidence['budget_bytes'],512,log,guard,remaining)
                    admission=load(admission_path)
                    if (admission['current_admission'].get('limit_bytes')!=evidence['budget_bytes'] or
                        admission['current_admission'].get('panel_tokens')!=512 or admission.get('cache_policy')!='clock'):
                        raise ResourceBlocked('Fixed CLOCK budget/panel not admitted')
                    guard.run([ROOT/'build/qwen/bin/freellm','bench',*common,'--workload-file',out/'workload.json','--repetitions','1',
                        '--temperature','0','--seed','0','--json',stem.with_suffix('.json')],stdout=log,timeout=min(150,remaining()),env=env)
                raw=load(stem.with_suffix('.json'));requests=validate_request(raw,evidence,config,workload,expected)
                check_residency(raw,mode)
                report['measurements'].append(dict(pair=pair,residency=mode,requests=requests,
                    source=stem.with_suffix('.json').name,sha256=sha(stem.with_suffix('.json'))))
                save(out/'summary.json',report)
            report.update(decide(report['measurements'],pair_count),complete=True,phase='finished',memory_plan=expected['memory_plan'])
    except ResourceBlocked as error:report.update(status='resource_blocked',error=str(error))
    except subprocess.TimeoutExpired:report.update(status='time_budget_exhausted',error='Deadline reached; incomplete pairs remain inconclusive.')
    except KeyboardInterrupt:report.update(status='interrupted',error='Interrupted; no performance conclusion.')
    except Exception as error:report.update(status='failed',error=str(error))
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(out/'summary.json',report);report['evidence_seal']=seal(out)
        print(json.dumps(report,indent=2),flush=True)
    return 0 if report['complete'] else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--pairs',type=int,choices=[2,5],default=2)
    parser.add_argument('--time-limit',type=int)
    args=parser.parse_args()
    if args.time_limit is None:args.time_limit=900 if args.pairs==5 else 480
    raise SystemExit(run(args))
