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

from benchmark_exact import configurations, config_args
from capture_routes import load
from qualification_evidence import EvidenceGuard, ResourceBlocked, identity, save, seal, sha, verify_seal
from screen_cache import validate_request

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'docs/benchmarks/2026-09-10-slru-screen/raw'
DIAGNOSTIC=ROOT/'docs/benchmarks/2026-09-10-decode-startup/raw'
ORDER=((0,'off'),(0,'core'),(1,'core'),(1,'off'))


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


def decide(measurements):
    if [(m['pair'],m['residency']) for m in measurements]!=list(ORDER):raise ValueError('Requires two alternating off/core pairs')
    if any(len(m['requests'])!=2 for m in measurements):raise ValueError('Missing initial or append request')
    pairs={(m['pair'],m['residency']):m for m in measurements};ratios=[]
    for request in range(2):
        for metric in ('request_ms','time_to_first_token_ms','decode_wall_ms'):
            values=[]
            for pair in range(2):
                a,b=[pairs[pair,mode]['requests'][request][metric] for mode in ('off','core')]
                if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in (a,b)):raise ValueError('Invalid request timing')
                values.append(b/a)
            ratios.append(dict(request=request,metric=metric,ratios=values,median=statistics.median(values)))
    totals=[sum(r['request_ms'] for r in pairs[i,'core']['requests'])/
            sum(r['request_ms'] for r in pairs[i,'off']['requests']) for i in range(2)]
    promising=all(v<1 for v in totals) and statistics.median(totals)<=.99 and all(r['median']<=1.03 for r in ratios)
    return dict(status='promising' if promising else 'improvement_not_demonstrated',conversation_ratios=totals,
        median_conversation_ratio=statistics.median(totals),ratios=ratios,confidence_95=None,
        advance_to_five_pairs=promising,normal_request_latency_qualified=False,production_promoted=False)


def run(args):
    if not 0<args.time_limit<=600:raise ValueError('Deadline must be 1..600 seconds')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    report=dict(kind='residency_short_screen_v1',status='running',complete=False,phase='preparation',measurements=[],
        time_limit_seconds=args.time_limit,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        criterion=dict(pairs=2,order=ORDER,median_conversation_ratio_max=.99,each_conversation_ratio_below=1,other_median_ratio_max=1.03),
        instrumentation=dict(decode_diagnostics=False,profile=False,metal_validation=False),
        normal_request_latency_qualified=False,production_promoted=False)
    def remaining():
        value=args.time_limit-(time.monotonic()-started)
        if value<=0:raise subprocess.TimeoutExpired('residency-screen',args.time_limit)
        return value
    def phase(name):
        report['phase']=name;save(out/'summary.json',report);print(name,flush=True)
    try:
        for directory in (SOURCE,DIAGNOSTIC):
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
        save(out/'identity.json',evidence);guard=EvidenceGuard(evidence,out)
        report.update(build=evidence['build'],artifact_revision=evidence['artifact_revision'],budget_bytes=evidence['budget_bytes'],configurations=configs())
        expected={r['name']:r['output_token_ids'] for r in load(SOURCE/'pair-0-clock.json')['runs']}
        env=dict(os.environ)
        for key in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION'):env.pop(key,None)
        with (ROOT/'.cache/qwen-qualification.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ResourceBlocked('Another workload owns the GPU lease')
            guard.check_resources(initial=True)
            for pair,mode in ORDER:
                config=next(c for c in configs() if c['name']==mode);stem=out/f'pair-{pair}-{mode}'
                phase(f'timing_pair_{pair}_{mode}')
                common=['--model',model,'--artifact','mixed-4_8bit','--prepared',prepared,'--memory-gb','12','--context','8192',*config_args(config)]
                with stem.with_suffix('.log').open('w') as log:
                    guard.run([ROOT/'build/qwen/bin/freellm','inspect',*common,'--json',stem.with_suffix('.admission.json')],stdout=log,timeout=remaining(),env=env)
                    admission=load(stem.with_suffix('.admission.json'))
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
            report.update(decide(report['measurements']),complete=True,phase='finished',memory_plan=expected['memory_plan'])
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
    parser.add_argument('--time-limit',type=int,default=480)
    raise SystemExit(run(parser.parse_args()))
