#!/usr/bin/env python3
"""Bounded observation of prefill-to-decode stalls; no promotion or latency qualification."""
import argparse
import datetime
import fcntl
import math
import os
from pathlib import Path
import subprocess
import time

from benchmark_exact import configurations,config_args
from capture_routes import load
from qualification_evidence import EvidenceGuard,ResourceBlocked,identity,save,seal,sha,verify_seal
from screen_cache import validate_request

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'docs/benchmarks/2026-09-10-slru-screen/raw'
ORDER=((0,'off'),(0,'core'),(1,'core'),(1,'off'))
COUNTERS={'process':('decompressions','page_faults','pageins','system_swap_used_bytes'),
          'metal':('cpu_encode_ns','cpu_gpu_wait_ns','gpu_command_ns','submissions'),
          'expert_cache':('hits','misses','application_read_bytes'),
          'expert_dependencies':('coordinator_wait_ns','read_service_sum_ns','read_queue_sum_ns')}


def delta(a,b):
    # Missing counters and resets/wraps stay unknown; never turn them into zero work.
    if type(a) is not int or type(b) is not int or a<0 or b<a: return None
    return b-a


def differences(a,b):
    return {group:{key:delta(a.get(group,{}).get(key),b.get(group,{}).get(key)) for key in keys}
            for group,keys in COUNTERS.items()}


def analyze_steps(row):
    d=row.get('decode_diagnostics',{});latency=row.get('token_latency_ms',[]);samples=d.get('samples',[])
    if (row.get('profiling_enabled') is not True or d.get('kind')!='decode_step_diagnostics_v1' or
        d.get('max_steps')!=32 or d.get('total_decode_steps')!=len(latency) or
        d.get('captured_steps')!=len(samples) or len(samples)!=min(32,len(latency)) or
        d.get('omitted_steps')!=len(latency)-len(samples)):
        raise ValueError('Missing or inconsistent bounded decode coverage')
    rows=[];previous_end=0
    for i,s in enumerate(samples):
        if (s.get('step')!=i or s.get('offset')!=row['prompt_tokens']+i or s.get('input_token_id')!=row['output_token_ids'][i] or
            s.get('forward_ms')!=latency[i] or type(s.get('begin_ns')) is not int or type(s.get('end_ns')) is not int or
            not previous_end<=s['begin_ns']<s['end_ns'] or not math.isfinite(latency[i]) or latency[i]<=0):
            raise ValueError('Decode observation does not match the committed token interval')
        previous_end=s['end_ns']
        for key in ('before','after'):
            if type(s[key].get('sample_ns')) is not int or s[key]['sample_ns']<0: raise ValueError('Missing observation cost')
            if s[key]['metal'].get('live_command_groups')!=0: raise ValueError('Counters require the existing completed forward boundary')
        rows.append(dict(step=i,offset=s['offset'],forward_ms=latency[i],**differences(s['before'],s['after']),
            observation_ns=s['before']['sample_ns']+s['after']['sample_ns'],
            compressed_bytes_before=s['before']['process'].get('compressed_bytes'),
            compressed_bytes_after=s['after']['process'].get('compressed_bytes')))
    windows=[]
    for name,start,end in [('first_4',0,4),('remaining_captured',4,len(rows))]:
        part=rows[start:end]
        if not part:continue
        def total(group,key):
            values=[r[group][key] for r in part]
            return sum(values) if all(v is not None for v in values) else None
        windows.append(dict(name=name,steps=len(part),forward_ms=sum(r['forward_ms'] for r in part),
            **{g:{k:total(g,k) for k in keys} for g,keys in COUNTERS.items()}))
    return dict(samples=rows,windows=windows,captured_steps=len(rows),omitted_steps=d['omitted_steps'],
        observation_ns=sum(r['observation_ns'] for r in rows),
        counter_scope='CPU waits, GPU durations and concurrent reads overlap; their sums are not a critical-path decomposition.')


def historical():
    summary=load(SOURCE/'summary.json');rows=[]
    if not summary.get('complete'): raise ValueError('Unfinished source screen')
    for m in summary['measurements']:
        p=SOURCE/m['source']
        if sha(p)!=m['sha256']:raise ValueError('Changed historical measurement')
        r=load(p)['runs'][0];a,b=[r['phases']['decode'][k] for k in ('before','after')]
        rows.append(dict(pair=m['pair'],policy=m['policy'],source=str(p),sha256=sha(p),decode_wall_ms=r['decode_wall_ms'],
            first_4_ms=sum(r['token_latency_ms'][:4]),remaining_28_ms=sum(r['token_latency_ms'][4:]),
            decode_process_delta={k:delta(a['process'].get(k),b['process'].get(k)) for k in COUNTERS['process']},
            decode_metal_delta={k:delta(a['metal'].get(k),b['metal'].get(k)) for k in COUNTERS['metal']}))
    return dict(kind='decode_startup_historical_analysis_v1',complete=True,measurements=rows,
        limitations=['The first-four split was selected after inspecting this evidence.',
                     'Per-request process counters cannot locate decompression at individual tokens.',
                     'Overlapping GPU, CPU-wait and read-duration sums cannot be added into request time.'])


def run(args):
    if not 0<args.time_limit<=600:raise ValueError('Deadline must be 1..600 seconds')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    report=dict(kind='decode_startup_diagnostic_v1',status='running',complete=False,phase='preparation',measurements=[],
        time_limit_seconds=args.time_limit,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        design=dict(order=ORDER,cache_policy='clock',only_configuration_change='residency',steps_per_request=32,
                    hypothesis='Requesting core residency reduces early decode stalls and decompression when they occur.'),
        instrumentation=dict(decode_diagnostics=True,metal_validation=False,profile=False),
        normal_request_latency_qualified=False,production_promoted=False)
    def remaining():
        value=args.time_limit-(time.monotonic()-started)
        if value<=0:raise subprocess.TimeoutExpired('decode-startup',args.time_limit)
        return value
    def phase(name):
        report['phase']=name;save(out/'summary.json',report);print(name,flush=True)
    try:
        verify_seal(SOURCE,sha(SOURCE/'evidence-files.json'))
        save(out/'historical.json',historical())
        workload=load(SOURCE/'workload.json')[:1];save(out/'workload.json',workload)
        configs=[dict(configurations()[0],name=p,residency=p,cache_policy='clock') for p in ('off','core')]
        model,prepared=ROOT/'.cache/qwen-mixed-reference',ROOT/'.cache/prepared/q4-records-v1'
        evidence=identity(ROOT,configs,model,prepared,out/'workload.json')
        for name in ('summary.json','workload.json','evidence-files.json','pair-0-clock.json'):
            evidence['files'][str((SOURCE/name).resolve())]=sha(SOURCE/name)
        save(out/'identity.json',evidence);guard=EvidenceGuard(evidence,out)
        report.update(build=evidence['build'],artifact_revision=evidence['artifact_revision'],budget_bytes=evidence['budget_bytes'],configurations=configs)
        expected={workload[0]['name']:load(SOURCE/'pair-0-clock.json')['runs'][0]['output_token_ids']}
        env=dict(os.environ)
        for k in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION'):env.pop(k,None)
        with (ROOT/'.cache/qwen-qualification.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ResourceBlocked('Another workload owns the GPU lease')
            guard.check_resources(initial=True)
            for pair,mode in ORDER:
                config=next(c for c in configs if c['name']==mode);stem=out/f'pair-{pair}-{mode}'
                phase(f'pair_{pair}_{mode}')
                common=['--model',model,'--artifact','mixed-4_8bit','--prepared',prepared,'--memory-gb','12','--context','8192',*config_args(config)]
                with stem.with_suffix('.log').open('w') as log:
                    guard.run([ROOT/'build/qwen/bin/zerocool','inspect',*common,'--json',stem.with_suffix('.admission.json')],stdout=log,timeout=remaining(),env=env)
                    admission=load(stem.with_suffix('.admission.json'))
                    if admission['current_admission'].get('limit_bytes')!=evidence['budget_bytes'] or admission['current_admission'].get('panel_tokens')!=512:
                        raise ResourceBlocked('Fixed budget or panel not admitted')
                    guard.run([ROOT/'build/qwen/bin/zerocool','bench',*common,'--workload-file',out/'workload.json','--repetitions','1',
                        '--temperature','0','--seed','0','--decode-diagnostics','--json',stem.with_suffix('.json')],stdout=log,timeout=min(120,remaining()),env=env)
                raw=load(stem.with_suffix('.json'))
                observed=validate_request(raw,evidence,config,workload,expected,instrumented=True)[0]
                row=raw['runs'][0];steps=analyze_steps(row)
                if steps['captured_steps']!=32 or steps['omitted_steps']: raise ValueError('Incomplete diagnostic capture')
                for state in (row['before'],row['after']):
                    reg=state['metal']['residency'];kernels=state['metal']['kernels']
                    if reg['mode']!=mode or reg['set_overhead_bytes']>64*1024**2 or reg['bytes_by_class'].get('expert',0):
                        raise ValueError('Changed residency enrollment or metadata budget')
                    if mode=='core' and reg['bytes_by_class'].get('resident')!=state['memory_plan']['resident_bytes']:
                        raise ValueError('Core matrices were not enrolled')
                    if kernels.get('profile') or kernels.get('counter_profile'):raise ValueError('Additional GPU profiling enabled')
                report['measurements'].append(dict(pair=pair,residency=mode,request=observed,diagnostics=steps,
                    residency_before=row['before']['metal']['residency'],residency_after=row['after']['metal']['residency'],
                    source=stem.with_suffix('.json').name,sha256=sha(stem.with_suffix('.json'))))
                save(out/'summary.json',report)
            report.update(status='captured',complete=True,phase='finished',memory_plan=expected['memory_plan'])
    except ResourceBlocked as error:report.update(status='resource_blocked',error=str(error))
    except subprocess.TimeoutExpired:report.update(status='time_budget_exhausted',error='Deadline reached; incomplete requests remain incomplete.')
    except KeyboardInterrupt:report.update(status='interrupted',error='Interrupted; no conclusion.')
    except Exception as error:report.update(status='failed',error=str(error))
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(out/'summary.json',report);report['evidence_seal']=seal(out)
        print(__import__('json').dumps(report,indent=2),flush=True)
    return 0 if report['complete'] else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--time-limit',type=int,default=360)
    raise SystemExit(run(parser.parse_args()))
