#!/usr/bin/env python3
"""Measure physical buffer creation and completed-group retirement before reuse."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from benchmark_exact import config_args,inspect_admission
from capture_routes import load
from diagnose_decode_startup import analyze_steps
from qualification_evidence import EvidenceGuard,ResourceBlocked,identity,save,seal,sha
from screen_cache import validate_request
from screen_expert_tail import configs

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'docs/benchmarks/2026-09-13-expert-tail/raw'
CLASSES={'temporary','resident','state','expert','snapshot','workspace'}
FIELDS={'allocations','allocated_bytes','allocation_ns','owner_releases','owner_released_bytes','outside_release_ns'}


def difference(a,b):
    if type(a) is not int or type(b) is not int or not 0<=a<=b:raise ValueError('Missing, negative or reset buffer counter')
    return b-a


def delta(a,b):
    x,y=a.get('buffer_costs'),b.get('buffer_costs')
    for cost in (x,y):
        if not isinstance(cost,dict) or cost.get('kind')!='buffer_costs_v1' or set(cost.get('classes',{}))!=CLASSES:
            raise ValueError('Buffer cost measurement is unavailable or incompatible')
        if any(set(c)!=FIELDS for c in cost['classes'].values()):raise ValueError('Missing per-class counters')
    groups=difference(x['retired_groups'],y['retired_groups'])
    counts={name:{k:difference(x['classes'][name][k],y['classes'][name][k]) for k in FIELDS} for name in sorted(CLASSES)}
    if sum(c['allocations'] for c in counts.values())!=difference(a['allocation_count'],b['allocation_count']):
        raise ValueError('Physical allocation counts do not match the executor')
    if groups!=difference(a['submissions'],b['submissions']) or a.get('live_command_groups')!=0 or b.get('live_command_groups')!=0:
        raise ValueError('Incomplete command retirement at sample boundary')
    return dict(classes=counts,retired_groups=groups,group_retirement_ns=difference(x['group_retirement_ns'],y['group_retirement_ns']))


def analyze(raw):
    if [r.get('name') for r in raw.get('runs',[])]!=['coding_routes_32','append_128']:
        raise ValueError('Requires both complete initial and append captures')
    rows=[]
    for row in raw['runs']:
        check=analyze_steps(row)
        if check['captured_steps']!=32 or check['omitted_steps']:raise ValueError('Requires all 32 decode samples')
        a,b=[row['phases']['decode'][k] for k in ('before','after')]
        costs=delta(a['metal'],b['metal'])
        per_step=[delta(s['before']['metal'],s['after']['metal']) for s in row['decode_diagnostics']['samples']]
        # Sampling itself must not allocate engine buffers or create command groups.
        for name in CLASSES:
            for key in FIELDS:
                if sum(s['classes'][name][key] for s in per_step)!=costs['classes'][name][key]:
                    raise ValueError('Phase counters exceed committed sample coverage')
        if any(sum(s[k] for s in per_step)!=costs[k] for k in ('retired_groups','group_retirement_ns')):
            raise ValueError('Missing retirement sample coverage')
        creation=sum(c['allocation_ns'] for c in costs['classes'].values())/32e6
        outside=sum(c['outside_release_ns'] for c in costs['classes'].values())/32e6
        retirement=costs['group_retirement_ns']/32e6
        temporary=costs['classes']['temporary']['allocation_ns']/32e6
        process=[s[k]['process'] for s in row['decode_diagnostics']['samples'] for k in ('before','after')]
        disturbed=any(p.get('compressed_bytes')!=0 for p in process) or difference(a['process']['decompressions'],b['process']['decompressions'])>0
        rows.append(dict(name=row['name'],decode_ms_per_token=row['decode_wall_ms']/32,
            allocation_ms_per_token=creation,temporary_allocation_ms_per_token=temporary,
            group_retirement_ms_per_token=retirement,outside_owner_release_ms_per_token=outside,
            observed_cpu_interval_ms_per_token=creation+retirement+outside,
            temporary_reuse_upper_cpu_ms_per_token=temporary+retirement+outside,
            allocation_count_per_token=sum(c['allocations'] for c in costs['classes'].values())/32,
            memory_disturbance_observed=disturbed,compressed_peak_bytes=b['process'].get('compressed_peak_bytes'),
            costs=costs,samples=per_step))
    return dict(kind='buffer_cost_analysis_v1',complete=True,requests=rows,
        consider_bounded_reuse=all(r['temporary_reuse_upper_cpu_ms_per_token']>=20 and not r['memory_disturbance_observed'] for r in rows),
        normal_request_latency_qualified=False,production_promoted=False,
        limitations=['Measured CPU intervals can overlap GPU and I/O work and cannot be subtracted from request latency.',
            'Temporary reuse bound includes all group-retirement work; some state, command and residency work would remain.',
            'Owner callbacks nested in the same executor group retirement are excluded from the outside sum.',
            'Retired bytes refer to engine-owner releases; they are not physical footprint or proof of immediate driver deallocation.',
            'Counters are opt-in; disabled or old reports have missing measurements, not zero cost.',
            '32 steps per request, short initial and append workloads; no 2K/4K or sustained acceptance.'])


def run(out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    report=dict(kind='buffer_cost_capture_v1',complete=False,status='running',phase='prepare',time_limit_seconds=360,
        normal_request_latency_qualified=False,production_promoted=False,
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    def remaining():
        left=360-(time.monotonic()-start)
        if left<=0:raise subprocess.TimeoutExpired('buffer-costs',360)
        return left
    try:
        config=configs()[0];work=load(SOURCE/'workload.json');save(out/'workload.json',work)
        control=load(SOURCE/'pair-0-control.json')
        expected={r['name']:r['output_token_ids'] for r in control['runs']}
        model,prepared=ROOT/'.cache/qwen-mixed-reference',ROOT/'.cache/prepared/q4-records-v1'
        frozen=identity(ROOT,[config],model,prepared,out/'workload.json')
        for name in ('workload.json','pair-0-control.json'):
            frozen['files'][str((SOURCE/name).resolve())]=sha(SOURCE/name)
        save(out/'identity.json',frozen);guard=EvidenceGuard(frozen,out)
        report.update(identity={k:frozen[k] for k in ('build','artifact_revision','prepared_manifest_sha256','budget_bytes','device','physical_bytes')},
            configuration=config,workload=work,prior_output_control=dict(source=str(SOURCE/'pair-0-control.json'),sha256=sha(SOURCE/'pair-0-control.json')))
        common=['--model',model,'--artifact','mixed-4_8bit','--prepared',prepared,'--memory-gb','12','--context','8192',*config_args(config)]
        env=dict(os.environ)
        for k in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION'):env.pop(k,None)
        with (ROOT/'.cache/qwen-qualification.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ResourceBlocked('Another workload owns the GPU lease')
            guard.check_resources(initial=True)
            for name in ('normal','measured'):
                report['phase']=name;save(out/'summary.json',report);print(name,flush=True);stem=out/name
                with stem.with_suffix('.log').open('w') as log:
                    p=inspect_admission(ROOT/'build/qwen/bin/zerocool',common,stem,12*1024**3,512,log,guard,remaining)
                    if load(p)['current_admission']['expert_slots']!=1848:raise ResourceBlocked('Fixed expert capacity not admitted')
                    extra=['--decode-diagnostics'] if name=='measured' else []
                    guard.run([ROOT/'build/qwen/bin/zerocool','bench',*common,'--workload-file',out/'workload.json',
                        '--repetitions','1','--temperature','0','--seed','0','--bench-progress',stem.with_suffix('.progress.jsonl'),
                        '--json',stem.with_suffix('.json'),*extra],stdout=log,timeout=min(150,remaining()),env=env)
                raw=load(stem.with_suffix('.json'));validate_request(raw,frozen,config,work,expected,instrumented=name=='measured')
                for row in raw['runs']:
                    if (row['after']['metal']['buffer_costs'] is not None)!=(name=='measured'):
                        raise ValueError('Unexpected buffer instrumentation')
                    if row['after']['metal']['kernels']['profile'] or row['after']['metal']['kernels']['counter_profile']:
                        raise ValueError('Additional GPU profiling')
                report[name]=dict(source=stem.with_suffix('.json').name,sha256=sha(stem.with_suffix('.json')))
            analysis=analyze(load(out/'measured.json'));save(out/'analysis.json',analysis)
            a,b=[load(out/(n+'.json'))['runs'] for n in ('normal','measured')]
            report.update(complete=True,status='captured',phase='finished',analysis=dict(source='analysis.json',sha256=sha(out/'analysis.json')),
                consider_bounded_reuse=analysis['consider_bounded_reuse'],
                measured_to_normal_decode_ratios=[y['decode_wall_ms']/x['decode_wall_ms'] for x,y in zip(a,b)])
    except ResourceBlocked as e:report.update(status='resource_blocked',error=str(e))
    except subprocess.TimeoutExpired:report.update(status='time_budget_exhausted')
    except KeyboardInterrupt:report.update(status='interrupted')
    except Exception as e:report.update(status='failed',error=str(e))
    finally:
        report.update(elapsed_seconds=time.monotonic()-start,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(out/'summary.json',report);seal(out);print(json.dumps({k:report.get(k) for k in ('status','complete','error','consider_bounded_reuse')}),flush=True)
    return 0 if report['complete'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(p.parse_args().output))
