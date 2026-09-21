#!/usr/bin/env python3
"""One explicit 12/18GiB tradeoff; all other allocations and arithmetic fixed."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from benchmark_exact import config_args, inspect_admission
from capacity_experiment import decide, order
from capture_routes import load
from confirm_q8_steady import revalidate as confirmation, screen_files
from qualification_evidence import EvidenceGuard, ResourceBlocked, identity, import_sealed, save, seal, sha, confined
from screen_cache import validate_request
from screen_q8_steady import configs as q8_configs

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'docs/benchmarks/2026-09-11-q8-confirmation/raw'
KIND='q8_memory_budget_screen_v1'
BUDGETS={'control':12*1024**3,'candidate':18*1024**3}
SLOTS={'control':1848,'candidate':4175}
VARIABLE={'limit_bytes','expert_slots','expert_bytes','planned_bytes'}
TIME_LIMIT=600


def configs():
    return [dict(q8_configs()[1],name=name,expert_slots=slots) for name,slots in SLOTS.items()]


def source_files(directory):
    _,files=screen_files(directory/'screen');prior=load(directory/'summary.json')
    files[sha(directory/'summary.json')]=directory/'summary.json'
    for row in prior['measurements']:
        p=confined(directory,row['source'])
        if sha(p)!=row['sha256']:raise ValueError('Changed Q8 confirmation source')
        files[row['sha256']]=p
    return prior,files


def observations(raw,evidence,config,workload,expected):
    name=config['name'];budget=BUDGETS[name]
    if raw.get('gpu_reference_mode')!='off' or raw.get('gpu_references')!=[]:raise ValueError('Changed instrumentation')
    plan=raw['runs'][0]['before']['memory_plan'];fixed={k:v for k,v in plan.items() if k not in VARIABLE}
    if expected.setdefault('fixed',fixed)!=fixed:raise ValueError('Non-expert allocation changed')
    if (plan['limit_bytes']!=budget or plan['expert_slots']!=SLOTS[name] or
        plan['expert_bytes']!=SLOTS[name]*2768896 or plan['panel_tokens']!=512 or
        plan['planned_bytes']!=sum(plan[k] for k in ('resident_bytes','session_bytes','scratch_bytes','panel_scratch_bytes',
            'runtime_control_bytes','pipeline_scratch_bytes','snapshot_bytes','ngram_bytes','reserve_bytes','expert_bytes'))):
        raise ValueError('Requested budget or actual expert allocation differs')
    shared=expected.setdefault('tokens',{})
    arm=expected.setdefault(name,{})
    for row in raw['runs']:
        key=row['name'];outputs=row['output_token_ids']
        if shared.setdefault(key,outputs)!=outputs:raise ValueError('Memory budget changed output tokens')
        dispatches=row['after']['metal']['kernel_dispatches']
        if expected.setdefault('dispatches_'+key,dispatches)!=dispatches:raise ValueError('Memory budget changed kernel dispatches')
        for s in [row['before'],row['after'],*[v[k] for v in row['phases'].values() for k in ('before','after')]]:
            kernels=s['metal']['kernels']
            if kernels.get('profile') or kernels.get('counter_profile') or kernels.get('operator_capture') or s.get('short_append_tokens')!=32:
                raise ValueError('Changed scheduling or profiling')
    return validate_request(raw,evidence,config,workload,arm,memory_budget_bytes=budget)


def revalidate(summary,resolve):
    if (summary.get('kind')!=KIND or summary.get('complete') is not True or summary.get('configurations')!=configs() or
        summary.get('budgets_bytes')!=BUDGETS or summary.get('gpu_reference_mode')!='off' or summary.get('metal_validation') is not False):
        raise ValueError('Incomplete or changed memory-budget experiment')
    prior=resolve(summary['q8_source']['sha256']);proof=confirmation(prior,resolve)
    if proof['confidence_95']['high']>=1:raise ValueError('Requires observed Q8 conversation benefit; no promotion is inferred')
    if summary['identity']!={k:v for k,v in prior['identity'].items() if k!='budget_bytes'} or summary['workload']!=prior['workload']:
        raise ValueError('Build, artifact, machine or workload differs')
    rows=summary['measurements'];used={r['sha256'] for r in prior['measurements']};expected={}
    if [(r['pair'],r['configuration']) for r in rows]!=order(2):raise ValueError('Requires two fresh alternating budget pairs')
    previous=next(r for r in prior['measurements'] if r['configuration']=='candidate')
    observations(resolve(previous['sha256']),summary['identity'],configs()[0],summary['workload'],expected)
    for row in rows:
        h=row['sha256']
        if h in used:raise ValueError('Reused request cannot create a budget pair')
        used.add(h);config=next(c for c in configs() if c['name']==row['configuration'])
        if observations(resolve(h),summary['identity'],config,summary['workload'],expected)!=row['requests']:
            raise ValueError('Summary differs from original budget requests')
    result=decide(rows,2)
    if any(summary.get(k)!=v for k,v in result.items()):raise ValueError('Changed budget decision')
    return result


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    report=dict(kind=KIND,complete=False,status='running',phase='preparation',configurations=configs(),budgets_bytes=BUDGETS,
        gpu_reference_mode='off',metal_validation=False,measurements=[],time_limit_seconds=TIME_LIMIT,
        normal_request_latency_qualified=False,production_promoted=False,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    def remaining():
        v=TIME_LIMIT-(time.monotonic()-started)
        if v<=0:raise subprocess.TimeoutExpired('memory-budget',TIME_LIMIT)
        return v
    try:
        import_sealed(SOURCE,out/'q8',sha(SOURCE/'evidence-files.json'));prior,files=source_files(out/'q8')
        proof=confirmation(prior,lambda h:load(files[h]))
        if proof['confidence_95']['high']>=1:raise ValueError('Missing Q8 speed evidence')
        workload=prior['workload'];save(out/'workload.json',workload)
        model,prepared=ROOT/'.cache/qwen-mixed-reference',ROOT/'.cache/prepared/q4-records-v1'
        frozen=identity(ROOT,configs(),model,prepared,out/'workload.json')
        compact={k:frozen[k] for k in prior['identity'] if k!='budget_bytes'}
        if compact!={k:v for k,v in prior['identity'].items() if k!='budget_bytes'}:raise ValueError('Wrong prerequisite identity')
        frozen.update(protocol=KIND,budget_bytes=None,budgets_bytes=BUDGETS)
        for p in (out/'q8').rglob('*'):
            if p.is_file():frozen['files'][str(p.resolve())]=sha(p)
        save(out/'identity.json',frozen);guard=EvidenceGuard(frozen,out)
        report.update(identity=compact,workload=workload,q8_source=dict(source='q8/summary.json',sha256=sha(out/'q8/summary.json')))
        expected={};old=next(r for r in prior['measurements'] if r['configuration']=='candidate')
        observations(load(files[old['sha256']]),compact,configs()[0],workload,expected)
        env=dict(os.environ)
        for k in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION'):env.pop(k,None)
        with (ROOT/'.cache/qwen-qualification.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ResourceBlocked('Another workload owns the GPU lease')
            guard.check_resources(initial=True)
            for pair,name in order(2):
                config=next(c for c in configs() if c['name']==name);budget=BUDGETS[name];stem=out/f'pair-{pair}-{name}'
                report['phase']=stem.name;save(out/'summary.json',report);print(stem.name,flush=True)
                common=['--model',model,'--artifact','mixed-4_8bit','--prepared',prepared,'--memory-gb',str(budget//1024**3),
                    '--context','8192',*config_args(config)]
                with stem.with_suffix('.admission.log').open('w') as log:
                    p=inspect_admission(ROOT/'build/qwen/bin/zerocool',common,stem,budget,512,log,guard,remaining)
                if load(p)['current_admission']['expert_slots']!=SLOTS[name]:raise ResourceBlocked('Full expert capacity not admitted')
                last=0
                def pulse():
                    nonlocal last
                    if time.monotonic()-last>=15:
                        print(f'{stem.name}: elapsed {time.monotonic()-started:.0f}s; remaining {max(0,TIME_LIMIT-(time.monotonic()-started)):.0f}s',flush=True)
                        last=time.monotonic()
                with stem.with_suffix('.log').open('w') as log:
                    guard.run([ROOT/'build/qwen/bin/zerocool','bench',*common,'--workload-file',out/'workload.json',
                        '--repetitions','1','--temperature','0','--seed','0','--gpu-reference','off',
                        '--bench-progress',stem.with_suffix('.progress.jsonl'),'--json',stem.with_suffix('.json')],
                        stdout=log,timeout=min(150,remaining()),env=env,progress=pulse)
                raw=load(stem.with_suffix('.json'));requests=observations(raw,compact,config,workload,expected)
                h=sha(stem.with_suffix('.json'));files[h]=stem.with_suffix('.json')
                report['measurements'].append(dict(pair=pair,configuration=name,requests=requests,source=stem.with_suffix('.json').name,sha256=h))
                save(out/'summary.json',report)
            report.update(decide(report['measurements'],2),complete=True,phase='finished');revalidate(report,lambda h:load(files[h]))
    except ResourceBlocked as error:report.update(status='resource_blocked',error=str(error),complete=False)
    except subprocess.TimeoutExpired:report.update(status='time_budget_exhausted',error='Partial evidence remains unfinished.',complete=False)
    except KeyboardInterrupt:report.update(status='interrupted',complete=False)
    except Exception as error:report.update(status='failed',error=str(error),complete=False)
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(out/'summary.json',report);seal(out)
        print(json.dumps({k:report[k] for k in ('status','complete','phase','elapsed_seconds')}),flush=True)
    return 0 if report['complete'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    raise SystemExit(run(p.parse_args()))
