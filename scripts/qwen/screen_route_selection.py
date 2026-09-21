#!/usr/bin/env python3
"""Screen one exact router selector before spending time on full-state checks."""
import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import time

from benchmark_exact import config_args, inspect_admission
from capacity_experiment import decide, order
from capture_routes import load
from qualification_evidence import EvidenceGuard, ResourceBlocked, identity, save, seal, sha
from qualify_exact_sessions import compare as compare_state, check_configuration
from screen_cache import CHECKS, correctness_case, validate_request
from screen_q8_steady import configs as q8_configs
from selector_qualification import check_machine

ROOT=Path(__file__).resolve().parents[2]
KIND='route_selection_screen_v1'
LIMITS=dict(operators=180,requests=600,correctness=360)


def configs():
    return [dict(q8_configs()[1],name=name,route_selection=variant)
            for name,variant in (('control','serial'),('candidate','simd'))]


def state_cases():
    return [correctness_case('clock'),dict(correctness_case('clock'),kernel_policy='candidate',q8_decode_rows=2,route_selection='simd')]


def operators(raw,evidence):
    manifest=raw['source'];cases=manifest['cases']
    expected=[('decode',l,1) for l in range(48)]+[('prefill',l,72) for l in (0,47)]
    if (raw.get('kind')!='captured_router_operator_check_v1' or raw.get('complete') is not True or raw.get('exact') is not True or
        manifest.get('kind')!='captured_router_logits_v1' or manifest.get('build')!=evidence['build'] or
        manifest.get('artifact_revision')!=evidence['artifact_revision'] or raw['machine']['build_fingerprint']!=evidence['build'] or
        [(c['phase'],c['layer'],c['tokens']) for c in cases]!=expected):raise ValueError('Changed or incomplete router operator proof')
    if [(c['case'],c['cpu_ranking_exact']) for c in raw['checks']]!=[(c['sha256'],True) for c in cases]:
        raise ValueError('Missing independent CPU ranking check')
    wanted=[(c['sha256'],c['tokens'],r,v) for c in cases for r in range(10)
            for v in (('serial','simd') if r%2==0 else ('simd','serial'))]
    rows=raw['measurements']
    if [(r['case'],r['tokens'],r['repetition'],r['variant']) for r in rows]!=wanted:raise ValueError('Changed operator timing order')
    for row in rows:
        if row.get('exact') is not True or row['dispatches']!=8 or any(type(row[k]) not in (int,float) or not math.isfinite(row[k]) or row[k]<=0 for k in ('wall_ns','gpu_ns')):
            raise ValueError('Missing exact positive operator timing')
    ratios=[rows[i+int(rows[i]['variant']=='serial')]['gpu_ns']/rows[i+int(rows[i]['variant']!='serial')]['gpu_ns'] for i in range(0,len(rows),2)]
    per_case=[statistics.median(ratios[i:i+10]) for i in range(0,len(ratios),10)]
    return dict(exact=True,cpu_ranking_exact=True,cases=len(cases),median_gpu_ratio=statistics.median(ratios),
        worst_case_median_gpu_ratio=max(per_case),screen_passed=all(v<.9 for v in per_case))


def observations(raw,evidence,config,workload,expected):
    result=validate_request(raw,evidence,config,workload,expected)
    for r in raw['runs']:
        counts=dict(r['after']['metal']['kernel_dispatches']);n=counts.pop('route_simd',0)
        if bool(n)!=(config['route_selection']=='simd'):raise ValueError('Requested routing kernel did not execute')
        if n and counts.get('route',0):raise ValueError('Candidate mixed routing implementations')
        counts['route']=counts.get('route',0)+n
        if expected.setdefault('dispatches_'+r['name'],counts)!=counts:raise ValueError('Non-routing dispatch changed')
        for s in (r['before'],r['after'],*[v[k] for v in r['phases'].values() for k in ('before','after')]):
            if (s['metal']['kernels'].get('profile') or s['metal']['kernels'].get('counter_profile') or
                s.get('short_append_tokens')!=32 or s['memory_plan']['expert_slots']!=1848):raise ValueError('Changed timing instrumentation or capacity')
    return result


def correctness(reports,evidence):
    for raw,case in zip(reports,state_cases()):
        if (raw.get('case')!=case or raw.get('passed') is not True or raw.get('layers')!=48 or raw.get('full_model') is not True or
            len(raw.get('runs',[]))!=1 or len(raw.get('checks',[]))!=len(CHECKS) or
            {c['name'] for c in raw['checks']}!=CHECKS or any(c['passed'] is not True for c in raw['checks'])):raise ValueError('Incomplete state/failure checks')
        for key in ('continued_statistics','after_fresh'):
            state=raw['runs'][0][key];check_machine(state,evidence);check_configuration(state,case)
            if state['memory_plan']['expert_slots']!=32 or state['expert_cache']['evictions']<=0:raise ValueError('Missing forced eviction')
    if len(reports)!=2:raise ValueError('Missing correctness arm')
    compare_state(*reports,evidence['build'],reference_config=state_cases()[0],candidate_config=state_cases()[1])
    return dict(exact_logits_routes_state=True,all_48_layers=True,continued_equals_fresh=True,cancellation_failure_checked=True)


def revalidate(summary,resolve):
    if (summary.get('kind')!=KIND or summary.get('complete') is not True or summary.get('configurations')!=configs() or
        summary.get('gpu_reference_mode')!='off' or summary.get('metal_validation') is not False):raise ValueError('Changed router experiment')
    proof=operators(resolve(summary['operator_source']['sha256']),summary['identity'])
    if proof!=summary['operators'] or not proof['screen_passed']:raise ValueError('Operator screen did not pass')
    rows=summary['measurements'];expected={};seen=set()
    if [(r['pair'],r['configuration']) for r in rows]!=order(2):raise ValueError('Missing two alternating normal pairs')
    for r in rows:
        if r['sha256'] in seen:raise ValueError('Repeated request cannot create a pair')
        seen.add(r['sha256']);c=next(c for c in configs() if c['name']==r['configuration'])
        if observations(resolve(r['sha256']),summary['identity'],c,summary['workload'],expected)!=r['requests']:raise ValueError('Changed request summary')
    decision=decide(rows,2)
    if any(summary.get(k)!=v for k,v in decision.items()):raise ValueError('Changed router timing decision')
    if decision['advance_to_confirmation']:
        if correctness([resolve(r['sha256']) for r in summary['correctness_sources']],summary['identity'])!=summary['correctness']:raise ValueError('Changed state proof')
    elif summary.get('correctness_sources') or summary.get('correctness') is not None:raise ValueError('Unplanned full-state qualification')
    return decision


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic();deadline=started
    report=dict(kind=KIND,complete=False,status='running',phase='preparation',configurations=configs(),measurements=[],correctness_sources=[],
        limits_seconds=LIMITS,gpu_reference_mode='off',metal_validation=False,normal_request_latency_qualified=False,production_promoted=False,
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    def remaining():
        n=deadline-time.monotonic()
        if n<=0:raise subprocess.TimeoutExpired('router-screen',0)
        return n
    def source(p):return dict(source=str(p.relative_to(out)),sha256=sha(p))
    try:
        workload=load(ROOT/'docs/benchmarks/2026-09-11-q8-confirmation/raw/workload.json');save(out/'workload.json',workload)
        save(out/'capture-workload.json',[dict(workload[0],max_tokens=2)])
        for name,case in zip(('control','candidate'),state_cases()):save(out/f'{name}.case.json',case)
        model,prepared=ROOT/'.cache/qwen-mixed-reference',ROOT/'.cache/prepared/q4-records-v1'
        frozen=identity(ROOT,configs(),model,prepared,out/'workload.json')
        for p in (ROOT/'build/qwen/qwen_route_check',out/'capture-workload.json',out/'control.case.json',out/'candidate.case.json'):frozen['files'][str(p.resolve())]=sha(p)
        save(out/'identity.json',frozen);guard=EvidenceGuard(frozen,out)
        report.update(workload=workload,identity={k:frozen[k] for k in ('build','artifact_revision','prepared_manifest_sha256','budget_bytes','device','physical_bytes')})
        env=dict(os.environ)
        for k in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION'):env.pop(k,None)
        def common(c):return ['--model',model,'--artifact','mixed-4_8bit','--prepared',prepared,'--memory-gb','12','--context','8192',*config_args(c)]
        def execute(cmd,stem,limit,validation=False):
            report['phase']=stem.name;save(out/'summary.json',report);print(stem.name,flush=True);last=0
            def pulse():
                nonlocal last
                if time.monotonic()-last>=15:
                    print(f'{stem.name}: elapsed {time.monotonic()-started:.0f}s; phase remaining {max(0,deadline-time.monotonic()):.0f}s',flush=True);last=time.monotonic()
            with stem.with_suffix('.log').open('w') as log:
                guard.run(cmd,stdout=log,timeout=min(limit,remaining()),env=dict(env,MTL_DEBUG_LAYER='1',MTL_SHADER_VALIDATION='1') if validation else env,progress=pulse)
        def admission(c,stem):
            with stem.with_suffix('.admission.log').open('w') as log:
                p=inspect_admission(ROOT/'build/qwen/bin/zerocool',common(c),stem,12*1024**3,512,log,guard,remaining)
            if load(p)['current_admission']['expert_slots']!=1848:raise ResourceBlocked('Expected expert capacity not admitted')
        with (ROOT/'.cache/qwen-qualification.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ResourceBlocked('Another workload owns the GPU lease')
            guard.check_resources(initial=True);deadline=time.monotonic()+LIMITS['operators']
            fixtures=out/'fixtures';fixtures.mkdir();admission(configs()[0],out/'capture')
            with tempfile.TemporaryDirectory(dir=out,prefix='temporary-trace-') as temporary:
                execute([ROOT/'build/qwen/bin/zerocool','bench',*common(configs()[0]),'--workload-file',out/'capture-workload.json',
                    '--temperature','0','--seed','0','--repetitions','1','--trace-dir',temporary,'--json',out/'capture.json'],out/'capture',90)
                cases=[]
                for phase,layer,tokens in [('decode',l,1) for l in range(48)]+[('prefill',l,72) for l in (0,47)]:
                    p=fixtures/f'{phase}-{layer}.bin';shutil.copyfile(Path(temporary)/f'step_{72 if phase=="decode" else 0}'/f'router_{layer}.bin',p)
                    if p.stat().st_size!=tokens*512*4:raise ValueError('Wrong captured router shape')
                    cases.append(dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p),phase=phase,layer=layer,tokens=tokens))
            save(fixtures/'manifest.json',dict(kind='captured_router_logits_v1',build=frozen['build'],artifact_revision=frozen['artifact_revision'],capture=source(out/'capture.json'),cases=cases))
            for p in fixtures.iterdir():frozen['files'][str(p.resolve())]=sha(p)
            save(out/'identity.json',frozen)
            execute([ROOT/'build/qwen/qwen_route_check',fixtures/'manifest.json',out/'operators.json'],out/'operators',90)
            report.update(operator_source=source(out/'operators.json'),operators=operators(load(out/'operators.json'),frozen))
            if not report['operators']['screen_passed']:
                report.update(complete=True,status='operator_improvement_not_demonstrated');return 0
            deadline=time.monotonic()+LIMITS['requests'];expected={}
            for pair,name in order(2):
                c=next(c for c in configs() if c['name']==name);stem=out/f'pair-{pair}-{name}';admission(c,stem)
                execute([ROOT/'build/qwen/bin/zerocool','bench',*common(c),'--workload-file',out/'workload.json','--repetitions','1',
                    '--temperature','0','--seed','0','--gpu-reference','off','--bench-progress',stem.with_suffix('.progress.jsonl'),'--json',stem.with_suffix('.json')],stem,150)
                report['measurements'].append(dict(pair=pair,configuration=name,requests=observations(load(stem.with_suffix('.json')),frozen,c,workload,expected),**source(stem.with_suffix('.json'))))
                save(out/'summary.json',report)
            report.update(decide(report['measurements'],2))
            if report['advance_to_confirmation']:
                deadline=time.monotonic()+LIMITS['correctness'];correct=[]
                for name in ('control','candidate'):
                    p=out/f'correctness-{name}.json';execute([ROOT/'build/qwen/qwen_panel_check',model,prepared,out/f'{name}.case.json',p],p.with_suffix(''),180,True)
                    correct.append(load(p));report['correctness_sources'].append(source(p))
                report['correctness']=correctness(correct,frozen)
            report.update(complete=True,phase='finished')
            files={r['sha256']:out/r['source'] for r in [report['operator_source'],*report['measurements'],*report['correctness_sources']]}
            revalidate(report,lambda h:load(files[h]))
    except ResourceBlocked as e:report.update(status='resource_blocked',error=str(e),complete=False)
    except subprocess.TimeoutExpired:report.update(status='time_budget_exhausted',complete=False)
    except KeyboardInterrupt:report.update(status='interrupted',complete=False)
    except Exception as e:report.update(status='failed',error=str(e),complete=False)
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(out/'summary.json',report);seal(out);print(json.dumps({k:report[k] for k in ('status','complete','phase','elapsed_seconds')}),flush=True)
    return 0 if report['complete'] else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    raise SystemExit(run(parser.parse_args()))
