#!/usr/bin/env python3
"""Bounded current-build capacity, dependency and Q3 experiments. No promotion."""
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

from benchmark_exact import config_args, inspect_admission
from capture_routes import load
from decode_timeline import summarize
from qualification_evidence import EvidenceGuard, ResourceBlocked, identity, save, seal, sha, tree_bytes, verify_seal
from screen_cache import validate_request
from screen_decode_scratch import configs as scratch_configs

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'docs/benchmarks/2026-09-13-scratch-confirmation/settled-screen'
ORDER=((0,'control'),(0,'candidate'),(1,'candidate'),(1,'control'))


def configuration(slots=1848,policy='observe',name='control'):
    return dict(scratch_configs()[1],name=name,expert_slots=slots,memory_pressure_policy=policy)


def clean_memory(raw):
    # A missing gauge or counter reset is not a clean observation.
    for row in raw['runs']:
        for phase in row['phases'].values():
            a,b=(phase[k]['process'] for k in ('before','after'))
            if any(s.get('compressed_bytes')!=0 for s in (a,b)):return False
            if type(a.get('decompressions')) is not int or b.get('decompressions')!=a['decompressions']:return False
    return True


def capacity_decision(measurements,control=1848,candidate=1460):
    if [(m['pair'],m['configuration']) for m in measurements]!=list(ORDER):raise ValueError('Missing alternating capacity pairs')
    by={(m['pair'],m['configuration']):m for m in measurements}
    ratios=[]
    for pair in range(2):
        a,b=(sum(r['request_ms'] for r in by[pair,k]['requests']) for k in ('control','candidate'))
        if not all(math.isfinite(v) and v>0 for v in (a,b)):raise ValueError('Invalid request timing')
        ratios.append(b/a)
    current_clean=all(by[p,'control']['clean_memory'] is True for p in range(2))
    candidate_clean=all(by[p,'candidate']['clean_memory'] is True for p in range(2))
    selected=control if current_clean else candidate if candidate_clean and statistics.median(ratios)<=1.05 else None
    return dict(status='provisional' if selected else 'unresolved_pressure',selected_slots=selected,
                current_clean=current_clean,candidate_clean=candidate_clean,conversation_ratios=ratios,
                median_conversation_ratio=statistics.median(ratios),stability_evidence=True,
                normal_request_latency_qualified=False,production_promoted=False)


def q3_decision(report):
    if not report.get('complete') or report.get('validation') is not False or len(report.get('cases',[]))!=32:
        raise ValueError('Incomplete Q3 timing coverage')
    groups={};clean=True
    for c in report['cases']:
        key=(c['layer'],c['expert']);groups.setdefault(key,set())
        if c['rows'] in groups[key]:raise ValueError('Duplicate Q3 case')
        groups[key].add(c['rows'])
        if not c.get('exact_expanded_reference') or [p['pair'] for p in c['pairs']]!=list(range(5)):
            raise ValueError('Missing Q3 arithmetic or paired coverage')
        if any(type(c[k]) is not int or c[k]<=0 for k in ('q3_aligned_bytes','q4_aligned_bytes')) or c['q3_aligned_bytes']/c['q4_aligned_bytes']>.86:
            raise ValueError('Insufficient padded-byte saving')
        for pair in c['pairs']:
            for arm in ('q3','q4'):
                value=pair[arm]['ns_per_chain']
                if type(value) not in (float,int) or not math.isfinite(value) or value<=0:raise ValueError('Invalid Q3 timing')
                a,b=(pair[arm].get('memory_'+k,{}) for k in ('before','after'))
                clean &= (a.get('compressed_bytes')==b.get('compressed_bytes')==0 and
                    type(a.get('decompressions')) is int and a['decompressions']==b.get('decompressions'))
    if (len(groups)!=8 or any(v!={1,2,4,8} for v in groups.values()) or
        {l for l,e in groups}!={0,16,32,47} or any(sum(l==layer for l,e in groups)!=2 for layer in (0,16,32,47))):
        raise ValueError('Q3 expert/row coverage differs')
    # A repeat pair is the unit. Expert fixtures within a pair are correlated.
    ratios=[]
    for p in range(5):
        values=[c['pairs'][p] for c in report['cases'] if c['rows']==1]
        for v in values:
            if any(type(v[k]['ns_per_chain']) not in (float,int) or not math.isfinite(v[k]['ns_per_chain']) or v[k]['ns_per_chain']<=0 for k in ('q3','q4')):
                raise ValueError('Invalid Q3 timing')
        ratios.append(sum(v['q3']['ns_per_chain'] for v in values)/sum(v['q4']['ns_per_chain'] for v in values))
    logs=[math.log(r) for r in ratios];mean=statistics.mean(logs)
    upper=math.exp(mean+2.776445105*statistics.stdev(logs)/math.sqrt(5))
    return dict(status='memory_disturbed' if not clean else 'worth_quality_investigation' if upper<=1.03 else 'operator_regression',
                one_token_geomean_ratio=math.exp(mean),one_token_upper_95=upper,ratios=ratios,
                clean_memory=clean,advance_to_quality_stage=clean and upper<=1.03,quality_qualified=False,normal_request_latency_qualified=False)


class StageGuard(EvidenceGuard):
    def __init__(self,evidence,output,limit):
        super().__init__(evidence,output);self.limit=limit
    def check_resources(self,initial=False):
        super().check_resources(initial)
        if tree_bytes(self.output)>=self.limit:raise ResourceBlocked('Stage evidence byte limit reached')


class Experiment:
    def __init__(self,out,kind,configs,work,seconds):
        self.out=out.resolve();self.out.mkdir(parents=True,exist_ok=False)
        self.start=time.monotonic();self.deadline=self.start+seconds
        self.report=dict(kind=kind,complete=False,status='running',phase='prepare',measurements=[],
            started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),time_limit_seconds=seconds,
            normal_request_latency_qualified=False,production_promoted=False)
        self.lock=None;self.persist()
        try:self.prepare(configs,work,kind)
        except BaseException as error:
            self.__exit__(type(error),error,error.__traceback__);raise
    def prepare(self,configs,work,kind):
        save(self.out/'workload.json',work)
        self.model=ROOT/'.cache/qwen-mixed-reference';self.prepared=ROOT/'.cache/prepared/q4-records-v1'
        self.frozen=identity(ROOT,configs,self.model,self.prepared,self.out/'workload.json')
        for p in (ROOT/'build/qwen/qwen_q3_probe',ROOT/'scripts/qwen/probe_q3.cpp'):
            self.frozen['files'][str(p.resolve())]=sha(p)
        self.frozen['files'][str((SOURCE/'pair-0-control.json').resolve())]=sha(SOURCE/'pair-0-control.json')
        save(self.out/'identity.json',self.frozen)
        self.report.update(identity={k:self.frozen[k] for k in ('build','artifact_revision','prepared_manifest_sha256','budget_bytes','device','physical_bytes')},
            configurations=configs,workload=work)
        self.guard=StageGuard(self.frozen,self.out,256*1024**2 if kind=='stage200_q3_v1' else 2*1024**3)
        self.env=dict(os.environ)
        for k in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION'):self.env.pop(k,None)
        self.persist()
    def persist(self):save(self.out/'summary.json',self.report)
    def left(self):
        value=self.deadline-time.monotonic()
        if value<=0:raise subprocess.TimeoutExpired('stage200',0)
        return value
    def __enter__(self):
        self.lock=(ROOT/'.cache/qwen-qualification.lock').open('a')
        try:fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close();self.lock=None
            error=ResourceBlocked('Another workload owns GPU lease')
            self.__exit__(ResourceBlocked,error,None);raise error
        return self
    def __exit__(self,kind,error,tb):
        if error:
            status='resource_blocked' if isinstance(error,ResourceBlocked) else 'time_budget_exhausted' if isinstance(error,subprocess.TimeoutExpired) else 'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
            self.report.update(complete=False,status=status,error=str(error))
        else:self.report.update(complete=True,phase='finished')
        self.report['elapsed_seconds']=time.monotonic()-self.start;self.persist();seal(self.out)
        if self.lock:self.lock.close()
        print(json.dumps({k:self.report.get(k) for k in ('status','complete','selected_slots','error')}),flush=True)
        return bool(error)
    def command(self,command,stem,limit=150,validation=False):
        self.report['phase']=stem;self.persist();print(stem+' begin',flush=True)
        path=self.out/stem;last=0
        def progress():
            nonlocal last
            if time.monotonic()-last<15:return
            last=time.monotonic();p=path.with_suffix('.progress.jsonl')
            try:phase=json.loads(p.read_text().splitlines()[-1]).get('phase','running')
            except (OSError,ValueError,IndexError):phase='running'
            print(f'{stem}: {phase}; {self.left():.0f}s budget remaining',flush=True)
        with path.with_suffix('.log').open('w') as log:
            self.guard.run(command,stdout=log,timeout=min(limit,self.left()),progress=progress,
                env=dict(self.env,MTL_DEBUG_LAYER='1',MTL_SHADER_VALIDATION='1') if validation else self.env)
        print(stem+' complete',flush=True)
    def bench(self,c,stem,extra=(),limit=150):
        common=['--model',self.model,'--artifact','mixed-4_8bit','--prepared',self.prepared,'--memory-gb','12','--context','8192',*config_args(c)]
        with (self.out/(stem+'.admission.log')).open('w') as log:
            p=inspect_admission(ROOT/'build/qwen/bin/freellm',common,self.out/stem,12*1024**3,512,log,self.guard,self.left)
        if load(p)['current_admission']['expert_slots']!=c['expert_slots']:raise ResourceBlocked('Requested capacity not admitted')
        self.command([ROOT/'build/qwen/bin/freellm','bench',*common,'--workload-file',self.out/'workload.json',
            '--repetitions','1','--temperature','0','--seed','0','--bench-progress',self.out/(stem+'.progress.jsonl'),
            '--json',self.out/(stem+'.json'),*extra],stem,limit)
        return load(self.out/(stem+'.json'))


def run_capacity(args):
    verify_seal(SOURCE,sha(SOURCE/'evidence-files.json'))
    work=load(SOURCE/'workload.json');configs=[configuration(args.slots),configuration(args.candidate_slots,name='candidate')]
    exp=Experiment(args.output,'stage200_capacity_v1',configs,work,600)
    with exp:
        exp.guard.check_resources(initial=True)
        expected={r['name']:r['output_token_ids'] for r in load(SOURCE/'pair-0-control.json')['runs']}
        for pair,arm in ORDER:
            c=configs[arm=='candidate'];stem=f'pair-{pair}-{arm}';raw=exp.bench(c,stem)
            requests=validate_request(raw,exp.frozen,c,work,expected,capacity_axis=True)
            for row in raw['runs']:
                for states in row['phases'].values():
                    for s in states.values():
                        if s['memory_pressure']['policy']!='observe':raise ValueError('Unexpected pressure policy')
            exp.report['measurements'].append(dict(pair=pair,configuration=arm,requests=requests,clean_memory=clean_memory(raw),source=stem+'.json',sha256=sha(exp.out/(stem+'.json'))))
            exp.persist()
        exp.report.update(capacity_decision(exp.report['measurements'],args.slots,args.candidate_slots))
    return exp.report


def run_pressure(args):
    verify_seal(SOURCE,sha(SOURCE/'evidence-files.json'))
    work=load(SOURCE/'workload.json');configs=[configuration(args.slots),configuration(args.slots,'shrink','candidate')]
    exp=Experiment(args.output,'stage200_pressure_v1',configs,work,600)
    with exp:
        exp.guard.check_resources(initial=True)
        expected={r['name']:r['output_token_ids'] for r in load(SOURCE/'pair-0-control.json')['runs']}
        for pair,arm in ORDER:
            c=configs[arm=='candidate'];stem=f'pair-{pair}-{arm}';raw=exp.bench(c,stem)
            requests=validate_request(raw,exp.frozen,c,work,expected,pressure_axis=True)
            pressure=raw['runs'][-1]['after']['memory_pressure']
            exp.report['measurements'].append(dict(pair=pair,configuration=arm,requests=requests,clean_memory=clean_memory(raw),
                pressure=pressure,source=stem+'.json',sha256=sha(exp.out/(stem+'.json'))));exp.persist()
        exercised=any(any(e['achieved_slots']<e['before_slots'] for e in r['pressure']['events']) for r in exp.report['measurements'] if r['configuration']=='candidate')
        exp.report.update(status='observed' if exercised else 'not_exercised',real_pressure_exercised=exercised)
    return exp.report


def profile_row(raw,index,profile,deps):
    row=raw['runs'][index];low=row['prompt_tokens'];high=low+len(row['token_latency_ms'])
    groups=[]
    for g in profile['command_groups']:
        inside=[low<=op['offset']<high and op['tokens']==1 for op in g['operations']]
        if any(inside) and not all(inside):raise ValueError('Command crosses captured token window')
        if inside and all(inside):groups.append(g)
    selected=[d for d in deps if d['tokens']==1 and low<=d['offset']<high]
    return summarize(dict(raw,runs=[row]),dict(profile,command_groups=groups),selected)


def run_profile(args):
    verify_seal(SOURCE,sha(SOURCE/'evidence-files.json'))
    work=load(SOURCE/'workload.json')
    if args.prompt_tokens==2048:work[0]['tokens']=load(ROOT/'docs/benchmarks/2026-09-08-validation/workload-2k.json')[0]['tokens'][:2048]
    for task in work:task['max_tokens']=17
    c=configuration(args.slots);exp=Experiment(args.output,'decode_target_capture_v2',[c],work,900)
    with exp:
        exp.guard.check_resources(initial=True);expected={}
        for stem in ('normal','traced'):
            extra=['--decode-diagnostics','--phase-profile',exp.out/'commands.json','--profile-decode-only','1',
                '--dependency-trace',exp.out/'dependencies.jsonl'] if stem=='traced' else []
            raw=exp.bench(c,stem,extra,300)
            validate_request(raw,exp.frozen,c,work,expected,instrumented=stem=='traced',output_tokens=17)
            if stem=='normal':
                required=sum(r['phases']['decode']['after']['metal']['dispatches']-r['phases']['decode']['before']['metal']['dispatches'] for r in raw['runs'])
                if required>120000:raise ValueError('Requested decode windows exceed the native profile bound')
            exp.report[stem]=dict(source=stem+'.json',sha256=sha(exp.out/(stem+'.json')),clean_memory=clean_memory(raw))
        normal=load(exp.out/'normal.json');raw=load(exp.out/'traced.json');profile=load(exp.out/'commands.json')
        deps=[json.loads(x) for x in (exp.out/'dependencies.jsonl').read_text().splitlines()]
        timelines=[profile_row(raw,i,profile,deps) for i in range(2)];save(exp.out/'timelines.json',timelines)
        ratios=[b['decode_wall_ms']/a['decode_wall_ms'] for a,b in zip(normal['runs'],raw['runs'])]
        opportunities=[]
        for name in ('gpu_active','gpu_idle_submitted','gpu_idle_ready_expert','gpu_idle_pending_read','gpu_idle_callback','gpu_idle_other'):
            values=[t['mean_buckets_ms'][name] for t in timelines]
            opportunities.append(dict(bucket=name,observed_ms=values,large_enough_to_investigate=max(values)>=20,
                attainable_saving_ms=None,causality_established=False))
        exp.report.update(status='captured',trace_to_normal_ratios=ratios,opportunities=opportunities,
            timelines=dict(source='timelines.json',sha256=sha(exp.out/'timelines.json')))
    return exp.report


def run_q3(args):
    source=load(ROOT/'docs/benchmarks/2026-09-08-validation/workload-2k.json')[0]['tokens']
    slots=min(args.slots,1460)
    exp=Experiment(args.output,'stage200_q3_v1',[configuration(slots)],[],900)
    with exp:
        exp.guard.check_resources(initial=True)
        for split,offset in (('tuning',0),('heldout',512)):
            path=exp.out/(split+'.workload.json');save(path,dict(tokens=source[offset:offset+72],split=split,source_offset=offset))
            exp.frozen['files'][str(path)]=sha(path);save(exp.out/'identity.json',exp.frozen)
            exp.command([ROOT/'build/qwen/qwen_q3_probe','capture',exp.model,exp.prepared,path,exp.out/split,slots],split+'-capture',150)
            manifest=load(exp.out/split/'manifest.json')
            if manifest.get('complete') is not True:raise ValueError('Incomplete Q3 fixtures')
            for file in (exp.out/split).iterdir():exp.frozen['files'][str(file)]=sha(file)
            save(exp.out/'identity.json',exp.frozen)
            for mode in ('validate','timing'):
                stem=split+'-'+mode
                exp.command([ROOT/'build/qwen/qwen_q3_probe','probe',exp.out/split,exp.out/(stem+'.json'),mode],stem,150,mode=='validate')
                result=load(exp.out/(stem+'.json'))
                if result.get('complete') is not True or result.get('validation') is not (mode=='validate') or len(result.get('cases',[]))!=32:
                    raise ValueError('Incomplete Q3 operator coverage')
            decision=q3_decision(load(exp.out/(split+'-timing.json')))
            exp.report[split]=decision;exp.persist()
            if not decision['advance_to_quality_stage']:
                exp.report.update(status=decision['status'],advance_to_quality_stage=False,heldout_status='not_run' if split=='tuning' else 'completed')
                return exp.report
        exp.report.update(status=exp.report['heldout']['status'],advance_to_quality_stage=all(exp.report[s]['advance_to_quality_stage'] for s in ('tuning','heldout')))
    return exp.report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['capacity','pressure','profile','q3'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--slots',type=int,choices=[1848,1460,1072],default=1848)
    p.add_argument('--candidate-slots',type=int,choices=[1460,1072],default=1460)
    p.add_argument('--prompt-tokens',type=int,choices=[72,2048],default=72);args=p.parse_args()
    if args.stage=='capacity' and args.candidate_slots>=args.slots:p.error('candidate capacity must be smaller')
    result={'capacity':run_capacity,'pressure':run_pressure,'profile':run_profile,'q3':run_q3}[args.stage](args)
    return 0 if result['complete'] else 2


if __name__=='__main__':raise SystemExit(main())
