#!/usr/bin/env python3
"""Reproduce the bounded exact sparse-attention experiment; never promotes defaults."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import struct
import sys
from benchmark_exact import config_args, inspect_admission, workloads
from build_identity import build_fingerprint
from summarize_cached_comparison import summarize as cached_summary

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'docs/benchmarks/2026-09-09-sparse-attention/configs.json'


def check_cached(report,build,revision,tokens,axis):
    if (report.get('artifact_revision')!=revision or report.get('prompt_tokens')!=len(tokens)-1 or
        report.get('comparison_axis')!=axis or report.get('continuation_token')!=tokens[-1] or
        report.get('input_sha256')!=hashlib.sha256(struct.pack('<'+'i'*len(tokens),*tokens)).hexdigest()):
        raise ValueError('Cached input, artifact or comparison identity differs')
    for row in report['runs']:
        for state in (row['before'],row['after']):
            metal=state['metal']
            if metal['build_fingerprint']!=build or metal['device']!='Apple M1 Pro' or metal['physical_bytes']!=32*1024**3:
                raise ValueError('Cached experiment requires the current build on the actual 32GiB M1 Pro')
            if state['memory_plan']['planned_bytes']>12*1024**3 or state['process']['physical_footprint_bytes']>12*1024**3:
                raise ValueError('Cached experiment exceeded its memory budget')
    return cached_summary(report)


def acceptance(summary):
    if not summary.get('complete') or summary.get('mode')!='paired' or summary.get('comparison_purpose')!='experiment':
        raise ValueError('Requires a completed paired experiment')
    comparisons=summary['comparisons']
    if len(comparisons)!=1:raise ValueError('Freeze one candidate before qualification')
    bounds=comparisons[0]['confidence_bounds']
    required={(c,m) for c in ('prompt_2k','prompt_4k','append_128') for m in ('ttft_ms','decode_ms_per_token','request_ms')}
    if {(b['case'],b['metric']) for b in bounds}!=required or len(bounds)!=len(required):raise ValueError('Missing or duplicate workload metrics')
    for b in bounds:
        if b['pairs']<5 or not all(math.isfinite(b[k]) and b[k]>0 for k in ('median','low','high')) or not b['low']<=b['median']<=b['high']:
            raise ValueError('Invalid confidence evidence')
    improved=any(b['case'] in ('prompt_4k','append_128') and b['metric']=='request_ms' and b['median']<=.97 and b['high']<1 for b in bounds)
    return dict(stage_latency_passed=improved and all(b['high']<=1.03 for b in bounds),
                full_product_acceptance=False,production_promoted=False)


def choose(screen):
    if not screen.get('complete') or screen.get('mode')!='screen':raise ValueError('Incomplete normal screen')
    rows=screen['measurements'];ratios={}
    expected={(name,case,pair) for name in ('cpu-selection','gpu-selection','gpu-selection-skip')
              for case in ('prompt_2k','prompt_4k','append_128','prompt_7k') for pair in (0,1)}
    if len(rows)!=len(expected) or {(r['configuration'],r['name'],r['pair']) for r in rows}!=expected:
        raise ValueError('Screen requires all three arms and all four workloads in two complete pairs')
    if not all(math.isfinite(r['request_ms']) and r['request_ms']>0 for r in rows):
        raise ValueError('Invalid screen timing')
    for name in ('gpu-selection','gpu-selection-skip'):
        values=[]
        for case in ('prompt_4k','append_128'):
            a={r['pair']:r['request_ms'] for r in rows if r['configuration']=='cpu-selection' and r['name']==case}
            b={r['pair']:r['request_ms'] for r in rows if r['configuration']==name and r['name']==case}
            if set(a)!=set(b) or len(a)!=2:raise ValueError('Screen requires two complete alternating pairs')
            values.extend(b[i]/a[i] for i in a)
        if not all(math.isfinite(x) and x>0 for x in values):raise ValueError('Invalid screen timing')
        ratios[name]=math.exp(sum(map(math.log,values))/len(values))
    b,c=ratios['gpu-selection'],ratios['gpu-selection-skip']
    return 'gpu-selection' if b<=c*1.01 else 'gpu-selection-skip'


def run(args):
    if getattr(args,'track','three-arm')=='selector':
        from selector_qualification import run as selector_run
        return selector_run(args)
    if args.phase not in ('capture','cached','screen','paired','qualify') or getattr(args,'resume_from',None):
        raise ValueError('Recovery, profiling, all-phase execution and resume require --track selector')
    if getattr(args,'contexts',None) and len(set(args.contexts))!=len(args.contexts):
        raise ValueError('Duplicate cached contexts')
    if any(os.environ.get(k) not in (None,'','0') for k in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION')):
        raise ValueError('Disable validation during performance experiments')
    args.output.mkdir(parents=True,exist_ok=False)
    configs=json.loads(CONFIG.read_text());build=build_fingerprint(ROOT)
    seed=json.loads((ROOT/'docs/benchmarks/2026-09-08-validation/workload-2k.json').read_text())[0]['tokens']
    model=ROOT/'.cache/qwen-mixed-reference';prepared=ROOT/'.cache/prepared/q4-records-v1';binary=ROOT/'build/qwen/bin/freellm'
    shared=['--model',str(model),'--artifact','mixed-4_8bit','--prepared',str(prepared),'--memory-gb','12','--context','8192']
    hashes={};completed=[]
    def save(name,obj):
        p=args.output/name;p.write_text(json.dumps(obj,indent=2)+'\n');return p
    def call(command,stem,timeout=7200):
        if build_fingerprint(ROOT)!=build:raise ValueError('Native sources changed during experiment')
        print(stem,flush=True)
        with (args.output/(stem+'.log')).open('w') as log:
            subprocess.run([str(x) for x in command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    try:
        if args.phase=='capture':
            total=0
            conversations=workloads(seed,2,True)
            for name in ('prompt_7k','prompt_4k','append_128'):
                conversation=conversations[name]
                stem=args.output/name;dest=args.output/(name+'-capture')
                inp=save(name+'-workload.json',conversation)
                with (args.output/(name+'-admission.log')).open('w') as log:
                    inspect_admission(binary,shared+config_args(configs[0]),stem,12*1024**3,512,log)
                call([binary,'bench',*shared,*config_args(configs[0]),'--workload-file',inp,'--repetitions','1',
                      '--sparse-capture',dest,'--json',stem.with_suffix('.json')],name)
                manifest=dest/'manifest.json';data=json.loads(manifest.read_text());total+=data['bytes']
                if total>256*1024**2:raise ValueError('Aggregate sparse captures exceed256MiB')
                call([ROOT/'build/qwen/qwen_sparse_replay',manifest,args.output/(name+'-replay.json')],name+'-replay',600)
                hashes[name]=hashlib.sha256(manifest.read_bytes()).hexdigest();completed.append(name)
        elif args.phase=='cached':
            selected=getattr(args,'contexts',None)
            for axis,contexts,config in [('sparse_selection',selected or (2048,4096,7168),configs[1]),('attention_score_tiles',[n for n in (selected or (4096,7168)) if n>=4096],configs[2])]:
                for n in contexts:
                    name=f'{axis}-{n}';tokens=(seed*((n+len(seed)-1)//len(seed)))[:n]+[760];inp=save(name+'-tokens.json',tokens)
                    dest=args.output/(name+'.json')
                    call([binary,'bench',*shared,*config_args(config),'--cached-token-replay','--cached-compare',
                          '--cached-compare-axis',axis,'--tokens-file',inp,'--repetitions','5','--json',dest],name)
                    revision=json.loads((ROOT/'mixed-models.lock.json').read_text())['revision']
                    save(name+'-summary.json',check_cached(json.loads(dest.read_text()),build,revision,tokens,axis));completed.append(name)
        elif args.phase in ('screen','paired'):
            config=CONFIG
            if args.phase=='paired':
                if not args.screen:raise ValueError('Paired phase requires a completed --screen summary')
                screen=json.loads(args.screen.read_text())
                if (screen.get('build')!=build or screen.get('configurations')!=configs or screen.get('budget_bytes')!=12*1024**3 or
                    screen.get('artifact')!='mixed-4_8bit' or screen.get('comparison_purpose')!='experiment' or
                    screen.get('workload_sha256')!=hashlib.sha256((ROOT/'docs/benchmarks/2026-09-08-validation/workload-2k.json').read_bytes()).hexdigest()):
                    raise ValueError('Screen build, configurations, artifact, budget or workload differ')
                candidate=choose(screen);config=save('frozen-configs.json',[configs[0],next(c for c in configs if c['name']==candidate)])
            dest=args.output/'normal'
            cmd=[sys.executable,ROOT/'scripts/qwen/benchmark_exact.py','--output',dest,'--config',config,
                 '--mode','screen' if args.phase=='screen' else 'paired','--comparison-purpose','experiment',
                 '--pairs','2' if args.phase=='screen' else '5','--memory-gb','12']
            if args.phase=='screen':cmd+=['--include-7k']
            call(cmd,'normal',43200)
            if args.phase=='paired':save('stage-acceptance.json',acceptance(json.loads((dest/'summary.json').read_text())))
            completed.append('normal')
        elif args.phase=='qualify':
            for case in ('boundary','append','7k'):
                cmd=[sys.executable,ROOT/'scripts/qwen/qualify_exact_sessions.py','--case',case,'--output',args.output/case,
                     '--memory-gb','12','--token-tile','8','--affine-rows','4','--q8-decode-rows','2','--gdn-path','precompute',
                     '--ready-group','2','--residency','core-cache','--decode-path','grouped','--prefill-pipeline','double',
                     '--phase-memory','reclaim','--sparse-selection','gpu','--attention-score-tiles','skip-masked']
                call(cmd,case,43200);completed.append(case)
        save('summary.json',dict(phase=args.phase,complete=True,build_fingerprint=build,completed=completed,
             manifest_hashes=hashes,full_product_acceptance=False,production_promoted=False))
    except Exception as error:
        save('summary.json',dict(phase=args.phase,complete=False,build_fingerprint=build,completed=completed,error=str(error)))
        raise


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--track',choices=['selector','three-arm'],default='three-arm')
    ap.add_argument('--phase',choices=['all','recovery','capture','cached','screen','paired','qualify','profile'],required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--screen',type=Path)
    ap.add_argument('--contexts',type=int,nargs='+',choices=[2048,4096,7168])
    ap.add_argument('--resume-from',type=Path)
    run(ap.parse_args())
