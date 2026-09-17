#!/usr/bin/env python3
"""Short, source-sealed real-expert command-ownership screen; no promotion."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics

from build_identity import build_fingerprint
from qualification_evidence import EvidenceGuard, save, seal, sha
from screen_residency import paired_log_interval

ROOT=Path(__file__).resolve().parents[2]


def analyze(raw):
    if raw.get('complete') is not True or raw.get('validation') is not False or len(raw.get('pairs',[]))!=15:
        raise ValueError('Incomplete isolated timing coverage')
    results=[];clean=True
    for size in (1,2,4):
        rows=[r for r in raw['pairs'] if r['experts_per_group']==size]
        if [r['pair'] for r in rows]!=list(range(5)):raise ValueError('Missing repeat pairs')
        ratios=[];savings=[]
        for p,r in enumerate(rows):
            if [a['retained_references'] for a in r['arms']]!=[p%2==0,p%2!=0]:raise ValueError('Changed alternating order')
            means={}
            for a in r['arms']:
                if len(a['samples'])!=64 or any(type(s['wall_ns']) is not int or s['wall_ns']<=0 for s in a['samples']):
                    raise ValueError('Missing or invalid group timings')
                before,after=a['memory_before'],a['memory_after']
                clean &= (before.get('compressed_bytes')==after.get('compressed_bytes')==0 and
                          type(before.get('decompressions')) is int and before['decompressions']==after.get('decompressions'))
                means[a['retained_references']]=statistics.mean(s['wall_ns'] for s in a['samples'])
            ratios.append(means[False]/means[True]);savings.append((means[True]-means[False])/1e6)
        interval=paired_log_interval(ratios)
        results.append(dict(experts_per_group=size,ratios=ratios,confidence_95=interval,savings_ms_per_group=savings))
    primary=results[0]
    # Prior complete short trace: 7,174 expert-only groups / 32 tokens. This
    # scaling is an optimistic estimate, not a request-latency prediction.
    estimated=statistics.median(primary['savings_ms_per_group'])*(7174/32)
    advance=(clean and estimated>=20 and primary['confidence_95']['high']<1 and
             all(v>0 for v in primary['savings_ms_per_group']))
    return dict(status='worth_request_screen' if advance else 'memory_disturbed' if not clean else 'insufficient_benefit',
        results=results,clean_memory=clean,optimistic_ms_per_token=estimated,advance_to_request_screen=advance,
        normal_request_latency_qualified=False,production_promoted=False)


def run(out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    fixtures=ROOT/'docs/benchmarks/2026-09-13-stage200/q3-02/tuning'
    paths=[*fixtures.glob('*'),ROOT/'scripts/qwen/probe_submission.cpp',Path(__file__),
           ROOT/'build/qwen/qwen_submission_probe',ROOT/'build/qwen/test_qwen',ROOT/'tests/test_qwen.cpp']
    frozen=dict(root=str(ROOT),build=build_fingerprint(ROOT),assets={},files={str(p.resolve()):sha(p) for p in paths if p.is_file()})
    save(out/'identity.json',frozen);guard=EvidenceGuard(frozen,out)
    report=dict(kind='command_ownership_screen_v1',complete=False,status='running',identity=frozen,
                normal_request_latency_qualified=False,production_promoted=False)
    save(out/'summary.json',report)
    env=dict(os.environ)
    for k in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION'):env.pop(k,None)
    try:
        with (ROOT/'.cache/qwen-qualification.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            commands=[('native-tests',[ROOT/'build/qwen/test_qwen'],True),
                ('validation',[ROOT/'build/qwen/qwen_submission_probe',fixtures,out/'validation.json','validate'],True),
                ('timing',[ROOT/'build/qwen/qwen_submission_probe',fixtures,out/'timing.json','timing'],False)]
            for name,command,validation in commands:
                print(name,flush=True)
                with (out/(name+'.log')).open('w') as log:
                    guard.run(command,stdout=log,timeout=60,env=dict(env,MTL_DEBUG_LAYER='1',MTL_SHADER_VALIDATION='1') if validation else env)
            timing=json.loads((out/'timing.json').read_text());validation=json.loads((out/'validation.json').read_text())
            if any(r.get('build')!=frozen['build'] or r.get('complete') is not True for r in (timing,validation)):
                raise ValueError('Wrong native binary or incomplete fixture check')
            report.update(analyze(timing),complete=True)
    except Exception as error:report.update(status='incomplete',error=str(error))
    finally:
        save(out/'summary.json',report);seal(out);print(json.dumps({k:report.get(k) for k in ('status','optimistic_ms_per_token','error')}),flush=True)
    return 0 if report['complete'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    raise SystemExit(run(p.parse_args().output))
