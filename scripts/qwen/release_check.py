#!/usr/bin/env python3
"""Fail closed unless every numerical, M1 performance, and session gate passes.

This consumes reviewable evidence; model-free CTest is never a release pass.
Missing reports, missing model files, short benchmark runs, or skipped cases
are failures. A manually supplied coding result must include its tool trace
and independent post-run test results for review.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
from verify_checkpoint import ROOT, verify
from build_identity import build_fingerprint

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--model',type=Path,default=ROOT/'.cache/models/qwen38-flash-next')
ap.add_argument('--artifact',choices=('q4-control','mixed-4_8bit'),default='q4-control')
ap.add_argument('--evidence',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args();checks=[]
current_build=build_fingerprint(ROOT)
selected_lock=json.loads((ROOT/('models.lock.json' if args.artifact=='q4-control' else 'mixed-models.lock.json')).read_text())


def check(name,fn):
    try:
        detail=fn();checks.append(dict(name=name,passed=True,detail=detail))
    except Exception as e: checks.append(dict(name=name,passed=False,error=str(e)))


def load(name): return json.loads((args.evidence/name).read_text())


def current(fingerprint):
    assert fingerprint==current_build,'Evidence does not match current native sources; rerun this check'


def identity():
    result=verify(args.model,selected_lock)
    return dict(revision=result['revision'],verified_files=len(result['files']))


def logits():
    meta=load('logits.json')
    current(meta.get('build_fingerprint'))
    assert meta.get('artifact_revision')==selected_lock['revision'],'Logit evidence belongs to a different or unidentified artifact'
    assert meta.get('reference',{}).get('source_revision')==selected_lock['revision'],'Reference artifact differs'
    for role in ('native','reference'):
        assert meta.get(role+'_logits_sha256')==hashlib.sha256((args.evidence/(role+'_logits.f32')).read_bytes()).hexdigest(),'Logit payload changed since comparison'
    assert meta['reference_tokens']==meta['native_tokens'] and meta['layers']==48,'Different logit fixtures'
    a=np.fromfile(args.evidence/'reference_logits.f32',np.float32).astype(np.float64)
    b=np.fromfile(args.evidence/'native_logits.f32',np.float32).astype(np.float64)
    assert a.size==b.size==248320 and np.isfinite(a).all() and np.isfinite(b).all(),'Invalid logit vectors'
    relative=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-9))
    cosine=float(np.clip(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-9),-1,1))
    assert relative<=0.02 and cosine>=0.9998,f'Logit mismatch: relative L2={relative:.6f}, cosine={cosine:.6f}'
    return dict(relative_l2=relative,cosine=cosine)


def performance():
    from benchmark_exact import fixed_sampling
    report=load('performance.json');results={}
    assert report.get('complete') is True,'Performance run did not complete'
    for name,length,latency in [('prompt_2k',2048,60000),('prompt_4k',4096,None),('append_128',None,10000),('prompt_7k',7168,None)]:
        rows=[r for r in report['runs'] if r['name']==name]
        assert len(rows)>=5,f'{name}: requires at least five measurements'
        for r in rows:
            assert report.get('model_revision')==selected_lock['revision'],'Performance artifact differs'
            assert r['after'].get('artifact_revision')==selected_lock['revision'],'Runtime performance artifact differs'
            assert r['output_tokens']==256,f'{name}: requires 256 output tokens'
            if length: assert r['prompt_tokens']==length
            if name=='append_128':
                assert r['reused_tokens']==4096 and r['prefill_tokens']==128 and r.get('pending_tokens_ingested')==0,'Append must measure exactly 128 new tokens after sample-free priming'
            machine=r['after']['metal'];assert machine['device']=='Apple M1 Pro' and machine['physical_bytes']==32*1024**3,'Requires the actual 32GiB M1 Pro'
            current(machine.get('build_fingerprint'))
            assert r['after'].get('diagnostic_stream_trunk') is False,'Diagnostic schedules cannot satisfy performance acceptance'
            assert r.get('profiling_enabled') is False,'Profiled timings cannot satisfy performance acceptance'
            assert r.get('repetition')==0,'Each independent comparison must run in a fresh process'
            assert r.get('runtime_cache_state')==('retained' if name=='append_128' else 'empty_at_process_start'),'Wrong runtime cache freshness'
            plan=r['after']['memory_plan'];budget=plan['limit_bytes']
            assert budget<=22*1024**3 and plan['planned_bytes']<=budget
            assert r['after']['process']['physical_footprint_bytes']<=budget
            assert r['before']['memory_plan']['limit_bytes']==budget and r['before']['memory_plan']['expert_slots']==plan['expert_slots'],'Memory admission changed during request'
            assert len(r['token_latency_ms'])==255 and r['finish_reason']=='length','Requires complete generation intervals'
        speed=statistics.median(r['wall_tokens_per_second'] for r in rows)
        first=statistics.median(r['time_to_first_token_ms'] for r in rows)
        if name in ('prompt_2k','prompt_4k'): assert speed>=5,f'{name}: {speed:.3f} tokens/s < 5'
        if latency: assert first<=latency,f'{name}: first-token latency {first/1000:.2f}s exceeds target'
        results[name]=dict(median_wall_tokens_per_second=speed,median_first_token_ms=first)
    assert fixed_sampling(report.get('sampling')),'Requires identified fixed greedy sampling settings'
    return results


def evidence(name):
    data=load(name+'.json');assert data['passed'] is True,f'{name} did not pass'
    if name=='api':
        responses=[c.get('detail',{}) for c in data['checks'] if isinstance(c.get('detail'),dict) and 'zerocool' in c['detail']]
        assert responses,'API evidence lacks real inference statistics'
        for response in responses:
            current(response['zerocool']['after']['metal'].get('build_fingerprint'))
            assert response['zerocool']['after'].get('artifact_revision')==selected_lock['revision'],'API artifact differs'
    elif name in ('moe-replay','attention-replay'): current(data['native'].get('build_fingerprint'))
    else: current(data.get('build_fingerprint'))
    assert data.get('checks'),f'{name} lacks individual checks'
    assert all(c.get('passed') is True for c in data['checks']),f'{name} has failed checks'
    return len(data['checks'])


def soak():
    data=load('soak.json');samples=data['samples']
    current(data.get('build_fingerprint'))
    assert len(samples)>=20 and data['elapsed_seconds']>=1200,'Requires a 20-minute session and minute-by-minute memory samples'
    tail=samples[len(samples)//2:]
    assert max(s['physical_footprint_bytes'] for s in samples)<=22*1024**3
    assert tail[-1]['physical_footprint_bytes']-tail[0]['physical_footprint_bytes']<=128*1024**2,'Progressive process-memory growth'
    assert tail[-1]['system_swap_used_bytes']-tail[0]['system_swap_used_bytes']<=64*1024**2,'Sustained system swap growth'
    assert data['completed_turns']>=10
    return dict(seconds=data['elapsed_seconds'],turns=data['completed_turns'])


def coding():
    data=load('coding.json');assert data['passed'] and data['independent_tests_exit_code']==0
    current(data.get('build_fingerprint'))
    assert {'read','edit','test','recovery'}<=set(data['observed_actions'])
    assert data['tool_trace'] and data['independent_test_output']
    return data['observed_actions']


for name,fn in [('checkpoint',identity),('full_model_logits',logits),('performance',performance),('api',lambda:evidence('api')),
                ('operator_parity',lambda:evidence('operator-parity')),('cache_invariance',lambda:evidence('cache-invariance')),
                ('moe_replay',lambda:evidence('moe-replay')),('attention_replay',lambda:evidence('attention-replay')),
                ('soak',soak),('coding',coding)]: check(name,fn)
result=dict(passed=all(c['passed'] for c in checks),build_fingerprint=current_build,
            artifact=args.artifact,artifact_revision=selected_lock['revision'],checks=checks)
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)
