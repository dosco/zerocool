#!/usr/bin/env python3
"""Compile and capture the isolated scratch-pool probe; never qualifies inference."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess

from build_identity import build_fingerprint
from qualification_evidence import EvidenceGuard, ResourceBlocked, save, seal, sha

ROOT=Path(__file__).resolve().parents[2]


def analyze(raw):
    if (raw.get('kind')!='buffer_reuse_probe_v1' or raw.get('complete') is not True or
        raw.get('synthetic') is not True or len(raw.get('pairs',[]))!=5):
        raise ValueError('Requires five complete isolated pairs')
    rows=[];clean=True
    for i,pair in enumerate(raw['pairs']):
        if pair.get('pair')!=i or pair.get('order')!=(['pool','control'] if i%2 else ['control','pool']):
            raise ValueError('Unexpected pair order')
        means={}
        for name in ('control','pool'):
            arm=pair[name];samples=arm['warm']
            if len(samples)!=8:raise ValueError('Incomplete warm samples')
            means[name]=statistics.mean(s['elapsed_ns']/1e6 for s in samples)
            for s in [arm['cold'],arm['warmup'],*samples]:
                if s['elapsed_ns']<=0 or s['validated_elements']!=15138816:raise ValueError('Invalid probe iteration')
                a,b=s['process_before'],s['process_after']
                # Missing measurements remain inconclusive, never zero disturbance.
                clean &= (a.get('compressed_bytes')==0 and b.get('compressed_bytes')==0 and
                    type(a.get('decompressions')) is int and a['decompressions']==b.get('decompressions'))
            for s in samples:
                if s['allocation_count']!=(0 if name=='pool' else 3072) or s['scratch_reuses']!=(3072 if name=='pool' else 0):
                    raise ValueError('Unexpected physical allocation/reuse count')
        rows.append(dict(pair=i,control_ms=means['control'],pool_ms=means['pool'],saved_ms=means['control']-means['pool']))
    median=statistics.median(r['saved_ms'] for r in rows)
    return dict(kind='buffer_reuse_probe_analysis_v1',complete=True,synthetic=True,pairs=rows,
        median_saved_ms_per_synthetic_iteration=median,clean_memory_observations=bool(clean),
        material_isolated_opportunity=bool(clean and median>=20 and all(r['saved_ms']>0 for r in rows)),
        normal_request_latency_qualified=False,production_promoted=False,
        limitations=['Synthetic 3072-buffer workload; no model, SSD reads, actual allocation sequence or token-latency prediction.',
            'Cold/warmup iterations and pool release are recorded separately from warm means.',
            'Five alternating pairs in one process; no inference qualification confidence bound.'])


def run(out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=out.with_suffix('.probe')
    report=dict(kind='buffer_reuse_probe_capture_v1',complete=False,status='running',synthetic=True,
        normal_request_latency_qualified=False,production_promoted=False)
    try:
        if binary.exists():raise ValueError('Probe binary already exists')
        command=['clang++','-std=c++23','-O2','-Wall','-Wextra','-Werror','-Iinclude','-Ibuild/_deps/json-src/include',
            'scripts/qwen/probe_buffer_reuse.cpp','build/qwen/libfreellm_lib.a',
            '-framework','Metal','-framework','Foundation','-framework','IOKit','-o',str(binary)]
        save(out/'compile-command.json',command)
        with (out/'compile.log').open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=30,check=True)
        paths=[*sorted((ROOT/'scripts/qwen').glob('*.py')),ROOT/'scripts/qwen/probe_buffer_reuse.cpp',ROOT/'build/qwen/libfreellm_lib.a',binary]
        frozen=dict(root=str(ROOT),build=build_fingerprint(ROOT),files={str(p):sha(p) for p in paths},assets={})
        save(out/'identity.json',frozen);guard=EvidenceGuard(frozen,out)
        env=dict(os.environ)
        for k in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION'):env.pop(k,None)
        with (ROOT/'.cache/qwen-qualification.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ResourceBlocked('Another workload owns the GPU lease')
            with (out/'probe.log').open('w') as log:guard.run([binary,out/'probe.json'],stdout=log,timeout=120,env=env)
        raw=json.loads((out/'probe.json').read_text())
        if raw['build']!=frozen['build'] or raw['device']!='Apple M1 Pro' or raw['physical_bytes']!=32*1024**3:
            raise ValueError('Unexpected native build or machine')
        analysis=analyze(raw);save(out/'analysis.json',analysis)
        report.update(complete=True,status='captured',build=frozen['build'],source='probe.json',sha256=sha(out/'probe.json'),
            material_isolated_opportunity=analysis['material_isolated_opportunity'])
    except ResourceBlocked as e:report.update(status='resource_blocked',error=str(e))
    except subprocess.TimeoutExpired:report.update(status='time_budget_exhausted')
    except KeyboardInterrupt:report.update(status='interrupted')
    except Exception as e:report.update(status='failed',error=str(e))
    finally:save(out/'summary.json',report);seal(out);print(json.dumps(report),flush=True)
    return 0 if report['complete'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(p.parse_args().output))
