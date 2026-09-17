#!/usr/bin/env python3
"""Split observed GPU-idle submission intervals using Metal's driver timestamps.

These are scheduling observations, never an estimate of removable runtime work.
"""
import argparse
import json
import math
from pathlib import Path
import statistics

from qualification_evidence import save, sha, verify_seal
from stage200 import clean_memory, profile_row


def split(begin,end,groups):
    if not 0<=begin<end:raise ValueError('Invalid forward interval')
    values=[]
    for g in groups:
        v=[g['submitted_ns'],g['commit_returned_ns'],g['driver_start_seconds']*1e9,
           g['driver_end_seconds']*1e9,g['gpu_start_seconds']*1e9,g['gpu_end_seconds']*1e9,g['completed_ns']]
        if any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in v):raise ValueError('Missing driver clock')
        s,c,ds,de,gs,ge,done=v
        if not s<=ds<=de<=done or not s<=gs<=ge<=done or c<s:raise ValueError('Reversed driver clock')
        values.append(v)
    cuts=sorted({begin,end,*[min(end,max(begin,x)) for v in values for x in v]})
    totals=dict.fromkeys(('before_driver','in_driver','after_driver','commit_api_overlap'),0.0)
    for a,b in zip(cuts,cuts[1:]):
        mid=(a+b)/2
        if any(v[4]<=mid<v[5] for v in values):continue
        pending=[v for v in values if v[0]<=mid<v[4]]
        if not pending:continue
        # Queue order identifies the next group to execute. Later groups can be
        # scheduling concurrently and must not double-count this idle interval.
        s,c,ds,de,gs,ge,done=min(pending)
        key='before_driver' if mid<ds else 'in_driver' if mid<de else 'after_driver'
        totals[key]+=(b-a)/1e6
        if mid<c:totals['commit_api_overlap']+=(b-a)/1e6
    return dict(totals,command_groups=len(groups),total_commit_api_ms=sum(v[1]-v[0] for v in values)/1e6)


def analyze(directory):
    directory=Path(directory);verify_seal(directory,sha(directory/'evidence-files.json'))
    read=lambda name:json.loads((directory/name).read_text())
    summary=read('summary.json');raw=read('traced.json');normal=read('normal.json');profile=read('commands.json')
    if summary.get('complete') is not True:raise ValueError('Incomplete capture')
    deps=[json.loads(line) for line in (directory/'dependencies.jsonl').read_text().splitlines()]
    phases=[]
    for i,row in enumerate(raw['runs']):
        joined=profile_row(raw,i,profile,deps);tokens=[]
        for sample,base in zip(row['decode_diagnostics']['samples'],joined['tokens']):
            groups=[g for g in profile['command_groups'] if any(o['request_phase']=='decode' and o['offset']==sample['offset'] for o in g['operations'])]
            result=split(sample['begin_ns'],sample['end_ns'],groups)
            idle=sum(result[k] for k in ('before_driver','in_driver','after_driver'))
            if not math.isclose(idle,base['buckets_ms']['gpu_idle_submitted'],abs_tol=.0001):raise ValueError('Idle split does not reconcile')
            tokens.append(dict(offset=sample['offset'],**result))
        phases.append(dict(name=row['name'],captured_tokens=len(tokens),tokens=tokens,
            means={k:statistics.mean(t[k] for t in tokens) for k in tokens[0] if k!='offset'}))
    return dict(kind='submission_driver_timeline_v1',complete=True,build=joined['build'],
        source_seal_sha256=sha(directory/'evidence-files.json'),normal_clean_memory=clean_memory(normal),
        traced_clean_memory=clean_memory(raw),phases=phases,normal_request_latency_qualified=False,
        limitations=['Driver timestamps measure CPU scheduling; GPU timestamps may contain device stalls or preemption.',
            'The first three means partition GPU-idle submitted time. Commit API overlap and total commit time are not additive.',
            'Memory compression or host variation can affect these observations; no causal savings prediction follows.',
            'Source capture preserves full decode coverage and original command groups; profiling still adds overhead.'],
        timestamp_documentation='https://developer.apple.com/documentation/metal/mtlcommandbuffer/kernelstarttime')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('capture',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();save(args.output,analyze(args.capture))
