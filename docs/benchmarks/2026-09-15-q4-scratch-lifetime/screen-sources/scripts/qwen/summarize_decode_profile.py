#!/usr/bin/env python3
"""Attribute an instrumented, exact, zero-read cached token to GPU passes."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import statistics


def summarize(report, profile):
    runs=report.get('runs',[])
    if (report.get('kind')!='cached_full_token_replay' or report.get('passed') is not True
            or report.get('profiling_enabled') is not True or not runs or profile.get('truncated')):
        raise ValueError('Requires complete instrumented cached-token evidence')
    identity=None
    for run in runs:
        if run.get('exact') is not True or run['ready_hits']!=480 or run['application_read_bytes']!=0:
            raise ValueError('Changed arithmetic or uncached model data')
        for state in (run['before'],run['after']):
            key=(state['metal']['build_fingerprint'],state['artifact_revision'],state['memory_plan']['limit_bytes'])
            if identity is None:identity=key
            if key!=identity or not state['metal']['kernels'].get('counter_profile'):
                raise ValueError('Profiling identity differs')
    stages=defaultdict(int);kernels=defaultdict(int);layers=defaultdict(int);counts=defaultdict(int)
    commands=0;dispatches=0
    for command in profile['command_groups']:
        used=False
        for op in command['operations']:
            if op['request_phase']!='cached_replay':continue
            ns=op.get('gpu_pass_ns')
            if not isinstance(ns,int) or ns<=0:raise ValueError('Missing GPU pass timestamp')
            stages[op['stage']]+=ns;kernels[op['kernel']]+=ns
            layers[op['layer']]+=ns;counts[op['kernel']]+=1;dispatches+=1;used=True
        commands+=used
    expected=sum(r['after']['metal']['dispatches']-r['before']['metal']['dispatches'] for r in runs)
    if not dispatches or dispatches!=expected:raise ValueError('Missing dispatch coverage')
    n=len(runs)
    def rows(values):
        return [dict(name=k,mean_ms_per_token=v/n/1e6) for k,v in sorted(values.items(),key=lambda item:-item[1])]
    return dict(kind='instrumented_cached_decode_attribution',build_fingerprint=identity[0],
                artifact_revision=identity[1],budget_bytes=identity[2],repetitions=n,
                instrumented_forward_median_ms=statistics.median(r['forward_ns']/1e6 for r in runs),
                mean_dispatches=dispatches/n,mean_submissions=commands/n,
                stages=rows(stages),kernels=rows(kernels),layers=rows(layers),
                mean_kernel_counts={k:v/n for k,v in counts.items()},normal_request_latency_qualified=False,
                note='One dispatch per timed compute pass changes encoder overhead. GPU pass sums and CPU waits are not additive wall-time attribution. Confirm changes with unprofiled complete requests.')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('report',type=Path);ap.add_argument('profile',type=Path);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();result=summarize(json.loads(args.report.read_text()),json.loads(args.profile.read_text()))
    result['sources']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.report,args.profile)}
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
