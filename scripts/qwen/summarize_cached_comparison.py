#!/usr/bin/env python3
"""Validate alternating, zero-read full-token control/candidate comparisons."""
import argparse
import json
from pathlib import Path
import statistics
from benchmark_cached_tokens import validate
from benchmark_exact import paired_interval


def summarize(report):
    if not report.get('paired_comparison') or report.get('profiling_enabled') or not report.get('runs'):
        raise ValueError('Requires an unprofiled paired cached comparison')
    runs=report['runs'];n=len(runs)//2
    if len(runs)%2 or n<5:raise ValueError('Missing cached pairs')
    build=runs[0]['before']['metal']['build_fingerprint'];selected={}
    for i,row in enumerate(runs):
        rep=i//2;variant='control' if (i%2+rep%2)%2==0 else 'candidate'
        if row['repetition']!=rep or row['variant']!=variant:raise ValueError('Pairs are incomplete or not alternating')
        if row['forward_ns']<=0:raise ValueError('Invalid cached timing')
        for state in (row['before'],row['after']):
            execution={key:state[key] for key in ('execution','memory_plan','ready_group','chunk_tokens','io_workers','short_append_tokens')}
            if 'execution' in selected and execution!=selected['execution']:
                raise ValueError('Comparison changed execution or memory allocation')
            selected['execution']=execution
        kernels=row['after']['metal']['kernels'];before=row['before']['metal']['kernels']
        if kernels!=before:raise ValueError('Kernel changed during forward')
        config=dict(kernels);rows=config.pop('q8_decode_rows',0)
        if (variant=='control' and rows!=0) or (variant=='candidate' and rows not in (2,4,8)):
            raise ValueError('Unexpected cached variant')
        if 'config' in selected and config!=selected['config']:raise ValueError('Comparison changed other kernels')
        selected['config']=config
        if variant=='candidate':
            if 'rows' in selected and rows!=selected['rows']:raise ValueError('Candidate changed between pairs')
            selected['rows']=rows
    by_variant={key:[r for r in runs if r['variant']==key] for key in ('control','candidate')}
    for rows in by_variant.values():
        validate(dict(report,runs=rows),build,report['artifact_revision'],report['prompt_tokens'],n)
    a=[r['forward_ns']/1e6 for r in by_variant['control']]
    b=[r['forward_ns']/1e6 for r in by_variant['candidate']]
    return dict(kind='paired_cached_decode_comparison',build_fingerprint=build,artifact_revision=report['artifact_revision'],
                prompt_tokens=report['prompt_tokens'],q8_decode_rows=selected['rows'],pairs=n,
                control_ms=a,candidate_ms=b,control_median_ms=statistics.median(a),candidate_median_ms=statistics.median(b),
                latency_ratio=paired_interval([y/x for x,y in zip(a,b)]),exact=True,application_read_bytes=0,
                normal_request_latency_qualified=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('report',type=Path);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();result=summarize(json.loads(args.report.read_text()))
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
