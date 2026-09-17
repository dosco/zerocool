#!/usr/bin/env python3
"""Validate alternating zero-read comparisons with one declared changed axis."""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics
from benchmark_cached_tokens import validate
from benchmark_exact import paired_interval


def summarize(report):
    if not report.get('paired_comparison') or report.get('profiling_enabled') or not report.get('runs'):
        raise ValueError('Requires an unprofiled paired cached comparison')
    axis=report.get('comparison_axis','q8_decode_rows')
    axes={'q8_decode_rows':(0,(2,4,8)), 'sparse_selection':('cpu',('gpu',)),
          'attention_score_tiles':('full',('skip-masked',))}
    if axis not in axes:raise ValueError('Unknown comparison axis')
    control, candidates=axes[axis]
    runs=report['runs'];n=len(runs)//2
    if len(runs)%2 or n<5:raise ValueError('Missing cached pairs')
    build=runs[0]['before']['metal']['build_fingerprint'];selected={}
    for i,row in enumerate(runs):
        rep=i//2;variant='control' if (i%2+rep%2)%2==0 else 'candidate'
        if row['repetition']!=rep or row['variant']!=variant:raise ValueError('Pairs are incomplete or not alternating')
        elapsed=row.get('forward_ns')
        if isinstance(elapsed,bool) or not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed) or elapsed<=0:
            raise ValueError('Cached timing must be positive and finite')
        def configuration(state):
            result={key:copy.deepcopy(state.get(key)) for key in
                    ('execution','memory_plan','completion_pipeline','ready_group','chunk_tokens','io_workers','short_append_tokens')}
            result['kernels']=copy.deepcopy(state['metal']['kernels'])
            return result
        before=configuration(row['before']);after=configuration(row['after'])
        if before['kernels'].get('profile') or before['kernels'].get('counter_profile'):
            raise ValueError('Instrumented kernels cannot qualify cached timing')
        if before!=after:raise ValueError('Configuration changed during forward')
        container=before['execution'] if axis=='sparse_selection' else before['kernels']
        value=container.pop(axis,control)
        if (variant=='control' and value!=control) or (variant=='candidate' and value not in candidates):
            raise ValueError('Unexpected cached variant')
        if 'config' in selected and before!=selected['config']:raise ValueError('Comparison changed other configuration')
        selected['config']=before
        if variant=='candidate':
            if 'value' in selected and value!=selected['value']:raise ValueError('Candidate changed between pairs')
            selected['value']=value
    by_variant={key:[r for r in runs if r['variant']==key] for key in ('control','candidate')}
    for rows in by_variant.values():
        validate(dict(report,runs=rows),build,report['artifact_revision'],report['prompt_tokens'],n)
    a=[r['forward_ns']/1e6 for r in by_variant['control']]
    b=[r['forward_ns']/1e6 for r in by_variant['candidate']]
    result=dict(kind='paired_cached_decode_comparison',build_fingerprint=build,artifact_revision=report['artifact_revision'],
                comparison_axis=axis,control_value=control,candidate_value=selected['value'],
                prompt_tokens=report['prompt_tokens'],pairs=n,
                control_ms=a,candidate_ms=b,control_median_ms=statistics.median(a),candidate_median_ms=statistics.median(b),
                latency_ratio=paired_interval([y/x for x,y in zip(a,b)]),exact=True,application_read_bytes=0,
                normal_request_latency_qualified=False)
    if axis=='q8_decode_rows':result['q8_decode_rows']=selected['value']
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('report',type=Path);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();result=summarize(json.loads(args.report.read_text()))
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
