#!/usr/bin/env python3
"""Compare packed-Q8 decode to the existing two-row kernel on captured inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from benchmark_exact import paired_interval
from build_identity import build_fingerprint

ROOT=Path(__file__).resolve().parents[2]


def summarize(report, rows, repetitions):
    if report.get('kind')!='captured_q8_decode_screen' or report.get('exact') is not True:
        raise ValueError('Requires exact Q8 operator evidence')
    groups={}
    for row in report['measurements']:
        if row.get('exact') is not True or row['wall_ns']<=0:raise ValueError('Invalid operator measurement')
        key=(json.dumps(row['matrix'],sort_keys=True),row['case'])
        values=groups.setdefault(key,{})
        identity=(row['q8_decode_rows'],row['repetition'])
        if identity in values:raise ValueError('Duplicate operator measurement')
        values[identity]=row['wall_ns']
    if not groups:raise ValueError('Missing Q8 decode shapes')
    result=[]
    expected={(r,i) for r in (0,rows) for i in range(repetitions)}
    for (shape,case),values in groups.items():
        if set(values)!=expected:raise ValueError('Incomplete or changed Q8 variants')
        a=[values[0,i] for i in range(repetitions)];b=[values[rows,i] for i in range(repetitions)]
        result.append(dict(shape=json.loads(shape),case=case,q8_decode_rows=rows,
                           bounds=paired_interval([y/x for x,y in zip(a,b)]),
                           baseline_ms=statistics.median(a)/1e6,candidate_ms=statistics.median(b)/1e6))
    return result


def run(args):
    args.output.mkdir(parents=True,exist_ok=False);build=build_fingerprint(ROOT)
    result=dict(build_fingerprint=build,complete=False,normal_request_latency_qualified=False,reports=[])
    try:
        for rows in args.rows:
            for i,manifest in enumerate(args.manifests):
                stem=args.output/f'fixture-{i}-r{rows}';dest=stem.with_suffix('.json')
                with stem.with_suffix('.log').open('w') as log:
                    subprocess.run([str(args.binary),'bench','--artifact',args.artifact,'--operator-fixtures',str(manifest),
                                    '--kernel-policy','candidate','--q8-decode-rows',str(rows),
                                    '--repetitions',str(args.repetitions),'--json',str(dest)],
                                   check=True,stdout=log,stderr=subprocess.STDOUT,timeout=300)
                report=json.loads(dest.read_text())
                if report['build_fingerprint']!=build:raise ValueError('Native build differs')
                result['reports'].append(dict(path=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                                               cases=summarize(report,rows,args.repetitions)))
        result['complete']=True
    except Exception as error:
        result['error']=str(error);raise
    finally:(args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifests',type=Path,nargs='+');ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--rows',type=int,choices=[2,4,8],nargs='+',default=[2,4,8])
    ap.add_argument('--repetitions',type=int,choices=[5,10],default=5)
    ap.add_argument('--binary',type=Path,default=ROOT/'build/qwen/bin/freellm')
    ap.add_argument('--artifact',choices=['q4-control','mixed-4_8bit'],default='mixed-4_8bit')
    run(ap.parse_args())
