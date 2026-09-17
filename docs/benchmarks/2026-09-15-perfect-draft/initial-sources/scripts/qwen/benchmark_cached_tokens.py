#!/usr/bin/env python3
"""Bounded full-token replay; never accepted as normal inference throughput."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import struct
from build_identity import build_fingerprint

ROOT=Path(__file__).resolve().parents[2]


def validate(report,build,revision,prompt_tokens,repetitions):
    if report.get('profiling_enabled',False):raise ValueError('Instrumented replay cannot qualify cached timing')
    if report.get('kind')!='cached_full_token_replay' or report.get('passed') is not True or report.get('normal_request_latency_qualified') is not False:
        raise ValueError('Not a successful cached diagnostic')
    if report['artifact_revision']!=revision or report['prompt_tokens']!=prompt_tokens or len(report['runs'])!=repetitions:
        raise ValueError('Wrong artifact, context or repetition coverage')
    for row in report['runs']:
        if row.get('exact') is not True or row['ready_hits']!=480 or row['application_read_bytes']!=0:
            raise ValueError('Replay changed arithmetic or issued reads')
        for state in (row['before'],row['after']):
            if state['metal']['build_fingerprint']!=build or state['diagnostic_stream_trunk'] or state['memory_plan']['expert_slots']!=480:
                raise ValueError('Wrong build or incomplete working set')
            if state['memory_plan']['limit_bytes']!=12*1024**3 or state['memory_plan']['snapshot_bytes']!=report['snapshot_bytes']:
                raise ValueError('Diagnostic budget or snapshot changed')
        if row['before']['ngram_misses']!=row['after']['ngram_misses']:
            raise ValueError('Ngram read during replay')
    return True


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    source=json.loads((ROOT/'docs/benchmarks/2026-09-08-validation/workload-2k.json').read_text())[0]['tokens']
    lock=json.loads((ROOT/('models.lock.json' if args.artifact=='q4-control' else 'mixed-models.lock.json')).read_text())
    build=build_fingerprint(ROOT);reports=[]
    for n in args.contexts:
        tokens=(source*((n+len(source)-1)//len(source)))[:n]+[760]
        inp=args.output/f'{n}-tokens.json';out=args.output/f'{n}.json';inp.write_text(json.dumps(tokens)+'\n')
        with (args.output/f'{n}.log').open('w') as log:
            subprocess.run([str(args.binary),'bench','--cached-token-replay','--model',str(args.model),'--artifact',args.artifact,
                '--prepared',str(args.prepared),'--memory-gb','12','--expert-slots','480','--context','8192',
                '--panel','512','--chunk','128','--tokens-file',str(inp),'--residency',args.residency,
                '--decode-path',args.decode_path,'--kernel-policy',args.kernel_policy,'--gdn-path',args.gdn_path,
                '--token-tile',str(args.token_tile),'--affine-rows',str(args.affine_rows),'--q8-decode-rows',str(args.q8_decode_rows),'--gate-pair',args.gate_pair,'--ready-group',str(args.ready_group),'--repetitions','5','--json',str(out)],check=True,
                stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
        report=json.loads(out.read_text());validate(report,build,lock['revision'],n,5)
        if report['input_sha256']!=hashlib.sha256(struct.pack('<'+'i'*len(tokens),*tokens)).hexdigest():raise ValueError('Input identity differs')
        for row in report['runs']:
            for snapshot in (row['before'],row['after']):
                if snapshot['execution']['residency']!=args.residency or snapshot['execution']['decode_path']!=args.decode_path:raise ValueError('Execution variant differs')
                kernels=snapshot['metal']['kernels']
                if kernels.get('q8_decode_rows',0)!=args.q8_decode_rows:raise ValueError('Cached Q8 decode variant differs')
                if kernels['policy']!=args.kernel_policy or kernels['gdn']!=args.gdn_path or kernels['token_tile']!=args.token_tile or kernels.get('affine_rows',1)!=args.affine_rows or kernels.get('gate_pair',False)!=(args.gate_pair=='on') or snapshot['ready_group']!=args.ready_group:raise ValueError('Cached kernel configuration differs')
        reports.append(dict(prompt_tokens=n,report=out.name,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                            forward_ms=[r['forward_ns']/1e6 for r in report['runs']]))
        print(json.dumps(reports[-1]),flush=True)
    summary=dict(complete=True,build_fingerprint=build,artifact_revision=lock['revision'],reports=reports,
                 normal_request_latency_qualified=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--contexts',type=int,nargs='+',choices=[2048,4096,7168],default=[2048,4096,7168])
    ap.add_argument('--artifact',choices=['q4-control','mixed-4_8bit'],default='mixed-4_8bit')
    ap.add_argument('--model',type=Path,default=ROOT/'.cache/qwen-mixed-reference')
    ap.add_argument('--prepared',type=Path,default=ROOT/'.cache/prepared/q4-records-v1')
    ap.add_argument('--binary',type=Path,default=ROOT/'build/qwen/bin/freellm')
    ap.add_argument('--residency',choices=['off','core','core-cache'],default='off')
    ap.add_argument('--decode-path',choices=['reference','direct','grouped'],default='reference')
    ap.add_argument('--kernel-policy',choices=['reference','candidate'],default='reference')
    ap.add_argument('--gdn-path',choices=['original','precompute'],default='original')
    ap.add_argument('--token-tile',type=int,choices=[1,2,4,8],default=1)
    ap.add_argument('--affine-rows',type=int,choices=[1,2,4],default=1)
    ap.add_argument('--q8-decode-rows',type=int,choices=[0,2,4,8],default=0)
    ap.add_argument('--gate-pair',choices=['off','on'],default='off')
    ap.add_argument('--ready-group',type=int,choices=[1,2,4,8],default=4)
    ap.add_argument('--timeout',type=int,default=21600)
    run(ap.parse_args())
