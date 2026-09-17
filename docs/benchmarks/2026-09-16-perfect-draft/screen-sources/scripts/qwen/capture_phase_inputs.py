#!/usr/bin/env python3
"""Serial, bounded real-input capture and affine replay with explicit coverage."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from build_identity import build_fingerprint
from benchmark_exact import config_args

ROOT=Path(__file__).resolve().parents[2]
CASES=[('prefill',0,'gdn'),('prefill',30,'gdn'),('prefill',31,'attention'),
       ('append',30,'gdn'),('append',47,'attention'),('decode',0,'gdn'),
       ('decode',30,'gdn'),('decode',31,'attention'),('decode',23,'routed_expert')]


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    seed=json.loads(args.workload.read_text())[0]['tokens']
    # Selection and final evaluation use different token windows, not relabelled payloads.
    offset=0 if args.split=='tuning' else 512
    if len(seed)<offset+386:raise ValueError('Workload requires two disjoint token windows')
    workload=[dict(name='prime',tokens=seed[offset:offset+257],prime=True),
              dict(name='append',tokens=seed[offset+257:offset+386],append=True,max_tokens=2)]
    source=args.output/'workload.json';source.write_text(json.dumps(workload)+'\n')
    config=dict(kernel_policy='candidate',token_tile=8,gdn_path='precompute',panel=512,chunk=128,
                ready_group=2,io_workers=8,residency='core-cache',decode_path='grouped',prefill_pipeline='double',phase_memory='reclaim')
    summary=dict(build=build_fingerprint(ROOT),split=args.split,workload_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 complete=False,normal_request_latency_qualified=False,cases=[])
    try:
        for phase,layer,operator in CASES:
            name=f'{phase}-{layer}-{operator}'
            if args.cases and name not in args.cases:continue
            directory=args.output/name;report=args.output/f'{name}.json'
            with (args.output/f'{name}.log').open('w') as log:
                subprocess.run([str(args.binary),'bench','--model',str(args.model),'--prepared',str(args.prepared),
                    '--artifact',args.artifact,'--memory-gb','12','--context','8192',*config_args(config),
                    '--workload-file',str(source),'--repetitions','1','--operator-capture',str(directory),
                    '--capture-phase',phase,'--capture-layer',str(layer),'--capture-operator',operator,
                    '--json',str(report)],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
                manifest=directory/'manifest.json'
                if not manifest.exists():raise ValueError(f'Missing requested capture coverage: {name}')
                captured=json.loads(manifest.read_text())
                if not captured['cases'] or any(c['phase']!=phase or c['context']['layer']!=layer or c['context']['stage']!=operator for c in captured['cases']):
                    raise ValueError(f'Capture coverage differs: {name}')
                result=args.output/f'{name}-operators.json'
                subprocess.run([str(args.binary),'bench','--artifact',args.artifact,'--operator-fixtures',str(manifest),
                    '--repetitions',str(args.repetitions),'--json',str(result)],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
                summary['cases'].append(dict(name=name,manifest=str(manifest),result=str(result),count=len(captured['cases']),bytes=captured['bytes']))
                (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        if not summary['cases']:raise ValueError('No selected capture cases')
        summary['complete']=True
    except Exception as error:
        summary['error']=str(error);raise
    finally:(args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--binary',type=Path,default=ROOT/'build/qwen/bin/freellm')
    ap.add_argument('--model',type=Path,default=ROOT/'.cache/qwen-mixed-reference')
    ap.add_argument('--prepared',type=Path,default=ROOT/'.cache/prepared/q4-records-v1')
    ap.add_argument('--artifact',choices=['q4-control','mixed-4_8bit'],default='mixed-4_8bit')
    ap.add_argument('--workload',type=Path,default=ROOT/'docs/benchmarks/2026-09-08-validation/workload-2k.json')
    ap.add_argument('--split',choices=['tuning','heldout'],default='tuning')
    ap.add_argument('--cases',nargs='+',choices=[f'{p}-{l}-{o}' for p,l,o in CASES])
    ap.add_argument('--repetitions',type=int,choices=[5,10],default=5)
    ap.add_argument('--timeout',type=int,default=7200)
    run(ap.parse_args())
