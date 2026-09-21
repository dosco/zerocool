#!/usr/bin/env python3
"""Finite residency/decode/prefill experiments using the normal-request gates."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import benchmark_exact as bench


def configurations(experiment,base):
    base=dict(base,name='reference')
    if experiment=='residency':
        return [dict(base,name=mode,residency=mode) for mode in ('off','core','core-cache')]
    if experiment=='decode':
        return [dict(base,decode_path='reference'),dict(base,name='direct',decode_path='direct')]+[
            dict(base,name=f'grouped-{n}',decode_path='grouped',ready_group=n) for n in (1,2,4,8)]
    if experiment=='prefill':
        return [dict(base,prefill_pipeline='serial'),dict(base,name='double',prefill_pipeline='double')]
    raise ValueError('Unknown experiment')


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    base=json.loads(args.base_config.read_text()) if args.base_config else bench.configurations()[0]
    configs=configurations(args.experiment,base)
    if args.experiment=='residency':
        capacities=[]
        for c in configs:
            try:
                out=bench.inspect_admission(args.binary,
                    ['--model',str(args.model),'--artifact',args.artifact,'--prepared',str(args.prepared),
                     '--memory-gb',str(args.memory_gb),'--context','8192',*bench.config_args(c)],
                    args.output/f"{c['name']}-setup",int(args.memory_gb*1024**3),c['panel'],None)
            except Exception as error:
                (args.output/'setup-error.json').write_text(json.dumps(dict(complete=False,
                    configuration=c['name'],error=str(error)),indent=2)+'\n')
                raise
            plan=json.loads(out.read_text())['current_admission']
            capacities.append(plan['expert_slots'])
        for c in configs:c['expert_slots']=min(capacities)
    config=args.output/'configurations.json';config.write_text(json.dumps(configs,indent=2)+'\n')
    bench.run(SimpleNamespace(binary=args.binary,artifact=args.artifact,model=args.model,prepared=args.prepared,
        workload=args.workload,config=config,output=args.output/'normal',mode=args.mode,pairs=args.pairs,
        memory_gb=args.memory_gb,cases=args.cases,include_7k=False,timeout=args.timeout))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment',choices=['residency','decode','prefill'],required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--base-config',type=Path)
    ap.add_argument('--memory-gb',type=float,default=12);ap.add_argument('--mode',choices=['screen','paired'],default='screen')
    ap.add_argument('--pairs',type=int,default=1)
    ap.add_argument('--cases',nargs='+',choices=['prompt_2k','prompt_4k','append_128','prompt_7k'])
    ap.add_argument('--artifact',choices=['q4-control','mixed-4_8bit'],default='mixed-4_8bit')
    ap.add_argument('--model',type=Path,default=bench.ROOT/'.cache/qwen-mixed-reference')
    ap.add_argument('--prepared',type=Path,default=bench.ROOT/'.cache/prepared/q4-records-v1')
    ap.add_argument('--binary',type=Path,default=bench.ROOT/'build/qwen/bin/zerocool')
    ap.add_argument('--workload',type=Path,default=bench.ROOT/'docs/benchmarks/2026-09-08-validation/workload-2k.json')
    ap.add_argument('--timeout',type=int,default=7200)
    run(ap.parse_args())
