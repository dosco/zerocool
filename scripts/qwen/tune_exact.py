#!/usr/bin/env python3
"""Finite one-factor screens; never promotes runtime defaults."""
import argparse
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace
import benchmark_exact as bench


def choose(summary):
    grouped={}
    for row in summary["measurements"]:
        grouped.setdefault(row["configuration"],{})[row["name"]]=row
    reference=grouped["baseline"];scores={}
    for name,rows in grouped.items():
        if rows.keys()!=reference.keys():raise ValueError("Unmatched tuning workloads")
        ratios=[rows[case][metric]/reference[case][metric] for case in rows for metric in ["ttft_ms","decode_ms_per_token"]]
        if not all(math.isfinite(x) and x>0 for x in ratios):raise ValueError("Invalid timing")
        scores[name]=statistics.geometric_mean(ratios)
    # Deterministic ties retain the baseline.
    return min(scores,key=lambda name:(scores[name],name!="baseline",name)),scores


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    base=json.loads(args.config.read_text());base["name"]="baseline"
    history=[]
    for axis,values in [("panel",[0,256,512,1024]),("chunk",[32,64,128]),("ready_group",[1,2,4,8]),("io_workers",[2,4,8])]:
        configs=[dict(base,name="baseline")]+[dict(base,**{axis:v,"name":f"{axis}-{v}"}) for v in values if v!=base[axis]]
        config_path=args.output/f"{axis}-config.json";config_path.write_text(json.dumps(configs,indent=2)+"\n")
        stage=args.output/axis
        options=SimpleNamespace(binary=args.binary,artifact=args.artifact,model=args.model,prepared=args.prepared,
            workload=args.workload,config=config_path,output=stage,mode="screen",pairs=1,memory_gb=args.memory_gb,
            cases=["prompt_2k","append_128"],include_7k=False,timeout=7200)
        bench.run(options)
        summary=json.loads((stage/"summary.json").read_text());winner,scores=choose(summary)
        base=next(c for c in configs if c["name"]==winner);base=dict(base,name="baseline")
        history.append(dict(axis=axis,scores=scores,selected=base))
        (args.output/"selection.json").write_text(json.dumps(dict(selected=base,history=history,promoted=False),indent=2)+"\n")
    print(json.dumps(dict(selected=base,requires_combined_recheck_and_paired_qualification=True),indent=2))

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config",type=Path,required=True,help="One exact-validated candidate configuration object")
    ap.add_argument("--output",type=Path,required=True);ap.add_argument("--memory-gb",type=float,required=True)
    ap.add_argument("--artifact",choices=["q4-control","mixed-4_8bit"],default="mixed-4_8bit")
    ap.add_argument("--model",type=Path,default=bench.ROOT/".cache/qwen-mixed-reference")
    ap.add_argument("--prepared",type=Path,default=bench.ROOT/".cache/prepared/q4-records-v1")
    ap.add_argument("--binary",type=Path,default=bench.ROOT/"build/qwen/bin/zerocool")
    ap.add_argument("--workload",type=Path,default=bench.ROOT/"docs/benchmarks/2026-09-08-validation/workload-2k.json")
    run(ap.parse_args())
