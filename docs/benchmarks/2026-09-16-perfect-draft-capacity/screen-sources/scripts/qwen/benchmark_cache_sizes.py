#!/usr/bin/env python3
"""Separate 12/18GiB normal-request cache screen, with identical candidate kernels."""
import argparse
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import benchmark_exact as bench


def run(args):
    args.output.mkdir(parents=True,exist_ok=args.resume)
    config=bench.configurations()[2]
    config_path=args.output/"configuration.json"
    if config_path.exists() and json.loads(config_path.read_text())!=[config]:raise ValueError("Cannot resume a different configuration")
    config_path.write_text(json.dumps([config],indent=2)+"\n")
    expected={};measurements=[]
    try:
        for budget in (12,18):
            dest=args.output/f"{budget}g"
            options=SimpleNamespace(binary=args.binary,artifact=args.artifact,model=args.model,
                prepared=args.prepared,workload=args.workload,config=config_path,output=dest,
                mode="screen",pairs=1,memory_gb=budget,cases=["prompt_2k"],include_7k=False,timeout=7200)
            native_path=dest/f"{config['name']}-prompt_2k-0.json"
            reused=args.resume and native_path.exists()
            if not reused:bench.run(options)
            lock=json.loads((bench.ROOT/("models.lock.json" if args.artifact=="q4-control" else "mixed-models.lock.json")).read_text())
            if not expected:expected=dict(build=bench.build_fingerprint(bench.ROOT),revision=lock["revision"])
            report=json.loads(native_path.read_text())
            row=bench.validate(report,config,expected,budget*1024**3,64)
            measurements.append(dict(budget_gib=budget,revalidated_existing_report=reused,
                native_report_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),**row))
            print(f"{budget}GiB validated: {row['ttft_ms']/1000:.2f}s TTFT, {row['tokens_per_second']:.3f} tokens/s",flush=True)
            (args.output/"progress.json").write_text(json.dumps(dict(complete=False,measurements=measurements),indent=2)+"\n")
        result=dict(complete=True,kind="cache_capacity_screen",build=expected["build"],
            artifact_revision=expected["revision"],configuration=config,measurements=measurements,
            generated_tokens_equal=True,performance_qualified=False,
            note="One normal 2K+64 request per budget, in 12/18GiB order. No paired confidence claim.")
    except Exception as error:
        result=dict(complete=False,error=str(error),measurements=measurements,performance_qualified=False)
        (args.output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
        raise
    (args.output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--resume",action="store_true",help="Revalidate complete native reports against current sources before resuming missing budgets")
    ap.add_argument("--artifact",choices=["q4-control","mixed-4_8bit"],default="mixed-4_8bit")
    ap.add_argument("--model",type=Path,default=bench.ROOT/".cache/qwen-mixed-reference")
    ap.add_argument("--prepared",type=Path,default=bench.ROOT/".cache/prepared/q4-records-v1")
    ap.add_argument("--binary",type=Path,default=bench.ROOT/"build/qwen/bin/freellm")
    ap.add_argument("--workload",type=Path,default=bench.ROOT/"docs/benchmarks/2026-09-08-validation/workload-2k.json")
    run(ap.parse_args())
