#!/usr/bin/env python3
"""All-layer native optimization parity; independent short oracle remains separate."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from build_identity import build_fingerprint

ROOT=Path(__file__).resolve().parents[2]


def compare(reference,candidate,build):
    if not reference.get("passed") or not candidate.get("passed"):
        raise ValueError("State/failure harness failed")
    if not reference["runs"] or len(reference["runs"])!=len(candidate["runs"]):raise ValueError("Different or missing admitted panel coverage")
    for a,b in zip(reference["runs"],candidate["runs"]):
        if a["panel"]!=b["panel"] or a["stages"]!=b["stages"]:
            raise ValueError("Changed logits, route identities, or persistent state")
        for r in (a,b):
            s=r["continued_statistics"]
            if s["metal"]["build_fingerprint"]!=build or s["diagnostic_stream_trunk"] or s["memory_plan"]["panel_tokens"]!=r["panel"]:
                raise ValueError("Wrong build, diagnostic execution, or reduced panel")
            for stage in r["stages"]:
                if len(stage["layers"])!=48 or len(stage["routes"])!=48:raise ValueError("Incomplete layer or route proof")
    return True


def check_prime(report,build,prefix,append):
    if report.get("complete") is not True or len(report.get("runs",[]))!=3:
        raise ValueError("Incomplete prime/append check")
    prime,continued,fresh=report["runs"]
    if prime["finish_reason"]!="primed" or prime["output_tokens"] or prime["prompt_tokens"]!=prefix:
        raise ValueError("Priming sampled a token or ingested the wrong prefix")
    if continued["reused_tokens"]!=prefix or continued["prefill_tokens"]!=append or continued["pending_tokens_ingested"]:
        raise ValueError("Append included unrequested work or failed to retain the prime")
    if fresh["reused_tokens"] or continued["output_token_ids"]!=fresh["output_token_ids"] or continued["output_tokens"]!=2:
        raise ValueError("Primed continuation differs from fresh replay")
    for row in report["runs"]:
        if row["after"]["metal"]["build_fingerprint"]!=build or row["after"]["diagnostic_stream_trunk"]:
            raise ValueError("Wrong build or diagnostic Session execution")
    return continued["output_token_ids"]


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    data=json.loads((ROOT/"docs/benchmarks/2026-09-08-validation/workload-2k.json").read_text())[0]["tokens"]
    if args.case=="short":data=json.loads((ROOT/"tests/fixtures/qwen/mixed-reference-tokens.json").read_text())
    def sized(n):return (data*((n+len(data)-1)//len(data)))[:n]
    n,a={"short":(5,3),"microchunks":(257,129),"boundary":(2053,129),"append":(4096,128),"7k":(7168,128)}[args.case]
    context=256 if args.case=="short" else 8192
    panel=256 if args.case=="short" else args.panel
    base=dict(context=context,chunk=1 if args.case=="short" else args.chunk,memory_gib=args.memory_gb,
              layers=48,artifact=args.artifact,expert_slots=32 if args.case=="short" else 0,
              diagnostic_stream_trunk=False,require_exact_panel=True,panels=[panel],
              prefix=sized(n),append=sized(a),continuation=[760,369])
    candidate=dict(kernel_policy="candidate",token_tile=args.token_tile,
                   gdn_path=args.gdn_path,gdn_rows=args.gdn_rows,gdn_block=args.gdn_block,
                   residency=args.residency,decode_path=args.decode_path,prefill_pipeline=args.prefill_pipeline,
                   ready_group=args.ready_group,phase_memory=args.phase_memory,affine_rows=args.affine_rows,gate_pair=args.gate_pair,q8_decode_rows=args.q8_decode_rows)
    if args.shape_policy:
        if args.token_tile!=1:raise ValueError("Shape policy requires --token-tile 1")
        candidate['shape_policy']=str(args.shape_policy.resolve(strict=True))
    reports=[];prime_outputs=[];build=build_fingerprint(ROOT)
    for name,extra in [("reference",{}),("candidate",candidate)]:
        config=args.output/f"{name}-case.json";dest=args.output/f"{name}.json"
        config.write_text(json.dumps(dict(base,**extra),indent=2)+"\n")
        with (args.output/f"{name}.log").open("w") as log:
            subprocess.run([str(args.binary),str(args.model),str(args.prepared),str(config),str(dest)],
                           check=True,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
        reports.append(json.loads(dest.read_text()))
        if args.case=="short":
            workload=args.output/f"{name}-prime.workload.json";report=args.output/f"{name}-prime.json"
            workload.write_text(json.dumps([dict(name="prime",tokens=sized(n),prime=True),
                dict(name="append",tokens=sized(a),append=True,max_tokens=2),
                dict(name="fresh",tokens=sized(n)+sized(a),max_tokens=2)])+"\n")
            options=[item for key,value in extra.items() for item in ["--"+key.replace("_","-"),str(value)]]
            with (args.output/f"{name}-prime.log").open("w") as log:
                subprocess.run([str(args.runner),"bench","--model",str(args.model),"--artifact",args.artifact,
                    "--prepared",str(args.prepared),"--context","256","--chunk","128","--panel","0",
                    "--expert-slots","32","--memory-gb",str(args.memory_gb),"--repetitions","1",
                    "--workload-file",str(workload),"--json",str(report),*options],
                    check=True,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
            prime_outputs.append(check_prime(json.loads(report.read_text()),build,n,a))
    compare(*reports,build)
    if prime_outputs and prime_outputs[0]!=prime_outputs[1]:raise ValueError("Session candidate changed sampled output")
    summary=dict(passed=True,kind="exact_native_session_parity",build_fingerprint=build,case=args.case,
                 all_48_layers=True,all_logits_and_state_equal=True,all_routes_equal=True,
                 session_prime_checked=bool(prime_outputs),independent_long_context_oracle=False,performance_qualified=False,
                 report_hashes={name:hashlib.sha256((args.output/f"{name}.json").read_bytes()).hexdigest() for name in ["reference","candidate"]})
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case",choices=["short","microchunks","boundary","append","7k"],required=True)
    ap.add_argument("--output",type=Path,required=True);ap.add_argument("--memory-gb",type=float,required=True)
    ap.add_argument("--artifact",choices=["q4-control","mixed-4_8bit"],default="mixed-4_8bit")
    ap.add_argument("--model",type=Path,default=ROOT/".cache/qwen-mixed-reference")
    ap.add_argument("--prepared",type=Path,default=ROOT/".cache/prepared/q4-records-v1")
    ap.add_argument("--binary",type=Path,default=ROOT/"build/qwen/qwen_panel_check")
    ap.add_argument("--runner",type=Path,default=ROOT/"build/qwen/bin/freellm")
    ap.add_argument("--panel",type=int,default=512);ap.add_argument("--chunk",type=int,default=128)
    ap.add_argument("--token-tile",type=int,default=8);ap.add_argument("--gdn-path",default="precompute")
    ap.add_argument("--gdn-rows",type=int,default=4);ap.add_argument("--gdn-block",type=int,default=8)
    ap.add_argument("--residency",choices=["off","core","core-cache"],default="off")
    ap.add_argument("--decode-path",choices=["reference","direct","grouped"],default="reference")
    ap.add_argument("--prefill-pipeline",choices=["serial","double"],default="serial")
    ap.add_argument("--affine-rows",type=int,choices=[1,2,4],default=1)
    ap.add_argument("--q8-decode-rows",type=int,choices=[0,2,4,8],default=0)
    ap.add_argument("--gate-pair",choices=["off","on"],default="off")
    ap.add_argument("--phase-memory",choices=["fixed","reclaim"],default="fixed")
    ap.add_argument("--ready-group",type=int,choices=[1,2,4,8],default=4)
    ap.add_argument("--shape-policy",type=Path)
    ap.add_argument("--timeout",type=int,default=21600)
    run(ap.parse_args())
