#!/usr/bin/env python3
"""Check a completed native memory soak; this does not score coding quality."""
import argparse
import json
from pathlib import Path
from benchmark_exact import validate_phase_memory


def evaluate(report):
    rows=report.get("runs",[])
    if report.get("complete") is not True or report.get("benchmark_elapsed_ns",0)<1200*10**9 or not rows:
        raise ValueError("A complete 20-minute native workload is required")
    identity=None;points=[];elapsed=0
    for row in rows:
        s=row["after"];machine=s["metal"];plan=s["memory_plan"]
        key=(machine["build_fingerprint"],s["artifact_revision"],plan["limit_bytes"],plan["expert_slots"])
        if identity is None:identity=key
        if key!=identity or s["diagnostic_stream_trunk"] or row.get("profiling_enabled"):
            raise ValueError("Build, budget, cache capacity, or execution mode changed")
        execution=s.get('execution',{})
        if s.get('phase_memory',{}).get('pressure_resizes',0):raise ValueError('Emergency pressure reduction during soak')
        if execution.get('phase_memory','fixed')=='reclaim':
            validate_phase_memory(row,dict(phase_memory='reclaim',prefill_pipeline=execution.get('prefill_pipeline'),chunk=s['chunk_tokens']),plan['limit_bytes'])
        if machine["device"]!="Apple M1 Pro" or machine["physical_bytes"]!=32*1024**3:
            raise ValueError("Wrong machine")
        if s["process"]["physical_footprint_bytes"]>plan["limit_bytes"]:
            raise ValueError("Process exceeded the admitted budget")
        elapsed+=row["request_ms"]/1000
        points.append((elapsed,s["process"]["physical_footprint_bytes"],s["process"]["system_swap_used_bytes"]))
    tail=[p for p in points if p[0]>=points[-1][0]-600]
    if len(tail)<3:raise ValueError("Insufficient samples in the final ten minutes")
    mx=sum(p[0] for p in tail)/len(tail);my=sum(p[1] for p in tail)/len(tail)
    denom=sum((p[0]-mx)**2 for p in tail)
    if denom==0:raise ValueError("Invalid time samples")
    projected_growth=max(0,sum((p[0]-mx)*(p[1]-my) for p in tail)/denom)*600
    swap_growth=points[-1][2]-points[0][2]
    passed=projected_growth<=64*1024**2 and swap_growth<=64*1024**2
    return dict(passed=passed,kind="controlled_coding_workload_memory_soak",elapsed_seconds=report["benchmark_elapsed_ns"]/1e9,
                build_fingerprint=identity[0],artifact_revision=identity[1],budget_bytes=identity[2],
                projected_10min_footprint_growth_bytes=projected_growth,observed_system_swap_growth_bytes=swap_growth,
                independently_checked_coding_quality=False,
                note="System swap includes other processes. Growth thresholds are 64MiB per final ten minutes and 64MiB total swap.")

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("report",type=Path);ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args();result=evaluate(json.loads(args.report.read_text()));args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2));raise SystemExit(0 if result["passed"] else 1)
