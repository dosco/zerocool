#!/usr/bin/env python3
"""Fresh-process exact-kernel comparisons. Screening never qualifies promotion."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import struct
import subprocess
import time
from build_identity import build_fingerprint
from qualification_evidence import ResourceBlocked, confined

ROOT = Path(__file__).resolve().parents[2]


def fixed_sampling(sampling):
    # Options stores top_p as a native float. JSON preserves that binary32
    # value rather than the nearest binary64 representation of decimal 0.95.
    native_top_p=struct.unpack("f",struct.pack("f",0.95))[0]
    return isinstance(sampling,dict) and sampling.get("top_p") in (0.95,native_top_p) and \
        dict(sampling,top_p=0.95)==dict(temperature=0,top_k=20,top_p=0.95,seed=0)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configurations():
    return [dict(name="reference", kernel_policy="reference", token_tile=1, gdn_path="original", panel=512, chunk=128, ready_group=4, io_workers=8),
            dict(name="precompute", kernel_policy="candidate", token_tile=1, gdn_path="precompute", panel=512, chunk=128, ready_group=4, io_workers=8),
            dict(name="tile8-precompute", kernel_policy="candidate", token_tile=8, gdn_path="precompute", panel=512, chunk=128, ready_group=4, io_workers=8)]


def workloads(seed, output, include_7k=False):
    def sized(n):
        return (seed * ((n + len(seed) - 1) // len(seed)))[:n]
    cases = {"prompt_2k": [dict(name="prompt_2k", tokens=sized(2048), max_tokens=output)],
             "prompt_4k": [dict(name="prompt_4k", tokens=sized(4096), max_tokens=output)],
             "append_128": [dict(name="prime_4k", tokens=sized(4096), prime=True),
                            dict(name="append_128", tokens=sized(128), append=True, max_tokens=output)]}
    if include_7k:
        cases["prompt_7k"] = [dict(name="prompt_7k", tokens=sized(7168), max_tokens=output)]
    return cases


def config_args(config):
    allowed = {"memory_pressure_policy", "decode_scratch", "expert_tail", "cache_policy", "kernel_policy", "token_tile", "gdn_path", "gdn_rows", "gdn_block", "panel", "chunk", "ready_group", "io_workers", "residency", "decode_path", "prefill_pipeline", "expert_slots", "shape_policy", "phase_memory", "affine_rows", "gate_pair", "q8_decode_rows", "route_selection", "sparse_selection", "attention_score_tiles"}
    if set(config) - allowed - {"name"}:
        raise ValueError("Unknown experiment option")
    result = []
    for key, value in config.items():
        if key != "name":
            result += ["--" + key.replace("_", "-"), str(value)]
    return result


def validate_phase_memory(row, config, budget):
    policy=config.get("phase_memory","fixed")
    before=row["before"].get("phase_memory",{})
    after=row["after"].get("phase_memory",{})
    if after.get("pressure_resizes",0) != before.get("pressure_resizes",0):
        raise ValueError("Emergency pressure resize invalidates comparison")
    if policy=="fixed":
        if after.get("policy","fixed")!="fixed": raise ValueError("Phase policy changed")
        return
    if policy!="reclaim" or config.get("prefill_pipeline")!="double": raise ValueError("Invalid phase policy")
    for state,phase in [(row["before"],before),(row["after"],after)]:
        if phase.get("policy")!=policy or phase.get("phase")!="generation": raise ValueError("Incomplete phase transition")
        for name in ("prompt_plan","generation_plan"):
            p=phase[name]
            if p["limit_bytes"]!=budget or p["planned_bytes"]>budget: raise ValueError("Phase exceeds budget")
        if state["memory_plan"]!=phase["generation_plan"]: raise ValueError("Generation allocation changed")
    if before["prompt_plan"]!=after["prompt_plan"] or before["generation_plan"]!=after["generation_plan"]:
        raise ValueError("Declared phase plans changed")
    count=before["transition_count"]
    events=[e for e in after["transitions"] if e["sequence"]>count]
    if len(events)!=after["transition_count"]-count: raise ValueError("Missing transition evidence")
    previous=before["generation_plan"]
    for index,e in enumerate(events):
        target="prompt_plan" if index%2==0 else "generation_plan"
        reason="ingest_panel" if index%2==0 else "ingest_complete"
        if e["sequence"]!=count+index+1 or e["reason"]!=reason or e["before"]!=previous or e["after"]!=after[target]:
            raise ValueError("Undeclared allocation transition")
        if e["live_buffer_bytes"]>budget: raise ValueError("Transition exceeds budget")
        if target=="generation_plan" and any(p["allocated_bytes"] for p in e["scratch_after"]):
            raise ValueError("Prompt workspaces retained after reclamation")
        previous=e["after"]
    if len(events)%2: raise ValueError("Unfinished memory transition")
    if 'prefill_tokens' in row:
        needs_panel=after['prompt_plan'].get('panel_tokens',0)>config.get('chunk',128) and row['prefill_tokens']>config.get('chunk',128)
        if len(events)!=(2 if needs_panel else 0): raise ValueError("Missing or unnecessary ingestion transition")


def validate(report, config, expected, budget, output):
    if report.get('gpu_reference_mode', 'off') != 'off' or report.get('gpu_references'):
        raise ValueError('GPU boundary probes cannot qualify normal timing')
    if report.get("complete") is not True or report.get("model_revision") != expected["revision"]:
        raise ValueError("Incomplete or changed artifact")
    if not fixed_sampling(report.get("sampling")):
        raise ValueError("Changed or missing sampling settings")
    rows = report.get("runs", [])
    if not rows or rows[0].get("runtime_cache_state") != "empty_at_process_start":
        raise ValueError("Conversation did not start with empty runtime caches")
    observation = None
    for row in rows:
        state = row["after"]; plan = state["memory_plan"]; machine = state["metal"]
        if row.get("repetition") != 0 or row.get("profiling_enabled") or row.get('gpu_reference_mode','off')!='off' or state["diagnostic_stream_trunk"]:
            raise ValueError("Warm repetitions or diagnostic timing cannot qualify")
        if not state.get("completion_pipeline") or any(state.get(k)!=config[c] for k,c in
                [("chunk_tokens","chunk"),("ready_group","ready_group"),("io_workers","io_workers")]):
            raise ValueError("Execution schedule changed")
        if machine["device"] != "Apple M1 Pro" or machine["physical_bytes"] != 32*1024**3:
            raise ValueError("Requires actual 32GiB M1 Pro")
        if machine["build_fingerprint"] != expected["build"] or state["artifact_revision"] != expected["revision"]:
            raise ValueError("Native build or artifact changed")
        if plan["limit_bytes"] != budget or plan["planned_bytes"] > budget or plan["panel_tokens"] != config["panel"]:
            raise ValueError("Requested budget or panel was not admitted")
        if (config.get("phase_memory","fixed")=="fixed" and plan["expert_slots"] != row["before"]["memory_plan"]["expert_slots"]) or state["process"]["physical_footprint_bytes"] > budget:
            raise ValueError("Cache resized or process exceeded budget")
        if row["before"]["memory_plan"]["limit_bytes"]!=budget:
            raise ValueError("Budget changed during request")
        validate_phase_memory(row, config, budget)
        execution=state.get("execution", {})
        if execution.get("cached_token_replay", False):raise ValueError("Cached replay cannot qualify normal requests")
        for key,default in [("cache_policy","clock"),("residency","off"),("decode_path","reference"),("prefill_pipeline","serial"),("phase_memory","fixed"),("sparse_selection","cpu")]:
            if execution.get(key,default)!=config.get(key,default):raise ValueError("Execution candidate changed")
        if config.get("expert_slots") and plan["expert_slots"]!=config["expert_slots"]:
            raise ValueError("Fixed expert capacity changed")
        kernels = machine["kernels"]
        if kernels.get('route_selection','serial')!=config.get('route_selection','serial'):raise ValueError('Route selection changed')
        if kernels.get('attention_score_tiles','full')!=config.get('attention_score_tiles','full'):raise ValueError('Attention score tiles changed')
        if kernels.get('q8_decode_rows',0)!=config.get('q8_decode_rows',0):raise ValueError('Q8 decode variant changed')
        for key, runtime in [("kernel_policy", "policy"), ("token_tile", "token_tile"), ("gdn_path", "gdn")]:
            if kernels[runtime] != config[key]:
                raise ValueError("Kernel configuration changed")
        if kernels.get("affine_rows",1)!=config.get("affine_rows",1) or kernels.get("gate_pair",False)!=(config.get("gate_pair","off")=="on"):
            raise ValueError("Affine variant changed")
        table=json.loads(Path(config["shape_policy"]).read_text()) if config.get("shape_policy") else None
        if kernels.get("shape_table")!=table:raise ValueError("Shape selection changed")
        if kernels.get("gdn_rows")!=config.get("gdn_rows",4) or kernels.get("gdn_block")!=config.get("gdn_block",8):
            raise ValueError("GDN staging geometry changed")
        name = row["name"]
        signature = (state["prepared"]["manifest_sha256"], row["prompt_tokens"], row["reused_tokens"], tuple(row["output_token_ids"]))
        if expected.setdefault(name, signature) != signature:
            raise ValueError("Changed prepared bytes, tokens, or computation reuse")
        if row["finish_reason"] == "primed":
            if row["output_tokens"] != 0 or row["prompt_tokens"] != 4096:
                raise ValueError("Prime must ingest exactly 4096 tokens without sampling")
            continue
        if row["output_tokens"] != output or len(row["token_latency_ms"]) != output-1 or row["finish_reason"]!="length":
            raise ValueError("Insufficient generated tokens for comparison")
        if name == "append_128" and (row["reused_tokens"] != 4096 or row["prefill_tokens"] != 128 or row["pending_tokens_ingested"] != 0):
            raise ValueError("Append does not measure exactly 4096 retained plus 128 new tokens")
        for key in ('time_to_first_token_ms','decode_wall_ms','request_ms','wall_tokens_per_second'):
            value=row.get(key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise ValueError('Normal request timing must be positive and finite')
        def reads(s):
            return s["checkpoint_application_read_bytes"] + s["prepared"]["application_read_bytes"]
        phases = {}
        for phase, snapshots in row["phases"].items():
            a, b = snapshots["before"], snapshots["after"]
            phases[phase] = dict(application_read_bytes=reads(b)-reads(a),
                                cache_before=a["expert_cache"], cache_after=b["expert_cache"],
                                dependencies_before=a.get("phase_dependencies",{}),dependencies_after=b.get("phase_dependencies",{}),
                                gpu_command_ns=b["metal"]["gpu_command_ns"]-a["metal"]["gpu_command_ns"],
                                cpu_encode_ns=b["metal"]["cpu_encode_ns"]-a["metal"]["cpu_encode_ns"],
                                cpu_gpu_wait_ns=b["metal"]["cpu_gpu_wait_ns"]-a["metal"]["cpu_gpu_wait_ns"],
                                device_before=a["storage"], device_after=b["storage"])
        observation = dict(name=name, ttft_ms=row["time_to_first_token_ms"],
                           decode_ms_per_token=row["decode_wall_ms"]/(output-1), tokens_per_second=row["wall_tokens_per_second"],
                           output_token_ids=row.get("output_token_ids"),
                           forward_tokens_per_second=row["tokens_per_second"],request_ms=row["request_ms"],
                           p95_ms=row["p95_token_ms"], expert_slots=plan["expert_slots"], phases=phases,
                           peak_metal_bytes=machine["peak_buffer_bytes"], footprint_bytes=state["process"]["physical_footprint_bytes"],
                           memory_before=row["before"].get("process"), memory_after=state["process"], residency=machine.get("residency"), execution=execution)
    if observation is None:
        raise ValueError("No generated request")
    return observation


def paired_interval(ratios):
    if len(ratios) < 5 or not all(math.isfinite(x) and x > 0 for x in ratios):
        raise ValueError("Confidence bounds require five valid pairs")
    rng = random.Random(20260908)
    samples = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(10000))
    return dict(median=statistics.median(ratios), low=samples[249], high=samples[9749], pairs=len(ratios))


def summarize(measurements, configs):
    reference = configs[0]["name"]; reports=[]
    for config in configs[1:]:
        bounds=[]
        for case in sorted({r["name"] for r in measurements}):
            for metric in ["ttft_ms", "decode_ms_per_token", "request_ms"]:
                baseline={r["pair"]:r[metric] for r in measurements if r["configuration"]==reference and r["name"]==case}
                candidate={r["pair"]:r[metric] for r in measurements if r["configuration"]==config["name"] and r["name"]==case}
                if baseline.keys()!=candidate.keys():
                    raise ValueError("Unpaired measurements")
                ratios=[candidate[i]/baseline[i] for i in sorted(baseline)]
                bounds.append(dict(case=case, metric=metric, **paired_interval(ratios)))
        coverage={"prompt_2k","prompt_4k","append_128"}.issubset({b["case"] for b in bounds})
        passes=coverage and any(b["high"]<1 for b in bounds) and all(b["high"]<=1.03 for b in bounds)
        reports.append(dict(configuration=config["name"], required_workloads_complete=coverage,latency_gate_passed=passes, confidence_bounds=bounds))
    return reports


def inspect_admission(binary,common,stem,budget,panel,log,guard=None,remaining=None):
    # Metal's process-exit cleanup can lag exit status. Retry metadata only;
    # never lower the budget, rewrite a rejected report, or retry inference.
    for attempt,delay in enumerate((0,2,5)):
        if delay:
            if remaining is not None and remaining()<=delay:raise subprocess.TimeoutExpired('admission',delay)
            time.sleep(delay)
        suffix=".admission.json" if not attempt else f".admission-retry-{attempt}.json"
        path=stem.with_suffix(suffix)
        command=[str(binary),"inspect",*common,"--json",str(path)]
        timeout=min(120,remaining()) if remaining is not None else 120
        if guard:guard.run(command,stdout=log,timeout=timeout)
        else:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        report=json.loads(path.read_text());admission=report["current_admission"]
        if guard and (report['machine']['device']!=guard.identity['device'] or
                      report['machine']['physical_bytes']!=guard.identity['physical_bytes'] or
                      report['machine']['build_fingerprint']!=guard.identity['build'] or
                      report['revision']!=guard.identity['artifact_revision'] or
                      report['prepared']['manifest_sha256']!=guard.identity['prepared_manifest_sha256']):
            raise ValueError('Admission machine or build differs')
        if admission.get("limit_bytes")==budget and admission.get("panel_tokens")==panel:return path
    raise ResourceBlocked("Common budget/panel unavailable after bounded admission retries")


def is_original_control(config):
    return all(config.get(key,default)==default for key,default in [
        ("kernel_policy","reference"),("token_tile",1),("gdn_path","original"),
        ("residency","off"),("decode_path","reference"),("prefill_pipeline","serial"),("phase_memory","fixed"),("affine_rows",1),("gate_pair","off"),("q8_decode_rows",0),("shape_policy",None),("sparse_selection","cpu"),("attention_score_tiles","full")])


def import_pairs(source,destination,identity,configs,cases,pairs,expected,output,validation_only=False):
    """Revalidate complete pairs; never reuse just one arm of an interrupted pair."""
    source,destination=Path(source),Path(destination)
    summary=source/'summary.json'
    saved=json.loads((summary if summary.exists() else source/'progress.json').read_text())
    for key,value in identity.items():
        if saved.get(key)!=value:raise ValueError('Resume experiment identity differs: '+key)
    rows=saved.get('measurements',[]);indexed={}
    allowed={(p,c['name'],name) for p in range(pairs) for c in configs for name in cases}
    for row in rows:
        key=(row['pair'],row['configuration'],row['name'])
        if key in indexed:raise ValueError('Duplicate resume measurement')
        if key not in allowed:raise ValueError('Unexpected resume pair or workload')
        indexed[key]=row
    imported=[]
    for pair in range(pairs):
        order=configs if pair%2==0 else configs[::-1]
        keys=[(pair,c['name'],name) for c in order for name in cases]
        if not all(key in indexed for key in keys):continue
        validated=[];files={}
        for key in keys:
            row=indexed[key];config=next(c for c in configs if c['name']==key[1])
            for label in ('report','admission_report','workload_report'):
                name=row[label];path=confined(source,name)
                if path.name!=name or sha(path)!=row[label+'_sha256']:
                    raise ValueError('Changed resume raw evidence')
                files[name]=path
            raw=json.loads(files[row['report']].read_text())
            if json.loads(files[row['workload_report']].read_text())!=cases[key[2]]:
                raise ValueError('Changed resume token workload')
            admission=json.loads(files[row['admission_report']].read_text())
            if admission['current_admission'].get('limit_bytes')!=identity['budget_bytes'] or admission['current_admission'].get('panel_tokens')!=config['panel']:
                raise ValueError('Changed resume admission')
            observation=validate(raw,config,expected,identity['budget_bytes'],output)
            observation.update({k:row[k] for k in ('pair','configuration','report','report_sha256',
                'admission_report','admission_report_sha256','workload_report','workload_report_sha256')})
            if observation!=row:raise ValueError('Resume measurement differs from raw evidence')
            validated.append(observation)
        if not validation_only:
            for name,path in files.items():
                target=destination/name
                if target.exists():raise ValueError('Resume would overwrite evidence')
                target.write_bytes(path.read_bytes())
        imported.extend(validated)
    return imported


def run(args,guard=None):
    if args.pairs<1 or not math.isfinite(args.memory_gb) or not 0<args.memory_gb<=22:
        raise ValueError("Requires positive repetitions and a memory budget of at most 22GiB")
    if args.mode=="paired" and args.pairs not in (5,10):
        raise ValueError("Paired qualification requires five or ten pairs")
    if any(os.environ.get(k) not in (None,"","0") for k in ["MTL_DEBUG_LAYER","MTL_SHADER_VALIDATION"]):
        raise ValueError("Disable GPU validation during timing")
    configs=json.loads(args.config.read_text()) if args.config else configurations()
    if not configs or (args.mode=="paired" and getattr(args,"comparison_purpose","promotion")=="promotion" and not is_original_control(configs[0])) or len({c["name"] for c in configs})!=len(configs):
        raise ValueError("Configurations need unique names; paired promotion requires the complete original control first")
    for c in configs:
        if not c["name"].replace("-","").replace("_","").isalnum(): raise ValueError("Unsafe configuration name")
        config_args(c)
    lock=json.loads((ROOT/("models.lock.json" if args.artifact=="q4-control" else "mixed-models.lock.json")).read_text())
    expected=dict(build=build_fingerprint(ROOT),revision=lock["revision"])
    source=json.loads(args.workload.read_text())
    seed=source[0]["tokens"] if isinstance(source,list) and isinstance(source[0],dict) else source
    output=256 if args.mode=="paired" else 64
    if not seed:raise ValueError("Workload seed is empty")
    cases=workloads(seed,output,args.include_7k or "prompt_7k" in (args.cases or []))
    if args.cases: cases={k:v for k,v in cases.items() if k in args.cases}
    if not cases: raise ValueError("No selected workloads")
    args.output.mkdir(parents=True,exist_ok=False)
    measurements=[]
    identity=dict(kind="exact_kernel_normal_requests", mode=args.mode, comparison_purpose=getattr(args,"comparison_purpose","promotion"), build=expected["build"], artifact=args.artifact,
                  artifact_revision=lock["revision"], workload_sha256=sha(args.workload),
                  runner_sha256=sha(Path(__file__)), configurations=configs,budget_bytes=int(args.memory_gb*1024**3))
    identity.update(cases=list(cases),pairs=args.pairs,output_tokens=output)
    if guard:identity['evidence_identity']=guard.identity
    def unchanged():
        if build_fingerprint(ROOT)!=expected['build']:raise ValueError('Native sources changed during normal comparison')
        if guard:guard.check_identity();guard.check_resources()
    try:
        unchanged()
        if getattr(args,'resume_from',None):
            measurements=import_pairs(args.resume_from,args.output,identity,configs,cases,args.pairs,expected,output)
            unchanged()
        done={row['pair'] for row in measurements}
        for pair in range(args.pairs):
            if pair in done:continue
            for config in (configs if pair%2==0 else configs[::-1]):
                for name, conversation in cases.items():
                    stem=args.output/f"{config['name']}-{name}-{pair}"
                    workload=stem.with_suffix(".workload.json");workload.write_text(json.dumps(conversation)+"\n")
                    report_path=stem.with_suffix(".json");admission_path=stem.with_suffix(".admission.json")
                    common=["--model",str(args.model),"--artifact",args.artifact,"--prepared",str(args.prepared),
                            "--memory-gb",str(args.memory_gb),"--context","8192"]+config_args(config)
                    with stem.with_suffix(".log").open("w") as log:
                        unchanged()
                        admission_path=inspect_admission(args.binary,common,stem,identity["budget_bytes"],config["panel"],log,guard)
                        command=[str(args.binary),"bench",*common,"--workload-file",str(workload),"--repetitions","1",
                                 "--temperature","0","--seed","0","--json",str(report_path)]
                        if guard:guard.run(command,stdout=log,timeout=args.timeout)
                        else:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
                        unchanged()
                    report=json.loads(report_path.read_text());observation=validate(report,config,expected,identity["budget_bytes"],output)
                    observation.update(pair=pair,configuration=config["name"],report=report_path.name,report_sha256=sha(report_path),
                        admission_report=admission_path.name,admission_report_sha256=sha(admission_path),
                        workload_report=workload.name,workload_report_sha256=sha(workload))
                    if observation['output_token_ids'] is None:raise ValueError("Missing generated-token evidence")
                    previous=next((m for m in measurements if m['name']==name),None)
                    if previous and previous['output_token_ids']!=observation['output_token_ids']:
                        raise ValueError("Exact candidate changed generated tokens")
                    measurements.append(observation)
                    print(f"{config['name']} {name} pair {pair}: {observation['ttft_ms']/1000:.2f}s TTFT, {observation['tokens_per_second']:.3f} tokens/s",flush=True)
                    (args.output/"progress.json").write_text(json.dumps(dict(identity,complete=False,measurements=measurements),indent=2)+"\n")
        summary=summarize(measurements,configs) if args.mode=="paired" else []
        (args.output/"summary.json").write_text(json.dumps(dict(identity,complete=True,measurements=measurements,comparisons=summary,
            generated_tokens_equal=True,full_acceptance_qualified=False),indent=2)+"\n")
    except BaseException as error:
        (args.output/"summary.json").write_text(json.dumps(dict(identity,complete=False,error=str(error),
            status='resource_blocked' if isinstance(error,ResourceBlocked) else 'failed',measurements=measurements),indent=2)+"\n")
        raise


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary",type=Path,default=ROOT/"build/qwen/bin/freellm")
    ap.add_argument("--artifact",choices=["q4-control","mixed-4_8bit"],default="mixed-4_8bit")
    ap.add_argument("--model",type=Path,default=ROOT/".cache/qwen-mixed-reference")
    ap.add_argument("--prepared",type=Path,default=ROOT/".cache/prepared/q4-records-v1")
    ap.add_argument("--workload",type=Path,default=ROOT/"docs/benchmarks/2026-09-08-validation/workload-2k.json")
    ap.add_argument("--config",type=Path)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--mode",choices=["screen","paired"],default="screen")
    ap.add_argument("--comparison-purpose",choices=["experiment","promotion"],default="promotion")
    ap.add_argument("--pairs",type=int,default=1)
    ap.add_argument("--memory-gb",type=float,required=True)
    ap.add_argument("--cases",nargs="+",choices=["prompt_2k","prompt_4k","append_128","prompt_7k"])
    ap.add_argument("--include-7k",action="store_true")
    ap.add_argument("--timeout",type=int,default=7200)
    ap.add_argument("--resume-from",type=Path)
    run(ap.parse_args())
