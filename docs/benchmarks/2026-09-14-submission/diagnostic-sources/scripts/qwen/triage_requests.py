#!/usr/bin/env python3
"""One bounded retained-history request pair, gated by verified cached triage."""
import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import benchmark_exact as normal
from qualification_evidence import EvidenceGuard, ResourceBlocked, identity, save, sha, seal, verify_seal
from selector_qualification import CONFIG, ROOT, WORKLOAD, check_cached, check_machine, configurations
from triage_selector import decide as cached_decision


def load(path):
    return json.loads(Path(path).read_text())


def prerequisites(source, output):
    source = Path(source).resolve()
    report = load(source/'summary.json')
    if (report.get('kind') != 'selector_cached_triage' or report.get('complete') is not True or
        report.get('status') != 'promising' or report.get('advance_to_normal_screen') is not True):
        raise ValueError('Requires completed, promising cached triage')
    previous = load(source/'identity.json')
    EvidenceGuard(previous, output).check_identity()
    current = identity(ROOT, configurations(), ROOT/'.cache/qwen-mixed-reference',
                       ROOT/'.cache/prepared/q4-records-v1', WORKLOAD)
    current['files'][str(CONFIG.resolve())] = sha(CONFIG)
    if ({k:v for k,v in current.items() if k!='files'} != {k:v for k,v in previous.items() if k!='files'} or
        any(current['files'].get(k)!=v for k,v in previous['files'].items())):
        raise ValueError('Cached triage dependencies or settings changed')
    phase = load(source/'cached-run/summary.json')['phases']['cached-4096']
    if phase['status'] != 'passed': raise ValueError('Missing completed cached phase')
    directory = source/'cached-run/cached-4096'
    verify_seal(directory, phase['files_sha256'])
    result = check_cached(load(directory/'report.json'), previous, load(directory/'tokens.json'))
    if result != phase['result'] or result != report['cached_result'] or not cached_decision(result)['advance_to_normal_screen']:
        raise ValueError('Cached measurements do not support advancing')
    bound = [source/'identity.json', source/'summary.json', source/'cached-run/summary.json',
             directory/'evidence-files.json', directory/'report.json', directory/'tokens.json']
    current['files'].update({str(p):sha(p) for p in bound})
    save(output/'prerequisites.json', dict(source=str(source), files_sha256={str(p):sha(p) for p in bound}))
    save(output/'identity.json', current)
    return current


def normal_identity(evidence):
    return dict(kind='exact_kernel_normal_requests', mode='short_screen', comparison_purpose='experiment',
        build=evidence['build'], artifact='mixed-4_8bit', artifact_revision=evidence['artifact_revision'],
        workload_sha256=sha(WORKLOAD), runner_sha256=sha(Path(__file__)),
        configurations=evidence['configurations'], budget_bytes=evidence['budget_bytes'],
        cases=['append_128'], pairs=1, output_tokens=8, evidence_identity=evidence)


def decide(rows):
    if len(rows) != 2 or [(r.get('pair'),r.get('configuration'),r.get('name')) for r in rows] != [
        (0,'cpu-selection','append_128'),(0,'gpu-selection','append_128')]:
        raise ValueError('Requires exactly one CPU/GPU append pair')
    outputs = [r.get('output_token_ids') for r in rows]
    if (not isinstance(outputs[0],list) or len(outputs[0]) != 8 or outputs[0] != outputs[1] or
        any(type(t) is not int or t<0 for t in outputs[0])):
        raise ValueError('Short screen requires eight identical output tokens')
    ratios = {}
    for key in ('request_ms','ttft_ms','decode_ms_per_token'):
        values = [r.get(key) for r in rows]
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in values):
            raise ValueError('Missing or invalid request timings')
        ratios[key] = values[1]/values[0]
    promising = ratios['request_ms'] <= .99 and all(v <= 1.03 for v in ratios.values())
    return dict(status='promising' if promising else 'improvement_not_demonstrated',
        advance_to_full_validation=promising, ratios=ratios, confidence_95=None,
        normal_request_latency_qualified=False, production_promoted=False)


def check_requests(directory, evidence):
    report = load(directory/'summary.json')
    if report.get('complete') is not True: raise ValueError('Normal request pair is unfinished')
    seed = load(WORKLOAD)[0]['tokens']
    cases = {'append_128':normal.workloads(seed,8)['append_128']}
    rows = normal.import_pairs(directory,directory/'unused-validation',normal_identity(evidence),
        evidence['configurations'],cases,1,dict(build=evidence['build'],revision=evidence['artifact_revision']),8,
        validation_only=True)
    if len(rows)!=2 or len(report['measurements'])!=2: raise ValueError('Missing or duplicate short-screen observations')
    if report.get('instrumentation') != dict(profile=False,metal_api_validation=False,metal_shader_validation=False):
        raise ValueError('Missing or changed timing instrumentation receipt')
    for row in rows:
        raw = load(directory/row['report'])
        if raw.get('workloads') != cases['append_128']: raise ValueError('Native workload changed')
        for native in raw['runs']:
            for key in ('before','after'): check_machine(native[key],evidence)
    return rows, decide(rows)


def worker(output):
    evidence = load(output/'identity.json'); guard = EvidenceGuard(evidence,output)
    directory = output/'normal'; directory.mkdir()
    workload = normal.workloads(load(WORKLOAD)[0]['tokens'],8)['append_128']
    expected = dict(build=evidence['build'],revision=evidence['artifact_revision'])
    report = dict(normal_identity(evidence),complete=False,status='running',measurements=[])
    save(directory/'summary.json',report)
    env = dict(os.environ)
    for key in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION'): env.pop(key,None)
    try:
        for config in evidence['configurations']:
            name = config['name']; stem = directory/f'{name}-append_128-0'
            inputs = stem.with_suffix('.workload.json'); save(inputs,workload)
            raw_path = stem.with_suffix('.json')
            common = ['--model',ROOT/'.cache/qwen-mixed-reference','--artifact','mixed-4_8bit',
                      '--prepared',ROOT/'.cache/prepared/q4-records-v1','--memory-gb','12','--context','8192',
                      *normal.config_args(config)]
            print(f'{name}: prime 4096, append 128, generate 8',flush=True)
            with stem.with_suffix('.log').open('w') as log:
                admission = normal.inspect_admission(ROOT/'build/qwen/bin/freellm',common,stem,
                    evidence['budget_bytes'],512,log,guard)
                guard.run([ROOT/'build/qwen/bin/freellm','bench',*common,'--workload-file',inputs,
                           '--repetitions','1','--temperature','0','--seed','0','--json',raw_path],
                          stdout=log,timeout=900,env=env)
            raw = load(raw_path)
            if raw.get('workloads') != workload: raise ValueError('Executed workload changed')
            row = normal.validate(raw,config,expected,evidence['budget_bytes'],8)
            row.update(pair=0,configuration=name,report=raw_path.name,report_sha256=sha(raw_path),
                admission_report=admission.name,admission_report_sha256=sha(admission),
                workload_report=inputs.name,workload_report_sha256=sha(inputs))
            report['measurements'].append(row); save(directory/'summary.json',report)
        decide(report['measurements'])
        report.update(complete=True,status='complete',generated_tokens_equal=True,full_acceptance_qualified=False,
                      instrumentation=dict(profile=False,metal_api_validation=False,metal_shader_validation=False))
    except BaseException as error:
        report.update(status='resource_blocked' if isinstance(error,ResourceBlocked) else
                      'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        raise
    finally:
        save(directory/'summary.json',report)


def run(args):
    if not 0 < args.time_limit <= 900: raise ValueError('Short screen deadline must be 1..900 seconds')
    output = args.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    report = dict(kind='selector_short_request_triage',status='running',complete=False,
        time_limit_seconds=args.time_limit,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        criterion=dict(pairs=1,prime_tokens=4096,append_tokens=128,output_tokens=8,request_ratio_max=.99,metric_ratio_max=1.03),
        advance_to_full_validation=False,normal_request_latency_qualified=False,production_promoted=False)
    save(output/'summary.json',report)
    try:
        evidence = prerequisites(args.cached_triage,output); guard = EvidenceGuard(evidence,output)
        report['build'] = evidence['build']; save(output/'summary.json',report)
        with (ROOT/'.cache/qwen-qualification.lock').open('a') as lock:
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: raise ResourceBlocked('Another qualification workload owns the GPU lease')
            guard.check_resources(initial=True)
            remaining = args.time_limit-(time.monotonic()-started)
            if remaining <= 0: raise subprocess.TimeoutExpired('short-screen',args.time_limit)
            env = dict(os.environ)
            for key in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION'): env.pop(key,None)
            with (output/'launch.log').open('w') as log:
                guard.run([sys.executable,Path(__file__),'--worker','--output',output],stdout=log,timeout=remaining,env=env)
            rows, decision = check_requests(output/'normal',evidence)
            report.update(decision,complete=True,measurements=rows,files_sha256=seal(output/'normal'))
    except ResourceBlocked as error: report.update(status='resource_blocked',error=str(error))
    except subprocess.TimeoutExpired: report.update(status='time_budget_exhausted',error='Stopped at total deadline; no performance decision.')
    except KeyboardInterrupt: report.update(status='interrupted',error='User interrupted; no performance decision.')
    except Exception as error: report.update(status='failed',error=str(error))
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        report['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(output/'summary.json',report)
        (output/'README.md').write_text('# Short request triage\n\nStatus: **'+report['status']+'**.\n\n'
            'One retained-history pair; no confidence or production qualification.\n'
            'The deadline includes both history primes. Cleanup may extend elapsed time.\n'
            'Full validation is never launched by this helper.\n')
        print(json.dumps(report,indent=2),flush=True)
    return 0 if report['complete'] else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cached-triage',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--time-limit',type=int,default=900)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker: worker(args.output.resolve())
    else:
        if args.cached_triage is None: parser.error('--cached-triage is required')
        sys.exit(run(args))
