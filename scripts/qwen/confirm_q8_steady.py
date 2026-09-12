#!/usr/bin/env python3
"""Five fresh Q8 request pairs after a source-verified short screen; never pool runs."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from benchmark_exact import config_args, inspect_admission
from capacity_experiment import decide, order
from capture_routes import load
from qualification_evidence import EvidenceGuard, ResourceBlocked, confined, identity, import_sealed, save, seal, sha
from screen_q8_steady import configs, observations, revalidate as revalidate_screen

ROOT = Path(__file__).resolve().parents[2]
KIND = 'q8_steady_confirmation_v1'
SCREEN = ROOT/'docs/benchmarks/2026-09-11-q8-steady/raw'
TIME_LIMIT = 900


def screen_files(directory):
    summary = load(directory/'summary.json')
    files = {sha(directory/'summary.json'): directory/'summary.json'}
    for key in ('correctness_sources','operator_sources','measurements'):
        for row in summary[key]:
            path = confined(directory,row['source'])
            if sha(path) != row['sha256']: raise ValueError('Changed prerequisite source')
            files[row['sha256']] = path
    return summary, files


def revalidate(summary, resolve, *, kind=KIND, configs=configs,
               observations=observations, revalidate_screen=revalidate_screen):
    if (summary.get('kind') != kind or summary.get('complete') is not True or summary.get('configurations') != configs() or
        summary.get('gpu_reference_mode') != 'off' or summary.get('metal_validation') is not False or
        summary.get('prior_pairs_pooled') is not False or summary.get('early_success_stopping') is not False):
        raise ValueError('Incomplete or changed confirmation protocol')
    prior = resolve(summary['screen_source']['sha256'])
    if not revalidate_screen(prior,resolve)['advance_to_confirmation']:
        raise ValueError('Requires a passing short screen')
    if summary['identity'] != prior['identity'] or summary['workload'] != prior['workload']:
        raise ValueError('Confirmation build, artifact, machine, allocation ceiling or workload differs')
    rows = summary['measurements']; expected = {}; used = {r['sha256'] for r in prior['measurements']}
    if [(r['pair'],r['configuration']) for r in rows] != order(5): raise ValueError('Requires five fresh alternating pairs')
    # Preserve the screen's actual allocation and tokens, without importing its timing samples.
    first = prior['measurements'][0]
    observations(resolve(first['sha256']),summary['identity'],configs()[0],summary['workload'],expected)
    for row in rows:
        digest = row['sha256']
        if digest in used: raise ValueError('Reused request cannot establish a fresh confirmation pair')
        used.add(digest)
        config = next(c for c in configs() if c['name'] == row['configuration'])
        if observations(resolve(digest),summary['identity'],config,summary['workload'],expected) != row['requests']:
            raise ValueError('Confirmation differs from original requests')
    decision = decide(rows,5)
    if any(summary.get(k) != v for k,v in decision.items()): raise ValueError('Changed confirmation decision')
    return decision


def run(args, *, kind=KIND, screen=SCREEN, configs=configs, screen_files=screen_files,
        revalidate_screen=revalidate_screen, observations=observations, revalidate=revalidate):
    out = args.output.resolve(); out.mkdir(parents=True,exist_ok=False); started = time.monotonic()
    report = dict(kind=kind,complete=False,status='running',phase='preparation',configurations=configs(),measurements=[],
        gpu_reference_mode='off',metal_validation=False,prior_pairs_pooled=False,early_success_stopping=False,
        time_limit_seconds=TIME_LIMIT,normal_request_latency_qualified=False,production_promoted=False,
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    def remaining():
        value = TIME_LIMIT-(time.monotonic()-started)
        if value <= 0: raise subprocess.TimeoutExpired(kind,TIME_LIMIT)
        return value
    try:
        import_sealed(screen,out/'screen',sha(screen/'evidence-files.json'))
        prior, files = screen_files(out/'screen')
        if not revalidate_screen(prior,lambda h:load(files[h]))['advance_to_confirmation']:
            raise ValueError('Requires a passing short screen')
        workload = prior['workload']; save(out/'workload.json',workload)
        model, prepared = ROOT/'.cache/qwen-mixed-reference',ROOT/'.cache/prepared/q4-records-v1'
        frozen = identity(ROOT,configs(),model,prepared,out/'workload.json')
        compact = {k:frozen[k] for k in prior['identity']}
        if compact != prior['identity']: raise ValueError('Prerequisite uses another native build or artifact')
        for p in (out/'screen').rglob('*'):
            if p.is_file(): frozen['files'][str(p.resolve())]=sha(p)
        save(out/'identity.json',frozen); guard = EvidenceGuard(frozen,out)
        report.update(identity=compact,workload=workload,screen_source=dict(source='screen/summary.json',sha256=sha(out/'screen/summary.json')))
        expected = {}; first=prior['measurements'][0]
        observations(load(files[first['sha256']]),frozen,configs()[0],workload,expected)
        env=dict(os.environ)
        for k in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION'):env.pop(k,None)
        with (ROOT/'.cache/qwen-qualification.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ResourceBlocked('Another workload owns the GPU lease')
            guard.check_resources(initial=True)
            for pair,name in order(5):
                config=next(c for c in configs() if c['name']==name);stem=out/f'pair-{pair}-{name}'
                report['phase']=stem.name;save(out/'summary.json',report);print(stem.name,flush=True)
                common=['--model',model,'--artifact','mixed-4_8bit','--prepared',prepared,'--memory-gb','12','--context','8192',*config_args(config)]
                with stem.with_suffix('.admission.log').open('w') as log:
                    path=inspect_admission(ROOT/'build/qwen/bin/freellm',common,stem,frozen['budget_bytes'],512,log,guard,remaining)
                if load(path)['current_admission']['expert_slots'] != 1848:raise ResourceBlocked('Explicit capacity not admitted')
                last=0
                def pulse():
                    nonlocal last
                    if time.monotonic()-last>=15:
                        print(f'{stem.name}: elapsed {time.monotonic()-started:.0f}s; remaining {max(0,TIME_LIMIT-(time.monotonic()-started)):.0f}s',flush=True)
                        last=time.monotonic()
                with stem.with_suffix('.log').open('w') as log:
                    guard.run([ROOT/'build/qwen/bin/freellm','bench',*common,'--workload-file',out/'workload.json',
                        '--repetitions','1','--temperature','0','--seed','0','--gpu-reference','off',
                        '--bench-progress',stem.with_suffix('.progress.jsonl'),'--json',stem.with_suffix('.json')],
                        stdout=log,timeout=min(150,remaining()),env=env,progress=pulse)
                raw=load(stem.with_suffix('.json')); requests=observations(raw,frozen,config,workload,expected)
                digest=sha(stem.with_suffix('.json')); files[digest]=stem.with_suffix('.json')
                report['measurements'].append(dict(pair=pair,configuration=name,requests=requests,source=stem.with_suffix('.json').name,sha256=digest))
                save(out/'summary.json',report)
            report.update(decide(report['measurements'],5),complete=True,phase='finished')
            revalidate(report,lambda h:load(files[h]))
    except ResourceBlocked as error:report.update(status='resource_blocked',error=str(error),complete=False)
    except subprocess.TimeoutExpired:report.update(status='time_budget_exhausted',error='Partial confirmation remains unfinished.',complete=False)
    except KeyboardInterrupt:report.update(status='interrupted',complete=False)
    except Exception as error:report.update(status='failed',error=str(error),complete=False)
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(out/'summary.json',report);seal(out)
        print(json.dumps({k:report[k] for k in ('status','complete','phase','elapsed_seconds')}),flush=True)
    return 0 if report['complete'] else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    raise SystemExit(run(parser.parse_args()))
