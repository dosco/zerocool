#!/usr/bin/env python3
"""Bounded GPU diagnosis, one cache-capacity screen, and conditional confirmation."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from benchmark_exact import config_args, inspect_admission
from capacity_experiment import PROBE, KINDS, configs, order, observations, decide, revalidate, validate_probe
from capture_routes import load
from qualification_evidence import EvidenceGuard, ResourceBlocked, identity, save, seal, sha, verify_seal
from screen_cache import correctness_case, CHECKS
from selector_qualification import check_machine

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'docs/benchmarks/2026-09-10-slru-screen/raw'
LIMITS = {'correctness':360, 'diagnostic':450, 'screen':600, 'confirm':900}


def validate_correctness(reports, evidence):
    for enabled, raw in zip((False, True), reports):
        case = dict(correctness_case('clock'), gpu_reference=enabled)
        checks = CHECKS | ({'probe_preserves_state'} if enabled else set())
        if (raw.get('passed') is not True or raw.get('case') != case or raw.get('full_model') is not True or
            raw.get('layers') != 48 or len(raw.get('runs', [])) != 1 or
            len(raw.get('checks', [])) != len(checks) or {c['name'] for c in raw['checks']} != checks or
            any(c['passed'] is not True for c in raw['checks'])):
            raise ValueError('Missing probe real-model correctness or failure checks')
        run = raw['runs'][0]
        if len(run['stages']) != 3 or any(len(s['layers']) != 48 or len(s['routes']) != 48 for s in run['stages']):
            raise ValueError('Missing all-layer state and routing proof')
        if len(run['gpu_references']) != (2 if enabled else 0): raise ValueError('Missing fixture probes')
        for p in run['gpu_references']: validate_probe(p)
        for key in ('continued_statistics','after_fresh'):
            check_machine(run[key], evidence)
            if run[key]['memory_plan']['expert_slots'] != 32 or run[key]['expert_cache']['evictions'] <= 0:
                raise ValueError('Correctness requires forced eviction')
    if len(reports) != 2 or reports[0]['runs'][0]['stages'] != reports[1]['runs'][0]['stages']:
        raise ValueError('Probe changed real logits, routes or persistent state')
    return dict(passed=True, exact_logits_routes_state=True, continued_equals_fresh=True,
                cancellation_failure_checked=True, independent_model_reference=False)


def run(args):
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(kind='bounded_memory_stage_v1', complete=False, status='running', phase='preparation', stages={},
                  started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), production_promoted=False,
                  normal_request_latency_qualified=False, limits_seconds=LIMITS)
    def progress(name):
        report['phase'] = name; save(out/'summary.json', report); print(name, flush=True)
    try:
        verify_seal(SOURCE, sha(SOURCE/'evidence-files.json'))
        workload = load(SOURCE/'workload.json'); save(out/'workload.json', workload)
        model, prepared = ROOT/'.cache/qwen-mixed-reference', ROOT/'.cache/prepared/q4-records-v1'
        for enabled in (False, True): save(out/f'correctness-{enabled}.case.json', dict(correctness_case('clock'), gpu_reference=enabled))
        frozen = identity(ROOT, configs(), model, prepared, out/'workload.json')
        for p in [SOURCE/'pair-0-clock.json', SOURCE/'workload.json', SOURCE/'evidence-files.json',
                  out/'correctness-False.case.json', out/'correctness-True.case.json']:
            frozen['files'][str(p.resolve())] = sha(p)
        save(out/'identity.json', frozen)
        guard = EvidenceGuard(frozen, out)
        report.update(build=frozen['build'], artifact_revision=frozen['artifact_revision'])
        compact_identity = {k:frozen[k] for k in ('build','artifact_revision','prepared_manifest_sha256','budget_bytes','device','physical_bytes')}
        env = dict(os.environ)
        for k in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION'): env.pop(k, None)
        deadline = 0
        def remaining():
            value = deadline - time.monotonic()
            if value <= 0: raise subprocess.TimeoutExpired('bounded-memory-stage', 0)
            return value
        def run_process(command, stem, timeout, validation=False):
            seen = None; last = 0
            def pulse():
                nonlocal seen, last
                p = stem.with_suffix('.progress.jsonl')
                try:
                    lines = p.read_text().splitlines(); event = json.loads(lines[-1]) if lines else {}
                except (OSError, ValueError): event = {}
                phase = event.get('phase', 'starting'); now = time.monotonic()
                if phase != seen or now-last >= 10:
                    print(f'{stem.name}: {phase}; elapsed {now-started:.0f}s; stage remaining {max(0,deadline-now):.0f}s', flush=True)
                    seen = phase; last = now
            with stem.with_suffix('.log').open('w') as log:
                guard.run(command, stdout=log, timeout=min(timeout,remaining()),
                          env=dict(env,MTL_DEBUG_LAYER='1',MTL_SHADER_VALIDATION='1') if validation else env, progress=pulse)
        with (ROOT/'.cache/qwen-qualification.lock').open('a') as lock:
            try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: raise ResourceBlocked('Another workload owns the GPU lease')
            guard.check_resources(initial=True)
            progress('correctness'); deadline = time.monotonic()+LIMITS['correctness']; correct = []
            for enabled in (False, True):
                stem = out/f'correctness-{enabled}'
                run_process([ROOT/'build/qwen/qwen_panel_check',model,prepared,out/f'correctness-{enabled}.case.json',stem.with_suffix('.json')],stem,180,True)
                correct.append(load(stem.with_suffix('.json')))
            report['correctness'] = validate_correctness(correct, frozen); save(out/'summary.json', report)
            expected = {r['name']:r['output_token_ids'] for r in load(SOURCE/'pair-0-clock.json')['runs']}
            for mode in ('diagnostic', 'screen', 'confirm'):
                directory = out/mode; directory.mkdir(); deadline = time.monotonic()+LIMITS[mode]
                probe = 'off' if mode == 'confirm' else PROBE
                sequence = [(i,'control') for i in range(3)] if mode == 'diagnostic' else order(2 if mode == 'screen' else 5)
                summary = dict(kind=KINDS.get(mode,'gpu_boundary_diagnostic_v1'), mode=mode, complete=False, status='running',
                    identity=compact_identity, configurations=configs(), gpu_reference_mode=probe, metal_validation=False,
                    workload=workload, measurements=[], normal_request_latency_qualified=False, production_promoted=False)
                save(directory/'summary.json', summary)
                try:
                    for pair, name in sequence:
                        progress(f'{mode}_{pair}_{name}')
                        stem = directory/f'pair-{pair}-{name}'; config = next(c for c in configs() if c['name']==name)
                        common = ['--model',model,'--artifact','mixed-4_8bit','--prepared',prepared,'--memory-gb','12','--context','8192',*config_args(config)]
                        with stem.with_suffix('.log').open('w') as log:
                            admission_path=inspect_admission(ROOT/'build/qwen/bin/freellm',common,stem,frozen['budget_bytes'],512,log,guard,remaining)
                        admission=load(admission_path)['current_admission']
                        if admission['expert_slots']!=config['expert_slots']: raise ResourceBlocked('Explicit capacity not admitted')
                        run_process([ROOT/'build/qwen/bin/freellm','bench',*common,'--workload-file',out/'workload.json',
                            '--repetitions','1','--temperature','0','--seed','0','--gpu-reference',probe,
                            '--bench-progress',stem.with_suffix('.progress.jsonl'),'--json',stem.with_suffix('.json')],stem,150)
                        raw = load(stem.with_suffix('.json'))
                        requests = observations(raw,frozen,config,workload,expected,probe)
                        summary['measurements'].append(dict(pair=pair,configuration=name,requests=requests,
                            source=stem.with_suffix('.json').name,sha256=sha(stem.with_suffix('.json'))))
                        save(directory/'summary.json', summary)
                    summary.update(complete=True,status='observed')
                    if mode != 'diagnostic':
                        summary.update(decide(summary['measurements'],2 if mode=='screen' else 5))
                        by_hash={r['sha256']:directory/r['source'] for r in summary['measurements']}
                        revalidate(summary,lambda h:load(by_hash[h]))
                except BaseException as error:
                    status = 'resource_blocked' if isinstance(error,ResourceBlocked) else 'time_budget_exhausted' if isinstance(error,subprocess.TimeoutExpired) else 'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
                    summary.update(status=status,complete=False,error=str(error)); raise
                finally:
                    save(directory/'summary.json',summary);seal(directory)
                    report['stages'][mode]=dict(source=f'{mode}/summary.json',sha256=sha(directory/'summary.json'),status=summary['status'],complete=summary['complete'])
                    save(out/'summary.json',report)
                if mode=='screen' and not summary['advance_to_confirmation']: break
            report.update(complete=True,status='completed',phase='finished',candidate_for_later_qualification=summary.get('candidate_for_later_qualification',False))
    except ResourceBlocked as error: report.update(status='resource_blocked',error=str(error))
    except subprocess.TimeoutExpired: report.update(status='time_budget_exhausted',error='Deadline reached; partial evidence remains unfinished.')
    except KeyboardInterrupt: report.update(status='interrupted',error='Interrupted; no performance conclusion.')
    except Exception as error: report.update(status='failed',error=str(error))
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(out/'summary.json',report);seal(out);print(json.dumps(report,indent=2),flush=True)
    return 0 if report['complete'] else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    raise SystemExit(run(parser.parse_args()))
