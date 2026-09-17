#!/usr/bin/env python3
"""Bounded real-weight state/recovery diagnostic; cannot qualify full-model inference."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from capture_routes import load
from qualification_evidence import EvidenceGuard,ResourceBlocked,identity,save,seal,sha
from qualify_exact_sessions import check_configuration
from screen_cache import CHECKS
from screen_decode_scratch import SCRATCH_CHECKS,check_state_workspace,state_cases
from selector_qualification import check_machine

ROOT=Path(__file__).resolve().parents[2]


def cases():
    return [dict(c,layers=4,memory_gib=4) for c in state_cases()]


def validate(reports,evidence):
    if len(reports)!=2:raise ValueError('Requires both four-layer diagnostic arms')
    for report,case in zip(reports,cases()):
        required=CHECKS|SCRATCH_CHECKS if case['decode_scratch']=='reuse' else CHECKS
        if (report.get('case')!=case or report.get('passed') is not True or report.get('layers')!=4 or
            report.get('full_model') is not False or len(report.get('runs',[]))!=1 or
            len(report.get('checks',[]))!=len(required) or {c['name'] for c in report['checks']}!=required or
            any(c.get('passed') is not True for c in report['checks'])):
            raise ValueError('Incomplete four-layer state/recovery evidence')
        row=report['runs'][0];stages=row.get('stages',[])
        if len(stages)!=3 or any(len(s.get('layers',[]))!=4 or len(s.get('routes',[]))!=4 for s in stages):
            raise ValueError('Missing layer/state coverage')
        for key in ('continued_statistics','after_fresh'):
            state=row[key];check_machine(state,evidence,budget_bytes=4*1024**3);check_configuration(state,case)
            if (state['diagnostic_stream_trunk'] or state['memory_plan']['expert_slots']!=32 or
                state['memory_plan']['panel_tokens']!=0 or state['expert_cache']['evictions']<=0 or
                state['expert_tail_pending'] is not False or state['metal']['live_command_groups']!=0):
                raise ValueError('Missing fixed-capacity eviction or completed users')
            check_state_workspace(state,case,key)
    if reports[0]['runs'][0]['stages']!=reports[1]['runs'][0]['stages']:raise ValueError('Scratch reuse changed partial state or routes')
    return dict(exact_four_layer_state_and_routes=True,continued_equals_fresh=True,forced_eviction=True,
        cancellation_failure_checked=True,one_token_replay_remainder_checked=True,logits_compared=False,
        full_model=False,independent_model_reference=False,normal_request_latency_qualified=False,production_promoted=False)


def run(out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    report=dict(kind='decode_scratch_state_slice_v1',complete=False,status='running',phase='prepare',sources=[],
        full_model=False,normal_request_latency_qualified=False,production_promoted=False,
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),time_limit_seconds=180,
        limitations=['Four layers and nine tokens at 4GiB; cannot replace 48-layer state or performance qualification.',
            'Truncated forward computes no model logits; equality concerns captured state and route identities.',
            'Real mixed weights with native control arithmetic; not an independent implementation oracle.'])
    try:
        for name,case in zip(('control','candidate'),cases()):save(out/(name+'.case.json'),case)
        model,prepared=ROOT/'.cache/qwen-mixed-reference',ROOT/'.cache/prepared/q4-records-v1'
        frozen=identity(ROOT,cases(),model,prepared,out/'control.case.json')
        frozen.update(protocol='decode-scratch-four-layer-diagnostic-v1',budget_bytes=4*1024**3,context=256)
        frozen['files'][str(out/'candidate.case.json')]=sha(out/'candidate.case.json')
        save(out/'identity.json',frozen);guard=EvidenceGuard(frozen,out)
        report['identity']={k:frozen[k] for k in ('build','artifact_revision','prepared_manifest_sha256','budget_bytes','device','physical_bytes')}
        with (ROOT/'.cache/qwen-qualification.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ResourceBlocked('Another workload owns the GPU lease')
            raws=[]
            for name in ('control','candidate'):
                report['phase']=name;save(out/'summary.json',report);print(name,flush=True)
                with (out/(name+'.log')).open('w') as log:
                    remaining=180-(time.monotonic()-started)
                    if remaining<=0:raise subprocess.TimeoutExpired('state-slice',180)
                    guard.run([ROOT/'build/qwen/qwen_panel_check',model,prepared,out/(name+'.case.json'),out/(name+'.json')],
                        stdout=log,timeout=min(90,remaining),env=dict(os.environ,MTL_DEBUG_LAYER='1',MTL_SHADER_VALIDATION='1'))
                raws.append(load(out/(name+'.json')))
                report['sources'].append(dict(source=name+'.json',sha256=sha(out/(name+'.json'))))
            report.update(result=validate(raws,frozen),complete=True,status='passed',phase='finished')
    except ResourceBlocked as e:report.update(status='resource_blocked',error=str(e))
    except subprocess.TimeoutExpired:report.update(status='time_budget_exhausted')
    except KeyboardInterrupt:report.update(status='interrupted')
    except Exception as e:report.update(status='failed',error=str(e))
    finally:
        report['elapsed_seconds']=time.monotonic()-started;save(out/'summary.json',report);seal(out)
        print(json.dumps({k:report.get(k) for k in ('status','complete','error')}),flush=True)
    return 0 if report['complete'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(p.parse_args().output))
