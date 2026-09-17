#!/usr/bin/env python3
"""Capture real verifier cache lifetimes and select a bounded offline hypothesis."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess

import block_cache_replay as replay
import build_block_cache_trace as builder
import perfect_draft as verifier
from benchmark_host import build_probe,preflight,observe as observe_host
from capture_routes import load
from combined_q4 import freeze
from qualification_evidence import ResourceBlocked,save,sha,verify_seal
from stage200 import Experiment
from verify_stage200 import verify_sources

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'docs/benchmarks/2026-09-16-block-cache'
SOURCE=ROOT/'docs/benchmarks/2026-09-16-perfect-draft-capacity/screen-02'
MTP=ROOT/'.cache/prepared/mtp-q4-q8-v1/manifest.json'
PROTOCOL=BASE/'protocol.md'
SLOTS=[1072,1460]
LIMITATIONS=[
    'Instrumented capture is not a latency measurement; no old timings are pooled.',
    'Native CLOCK replay includes actual admission order and leases and must exactly reproduce captured snapshots and counters.',
    'Counterfactuals hold captured admission/pin/release order fixed. Alternative read completion, GPU timing and policy-specific hit-first ordering are not predicted.',
    'The short fixed continuation is not representative coding-quality, long-context, real-draft acceptance or sustained-use qualification.',
    'Application read estimates are simulated, not observed device traffic or predicted tokens/s. Draft memory/traffic remain unmeasured.']
CRITERIA=dict(slots=SLOTS,width=4,context=8192,memory_bytes=verifier.BUDGET,trace_workspace_bytes=32*1024**2,
    process_seconds=120,stage_seconds=300,policies=['clock','slru'],simulated_slots=[1072,1460,1536],
    minimum_miss_reduction=.10,draft_cache_slots=128,minimum_remaining_bytes=128*1024**2,
    selection_order=[['slru',1460],['clock',1536]])
require=replay.require


def memory_plan(raw,manifest,slots):
    draft=manifest['memory_plan'];target=raw['memory_plan']
    target_total=target['planned_bytes']-target['expert_bytes']+slots*replay.SLOT_BYTES+raw['snapshot_allocated_bytes']+raw['host_logits_bound_bytes']
    draft_total=draft['incremental_budget_bytes']-draft['expert_allocated_bytes']+128*replay.SLOT_BYTES
    total=target_total+draft_total
    return dict(target_slots=slots,draft_slots=128,target_bytes=target_total,draft_bytes=draft_total,
        total_bytes=total,remaining_bytes=verifier.BUDGET-total,
        fits_with_headroom=total+128*1024**2<=verifier.BUDGET,native_draft_allocation_measured=False)


def observe(raw,frozen,work,slots,input_sha,trace_bytes,reference):
    result=verifier.observe(raw,frozen,work,4,False,input_sha,expert_slots=slots)
    expected=verifier.observe(reference,frozen,work,4,False,input_sha,expert_slots=slots)
    require(all(result[k]==expected[k] for k in ('prime','row_logits_sha256','endpoints',
            'snapshot_allocated_bytes','host_logits_bound_bytes')),'Capture changed exact model results or initial cache')
    require(raw['memory_plan']['planned_bytes']+raw['snapshot_allocated_bytes']+raw['host_logits_bound_bytes']+32*1024**2
            <=verifier.BUDGET,'Trace workspace is not admitted')
    trace=replay.decode(trace_bytes,raw,work,frozen['build'])
    compact={k:v for k,v in trace.items() if k not in ('events','requests')}
    for f in compact['forwards']:f.pop('demands')
    return dict(exact_logits_state_routes=True,exact_initial_cache=True,
        clean_memory=result['clean_memory'],clean_host=result['clean_host'],
        peak_physical_bytes=result['peak_physical_bytes'],peak_compressed_bytes=result['peak_compressed_bytes'],
        native_replay=compact,curves=replay.curves(trace),performance_measurement=False)


def select(captures,budgets):
    require([c['slots'] for c in captures]==SLOTS,'Missing both captured native orders')
    require(all(c['observation']['clean_memory'] and c['observation']['clean_host'] and
                c['observation']['native_replay']['native_counts_exact'] and
                c['observation']['native_replay']['native_slot_state_exact'] for c in captures),
            'Disturbed or inexact capture cannot select a candidate')
    candidates=[]
    for policy,capacity in [('slru',1460),('clock',1536)]:
        savings=[];passes=[]
        for c in captures:
            curves={(r['policy'],r['capacity']):r for r in c['observation']['curves']}
            control=curves['clock',1460]['decode_misses'];candidate=curves[policy,capacity]['decode_misses']
            require(control>0,'Missing baseline misses')
            savings.append(1-candidate/control)
            passes.append(10*candidate<=9*control) # Exact 10% boundary, without float rounding.
        admitted=next(b['fits_with_headroom'] for b in budgets if b['target_slots']==capacity)
        candidates.append(dict(policy=policy,slots=capacity,simulated_miss_reductions=savings,
            fits_joint_plan=admitted,meets_screen=admitted and all(passes)))
    selected=next((dict(policy=c['policy'],slots=c['slots']) for c in candidates if c['meets_screen']),None)
    return dict(status='offline_candidate_selected' if selected else 'no_cache_candidate_selected',
        selected_candidate=selected,candidates=candidates,normal_request_latency_qualified=False,
        production_promoted=False,actual_draft_measured=False,latency_prediction=None)


def run(output,binary):
    work,token_source=verifier.source_input()
    exp=Experiment(output,'block_cache_capture_v1',[verifier.configuration(4,expert_slots=s) for s in SLOTS],work,300)
    with exp:
        exp.report.update(criteria=CRITERIA,limitations=LIMITATIONS,captures=[],token_source=token_source,
                          binary=str(binary),actual_draft_measured=False)
        verify_seal(SOURCE,sha(SOURCE/'evidence-files.json'))
        source_summary=load(SOURCE/'summary.json')
        require(source_summary['complete'] is True and source_summary['paired_comparison_complete'] is False,
                'Changed correctness source status')
        anchors=[SOURCE/f'pair-0-slots-{s}-width-4.json' for s in SLOTS]
        exp.report['reference_sources']=[dict(path=str(p),sha256=sha(p)) for p in anchors]
        proof=builder.verify(binary,exp.frozen['build']);save(exp.out/'producer.json',proof['producer'])
        host=build_probe(ROOT/'.cache/benchmark-host'/hashlib.sha256(str(exp.out).encode()).hexdigest())
        save(exp.out/'host-producer.json',host['producer']);exp.report['host_preflight']=[]
        shutil.copyfile(PROTOCOL,exp.out/'protocol.md')
        exp.report.update(protocol_sha256=sha(PROTOCOL),producer_sha256=sha(exp.out/'producer.json'),
            host_producer_sha256=sha(exp.out/'host-producer.json'),mtp_manifest_sha256=sha(MTP))
        freeze(exp,[*proof['files'],*host['files'],*anchors,MTP,PROTOCOL,exp.out/'protocol.md',
                    exp.out/'producer.json',exp.out/'host-producer.json',
                    *[p for p in verifier.SOURCE.iterdir() if p.is_file()]])
        exp.persist();exp.guard.check_resources(initial=True)
        for slots,anchor in zip(SLOTS,anchors):
            stem=f'capture-{slots}';preflight(exp,host,stem);raw_path=exp.out/(stem+'.json')
            try:
                exp.command([binary,exp.model,exp.prepared,exp.out/'workload.json',raw_path,'4','timing',str(slots)],stem,limit=120)
            except subprocess.CalledProcessError:
                if raw_path.is_file() and load(raw_path).get('error')=='fixed memory admission failed':
                    raise ResourceBlocked('Fixed capture memory admission failed') from None
                raise
            trace_path=exp.out/(stem+'.cache.jsonl');raw=load(raw_path)
            result=observe(raw,exp.frozen,work,slots,sha(exp.out/'workload.json'),trace_path.read_bytes(),load(anchor))
            exp.report['captures'].append(dict(slots=slots,source=raw_path.name,sha256=sha(raw_path),
                trace_source=trace_path.name,trace_sha256=sha(trace_path),observation=result));exp.persist()
            if not result['clean_memory'] or not result['clean_host']:raise ResourceBlocked('Capture memory or host is disturbed')
        exp.report['joint_memory_plans']=[memory_plan(raw,load(MTP),s) for s in (1072,1460,1536)]
        exp.report.update(select(exp.report['captures'],exp.report['joint_memory_plans']))
    return exp.report


def audit(output):
    seal=sha(output/'evidence-files.json');verify_seal(output,seal)
    saved,frozen=load(output/'summary.json'),load(output/'identity.json')
    require(saved['kind']=='block_cache_capture_v1' and saved['criteria']==verifier.load_json(CRITERIA) and
            saved['limitations']==LIMITATIONS and saved['identity']=={k:frozen[k] for k in saved['identity']} and
            saved['configurations']==[verifier.configuration(4,expert_slots=s) for s in SLOTS], 'Changed capture protocol')
    provenance=verify_sources(output.parent,frozen)
    proof=builder.verify(Path(saved['binary']),frozen['build'])
    require(saved['producer_sha256']==sha(output/'producer.json') and proof['producer']==load(output/'producer.json') and
            all(frozen['files'][str(Path(p).resolve())]==sha(p) for p in proof['files']),'Changed producer')
    work,source=verifier.source_input()
    require(work==saved['workload']==load(output/'workload.json') and source==saved['token_source'] and
            sha(output/'protocol.md')==saved['protocol_sha256']==frozen['files'][str(PROTOCOL)] and
            sha(MTP)==saved['mtp_manifest_sha256']==frozen['files'][str(MTP)],'Changed workload, protocol or MTP plan')
    host=load(output/'host-producer.json')
    require(saved['host_producer_sha256']==sha(output/'host-producer.json') and host['complete'] is True and
            host['base_native_fingerprint']==frozen['build'] and
            host['binary_sha256']==sha(host['binary'])==frozen['files'][host['binary']] and
            all(frozen['files'][p]==h for p,h in host['files'].items()),'Changed host producer')
    checked=[];checks=saved['host_preflight'];captures=saved['captures']
    require(len(captures)<=len(checks)<=min(len(captures)+1,2),'Missing host checks or extra processes')
    for i,item in enumerate(checks):
        p=output/f'capture-{SLOTS[i]}-host.json';o=observe_host(load(p),frozen['build'])
        require(item==dict(source=p.name,sha256=sha(p),observation=o),'Changed host observation')
        require(o['clean_host'] or (i==len(checks)-1 and saved['status']=='resource_blocked' and
                not (output/f'capture-{SLOTS[i]}.json').exists()),'Work ran after blocked host')
    for i,c in enumerate(captures):
        slots=SLOTS[i];anchor=SOURCE/f'pair-0-slots-{slots}-width-4.json'
        require(saved['reference_sources'][i]==dict(path=str(anchor),sha256=sha(anchor)) and
                frozen['files'][str(anchor)]==sha(anchor),'Changed untraced reference')
        path=output/f'capture-{slots}.json';trace_path=output/f'capture-{slots}.cache.jsonl';raw=load(path)
        result=observe(raw,frozen,work,slots,sha(output/'workload.json'),trace_path.read_bytes(),load(anchor))
        expected=dict(slots=slots,source=path.name,sha256=sha(path),trace_source=trace_path.name,
                      trace_sha256=sha(trace_path),observation=result)
        require(c==expected,'Changed capture or replay')
        checked.append(c)
        if not result['clean_memory'] or not result['clean_host']:
            require(i==len(captures)-1 and len(checks)==len(captures) and saved['status']=='resource_blocked',
                    'Work ran after disturbed capture')
    if saved['complete']:
        require(len(checked)==2 and len(checks)==2,'Missing completed captures')
        budgets=[memory_plan(raw,load(MTP),s) for s in (1072,1460,1536)]
        require(saved['joint_memory_plans']==budgets and all(saved.get(k)==v for k,v in select(checked,budgets).items()),
                'Changed memory plan or selected candidate')
    else:require(saved['status'] in ('failed','resource_blocked','interrupted','time_budget_exhausted'),'Invalid incomplete status')
    return dict(kind='block_cache_audit_v1',complete=True,audit_passed=True,recorded_complete=saved['complete'],
        status=saved['status'],source_seal_sha256=seal,source_provenance=provenance,production_promoted=False,latency_prediction=None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['run','verify'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--binary',type=Path);p.add_argument('--source',type=Path)
    a=p.parse_args()
    if a.action=='run':
        if a.binary is None:p.error('--binary required')
        raise SystemExit(0 if run(a.output,a.binary.resolve())['complete'] else 2)
    else:
        if a.source is None:p.error('--source required')
        save(a.output,audit(a.source.resolve()))
