#!/usr/bin/env python3
"""Fresh fixed-budget full-model validation and alternating expanded-Q8 timings."""
import argparse
from pathlib import Path
import subprocess

import build_q8_expanded as builder
import perfect_draft as verifier
import screen_q8_expanded as operators
from benchmark_host import observe as observe_host
from cache_residency import require
from capture_block_profile import admission_failure
from capture_routes import load
from combined_q4 import freeze
from perfect_draft_capacity import counters
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
from q8_expanded_contract import CASES, KERNEL
from stage200 import Experiment
from verify_stage200 import verify_sources

VALIDATION=[('serial',1,False),('control',4,False),('candidate',4,True)]
ORDER=[(0,'control'),(0,'candidate'),(1,'candidate'),(1,'control')]
CRITERIA=dict(memory_bytes=verifier.BUDGET,expert_slots=1460,context=8192,
    validation=[list(v) for v in VALIDATION],timing_order=[list(v) for v in ORDER],
    validation_tokens=4,timing_tokens=16,floor_tokens_per_second=5,stop_first_clean_candidate_below_floor=True,
    pairs=2,stage_seconds=600,validation_seconds=150,timing_seconds=90,
    packed_dispatches_per_block=sum(c['frequency'] for c in CASES),actual_draft_measured=False)
LIMITATIONS=verifier.LIMITATIONS+[
    'Both timing arms use four-token perfect proposals and the same developer backend; only the scoped packed-Q8 selector changes.',
    'Two fresh pairs are a short screening result, not confidence-bounded evidence at 2K/4K context or with a real draft.',
    'A below-floor early stop settles only that screening decision and leaves the paired comparison incomplete.']


def stages():
    return [('validate',None,name,width,packed,'validate-'+name) for name,width,packed in VALIDATION]+[
        ('timing',pair,arm,4,arm=='candidate',f'pair-{pair}-{arm}') for pair,arm in ORDER]


def observe(raw,frozen,work,width,validation,input_hash,packed):
    require(raw.get('expanded_q8') is packed,'Changed expanded selector')
    o=verifier.observe(raw,frozen,work,width,validation,input_hash,expert_slots=1460)
    require(all(raw[k]['power_source']=='AC Power' for k in ('host_before','host_after')),'Full verifier requires AC power')
    a,b=(raw[k]['metal']['kernel_dispatches'] for k in ('before','after'))
    require(a.get(KERNEL,0)==0,'Packed path changed priming')
    count=b.get(KERNEL,0)-a.get(KERNEL,0)
    expected=(3 if validation else 4)*CRITERIA['packed_dispatches_per_block'] if packed else 0
    require(type(count) is int and count==expected,'Packed selector did not execute the exact declared scope')
    return dict(o,packed_dispatches=count)


def check_progress(o,validated,measured):
    verifier.check_prime_reference(o['prime'],load(verifier.PRIME_REFERENCE)['prime'])
    if validated:require(o['prime']==validated[0]['observation']['prime'],'Changed initial CLOCK state')
    if o['validation']:
        if validated:verifier.compare(validated[0]['observation'],o)
    else:
        require(len(validated)==3,'Timing before complete correctness/recovery checks')
        serial=validated[0]['observation']
        require(o['row_logits_sha256'][:4]==serial['row_logits_sha256'] and o['endpoints'][0]==serial['endpoints'][-1] and
            all(o[k]==serial[k] for k in ('snapshot_allocated_bytes','host_logits_bound_bytes')),'Timed outputs/state/workspace differ from validation')
        if measured:
            old=measured[0]['observation']
            require(o['row_logits_sha256']==old['row_logits_sha256'] and o['endpoints']==old['endpoints'],'Timed logits/routes/state changed')


def decide(rows):
    require([(r['pair'],r['arm']) for r in rows]==ORDER[:len(rows)] and len(rows)<=len(ORDER),'Wrong paired timing order')
    for row in rows:
        o=row['observation'];require(o['clean_memory'] and o['clean_host'] and not o['validation'] and
            o['verified_tokens']==16 and o['width']==4 and o['decode_wall_ns']>0 and
            o['verified_tokens_per_second']==16e9/o['decode_wall_ns'],'Disturbed or invalid timing cannot decide')
    common=dict(paired_comparison_complete=False,expanded_q8_promising=False,confidence_95=None,**verifier.FLAGS)
    weak=[i for i,r in enumerate(rows) if r['arm']=='candidate' and r['observation']['verified_tokens_per_second']<5]
    if weak:
        require(weak[0]==len(rows)-1,'Ran beyond declared early stop')
        return dict(common,status='expanded_verifier_below_floor',early_stop=True,
            stop_source=rows[-1]['source'],remaining_timing_processes=len(ORDER)-len(rows))
    if len(rows)<len(ORDER):return dict(common,status='running',early_stop=False)
    by={(r['pair'],r['arm']):r['observation'] for r in rows}
    ratios=[by[p,'candidate']['decode_wall_ns']/by[p,'control']['decode_wall_ns'] for p in (0,1)]
    promising=all(r<1 for r in ratios)
    return dict(common,status='expanded_verifier_promising' if promising else 'expanded_verifier_no_advantage',
        early_stop=False,paired_comparison_complete=True,expanded_q8_promising=promising,paired_wall_ratios=ratios)


def operator_proof(source):
    audit=operators.audit(source);report=load(source/'summary.json')
    require(audit['audit_passed'] and report['complete'] is True and report['advance_to_verifier_screen'] is True,
        'Full verifier requires a complete promising expanded operator screen')
    return dict(path=str(source),seal_sha256=sha(source/'evidence-files.json'),summary_sha256=sha(source/'summary.json'))


def run(output,directory,source):
    work,tokens=verifier.source_input();exp=Experiment(output,'q8_expanded_verifier_v1',
        [verifier.configuration(w,expert_slots=1460) for w in (1,4)],work,600)
    with exp:
        cfg,host=operators.setup(exp,directory);gate=operator_proof(source)
        exp.report.update(criteria=CRITERIA,limitations=LIMITATIONS,token_source=tokens,validation=[],operator_source=gate,**verifier.FLAGS)
        freeze(exp,[source/'summary.json',source/'evidence-files.json',verifier.PRIME_REFERENCE]);exp.persist()
        for mode,pair,arm,width,packed,stem in stages():
            operators.host_check(exp,host,stem)
            if packed:exp.env['FREELLM_Q8_EXPANDED']='packed'
            else:exp.env.pop('FREELLM_Q8_EXPANDED',None)
            path=exp.out/(stem+'.json')
            try:exp.command([cfg['verifier']['binary'],exp.model,exp.prepared,exp.out/'workload.json',path,str(width),mode,'1460'],
                stem,limit=150 if mode=='validate' else 90,validation=mode=='validate')
            except subprocess.CalledProcessError:
                if path.exists() and admission_failure(load(path)):raise ResourceBlocked('Fixed verifier memory admission failed') from None
                raise
            raw=load(path);o=observe(raw,exp.frozen,work,width,mode=='validate',sha(exp.out/'workload.json'),packed)
            check_progress(o,exp.report['validation'],exp.report['measurements'])
            row=dict(source=path.name,sha256=sha(path),arm=arm,observation=o)
            if mode=='validate':exp.report['validation'].append(row)
            else:exp.report['measurements'].append(dict(row,pair=pair,counters=counters(raw)))
            exp.persist()
            if not o['clean_memory'] or not o['clean_host']:raise ResourceBlocked('Verifier memory or host disturbed')
            if mode=='timing':
                result=decide(exp.report['measurements']);exp.report.update(result);exp.persist()
                if result['early_stop']:break
    return exp.report


def audit(output):
    seal=sha(output/'evidence-files.json');verify_seal(output,seal);saved=load(output/'summary.json');frozen=load(output/'identity.json')
    require(saved['kind']=='q8_expanded_verifier_v1' and saved['criteria']==CRITERIA and saved['limitations']==LIMITATIONS and
        saved['identity']=={k:frozen[k] for k in saved['identity']} and saved['source_timing_reused'] is False and
        sha(output/'protocol.md')==frozen['files'][str(operators.PROTOCOL)],'Changed verifier scope')
    provenance=verify_sources(output.parent,frozen);proof=builder.verify(Path(saved['build_directory']),frozen['build'])
    require(proof['producer']==load(output/'producer.json') and all(frozen['files'][str(p)]==sha(p) for p in proof['files']),'Changed verifier producer')
    require(saved['operator_source']==operator_proof(Path(saved['operator_source']['path'])),'Changed operator gate')
    work,tokens=verifier.source_input();require(saved['workload']==load(output/'workload.json')==work and saved['token_source']==tokens,'Changed workload')
    host=load(output/'host-producer.json');require(host['complete'] and host['base_native_fingerprint']==frozen['build'] and
        host['binary_sha256']==sha(host['binary'])==frozen['files'][host['binary']],'Changed host producer')
    checks=saved['host_preflight'];schedule=stages();rows=saved['validation']+saved['measurements'];validated=[];measured=[]
    require(len(rows)<=len(checks)<=min(len(rows)+1,len(schedule)),'Missing host checks or extra processes')
    for i,item in enumerate(checks):
        p=output/(schedule[i][-1]+'-host.json');require(item==dict(source=p.name,sha256=sha(p),observation=observe_host(load(p),frozen['build'])),'Changed preflight')
    for i,row in enumerate(rows):
        mode,pair,arm,width,packed,stem=schedule[i];p=output/(stem+'.json');raw=load(p)
        require(row['arm']==arm and row['source']==p.name and row['sha256']==sha(p),'Reordered or changed native source')
        o=observe(raw,frozen,work,width,mode=='validate',sha(output/'workload.json'),packed);check_progress(o,validated,measured)
        require(o==row['observation'],'Changed derived result')
        if mode=='validate':validated.append(row)
        else:
            require(row['pair']==pair and row['counters']==counters(raw),'Changed timing pair/counters');measured.append(row)
        if not o['clean_memory'] or not o['clean_host']:
            require(not saved['complete'] and saved['status']=='resource_blocked' and i==len(rows)-1 and len(checks)==len(rows),'Work continued after resource disturbance')
    if saved['complete']:
        require(len(validated)==3 and len(checks)==len(rows),'Incomplete verification')
        decision=decide(measured);require(decision['status']!='running' and all(saved[k]==v for k,v in decision.items()),'Changed terminal decision')
    else:require(saved['status'] in ('failed','resource_blocked','interrupted','time_budget_exhausted'),'Invalid unfinished result')
    return dict(kind='q8_expanded_verifier_audit_v1',complete=True,audit_passed=True,recorded_complete=saved['complete'],recorded_status=saved['status'],
        source_seal_sha256=seal,source_provenance=provenance,production_promoted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['run','verify']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--build-directory',type=Path);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    if a.action=='verify':save(a.output,audit(a.source.resolve()))
    else:raise SystemExit(0 if run(a.output,a.build_directory.resolve(),a.source.resolve())['complete'] else 2)
