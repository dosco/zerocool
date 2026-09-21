#!/usr/bin/env python3
"""Exactness and small paired request screen for bounded verifier scratch reuse."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import build_mtp_expert_scratch as builder
from benchmark_host import build_probe
from cache_residency import require
from combined_q4 import freeze
from perfect_draft import configuration,source_input
from prepare_mtp import verify as verify_artifact
from qualification_evidence import ResourceBlocked,save,sha,verify_seal
from screen_mtp_continuation import ROOT,PREPARED,host_check,observe,workloads,select_cases
from screen_mtp_recovery import equivalent
from stage200 import Experiment

BASE=ROOT/'docs/benchmarks/2026-09-16-mtp-expert-scratch'


def setup(exp,directory):
    cfg,proof=builder.verify(directory)
    require(proof['base_native_fingerprint']==exp.frozen['build'],'Changed native base')
    save(exp.out/'producer.json',proof);save(exp.out/'draft-audit.json',verify_artifact(PREPARED))
    host=build_probe(ROOT/'.cache/benchmark-host'/hashlib.sha256(str(exp.out).encode()).hexdigest())
    freeze(exp,[*builder.inputs(cfg),*builder.generated(cfg['output']),*cfg['objects'],cfg['binary'],*host['files'],
        PREPARED/'manifest.json',PREPARED/'dense.bin',PREPARED/'experts.bin',BASE/'protocol.md',
        exp.model/'tokenizer.json',exp.model/'generation_config.json'])
    exp.env['ZEROCOOL_Q8_EXPANDED']='packed';exp.guard.check_resources(initial=True)
    return cfg,host


def comparison(a,b):
    equivalent(a,b)
    require(all(a[k]==b[k] for k in ('mode','validation','committed_token_ids','row_logits_sha256',
        'final_draft_state','next_id','stop_reason')),'Scratch reuse changed full outputs or draft state')
    require(a['before']['expert_cache']['diagnostic_cache_state']==b['before']['expert_cache']['diagnostic_cache_state'],
        'Changed initial expert cache')
    require(a['expert_scratch_after']['enabled'] is False and b['expert_scratch_after']['enabled'] is True,
        'Missing controlled scratch difference')
    require(a['expert_scratch_after']['forwards']==0 and b['expert_scratch_after']['forwards']>0 and
        b['expert_scratch_after']['groups']>0,'Scratch candidate not exercised')
    for r in (a,b):
        require(r['admission']['expert_scratch_reserve_bytes']==2*1024**2 and
            r['admission']['combined_bytes']<=12*1024**3,'Unadmitted scratch capacity')
        require(r['after']['metal']['active_scratch_slot']==-1 and r['after']['metal']['live_command_groups']==0,
            'Outstanding scratch/GPU users')
    def counters(r):
        return {k:r['after']['metal'][k]-r['before']['metal'][k]
            for k in ('allocation_count','scratch_reuses','dispatches','submissions')}
    return dict(exact_all_logits_tokens_and_state=True,ratio=b['decode_wall_ns']/a['decode_wall_ns'],
        control_tps=a['tokens_per_second'],candidate_tps=b['tokens_per_second'],
        control_counters=counters(a),candidate_counters=counters(b))


def prerequisite(exp,directory,kind,status):
    directory=Path(directory).resolve();verify_seal(directory,sha(directory/'evidence-files.json'))
    r=json.loads((directory/'summary.json').read_text())
    require(r['kind']==kind and r['complete'] is True and r['status']==status,'Unqualified prerequisite')
    require(json.loads((directory/'producer.json').read_text())==json.loads((exp.out/'producer.json').read_text()),
        'Prerequisite producer differs')
    freeze(exp,[directory/p for p in ('summary.json','producer.json','evidence-files.json')]);return r


def fixture(output,directory):
    exp=Experiment(output,'mtp_expert_scratch_fixture_v1',[],{},240)
    with exp:
        cfg,host=setup(exp,directory);host_check(exp,host,'fixture',False)
        exp.command([cfg['binary'],'--expert-scratch-test',ROOT/'.cache/prepared/q4-records-v1',exp.out/'native.json'],
            'fixture',limit=120,validation=True)
        r=json.loads((exp.out/'native.json').read_text())
        require(r.get('complete') is True and len(r['cases'])==4 and
            all(r.get(k) is True for k in ('rows_exact','reversed_reads','forced_eviction','all_hits',
                'cancellation_drained','encode_failure_drained','read_failure_drained','delayed_completion_safe','pools_released')),
            'Incomplete expert scratch fixture')
        exp.report.update(status='fixture_validated',native_source='native.json',native_sha256=sha(exp.out/'native.json'))
    return exp.report


def run(output,directory,fixture_source,validation_source=None,long=False,selected_cases=None,short_source=None):
    validation=validation_source is None
    if validation:
        work,_=source_input();work.update(eos_ids=[248046,248044],max_tokens=16,draft_slots=32,
            target_prepared=str(ROOT/'.cache/prepared/q4-records-v1'));cases=[work]
    else:cases=select_cases(workloads(ROOT/'.cache/qwen-mixed-reference',128 if long else 16),
        selected_cases if long else ['merge_intervals'])
    kind='mtp_expert_scratch_validation_v1' if validation else 'mtp_expert_scratch_screen_v1'
    exp=Experiment(output,kind,[configuration(4,expert_slots=1460)],cases,600 if validation else 1500)
    with exp:
        cfg,host=setup(exp,directory)
        prerequisite(exp,fixture_source,'mtp_expert_scratch_fixture_v1','fixture_validated')
        if not validation:prerequisite(exp,validation_source,'mtp_expert_scratch_validation_v1','numerically_validated')
        if long:
            require(short_source is not None,'Long screen requires a promising short screen')
            short=prerequisite(exp,short_source,'mtp_expert_scratch_screen_v1','promising_short_screen')
            require(len(short['pairs'])==2 and short['geometric_mean_ratio']<=0.98,'Incomplete short gate')
            exp.report['short_source']=str(Path(short_source).resolve())
        exp.report.update(pairs=[],samples=[],paired_confidence_qualified=False,
            fixture_source=str(Path(fixture_source).resolve()),validation_source=str(validation_source) if validation_source else None,
            selected_case_names=[w.get('name','forced-rejection') for w in cases],early_stop=False)
        mode='fast-validate' if validation else 'fast-timing'
        for i,work in enumerate(cases):
            input_path=exp.out/f'case-{i}.json';save(input_path,work);freeze(exp,[input_path])
            for pair in range(1 if validation or long else 2):
                values={}
                for arm in (('off','on') if (i+pair)%2==0 else ('on','off')):
                    stem=f'case-{i}-pair-{pair}-{arm}';host_check(exp,host,stem)
                    exp.env['ZEROCOOL_MTP_EXPERT_SCRATCH']=arm
                    exp.command([cfg['binary'],exp.model,PREPARED,input_path,exp.out/(stem+'.json'),mode],stem,
                        limit=200 if validation else 360,validation=validation)
                    raw=json.loads((exp.out/(stem+'.json')).read_text());observed=observe(raw,work,sha(input_path),mode)
                    exp.report['samples'].append(dict(case=i,pair=pair,arm=arm,source=stem+'.json',sha256=sha(exp.out/(stem+'.json')),**observed));exp.persist()
                    if not observed['clean_memory'] or not observed['clean_host']:raise ResourceBlocked('Scratch sample memory or host disturbed')
                    values[arm]=raw
                result=dict(case=i,pair=pair,name=work.get('name','forced-rejection'),**comparison(values['off'],values['on']))
                exp.report['pairs'].append(result);exp.persist()
                if not validation and not long and (result['ratio']>0.98 if pair==0 else
                        result['ratio']>=1 or math.sqrt(result['ratio']*exp.report['pairs'][0]['ratio'])>0.98):
                    exp.report.update(status='insufficient_short_gain',early_stop=True);return exp.report
        exp.report['status']='numerically_validated' if validation else 'continuation_screen_complete' if long else 'promising_short_screen'
        if not validation:
            ratios=[p['ratio'] for p in exp.report['pairs']]
            exp.report['geometric_mean_ratio']=math.exp(sum(map(math.log,ratios))/len(ratios))
            exp.report['all_candidates_at_least_5_tps']=all(p['candidate_tps']>=5 for p in exp.report['pairs'])
    return exp.report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['fixture','validate','screen'])
    for k in ('output','build'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--fixture',type=Path);p.add_argument('--validation-source',type=Path)
    p.add_argument('--short-source',type=Path)
    p.add_argument('--long',action='store_true');p.add_argument('--case',dest='selected_cases',action='append')
    a=p.parse_args()
    if a.mode=='fixture':fixture(a.output,a.build)
    else:
        require(a.fixture is not None,'Fixture required')
        require((a.mode=='screen')==(a.validation_source is not None),'Screen needs validation source')
        require(a.mode=='screen' or not a.long,'Long validation is unsupported')
        require(a.long or a.selected_cases is None,'Case selection needs --long')
        run(a.output,a.build,a.fixture,a.validation_source,a.long,a.selected_cases,a.short_source)
