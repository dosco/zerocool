#!/usr/bin/env python3
"""Validate direct expert destinations and stop weak normal-request candidates early."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import build_mtp_direct_output as builder
from benchmark_host import build_probe
from cache_residency import require
from combined_q4 import freeze
from perfect_draft import configuration,source_input
from prepare_mtp import verify as verify_artifact
from qualification_evidence import ResourceBlocked,save,sha,verify_seal
from screen_mtp_continuation import ROOT,PREPARED,host_check,observe,workloads,select_cases
from screen_mtp_expert_scratch import prerequisite
from screen_mtp_recovery import equivalent
from stage200 import Experiment

BASE=ROOT/'docs/benchmarks/2026-09-16-mtp-direct-output'
REFERENCE=ROOT/'docs/benchmarks/2026-09-16-mtp-ngram-init/validation-03'
FIXTURE_FLAGS=('destinations_and_sentinels_exact','mixed_rows_exact','reversed_reads','forced_eviction',
    'all_hits','invalid_destinations_rejected','scope_exclusion','cancellation_drained',
    'read_failure_drained','encode_failure_drained','delayed_completion_safe','all_buffers_released')


def setup(exp,directory):
    cfg,proof=builder.verify(directory)
    require(proof['base_native_fingerprint']==exp.frozen['build'],'Changed native base')
    save(exp.out/'producer.json',proof);save(exp.out/'draft-audit.json',verify_artifact(PREPARED))
    host=build_probe(ROOT/'.cache/benchmark-host'/hashlib.sha256(str(exp.out).encode()).hexdigest())
    freeze(exp,[*builder.inputs(cfg),*builder.generated(cfg['output']),*cfg['objects'],cfg['binary'],*host['files'],
        PREPARED/'manifest.json',PREPARED/'dense.bin',PREPARED/'experts.bin',BASE/'protocol.md',
        exp.model/'tokenizer.json',exp.model/'generation_config.json'])
    verify_seal(REFERENCE,sha(REFERENCE/'evidence-files.json'))
    require(json.loads((REFERENCE/'summary.json').read_text())['status']=='numerically_validated','Unqualified numerical reference')
    freeze(exp,[REFERENCE/p for p in ('summary.json','case-0.json','case-0-pair-0-off.json','evidence-files.json')])
    exp.env.update(FREELLM_Q8_EXPANDED='packed',FREELLM_MTP_NGRAM_INIT='lazy',FREELLM_MTP_EXPERT_SCRATCH='off')
    exp.report.update(numerical_reference=str(REFERENCE),historical_timing_reused=False,ngram_initialization='lazy',expert_scratch=False)
    exp.guard.check_resources(initial=True);return cfg,host


def counter_delta(r,key):
    a,b=r['direct_output_before'][key],r['direct_output_after'][key]
    require(type(a) is int and type(b) is int and 0<=a<=b,'Invalid direct-output counter')
    return b-a


def comparison(a,b):
    equivalent(a,b)
    fields=('mode','validation','committed_token_ids','row_logits_sha256','final_draft_state','next_id','stop_reason')
    require(all(a[k]==b[k] for k in fields),'Direct output changed full results or draft state')
    require(a['before']['expert_cache']['diagnostic_cache_state']==b['before']['expert_cache']['diagnostic_cache_state'],
        'Changed initial expert cache')
    for r,enabled in ((a,False),(b,True)):
        require(all(r['direct_output_'+p]['enabled'] is enabled and r['direct_output_'+p]['active'] is False
            for p in ('before','after')),'Missing controlled direct-output setting or leaked scope')
        require(r['expert_scratch_after']['enabled'] is False and r['expert_scratch_after']['forwards']==0 and
            r['expert_scratch_after']['groups']==0,'Uncontrolled scratch difference')
        require(r['admission']['expert_scratch_reserve_bytes']==2*1024**2 and
            r['admission']['combined_bytes']<=12*1024**3,'Changed joint admission')
        require(r['after']['metal']['active_scratch_slot']==-1 and r['after']['metal']['live_command_groups']==0,
            'Outstanding GPU users')
        require(counter_delta(r,'forwards')==sum(c['width']==4 for c in r['cycles'])>0,'Incomplete verifier scope coverage')
        require(counter_delta(r,'avoided_copy_bytes')==counter_delta(r,'direct_writes')*2560*4,'Incorrect copy byte accounting')
        require(r['ngram_initialization']=='lazy','Changed ngram mode')
        for p in ('before','after'):
            n=r['ngram_'+p+'_decode']
            require(0<n['constructed_rows']==n['cached_rows']<=n['capacity_rows'] and
                n['initialized_row_bytes']<=n['reserved_row_bytes']<64*1024**2 and
                n['hits']==r[p]['ngram_hits'] and n['misses']==r[p]['ngram_misses'],'Invalid ngram accounting')
    eligible=counter_delta(a,'eligible_calls');direct=counter_delta(b,'direct_writes')
    require(eligible==counter_delta(b,'eligible_calls')==direct>0 and counter_delta(a,'direct_writes')==0,
        'Single-row direct output coverage differs')
    require(all(a['ngram_'+p+'_decode']==b['ngram_'+p+'_decode'] for p in ('before','after')),'Changed logical ngram cache')
    def counters(r):
        values={k:r['after']['metal'][k]-r['before']['metal'][k]
            for k in ('allocation_count','scratch_reuses','dispatches','submissions')}
        require(all(type(v) is int and v>=0 for v in values.values()),'Invalid Metal counter delta');return values
    ca,cb=counters(a),counters(b)
    require(ca['dispatches']-cb['dispatches']==direct,'Dispatch difference exceeds controlled scatter removal')
    return dict(exact_all_logits_tokens_and_state=True,ratio=b['decode_wall_ns']/a['decode_wall_ns'],
        control_tps=a['tokens_per_second'],candidate_tps=b['tokens_per_second'],control_counters=ca,candidate_counters=cb,
        direct_writes=direct,avoided_copy_bytes=direct*2560*4,lazy_cache_exact=True)


def reference_exact(raw):
    old=json.loads((REFERENCE/'case-0-pair-0-off.json').read_text())
    fields=('input_sha256','admission','prime_logits_sha256','row_logits_sha256','committed_token_ids',
        'final_target_state','final_draft_state','boundaries','next_id','stop_reason','ngram_before_decode','ngram_after_decode')
    require(raw['validation'] is True and all(old[k]==raw[k] for k in fields),'Changed qualified numerical reference')


def short_rejected(ratios):
    require(1<=len(ratios)<=2 and all(type(r) in (float,int) and math.isfinite(r) and r>0 for r in ratios),
        'Invalid short comparison ratios')
    return ratios[0]>0.98 or (len(ratios)==2 and (ratios[1]>=1 or math.sqrt(ratios[0]*ratios[1])>0.98))


def fixture(output,directory):
    exp=Experiment(output,'mtp_direct_output_fixture_v1',[],{},240)
    with exp:
        cfg,host=setup(exp,directory);host_check(exp,host,'fixture',False)
        exp.command([cfg['binary'],'--ngram-init-test',exp.model,exp.prepared,exp.out/'ngram.json'],'ngram',limit=90)
        n=json.loads((exp.out/'ngram.json').read_text());require(n['complete'] is True and n['exact_outputs_and_cache'] is True,
            'Incomplete shared ngram fixture')
        exp.command([cfg['binary'],'--direct-output-test',exp.prepared,exp.out/'native.json'],'native',limit=120,validation=True)
        raw=json.loads((exp.out/'native.json').read_text())
        require(raw['kind']=='mtp_direct_output_fixture_v1' and raw['complete'] is True and len(raw['cases'])==4 and
            all(raw[k] is True for k in FIXTURE_FLAGS),'Incomplete native destination fixture')
        exp.report.update(status='fixture_validated',native_sha256=sha(exp.out/'native.json'),ngram_exact=True)
    return exp.report


def run(output,directory,fixture_source,validation_source=None,long=False,selected_cases=None,short_source=None):
    validation=validation_source is None
    if validation:
        work,_=source_input();work.update(eos_ids=[248046,248044],max_tokens=16,draft_slots=32,
            target_prepared=str(ROOT/'.cache/prepared/q4-records-v1'));cases=[work]
    else:cases=select_cases(workloads(ROOT/'.cache/qwen-mixed-reference',128 if long else 16),
        selected_cases if long else ['merge_intervals'])
    kind='mtp_direct_output_validation_v1' if validation else 'mtp_direct_output_screen_v1'
    exp=Experiment(output,kind,[configuration(4,expert_slots=1460)],cases,600 if validation else 1500)
    with exp:
        cfg,host=setup(exp,directory)
        prerequisite(exp,fixture_source,'mtp_direct_output_fixture_v1','fixture_validated')
        if not validation:prerequisite(exp,validation_source,'mtp_direct_output_validation_v1','numerically_validated')
        if long:
            require(short_source is not None,'Long screen requires a promising short screen')
            short=prerequisite(exp,short_source,'mtp_direct_output_screen_v1','promising_short_screen')
            require(len(short['pairs'])==2 and not short_rejected([p['ratio'] for p in short['pairs']]),'Incomplete short gate')
        exp.report.update(pairs=[],samples=[],paired_confidence_qualified=False,early_stop=False,
            fixture_source=str(Path(fixture_source).resolve()),validation_source=str(validation_source) if validation_source else None,
            short_source=str(short_source) if short_source else None,selected_case_names=[w.get('name','forced-rejection') for w in cases])
        mode='fast-validate' if validation else 'fast-timing'
        for i,work in enumerate(cases):
            input_path=exp.out/f'case-{i}.json';save(input_path,work);freeze(exp,[input_path])
            for pair in range(1 if validation or long else 2):
                values={}
                for arm in (('off','on') if (i+pair)%2==0 else ('on','off')):
                    stem=f'case-{i}-pair-{pair}-{arm}';host_check(exp,host,stem)
                    exp.env['FREELLM_MTP_DIRECT_OUTPUT']=arm
                    exp.command([cfg['binary'],exp.model,PREPARED,input_path,exp.out/(stem+'.json'),mode],stem,
                        limit=200 if validation else 360,validation=validation)
                    raw=json.loads((exp.out/(stem+'.json')).read_text());observed=observe(raw,work,sha(input_path),mode)
                    exp.report['samples'].append(dict(case=i,pair=pair,arm=arm,source=stem+'.json',sha256=sha(exp.out/(stem+'.json')),**observed));exp.persist()
                    if not observed['clean_memory'] or not observed['clean_host']:raise ResourceBlocked('Direct-output sample memory or host disturbed')
                    if validation:reference_exact(raw)
                    values[arm]=raw
                result=dict(case=i,pair=pair,name=work.get('name','forced-rejection'),**comparison(values['off'],values['on']))
                exp.report['pairs'].append(result);exp.persist()
                if not validation and not long and short_rejected([p['ratio'] for p in exp.report['pairs']]):
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
    for k in ('fixture','validation-source','short-source'):p.add_argument('--'+k,type=Path)
    p.add_argument('--long',action='store_true');p.add_argument('--case',dest='selected_cases',action='append')
    a=p.parse_args()
    if a.mode=='fixture':fixture(a.output,a.build)
    else:
        require(a.fixture is not None,'Fixture required')
        require((a.mode=='screen')==(a.validation_source is not None),'Screen needs validation source')
        require(a.mode=='screen' or not a.long,'Long validation unsupported')
        require(a.long or a.selected_cases is None,'Case selection needs --long')
        run(a.output,a.build,a.fixture,a.validation_source,a.long,a.selected_cases,a.short_source)
