#!/usr/bin/env python3
"""Validate lazy ngram construction before resuming the unchanged scratch timing screen."""
import argparse
import json
from pathlib import Path

import build_mtp_ngram_init as builder
import screen_mtp_expert_scratch as screen
from cache_residency import require
from combined_q4 import freeze
from qualification_evidence import sha,verify_seal

ROOT=screen.ROOT
BASE=ROOT/'docs/benchmarks/2026-09-16-mtp-ngram-init'
REFERENCE=screen.BASE/'validation-02'
original_setup=screen.setup
original_comparison=screen.comparison


def setup(exp,directory):
    cfg,host=original_setup(exp,directory)
    freeze(exp,[BASE/'protocol.md']);exp.env['ZEROCOOL_MTP_NGRAM_INIT']='lazy'
    verify_seal(REFERENCE,sha(REFERENCE/'evidence-files.json'))
    freeze(exp,[REFERENCE/p for p in ('summary.json','case-0.json','case-0-pair-0-off.json','case-0-pair-0-on.json','evidence-files.json')])
    exp.report.update(ngram_initialization='lazy',numerical_reference=str(REFERENCE),
        historical_timing_reused=False)
    return cfg,host


def comparison(a,b):
    result=original_comparison(a,b)
    for r in (a,b):
        require(r['ngram_initialization']=='lazy','Incorrect ngram mode')
        for phase in ('before_decode','after_decode'):
            n=r['ngram_'+phase]
            require(0<n['constructed_rows']==n['cached_rows']<=n['capacity_rows'] and
                n['initialized_row_bytes']<=n['reserved_row_bytes']<64*1024**2,'Invalid lazy cache accounting')
            require(n['hits']==r[phase.split('_')[0]]['ngram_hits'] and n['misses']==r[phase.split('_')[0]]['ngram_misses'],
                'Ngram observation differs from model counters')
        if r['validation']:
            arm='on' if r['expert_scratch_after']['enabled'] else 'off'
            old=json.loads((REFERENCE/f'case-0-pair-0-{arm}.json').read_text())
            fields=('input_sha256','admission','prime_logits_sha256','row_logits_sha256','committed_token_ids',
                'final_target_state','final_draft_state','boundaries','next_id','stop_reason')
            require(all(old[k]==r[k] for k in fields),'Lazy full model differs from qualified eager reference')
            require(all(old[p][k]==r[p][k] for p in ('before','after') for k in ('ngram_hits','ngram_misses')),
                'Lazy initialization changed full-model cache hits/misses')
    require(all(a['ngram_'+p]==b['ngram_'+p] for p in ('before_decode','after_decode')),'Scratch reuse changed ngram cache')
    result['lazy_cache_exact']=True;return result


def configure():
    screen.builder=builder;screen.setup=setup;screen.comparison=comparison


def fixture(output,directory):
    exp=screen.Experiment(output,'mtp_expert_scratch_fixture_v1',[],{},240)
    with exp:
        cfg,host=setup(exp,directory);screen.host_check(exp,host,'fixture',False)
        exp.command([cfg['binary'],'--ngram-init-test',exp.model,exp.prepared,exp.out/'ngram.json'],'ngram',limit=90)
        raw=json.loads((exp.out/'ngram.json').read_text())
        require(raw['complete'] is True and raw['exact_outputs_and_cache'] is True and len(raw['cases'])==2,
            'Incomplete ngram cache fixture')
        exp.command([cfg['binary'],'--expert-scratch-test',exp.prepared,exp.out/'native.json'],'scratch',limit=120,validation=True)
        raw=json.loads((exp.out/'native.json').read_text())
        require(raw['complete'] is True and all(raw[k] is True for k in ('rows_exact','reversed_reads','forced_eviction',
            'all_hits','cancellation_drained','encode_failure_drained','read_failure_drained','delayed_completion_safe','pools_released')),
            'Incomplete scratch fixture')
        exp.report.update(status='fixture_validated',ngram_exact=True,scratch_exact=True)
    return exp.report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['fixture','validate','screen'])
    for k in ('output','build'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('fixture','validation-source','short-source'):p.add_argument('--'+k,type=Path)
    p.add_argument('--long',action='store_true');p.add_argument('--case',dest='selected_cases',action='append')
    a=p.parse_args();configure()
    if a.mode=='fixture':fixture(a.output,a.build)
    else:
        require(a.fixture is not None,'Fixture required')
        require((a.mode=='screen')==(a.validation_source is not None),'Screen needs validation source')
        require(a.mode=='screen' or not a.long,'Long validation unsupported')
        require(a.long or a.selected_cases is None,'Case selection needs --long')
        screen.run(a.output,a.build,a.fixture,a.validation_source,a.long,a.selected_cases,a.short_source)
