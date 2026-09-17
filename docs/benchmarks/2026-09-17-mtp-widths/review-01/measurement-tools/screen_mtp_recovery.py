#!/usr/bin/env python3
"""Paired screen of exact state-only MTP catch-up versus the complete draft layer."""
import argparse
import json
from pathlib import Path
from cache_residency import require
from combined_q4 import freeze
from perfect_draft import source_input,configuration
from qualification_evidence import ResourceBlocked,sha,verify_seal
from screen_mtp_forward import ROOT,PREPARED,BASE,setup,host_check,clean
from stage200 import Experiment


def equivalent(a,b):
    require(a['complete'] is True and b['complete'] is True and a['input_sha256']==b['input_sha256'] and
        a['admission']==b['admission'] and a['draft_manifest_sha256']==b['draft_manifest_sha256'], 'incompatible recovery runs')
    require(a['prime_logits_sha256']==b['prime_logits_sha256'] and a['generated_tokens']==b['generated_tokens'] and
        a['final_target_state']==b['final_target_state'],'recovery changed target output/state')
    fields=('width','proposals','accepted_proposals','committed_tokens','forced_rejection','next_id')
    require([{k:c[k] for k in fields} for c in a['cycles']]==[{k:c[k] for k in fields} for c in b['cycles']],
        'recovery changed proposals or acceptance')
    require(a.get('boundaries')==b.get('boundaries'),'recovery changed intermediate target/draft state')
    for r in (a,b):
        require(r['decode_wall_ns']==sum(c['wall_ns'] for c in r['cycles']) and r['decode_wall_ns']>0 and
            r['generated_tokens']==sum(c['committed_tokens'] for c in r['cycles']),'incomplete recovery timing coverage')


def run(output,directory,fixture_source,validation_source=None):
    verify_seal(fixture_source,sha(fixture_source/'evidence-files.json'))
    old=json.loads((fixture_source/'summary.json').read_text());native=json.loads((fixture_source/'native.json').read_text())
    require(old['complete'] is True and old['status']=='forward_fixture_validated' and native['state_only_catchup_exact'] is True,
        'state-only MTP fixture not qualified')
    work,_=source_input();work.update(target_prepared=str(ROOT/'.cache/prepared/q4-records-v1'),draft_slots=32)
    exp=Experiment(output,'mtp_recovery_screen_v1',[configuration(4,expert_slots=1460)],work,600)
    with exp:
        cfg,host=setup(exp,directory);freeze(exp,[BASE/'recovery-protocol.md'])
        if validation_source is not None:freeze(exp,[BASE/'recovery-timing-protocol.md'])
        require(json.loads((fixture_source/'producer.json').read_text())==json.loads((exp.out/'producer.json').read_text()),'changed fixture producer')
        exp.report.update(fixture_source=str(fixture_source),pairs=[],timing_samples_reused=False)
        checks=[]
        if validation_source is not None:
            verify_seal(validation_source,sha(validation_source/'evidence-files.json'))
            prior=json.loads((validation_source/'summary.json').read_text())
            require(prior['kind']=='mtp_recovery_screen_v1' and prior['status']=='resource_blocked' and
                prior['workload']==work and json.loads((validation_source/'producer.json').read_text())==
                json.loads((exp.out/'producer.json').read_text()),'incompatible numerical validation source')
            for mode in ('validate','fast-validate'):
                raw=json.loads((validation_source/(mode+'.json')).read_text())
                require(raw['mode']==mode and raw['validation'] is True and raw['complete'] is True and
                    raw['input_sha256']==sha(validation_source/'workload.json'),'incomplete numerical validation')
                checks.append(raw)
            freeze(exp,[validation_source/p for p in ('summary.json','producer.json','workload.json','validate.json','fast-validate.json','evidence-files.json')])
            exp.report['numerical_source']=dict(path=str(validation_source.resolve()),recorded_status=prior['status'],
                recorded_complete=prior['complete'],use='Numerical identity only. No validation timing or memory cleanliness reused.')
        else:
            for mode in ('validate','fast-validate'):
                host_check(exp,host,mode);exp.command([cfg['binary'],exp.model,PREPARED,exp.out/'workload.json',exp.out/(mode+'.json'),mode],mode,limit=120,validation=True)
                raw=json.loads((exp.out/(mode+'.json')).read_text());observed=clean(raw);exp.report[mode]=observed;exp.persist()
                if not observed['clean_memory'] or not observed['clean_host']:raise ResourceBlocked('recovery validation memory or host disturbed')
                checks.append(raw)
        equivalent(*checks);exp.report['exact_recovery']=True;exp.persist()
        # Five pairs are a final timing screen only if the first candidate reaches
        # the absolute 5-token/s floor. Every process is fresh; order alternates.
        for pair in range(5):
            values={}
            for arm in (('control','candidate') if pair%2==0 else ('candidate','control')):
                stem=f'pair-{pair}-{arm}';mode='timing' if arm=='control' else 'fast-timing'
                host_check(exp,host,stem);exp.command([cfg['binary'],exp.model,PREPARED,exp.out/'workload.json',exp.out/(stem+'.json'),mode],stem,limit=100)
                raw=json.loads((exp.out/(stem+'.json')).read_text());observed=clean(raw)
                require(raw['complete'] is True and raw['generated_tokens']==16,'incomplete recovery sample')
                exp.report.setdefault('samples',[]).append(dict(source=stem+'.json',sha256=sha(exp.out/(stem+'.json')),
                    pair=pair,arm=arm,tokens_per_second=raw['tokens_per_second'],**observed));exp.persist()
                if not observed['clean_memory'] or not observed['clean_host']:raise ResourceBlocked('recovery timing memory or host disturbed')
                values[arm]=raw
            equivalent(values['control'],values['candidate'])
            ratio=values['candidate']['decode_wall_ns']/values['control']['decode_wall_ns']
            exp.report['pairs'].append(dict(pair=pair,ratio=ratio,control_tps=values['control']['tokens_per_second'],candidate_tps=values['candidate']['tokens_per_second']))
            exp.persist()
            if pair==0 and values['candidate']['tokens_per_second']<5:
                exp.report.update(status='below_absolute_target',early_stop=True,paired_confidence_qualified=False)
                break
        else:
            from screen_residency import paired_log_interval
            interval=paired_log_interval([p['ratio'] for p in exp.report['pairs']]);floor=min(p['candidate_tps'] for p in exp.report['pairs'])
            exp.report.update(status='promising_short_screen' if interval['high']<1 and floor>=5 else 'inconclusive_short_screen',
                confidence_95=interval,all_candidates_at_least_5_tps=floor>=5,paired_confidence_qualified=True)
    return exp.report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('output','build','fixture'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--numerical-source',type=Path)
    a=p.parse_args();run(a.output,a.build,a.fixture,a.numerical_source)
