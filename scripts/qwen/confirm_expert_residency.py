#!/usr/bin/env python3
"""State qualification then five fresh pairs for expert-cache residency."""
import argparse
from pathlib import Path
import statistics

from capacity_experiment import decide as paired_decision, order
from capture_routes import load
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
from qualify_exact_sessions import check_configuration
from screen_cache import CHECKS, correctness_case
from screen_decode_scratch import SCRATCH_CHECKS, check_state_workspace
from screen_expert_residency import configs, observe
from screen_coalesced import decide as screen_decision
from stage200 import Experiment, SOURCE, clean_memory


def require(ok,message):
    if not ok:raise ValueError(message)


def validate_requests(directory,pairs):
    verify_seal(directory,sha(directory/'evidence-files.json'))
    report=load(directory/'summary.json');frozen=load(directory/'identity.json')
    require(report.get('complete') is True and report['configurations']==configs(),'Incomplete or changed residency prerequisite')
    require([(m['pair'],m['configuration']) for m in report['measurements']]==order(pairs),'Missing alternating requests')
    expected={r['name']:r['output_token_ids'] for r in load(SOURCE/'pair-0-control.json')['runs']};seen=set()
    for m in report['measurements']:
        path=directory/m['source'];digest=sha(path);require(digest==m['sha256'] and digest not in seen,'Changed or reused request')
        seen.add(digest);raw=load(path);c=next(c for c in configs() if c['name']==m['configuration'])
        require(observe(raw,frozen,c,report['workload'],expected)==m['requests'],'Changed request observations')
        require(clean_memory(raw) and m['clean_memory'] is True,'Disturbed request cannot qualify')
    decision=screen_decision(report['measurements']) if pairs==2 else confirmation_decision(report['measurements'])
    expected_decision=decision
    if pairs==5 and report.get('kind')=='expert_residency_confirm_v1' and 'request_gain_demonstrated' not in report:
        # V1 conflated missing the 20ms stage target with no measured gain.
        # Preserve its original disposition while reconstructing every number.
        expected_decision={k:v for k,v in decision.items() if k not in ('request_gain_demonstrated','stage_target_met')}
        expected_decision['status']='qualified_short_gain' if decision['short_gain_qualified'] else 'improvement_not_demonstrated'
    require(all(report.get(k)==v for k,v in expected_decision.items()),'Changed prerequisite decision')
    return report


def cases():
    return [dict(correctness_case('clock'),kernel_policy='candidate',q8_decode_rows=2,route_selection='simd',
                 decode_scratch='reuse',decode_submission='immediate',residency=c['residency']) for c in configs()]


def validate_state(reports,build):
    require(len(reports)==2,'Missing state arm')
    for raw,case in zip(reports,cases()):
        require(raw.get('case')==case and raw.get('passed') is True and raw.get('full_model') is True and raw.get('layers')==48,
                'Wrong or incomplete full-model state case')
        checks=raw.get('checks',[])
        require(len(checks)==len(CHECKS|SCRATCH_CHECKS) and {c['name'] for c in checks}==CHECKS|SCRATCH_CHECKS and
                all(c['passed'] is True for c in checks),'Missing state/cancellation/failure check')
        require(len(raw['runs'])==1,'Changed state repetition count');run=raw['runs'][0]
        require(len(run['stages'])==3 and all(len(s['layers'])==len(s['routes'])==48 for s in run['stages']),'Missing layer/state coverage')
        for key in ('continued_statistics','after_fresh'):
            s=run[key];check_configuration(s,case);check_state_workspace(s,case,key)
            require(s['metal']['build_fingerprint']==build and s['memory_plan']['expert_slots']==32 and
                    s['expert_cache']['evictions']>0 and not s['diagnostic_stream_trunk'] and
                    s['metal']['live_command_groups']==0,'Missing forced eviction or native identity')
    require(reports[0]['runs'][0]['stages']==reports[1]['runs'][0]['stages'],'Residency changed logits, routes or persistent state')
    return dict(exact_logits_routes_state=True,all_48_layers=True,continued_equals_fresh=True,
                cancellation_failure_checked=True,forced_eviction=True,independent_model_reference=False)


def confirmation_decision(rows):
    decision=paired_decision(rows,5)
    by={(r['pair'],r['configuration']):r for r in rows}
    savings=[[(by[p,'control']['requests'][i]['decode_wall_ms']-by[p,'candidate']['requests'][i]['decode_wall_ms'])/32
              for p in range(5)] for i in range(2)]
    gain=decision['candidate_for_later_qualification'] and all(r.get('clean_memory') is True for r in rows)
    passed=gain and all(statistics.median(s)>=20 for s in savings)
    return dict(decision,status='qualified_short_gain' if passed else 'gain_below_stage_target' if gain else 'improvement_not_demonstrated',
        request_gain_demonstrated=gain,stage_target_met=passed,
        candidate_for_later_qualification=passed,short_gain_qualified=passed,decode_savings_ms_per_token=savings,
        prior_pairs_pooled=False,early_success_stopping=False,production_promoted=False)


def run(args):
    prior=validate_requests(args.screen,2);require(prior['advance_to_confirmation'],'Short screen did not pass')
    cs=configs();exp=Experiment(args.output,'expert_residency_'+args.mode+('_v1' if args.mode=='state' else '_v2'),cs,prior['workload'],400 if args.mode=='state' else 900)
    with exp:
        require(exp.report['identity']==prior['identity'],'Native build/artifact/budget changed after screen')
        for p in args.screen.rglob('*'):
            if p.is_file():exp.frozen['files'][str(p.resolve())]=sha(p)
        exp.report['screen_source']=dict(path=str(args.screen.resolve()),sha256=sha(args.screen/'summary.json'))
        if args.mode=='state':
            reports=[]
            for c,case in zip(cs,cases()):
                path=exp.out/(c['name']+'.case.json');save(path,case);exp.frozen['files'][str(path)]=sha(path)
            save(exp.out/'identity.json',exp.frozen)
            for c in cs:
                stem='state-'+c['name']
                exp.command([exp.frozen['root']+'/build/qwen/qwen_panel_check',exp.model,exp.prepared,
                    exp.out/(c['name']+'.case.json'),exp.out/(stem+'.json')],stem,180,True)
                reports.append(load(exp.out/(stem+'.json')))
            exp.report.update(status='state_qualified',correctness=validate_state(reports,exp.frozen['build']))
        else:
            require(args.state is not None,'State proof is required before five pairs')
            verify_seal(args.state,sha(args.state/'evidence-files.json'));state=load(args.state/'summary.json')
            require(state.get('complete') is True and state.get('identity')==prior['identity'],'Incomplete or incompatible state proof')
            proof=validate_state([load(args.state/f'state-{c["name"]}.json') for c in cs],exp.frozen['build'])
            require(state.get('correctness')==proof,'Changed state proof')
            for p in args.state.rglob('*'):
                if p.is_file():exp.frozen['files'][str(p.resolve())]=sha(p)
            save(exp.out/'identity.json',exp.frozen)
            exp.report['state_source']=dict(path=str(args.state.resolve()),sha256=sha(args.state/'summary.json'))
            exp.report.update(prior_pairs_pooled=False,early_success_stopping=False,correctness=proof)
            expected={r['name']:r['output_token_ids'] for r in load(SOURCE/'pair-0-control.json')['runs']}
            prior_hashes={m['sha256'] for m in prior['measurements']}
            for pair,arm in order(5):
                c=cs[arm=='candidate'];stem=f'pair-{pair}-{arm}';raw=exp.bench(c,stem)
                digest=sha(exp.out/(stem+'.json'));require(digest not in prior_hashes,'Reused screen timing')
                exp.report['measurements'].append(dict(pair=pair,configuration=arm,requests=observe(raw,exp.frozen,c,prior['workload'],expected),
                    clean_memory=clean_memory(raw),source=stem+'.json',sha256=digest));exp.persist()
                if not clean_memory(raw):raise ResourceBlocked('Compression invalidates fresh confirmation; stop without pooling')
            exp.report.update(confirmation_decision(exp.report['measurements']))
    return 0 if exp.report['complete'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('state','confirm'))
    p.add_argument('--screen',type=Path,required=True);p.add_argument('--state',type=Path);p.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
