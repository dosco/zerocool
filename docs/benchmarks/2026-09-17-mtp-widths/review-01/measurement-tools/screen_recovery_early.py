#!/usr/bin/env python3
"""Reject weak recovery candidates early; this screen cannot qualify adoption.

Uses a sealed clean real-state fixture and an exact full-model diagnostic pair.
Every timing sample must still pass the original strict resource and numerical
checks. The complete forced-prefix/EOS stage remains mandatory for advancement.
"""
import argparse
from pathlib import Path
import sys


def diagnostic_pair(trial, source, proof):
    source = Path(source).resolve()
    trial.verify_seal(source, trial.sha(source/'evidence-files.json'))
    summary = trial.read(source/'summary.json')
    trial.require(summary['kind']=='mtp_target_recovery_validation_v1',
                  'Reference must be a full-model validation stage')
    trial.require(trial.read(source/'producer.json')==proof, 'Changed diagnostic producer')
    identity = trial.read(source/'identity.json')
    trial.require(identity['files'].get(proof['binary'])==proof['binary_sha256'],
                  'Diagnostic producer was not frozen')
    work = trial.read(source/'case-0.json')
    trial.require(work==trial.correctness_cases()[0], 'Reference must force accepted prefix one')
    values = []
    observations = []
    files = [source/p for p in ('summary.json','producer.json','identity.json','evidence-files.json','case-0.json')]
    for arm in trial.ARMS:
        path = source/f'case-0-pair-0-{arm}.json'
        raw = trial.read(path)
        trial.require(raw['producer_binary_sha256']==proof['binary_sha256'] and raw['target_recovery']==arm,
                      'Missing or changed diagnostic arm')
        observations.append(dict(arm=arm,source=str(path),sha256=trial.sha(path),
            **trial.observe(raw,work,trial.sha(source/'case-0.json'),True)))
        values.append(raw);files.append(path)
    # Resource-disturbed validation proves only arithmetic. No validation
    # timing or resource qualification is inherited by the early screen.
    trial.comparison(*values)
    return dict(source=str(source),recorded_status=summary['status'],
                use='Numerical diagnostic only; no validation timing or resource eligibility reused',
                observations=observations),files


def run(source_root, prior_trial, reference_pair, output):
    source_root = Path(source_root).resolve()
    sys.path.insert(0, str(source_root/'scripts/qwen'))
    import trial_recovery as trial
    trial.require(trial.ROOT.resolve()==source_root, 'Imports do not belong to the selected frozen source')
    recipe_path = source_root/'scripts/qwen/recipes/target_recovery.json'
    registered = trial.recipe(recipe_path)
    prior_trial = Path(prior_trial).resolve()
    trial.verify_seal(prior_trial,trial.sha(prior_trial/'evidence-files.json'))
    prior = trial.read(prior_trial/'summary.json')
    trial.require(prior['kind']==trial.KIND and prior['recipe_sha256']==trial.sha(recipe_path),
                  'Incompatible real-fixture recipe')
    directory = Path(prior['build_directory'])
    _,proof = trial.builder.verify(directory)
    stages = trial.resumed_stages(prior,proof)
    trial.require('fixtures' in stages, 'A clean real-state fixture is required')
    diagnostic,files = diagnostic_pair(trial,reference_pair,proof)
    work = trial.select_cases(trial.workloads(source_root/'.cache/qwen-mixed-reference',128),['lru_cache'])[0]
    work['max_tokens'] = 64
    exp = trial.Experiment(Path(output),'mtp_target_recovery_screen_v1',
        [trial.configuration(4,expert_slots=1460)],[work],registered['stage_seconds']['short'])
    with exp:
        cfg,host = trial.setup(exp,directory,recipe_path)
        trial.prerequisite(exp,stages['fixtures'],'fixture_validated')
        trial.freeze(exp,[Path(__file__).resolve(),*files,prior_trial/'summary.json',prior_trial/'evidence-files.json'])
        exp.report.update(stage='early',preliminary=True,full_correctness_stage_passed=False,
            advancement_allowed=False,fixture_source=str(stages['fixtures']),
            diagnostic_source=diagnostic,
            pairs=[],samples=[],early_stop=False,remaining_correctness=['prefix-2','prefix-3','prefix-4','immediate-eos'],
            limitations=['A promising result still requires the entire clean forced-prefix/EOS stage.',
                'Early timing is never reused by the registered qualification recipe.',
                'Two short pairs do not qualify sustained performance or establish a confidence bound.'])
        # The first pair may reject this mechanism, but cannot promote it.
        for pair in range(2):
            values = trial.samples(exp,cfg,host,work,0,pair,False)
            trial.require(all(v['generated_tokens']==64 and v['stop_reason']=='length' and
                any(c['committed_tokens']<c['width'] for c in v['cycles']) for v in values.values()),
                'Early screen must complete 64 tokens with natural rejection')
            if not trial.short_gate(exp.report['pairs']):
                exp.report.update(status='insufficient_early_gain',early_stop=True,
                                  decision_reason='Below the unchanged 5% short-screen gate')
                break
        else:
            exp.report.update(status='promising_early_screen',
                              decision_reason='Finish full correctness before any advancement; no adoption qualified')
    # Reports and all raw measurements are sealed by Experiment, including
    # failure paths. The normal importer and comparison commands can read them.
    return exp.report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','prior-trial','reference-pair','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    result=run(a.source_root,a.prior_trial,a.reference_pair,a.output)
    print({k:result.get(k) for k in ('status','complete','error','advancement_allowed')})
    raise SystemExit(0 if result['complete'] else 2)
