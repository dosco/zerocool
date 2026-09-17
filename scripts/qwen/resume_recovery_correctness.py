#!/usr/bin/env python3
"""Reuse individually clean correctness runs, never incomplete runs or timing.

Original stages retain their recorded status. A new stage can qualify only after
all five independent cases satisfy the unchanged raw numerical/resource checks.
"""
import argparse
from pathlib import Path
import sys


def validation_inventory(trial, sources, proof):
    chosen = {}
    rejected = []
    for directory in sources:
        root = Path(directory).resolve()
        trial.verify_seal(root,trial.sha(root/'evidence-files.json'))
        report = trial.read(root/'summary.json')
        trial.require(report.get('kind') in ('mtp_target_recovery_validation_v1',
            'mtp_target_recovery_numerical_diagnostic_v1'), 'Only numerical validation can be reused')
        trial.require(trial.read(root/'producer.json')==proof, 'Changed validation producer')
        identity = trial.read(root/'identity.json')
        trial.require(identity['files'].get(proof['binary'])==proof['binary_sha256'] and
            all(Path(p).is_file() and trial.sha(p)==h for p,h in identity['files'].items()),
            'Validation inputs changed')
        for i,work in enumerate(trial.correctness_cases()):
            rows = [r for r in report.get('samples',[]) if r.get('case')==i]
            trial.require(len(rows)<=2 and len({r['arm'] for r in rows})==len(rows) and
                all(r['arm'] in trial.ARMS and r.get('pair',0)==0 for r in rows),
                'Duplicate or incompatible validation samples')
            if not rows:
                continue
            input_path=root/f'case-{i}.json'
            trial.require(trial.read(input_path)==work, 'Changed forced-prefix/EOS workload')
            values={};paths={};resources={}
            for row in rows:
                path=(root/row['source']).resolve()
                trial.require(path.is_relative_to(root) and trial.sha(path)==row['sha256'],
                              'Changed or unconfined validation sample')
                raw=trial.read(path);arm=row['arm']
                trial.require(raw.get('producer_binary_sha256')==proof['binary_sha256'] and
                    raw.get('target_recovery')==arm, 'Changed validation sample producer or arm')
                observed=trial.observe(raw,work,trial.sha(input_path),True)
                values[arm]=raw;paths[arm]=path;resources[arm]=observed
            if len(values)==2:
                trial.comparison(*(values[a] for a in trial.ARMS))
            for arm,observed in resources.items():
                if not observed['clean_memory'] or not observed['clean_host']:
                    rejected.append(dict(source=str(root),case=i,arm=arm,reason='Run does not pass strict resource checks'))
                    continue
                if (i,arm) not in chosen:
                    chosen[i,arm]=dict(root=root,input=input_path,path=paths[arm],identity=identity,
                        seal_sha256=trial.sha(root/'evidence-files.json'))
    return chosen,rejected


def correctness_stage(trial, output, directory, recipe_path, fixture, sources):
    r=trial.recipe(recipe_path);cases=trial.correctness_cases()
    exp=trial.Experiment(output,'mtp_target_recovery_validation_v1',
        [trial.configuration(4,expert_slots=1460)],cases,r['stage_seconds']['correctness'])
    with exp:
        cfg,host=trial.setup(exp,directory,recipe_path)
        trial.prerequisite(exp,fixture,'fixture_validated')
        proof=trial.read(exp.out/'producer.json')
        chosen,rejected=validation_inventory(trial,sources,proof)
        trial.freeze(exp,[Path(__file__).resolve()])
        exp.report.update(pairs=[],samples=[],fixture_source=str(fixture),
            performance_measurement=False,reused_validation_runs=[],excluded_validation_runs=rejected,
            reuse_policy='Complete independently verified clean correctness runs only; every pair is compared again; timing is never reused')
        for i,work in enumerate(cases):
            input_path=exp.out/f'case-{i}.json'
            existing=next((chosen[i,a] for a in trial.ARMS if (i,a) in chosen),None)
            if existing:
                input_path.write_bytes(existing['input'].read_bytes())
            else:
                trial.save(input_path,work)
            trial.freeze(exp,[input_path]);values={}
            for arm in (trial.ARMS if i%2==0 else trial.ARMS[::-1]):
                stem=f'case-{i}-pair-0-{arm}';path=exp.out/(stem+'.json')
                entry=chosen.get((i,arm))
                if entry:
                    root=entry['root']
                    trial.require(trial.sha(input_path)==trial.sha(entry['input']), 'Validation input bytes differ')
                    trial.freeze(exp,[*[Path(p) for p in entry['identity']['files']],
                        *[root/p for p in ('summary.json','producer.json','identity.json','evidence-files.json')],
                        entry['input'],entry['path']])
                    path.write_bytes(entry['path'].read_bytes())
                    exp.report['reused_validation_runs'].append(dict(case=i,arm=arm,source=str(root),
                        seal_sha256=entry['seal_sha256'],sample=str(entry['path']),sha256=trial.sha(path)))
                    print(f'correctness case {i} {arm}: reusing complete clean validation',flush=True)
                else:
                    trial.host_check(exp,host,stem);exp.env['FREELLM_TARGET_RECOVERY']=arm
                    exp.command([cfg['binary'],exp.model,trial.PREPARED,input_path,path,'fast-validate'],
                                stem,limit=300,validation=True)
                raw=trial.read(path);observed=trial.observe(raw,work,trial.sha(input_path),True)
                trial.freeze(exp,[path]);values[arm]=raw
                exp.report['samples'].append(dict(case=i,pair=0,arm=arm,source=path.name,sha256=trial.sha(path),**observed))
                exp.persist()
                if not observed['clean_memory'] or not observed['clean_host']:
                    raise trial.ResourceBlocked('Correctness run does not pass unchanged strict resource checks')
            exp.report['pairs'].append(dict(case=i,pair=0,name=work['name'],
                **trial.comparison(*(values[a] for a in trial.ARMS))))
            exp.persist()
        exp.report.update(status='numerically_validated',checked_prefixes=[1,2,3,4],eos_checked=True)
    return exp.report


def run(source_root, prior_trial, sources, output):
    source_root=Path(source_root).resolve()
    sys.path.insert(0,str(source_root/'scripts/qwen'))
    import trial_recovery as trial
    from evidence_index import Index
    trial.require(trial.ROOT.resolve()==source_root, 'Imports differ from selected frozen source')
    # The explicit callback changes only validation scheduling inside this
    # process. The registered recipe, native producer and all gates are reused.
    trial.correctness_stage=lambda *args: correctness_stage(trial,*args,sources=sources)
    index=Index(source_root/'.cache/evidence/index.sqlite')
    try:
        return trial.run(source_root/'scripts/qwen/recipes/target_recovery.json',Path(output),
                         resume=Path(prior_trial),index=index)
    finally:
        index.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','prior-trial','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--validation-source',type=Path,action='append',required=True)
    a=p.parse_args();result=run(a.source_root,a.prior_trial,a.validation_source,a.output)
    print({k:result.get(k) for k in ('status','complete','error')})
    raise SystemExit(0 if result['complete'] else 2)
