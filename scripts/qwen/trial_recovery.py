#!/usr/bin/env python3
"""One registered, bounded recovery recipe. Historical runners are unchanged."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

import build_target_recovery as builder
from benchmark_host import build_probe
from cache_residency import require
from combined_q4 import freeze
from perfect_draft import configuration, source_input
from prepare_mtp import verify as verify_artifact
from qualification_evidence import ResourceBlocked, save, seal, sha, verify_seal
from screen_mtp_continuation import ROOT, PREPARED, host_check, workloads, select_cases
from screen_mtp_forward import clean
from stage200 import Experiment
from target_recovery_checks import ARMS, observe, comparison, short_gate, long_gate, replay_resources, resource_failure
from mtp_evidence import related_history

KIND='mtp_target_recovery_trial_v1'
FIXTURE_FLAGS=('invalid_geometry_atomic','missing_coverage_atomic','invalid_prefix_atomic',
    'cancellation_drained','failure_drained','delayed_completion_safe','all_buffers_released','corrupt_tensor_rejected')


def read(path):return json.loads(Path(path).read_text())


def recipe(path):
    value=read(path)
    require(value.get('kind')=='target_recovery_recipe_v1' and value.get('arms')==list(ARMS) and
        value.get('controlled_change')=='target_recovery' and value.get('memory_bytes')==12*1024**3 and
        value.get('journal_bytes')==16*1024**2 and value.get('target_slots')==1460 and value.get('draft_slots')==32 and
        value.get('thresholds')==dict(first_gain=.05,reverse_gain=.05,all_accepted_regression=.02,long_gain=.05,case_regression=.02) and
        value.get('workloads')==dict(short_case='lru_cache',short_tokens=64,all_accepted_case='merge_intervals',all_accepted_tokens=16,
            long_cases=['merge_intervals','lru_cache','retry_backoff'],long_tokens=128,long_pairs=2), 'Unsupported or weakened recovery recipe')
    require(set(value['stage_seconds'])=={'fixtures','correctness','short','all_accepted','long'} and
        all(type(n) is int and 30<=n<=3600 for n in value['stage_seconds'].values()), 'Invalid stage deadline')
    require(value['prerequisites']==['exact_saved_state','exact_full_model','clean_memory_power_thermal'], 'Changed advancement prerequisites')
    return value


def setup(exp, directory, recipe_path):
    cfg,proof=builder.verify(directory);require(proof['base_native_fingerprint']==exp.frozen['build'],'Changed native base')
    save(exp.out/'producer.json',proof);save(exp.out/'draft-audit.json',verify_artifact(PREPARED))
    host=build_probe(ROOT/'.cache/benchmark-host'/hashlib.sha256(str(exp.out).encode()).hexdigest())
    freeze(exp,[*builder.inputs(cfg),*builder.generated(cfg['output']),*cfg['objects'],cfg['binary'],*host['files'],
        PREPARED/'manifest.json',PREPARED/'dense.bin',PREPARED/'experts.bin',Path(recipe_path).resolve(),
        exp.model/'tokenizer.json',exp.model/'generation_config.json'])
    exp.env.update(FREELLM_Q8_EXPANDED='packed',FREELLM_MTP_NGRAM_INIT='lazy',FREELLM_MTP_EXPERT_SCRATCH='off',
        FREELLM_MTP_DIRECT_OUTPUT='on')
    exp.report.update(historical_timing_reused=False,controlled_change='target_recovery',paired_confidence_qualified=False)
    exp.guard.check_resources(initial=True);return cfg,host


def reference_work():
    work,_=source_input();work.update(eos_ids=[248046,248044],max_tokens=16,draft_slots=32,
        target_prepared=str(ROOT/'.cache/prepared/q4-records-v1'))
    return work


def correctness_cases():
    cases=[dict(reference_work(),force_prefix=k,name=f'prefix-{k}') for k in range(1,5)]
    eos=dict(reference_work(),force_prefix=4,name='immediate-eos')
    eos['eos_ids']=[eos['expected_prompt_id']]
    return [*cases,eos]


def fixture_identity(data, proof, target_sha, draft_sha):
    require(data.get('kind')=='target_recovery_fixture_v1' and data.get('complete') is True and
        data.get('reference_origin')=='full-target-forward' and
        data['artifact_revision']=='b2c422f3c643e36f04227a64d61796b44a4b1029' and
        data['context']==8192 and data['layers']==48,'Fixture is not a real target reference')
    source=data['source'];policy=source['kernel_policy']
    require(source['producer_binary_sha256']==proof['binary_sha256'] and
        source['native_build_fingerprint']==proof['base_native_fingerprint'] and
        source['target_prepared_sha256']==target_sha and source['draft_manifest_sha256']==draft_sha and
        isinstance(source['input_sha256'],str) and len(source['input_sha256'])==64 and
        source['request_id'].startswith(source['input_sha256']+':'),'Incompatible fixture source identity')
    expected=dict(q4_decode='reference',q8_decode_rows=2,gdn='original',route_selection='simd',
        token_tile=1,affine_rows=1,gate_pair=False,profile=False,counter_profile=False)
    require(all(policy.get(k)==v for k,v in expected.items()),'Changed fixture kernel policy')
    require([e['keep'] for e in data['expected']]==[1,2,3,4] and
        all(len(e['row_logits_sha256'])==e['keep'] and all(isinstance(h,str) and len(h)==64
            for h in e['row_logits_sha256']) for e in data['expected']), 'Missing independent target prefix evidence')


def replay_checks(raw, manifest):
    require(raw.get('kind')=='target_recovery_replay_v1' and raw.get('complete') is True and
        raw['source_manifest_sha256']==sha(manifest) and raw['full_model_loaded'] is False and
        raw['process_limit_bytes']==2*1024**3 and all(raw.get(k) is True for k in FIXTURE_FLAGS) and
        len(raw['cases'])==8 and {(c['keep'],c['repetition']) for c in raw['cases']}=={(k,r) for k in range(1,5) for r in range(2)} and
        all(c['every_buffer_exact'] is True for c in raw['cases']), 'Incomplete recovery fixture validation')
    data=read(manifest)
    require(data.get('reference_origin')==raw.get('reference_origin')=='full-target-forward',
        'Synthetic replay cannot qualify a real fixture')
    states=data['expected']
    require(all(c['state']==states[c['keep']-1]['state'] for c in raw['cases']), 'Fixture expected state changed')
    require(raw['journal']['reserved_bytes']==16*1024**2 and raw['journal']['peak_incremental_bound_bytes']<=16*1024**2,
        'Recovery reserve exceeds 16MiB')
    resources=replay_resources(raw)
    if not resources['clean_memory'] or not resources['clean_host']:raise ResourceBlocked(resource_failure(raw,'Replay resources disturbed'))
    return resources


def capture_checks(raw, manifest):
    data=read(manifest);source=data['source']
    require(raw.get('complete') is True and raw.get('kind')=='target_recovery_capture_v1' and
        raw['mode']=='fast-validate' and raw['validation'] is True and raw['performance_measurement'] is False and
        raw['producer_binary_sha256']==source['producer_binary_sha256'] and
        raw['input_sha256']==source['input_sha256'] and raw['request_id']==source['request_id'] and
        raw['draft_manifest_sha256']==source['draft_manifest_sha256'] and
        raw['before']['prepared']['manifest_sha256']==source['target_prepared_sha256'] and
        raw['before']['metal']['build_fingerprint']==source['native_build_fingerprint'] and
        raw['before']['metal']['kernels']==source['kernel_policy'] and
        raw['capture']['sha256']==sha(manifest) and
        Path(raw['capture']['manifest']).resolve()==Path(manifest).resolve() and
        raw['capture']['payload_bytes']==data['payload_bytes'] and raw['capture']['fixture_limit_bytes']==2*1024**3 and
        raw['capture']['independent_full_target_prefixes']==[1,2,3,4], 'Capture source differs from fixture')
    return clean(raw)


def external_capture(bundle):
    ref_path=bundle/'capture-source.json';ref=read(ref_path)
    require(ref.get('kind')=='target_recovery_capture_source_v1' and
        ref['manifest_sha256']==sha(bundle/'manifest.json'),'Missing or changed capture provenance')
    path=Path(ref['report']).resolve();work=path.parent/'workload.json'
    verify_seal(path.parent,sha(path.parent/'evidence-files.json'))
    raw=read(path)
    require(sha(path)==ref['sha256'] and sha(work)==raw['input_sha256'],'Changed original capture evidence')
    return raw,[ref_path,path,work,path.parent/'evidence-files.json']


def fixture_stage(output,directory,recipe_path,source=None):
    r=recipe(recipe_path);work=reference_work();work['force_prefix']=4
    bundle=ROOT/'.cache/target-recovery-fixtures'/hashlib.sha256(str(Path(output).resolve()).encode()).hexdigest()
    if source is not None:bundle=Path(source).resolve()
    else:work['capture_recovery']=str(bundle)
    exp=Experiment(output,'mtp_target_recovery_fixture_v1',[configuration(4,expert_slots=1460)],work,r['stage_seconds']['fixtures'])
    with exp:
        cfg,host=setup(exp,directory,recipe_path)
        if source is None:
            host_check(exp,host,'capture-recovery');exp.env['FREELLM_TARGET_RECOVERY']='full-replay'
            exp.command([cfg['binary'],exp.model,PREPARED,exp.out/'workload.json',exp.out/'capture.json','fast-validate'],
                'capture',limit=min(600,exp.left()),validation=True)
            raw=read(exp.out/'capture.json')
            require(raw['input_sha256']==sha(exp.out/'workload.json'),'Capture workload changed')
            save(bundle/'capture-source.json',dict(kind='target_recovery_capture_source_v1',report=str(exp.out/'capture.json'),
                sha256=sha(exp.out/'capture.json'),manifest_sha256=sha(bundle/'manifest.json')))
        else:
            raw,files=external_capture(bundle);freeze(exp,files)
        manifest=bundle/'manifest.json';data=read(manifest)
        fixture_identity(data,read(exp.out/'producer.json'),exp.frozen['prepared_manifest_sha256'],sha(PREPARED/'manifest.json'))
        observed=capture_checks(raw,manifest);exp.report['capture_resources']=observed;exp.persist()
        if not observed['clean_memory'] or not observed['clean_host']:raise ResourceBlocked(resource_failure(raw,'Capture resources disturbed'))
        require(data['payload_bytes']+manifest.stat().st_size+(bundle/'capture-source.json').stat().st_size<=2*1024**3 and
            sha(bundle/'payload.bin')==data['payload_sha256'],'Incompatible or corrupt fixture bundle')
        freeze(exp,[manifest,bundle/'payload.bin',bundle/'capture-source.json'])
        save(exp.out/'fixture-manifest.json',data)
        exp.report['fixture']=dict(directory=str(bundle),manifest_sha256=sha(manifest),payload_sha256=data['payload_sha256'])
        exp.persist();host_check(exp,host,'replay-recovery',False)
        exp.command([cfg['binary'],'--replay-recovery',bundle,exp.out/'replay.json'],'replay-recovery',limit=180,validation=True)
        exp.report['replay_resources']=replay_checks(read(exp.out/'replay.json'),manifest)
        exp.report.update(status='fixture_validated',native_sha256=sha(exp.out/'replay.json'))
    return exp.report


def verified_stage(source, status, proof):
    source=Path(source).resolve();verify_seal(source,sha(source/'evidence-files.json'));raw=read(source/'summary.json')
    require(raw.get('complete') is True and raw['status']==status and
        read(source/'producer.json')==proof,'Incomplete or incompatible prerequisite')
    # Every source file in the old evidence identity must still match. Timing
    # prerequisites are never reused by resume, even when these identities match.
    old=read(source/'identity.json')
    require(all(Path(p).is_file() and sha(Path(p))==h for p,h in old['files'].items()), 'Prerequisite inputs changed')
    if status=='fixture_validated':
        require(raw['kind']=='mtp_target_recovery_fixture_v1','Wrong fixture stage')
        fixture=raw['fixture'];manifest=Path(fixture['directory'])/'manifest.json'
        require(sha(manifest)==fixture['manifest_sha256'] and sha(manifest.parent/'payload.bin')==fixture['payload_sha256'] and
            sha(source/'replay.json')==raw['native_sha256'],'Fixture payload or replay changed')
        fixture_identity(read(manifest),proof,old['prepared_manifest_sha256'],sha(PREPARED/'manifest.json'))
        captured,_=external_capture(manifest.parent);resources=capture_checks(captured,manifest)
        require(resources==raw['capture_resources'] and resources['clean_memory'] and resources['clean_host'],
            'Unclean original fixture capture')
        replay_checks(read(source/'replay.json'),manifest)
    if status=='numerically_validated':
        require(raw['kind']=='mtp_target_recovery_validation_v1' and raw['checked_prefixes']==[1,2,3,4] and
            raw['eos_checked'] is True,'Incomplete forced-prefix/EOS qualification')
        pairs=[];samples=[]
        for i,work in enumerate(correctness_cases()):
            input_path=source/f'case-{i}.json';require(read(input_path)==work,'Wrong forced-prefix/EOS workload')
            values={}
            for arm in (ARMS if i%2==0 else ARMS[::-1]):
                sample=source/f'case-{i}-pair-0-{arm}.json';value=read(sample)
                require(value['target_recovery']==arm and value['producer_binary_sha256']==proof['binary_sha256'],
                    'Changed correctness producer or arm')
                result=observe(value,work,sha(input_path),True)
                require(result['clean_host'] and result['clean_memory'],'Unclean correctness prerequisite')
                samples.append(dict(case=i,pair=0,arm=arm,source=sample.name,sha256=sha(sample),**result));values[arm]=value
            pairs.append(dict(case=i,pair=0,name=work['name'],**comparison(*(values[a] for a in ARMS))))
        require(raw['pairs']==pairs and raw['samples']==samples,'Changed correctness sample inventory or summary')
    if status in ('promising_short_screen','all_accepted_safe'):
        from evidence_index import Index
        from mtp_evidence import compare as revalidate
        index=Index(':memory:')
        try:
            index.import_paths([source/'summary.json'])
            checked=revalidate(index,str(source/'summary.json'),*ARMS,['target_recovery'])
        finally:index.close()
        if status=='promising_short_screen':
            require(len(checked['pairs'])==2 and short_gate(checked['pairs']) and
                all(s['generated_tokens']==64 and s['rejected_cycles']>0 and s['completed_requested_length'] for s in raw['samples']),
                'Incomplete short prerequisite coverage')
        else:require(len(checked['pairs'])==1 and checked['pairs'][0]['ratio']<=1.02 and
            all(s['generated_tokens']==16 and s['rejected_cycles']==0 for s in raw['samples']), 'Invalid all-accepted prerequisite')
    return raw,old


def prerequisite(exp, source, status):
    source=Path(source).resolve();raw,old=verified_stage(source,status,read(exp.out/'producer.json'))
    freeze(exp,[*[Path(p) for p in old['files']],*[source/p for p in ('summary.json','producer.json','identity.json','evidence-files.json')]])
    return raw


def resumed_stages(old, proof):
    result={};statuses=dict(fixtures='fixture_validated',correctness='numerically_validated')
    for item in old['stages']:
        name=item['name']
        if name not in statuses or item['complete'] is not True:continue
        path=Path(item['directory']).resolve()
        require(name not in result and sha(path/'summary.json')==item['summary_sha256'] and
            sha(path/'evidence-files.json')==item['seal_sha256'] and item['status']==statuses[name],
            'Changed resumed stage reference')
        verified_stage(path,statuses[name],proof);result[name]=path
    require('correctness' not in result or 'fixtures' in result,'Missing resumed fixture stage')
    if 'correctness' in result:
        require(Path(read(result['correctness']/'summary.json')['fixture_source']).resolve()==result['fixtures'],
            'Resumed correctness belongs to a different fixture stage')
    return result


def samples(exp,cfg,host,work,case,pair,validation):
    input_path=exp.out/f'case-{case}.json'
    if input_path.exists():require(read(input_path)==work,'Workload changed between pairs')
    else:save(input_path,work);freeze(exp,[input_path])
    values={};mode='fast-validate' if validation else 'fast-timing'
    for arm in (ARMS if (case+pair)%2==0 else ARMS[::-1]):
        stem=f'case-{case}-pair-{pair}-{arm}';host_check(exp,host,stem);exp.env['FREELLM_TARGET_RECOVERY']=arm
        exp.command([cfg['binary'],exp.model,PREPARED,input_path,exp.out/(stem+'.json'),mode],stem,
            limit=300 if validation else 360,validation=validation)
        raw=read(exp.out/(stem+'.json'));observed=observe(raw,work,sha(input_path),validation)
        exp.report['samples'].append(dict(case=case,pair=pair,arm=arm,source=stem+'.json',sha256=sha(exp.out/(stem+'.json')),**observed));exp.persist()
        if not observed['clean_memory'] or not observed['clean_host']:raise ResourceBlocked(resource_failure(raw,'Recovery sample resources disturbed'))
        values[arm]=raw
    result=dict(case=case,pair=pair,name=work.get('name','forced-prefix'),**comparison(*(values[a] for a in ARMS)))
    exp.report['pairs'].append(result);exp.persist();return values


def correctness_stage(output,directory,recipe_path,fixture):
    r=recipe(recipe_path);cases=correctness_cases()
    exp=Experiment(output,'mtp_target_recovery_validation_v1',[configuration(4,expert_slots=1460)],cases,r['stage_seconds']['correctness'])
    with exp:
        cfg,host=setup(exp,directory,recipe_path);prerequisite(exp,fixture,'fixture_validated')
        exp.report.update(pairs=[],samples=[],fixture_source=str(fixture))
        for i,work in enumerate(cases):samples(exp,cfg,host,work,i,0,True)
        exp.report.update(status='numerically_validated',checked_prefixes=[1,2,3,4],eos_checked=True)
    return exp.report


def screen_stage(output,directory,recipe_path,validation,stage,short=None,all_accepted=None):
    r=recipe(recipe_path);require(stage in ('short','all_accepted','long'),'Unknown screening stage')
    names=['lru_cache'] if stage=='short' else ['merge_intervals'] if stage=='all_accepted' else r['workloads']['long_cases']
    length=64 if stage=='short' else 16 if stage=='all_accepted' else 128
    cases=select_cases(workloads(ROOT/'.cache/qwen-mixed-reference',16 if length==16 else 128),names)
    for work in cases:work['max_tokens']=length
    exp=Experiment(output,'mtp_target_recovery_screen_v1',[configuration(4,expert_slots=1460)],cases,r['stage_seconds'][stage])
    with exp:
        cfg,host=setup(exp,directory,recipe_path);prerequisite(exp,validation,'numerically_validated')
        if stage!='short':
            prior=prerequisite(exp,short,'promising_short_screen');require(short_gate(prior['pairs']),'Failed short prerequisite')
        if stage=='long':prerequisite(exp,all_accepted,'all_accepted_safe')
        exp.report.update(pairs=[],samples=[],stage=stage,validation_source=str(validation),early_stop=False)
        for i,work in enumerate(cases):
            for pair in range(1 if stage=='all_accepted' else 2):
                values=samples(exp,cfg,host,work,i,pair,False)
                require(all(v['generated_tokens']==length and v['stop_reason']=='length' for v in values.values()),
                    'EOS-shortened screen is diagnostic, not the required length gate')
                if stage=='short':
                    require(all(any(c['committed_tokens']<c['width'] for c in v['cycles']) for v in values.values()), 'Short screen did not exercise natural rejection')
                    if not short_gate(exp.report['pairs']):
                        exp.report.update(status='insufficient_short_gain',early_stop=True,decision_reason='Below 5% gain or reverse-order regression');return exp.report
                if stage=='all_accepted':
                    require(all(all(c['committed_tokens']==c['width'] for c in v['cycles']) for v in values.values()), 'All-accepted workload rejected a proposal')
                    if exp.report['pairs'][-1]['ratio']>1.02:
                        exp.report.update(status='all_accepted_regression',early_stop=True,decision_reason='More than 2% all-accepted regression');return exp.report
        if stage=='long':
            result=long_gate(exp.report['pairs']);exp.report.update(gate=result,status='promising_long_screen' if result['passed'] else 'insufficient_long_gain')
        else:exp.report['status']='promising_short_screen' if stage=='short' else 'all_accepted_safe'
    return exp.report


def run(recipe_path,output,resume=None,index=None):
    recipe_path=Path(recipe_path).resolve();r=recipe(recipe_path);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    save(output/'recipe.json',r);recipe_sha=sha(recipe_path);start=time.monotonic()
    report=dict(kind=KIND,complete=False,status='running',phase='build',stages=[],recipe_sha256=recipe_sha,
        controlled_change=r['controlled_change'],historical_timing_reused=False,production_promoted=False,
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if index is not None:report['related_experiments']=related_history(index)
    directory=ROOT/'.cache/target-recovery-builds'/hashlib.sha256(str(output).encode()).hexdigest()
    prior={}
    try:
        if resume:
            resume=Path(resume).resolve();verify_seal(resume,sha(resume/'evidence-files.json'));old=read(resume/'summary.json')
            require(old['kind']==KIND and old['recipe_sha256']==recipe_sha,'Incompatible resumed recipe')
            directory=Path(old['build_directory']);_,proof=builder.verify(directory)
            prior=resumed_stages(old,proof)
            report['resume']=dict(source=str(resume),use='Verified correctness prerequisites only; all timing samples are fresh')
        else:
            report['build_directory']=str(directory);save(output/'summary.json',report);print('build: compiling isolated target recovery candidate',flush=True)
            builder.build(directory)
        report['build_directory']=str(directory)
        # Later stages consume the directories the earlier ones produced.
        fixtures=correctness=None
        for name in ('fixtures','correctness','short','all_accepted','long'):
            report['phase']=name;save(output/'summary.json',report)
            print(f'trial phase {name}; stage budget {r["stage_seconds"][name]}s',flush=True)
            destination=output/name
            if name in prior:
                # References, seals, identities and raw correctness were verified
                # before exposing these stages as reusable evidence.
                destination=prior[name];value=read(destination/'summary.json')
            elif name=='fixtures':value=fixture_stage(destination,directory,recipe_path)
            elif name=='correctness':value=correctness_stage(destination,directory,recipe_path,fixtures)
            else:value=screen_stage(destination,directory,recipe_path,correctness,name,
                output/'short' if name!='short' else None,output/'all_accepted' if name=='long' else None)
            report['stages'].append(dict(name=name,directory=str(destination),complete=value['complete'],status=value['status'],
                summary_sha256=sha(destination/'summary.json'),seal_sha256=sha(destination/'evidence-files.json')))
            save(output/'summary.json',report)
            if name=='fixtures':fixtures=destination
            if name=='correctness':correctness=destination
            if not value['complete']:
                report.update(status=value['status'],error=value.get('error'));break
            if value['status'] in ('insufficient_short_gain','all_accepted_regression','insufficient_long_gain'):
                report.update(complete=True,status='rejected',decision_reason=value.get('decision_reason',value['status']));break
        else:report.update(complete=True,status='promising',decision_reason='Passed screening only; production qualification remains outstanding')
    except BaseException as error:
        report.update(status='resource_blocked' if isinstance(error,ResourceBlocked) else 'time_budget_exhausted' if isinstance(error,subprocess.TimeoutExpired) else
            'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
    finally:
        report['elapsed_seconds']=time.monotonic()-start;save(output/'summary.json',report)
        if index is not None:
            index.import_paths([output])
            from query_evidence import record
            decision='promising' if report['status']=='promising' else 'rejected' if report['status']=='rejected' else 'blocked' if report['status']=='resource_blocked' else 'inconclusive'
            entry=dict(hypothesis='Avoid repeated full target forwards after partial draft rejection',expected_effect='Reduce target recovery time and expert reads',
                controlled_change='full-replay versus state-only; equal 16MiB journal capacity and fixed 12GiB admission',
                correctness='See individually sealed fixture and full-model stages',outcome=report.get('decision_reason') or report.get('error') or report['status'],
                decision=decision,smallest_experiment='Follow the first failed or missing stage; never reuse timing samples',
                limitations=['Experimental greedy candidate; no production promotion or sustained/long-context qualification'],
                evidence=[str(output/'summary.json')],tags=dict(optimization=['target-state-recovery'],workload=['lru_cache','merge_intervals','retry_backoff'],
                    configuration=['12GiB','1460-target-slots','32-draft-slots','direct-output'],rejection_reason=[report['status']] if decision=='rejected' else []),
                rerun_rationale='Separates target recovery from earlier draft catch-up and direct-output experiments; resumes retain only compatible correctness evidence')
            save(output/'ledger-input.json',entry)
            save(output/'ledger-reference.json',record(index,output/'ledger-input.json',ROOT/'docs/experiments'))
        seal(output)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['capture-recovery','replay-recovery'])
    for key in ('recipe','build','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--fixture',type=Path);a=p.parse_args()
    require((a.operation=='replay-recovery')==(a.fixture is not None),'Replay needs a fixture; capture creates its own')
    value=fixture_stage(a.output,a.build,a.recipe,a.fixture)
    raise SystemExit(0 if value['complete'] else 2)
