#!/usr/bin/env python3
"""Capture and screen expanded packed-Q8 scope; old samples are never pooled."""
import argparse
import hashlib
from pathlib import Path
import re
import shutil
import statistics
import subprocess

import block_compute_profile as profile
import build_q8_expanded as builder
import perfect_draft as verifier
import q8_expanded_fixtures as fixtures
from benchmark_host import build_probe, preflight, observe as observe_host
from cache_residency import require
from capture_block_profile import ANCHOR, admission_failure
from capture_routes import load
from combined_q4 import freeze
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
from q8_expanded_contract import ROOT, BASE, CASES, KERNEL, CRITERIA, LIMITATIONS
from screen_residency import paired_log_interval
from stage200 import Experiment
from verify_stage200 import verify_sources

PROTOCOL=BASE/'protocol.md'


def analyze(raw,validation,frozen,manifest):
    require(raw.get('kind')=='q8_expanded_operator_v1' and raw.get('complete') is True and raw.get('validation') is validation and
        raw.get('build')==frozen['build'] and raw.get('artifact_revision')==frozen['artifact_revision'] and raw.get('source')==manifest and
        raw.get('device')=='Apple M1 Pro' and 0<raw.get('peak_gpu_bytes',0)<=CRITERIA['operator_gpu_bytes'] and
        raw.get('production_promoted') is False and raw.get('normal_request_latency_qualified') is False,'Changed operator identity or bounds')
    require([c['name'] for c in raw['cases']]==[c['name'] for c in CASES],'Missing or reordered shapes')
    host=[raw['host_before'],raw['host_after']]
    require(host[0]['monotonic_ns']<=host[1]['monotonic_ns'],'Reversed host clock')
    clean_host=all(h['thermal_state']==0 and h['low_power_mode'] is False and h['power_source']=='AC Power' for h in host)
    memory=[raw['memory_before'],raw['memory_after_destroy']];pairs=1 if validation else 5
    weighted=[[0.,0.] for _ in range(pairs)];cases=[]
    for case,spec in zip(raw['cases'],CASES):
        require((case['K'],case['N'],case['tokens'],case['frequency'])==(spec['K'],spec['N'],4,spec['frequency']) and
            re.fullmatch('[0-9a-f]{64}',case['reference_sha256']) is not None and
            [p['pair'] for p in case['pairs']]==list(range(pairs)) and
            [a['candidate'] for a in case['warmup']]==[False,True],'Changed paired coverage')
        samples=[*case['warmup'],*[a for p in case['pairs'] for a in p['arms']]]
        for i,arm in enumerate(samples):
            n=1 if validation or i<2 else 32
            require(type(arm['candidate']) is bool and type(arm['repeats']) is int and arm['repeats']==n and
                arm['exact'] is True and arm['output_sha256']==case['reference_sha256'] and
                type(arm['gpu_ns']) is int and type(arm['wall_ns']) is int and 0<arm['gpu_ns']<=arm['wall_ns'] and
                arm['kernel_dispatches']=={KERNEL if arm['candidate'] else 'q8_mm_t4':n},'Changed output, timing or dispatch selection')
            memory += [arm['memory_before'],arm['memory_after']]
        ratios=[]
        for p in case['pairs']:
            require([a['candidate'] for a in p['arms']]==[bool(p['pair']%2),not bool(p['pair']%2)],'Changed alternating order')
            times={a['candidate']:a['gpu_ns']/a['repeats'] for a in p['arms']}
            for arm in (False,True):weighted[p['pair']][int(arm)]+=times[arm]*spec['frequency']/4e6
            ratios.append(times[True]/times[False])
        cases.append(dict(name=case['name'],paired_ratios=ratios))
    fields=('physical_footprint_bytes','physical_footprint_peak_bytes','compressed_bytes','compressed_peak_bytes','decompressions','system_swap_used_bytes')
    require(all(type(v.get(k)) is int and v[k]>=0 for v in memory for k in fields) and
        all(0<v['physical_footprint_bytes']<=v['physical_footprint_peak_bytes']<=CRITERIA['operator_physical_bytes'] and
            v['compressed_bytes']<=v['compressed_peak_bytes'] for v in memory),'Missing or excessive operator memory')
    clean=(all(v['compressed_bytes']==v['compressed_peak_bytes']==0 for v in memory) and
        len({v['decompressions'] for v in memory})==len({v['system_swap_used_bytes'] for v in memory})==1)
    result=dict(exact=True,clean_memory=clean,clean_host=clean_host,cases=cases,
        peak_physical_bytes=max(v['physical_footprint_peak_bytes'] for v in memory),normal_request_latency_qualified=False,production_promoted=False)
    if validation:return result
    savings=[a-b for a,b in weighted];ratios=[b/a for a,b in weighted];interval=paired_log_interval(ratios)
    promising=clean and clean_host and interval['high']<1 and statistics.median(savings)>=10
    return dict(result,status='worth_verifier_screen' if promising else 'insufficient_isolated_benefit',advance_to_verifier_screen=promising,
        shape_frequency_projection_ms=savings,median_projection_ms_per_token=statistics.median(savings),weighted_ratios=ratios,
        confidence_95=interval,limitations=LIMITATIONS)


def exact_capture(output,frozen,work):
    return profile.exact_observation(load(output/'capture.json'),frozen,work,sha(output/'workload.json'),load(ANCHOR),'commands')


def setup(exp,directory):
    proof=builder.verify(directory,exp.frozen['build']);cfg=builder.settings(directory)
    host=build_probe(ROOT/'.cache/benchmark-host'/hashlib.sha256(str(exp.out).encode()).hexdigest())
    save(exp.out/'producer.json',proof['producer']);save(exp.out/'host-producer.json',host['producer'])
    shutil.copyfile(PROTOCOL,exp.out/'protocol.md')
    exp.report.update(criteria=CRITERIA,limitations=LIMITATIONS,build_directory=str(directory),host_preflight=[],source_timing_reused=False)
    freeze(exp,[*proof['files'],*host['files'],PROTOCOL,exp.out/'protocol.md',exp.out/'producer.json',exp.out/'host-producer.json',ANCHOR])
    exp.env.pop('FREELLM_Q8_EXPANDED',None);exp.persist();exp.guard.check_resources(initial=True)
    return cfg,host


def host_check(exp,host,mode):
    observed=preflight(exp,host,mode)
    if observed['host']['power_source']!='AC Power':raise ResourceBlocked('AC power required before '+mode)


def capture(output,directory):
    work,tokens=verifier.source_input();exp=Experiment(output,'q8_expanded_capture_v1',[verifier.configuration(4,expert_slots=1460)],work,210)
    with exp:
        cfg,host=setup(exp,directory);exp.report['token_source']=tokens;exp.persist();host_check(exp,host,'capture')
        try:exp.command([cfg['capture_binary'],exp.model,exp.prepared,exp.out/'workload.json',exp.out/'capture.json','4','timing','1460'],'capture',limit=150)
        except subprocess.CalledProcessError:
            if (exp.out/'capture.json').exists() and admission_failure(load(exp.out/'capture.json')):raise ResourceBlocked('Fixed input-capture memory admission failed') from None
            raise
        exact=exact_capture(exp.out,exp.frozen,work);exp.report['capture']=exact;exp.persist()
        manifest=fixtures.prepare(exp.out/'inputs',exp.out/'fixtures',exp.model,exp.frozen['build'])
        freeze(exp,[*exp.out.joinpath('inputs').iterdir(),*exp.out.joinpath('fixtures').iterdir()])
        exp.report['fixture_manifest_sha256']=sha(exp.out/'fixtures/manifest.json');exp.report['input_bytes']=manifest['input_bytes'];exp.persist()
        if not exact['clean_memory'] or not exact['clean_host']:raise ResourceBlocked('Input capture memory or host disturbed; no capture timing is qualified')
        exp.report.update(status='captured',native_model_loaded=True)
    return exp.report


def source_proof(source,frozen,work):
    seal=sha(source/'evidence-files.json');verify_seal(source,seal);saved=load(source/'summary.json');original=load(source/'identity.json')
    require(saved['kind']=='q8_expanded_capture_v1' and saved['status'] in ('captured','resource_blocked') and
        all(original[k]==frozen[k] for k in ('build','artifact_revision','prepared_manifest_sha256')) and saved['workload']==work,
        'Changed fixture capture identity')
    exact=exact_capture(source,original,work);require(exact==saved['capture'],'Changed numerical capture proof')
    proof=builder.verify(Path(saved['build_directory']),frozen['build']);require(proof['producer']==load(source/'producer.json'),'Changed fixture producer')
    manifest=fixtures.verify(source/'inputs',source/'fixtures',ROOT/'.cache/qwen-mixed-reference',frozen['build'])
    require(saved['fixture_manifest_sha256']==sha(source/'fixtures/manifest.json'),'Changed fixture manifest')
    return dict(path=str(source),seal_sha256=seal,recorded_complete=saved['complete'],recorded_status=saved['status'],
        exact=exact,manifest_sha256=sha(source/'fixtures/manifest.json'),use='input bytes and numerical identity only'),manifest


def run(output,directory,source):
    work,tokens=verifier.source_input();exp=Experiment(output,'q8_expanded_screen_v1',[],work,CRITERIA['stage_seconds'])
    with exp:
        cfg,host=setup(exp,directory);proof,manifest=source_proof(source,exp.frozen,work)
        exp.report.update(token_source=tokens,source_proof=proof,native_model_loaded=False);freeze(exp,[*source.joinpath('fixtures').iterdir(),source/'summary.json',source/'evidence-files.json'])
        exp.persist();hashes=None
        for mode in ('validate','timing'):
            host_check(exp,host,mode)
            exp.command([cfg['binary'],source/'fixtures/manifest.json',exp.model,exp.out/(mode+'.json'),mode],mode,limit=90,validation=mode=='validate')
            raw=load(exp.out/(mode+'.json'));result=analyze(raw,mode=='validate',exp.frozen,manifest)
            current=[c['reference_sha256'] for c in raw['cases']]
            require(hashes is None or hashes==current,'Cross-process references changed');hashes=current
            exp.report[mode]=result;exp.persist()
            if not result['clean_memory'] or not result['clean_host']:raise ResourceBlocked('Expanded operator memory or host disturbed')
        exp.report.update(status=result['status'],advance_to_verifier_screen=result['advance_to_verifier_screen'])
    return exp.report


def audit(output):
    seal=sha(output/'evidence-files.json');verify_seal(output,seal);saved=load(output/'summary.json');frozen=load(output/'identity.json')
    require(saved['criteria']==CRITERIA and saved['limitations']==LIMITATIONS and saved['source_timing_reused'] is False and
        saved['identity']=={k:frozen[k] for k in saved['identity']} and sha(output/'protocol.md')==frozen['files'][str(PROTOCOL)],'Changed protocol')
    provenance=verify_sources(output.parent,frozen);proof=builder.verify(Path(saved['build_directory']),frozen['build'])
    require(proof['producer']==load(output/'producer.json') and all(frozen['files'][str(p)]==sha(p) for p in proof['files']),'Changed build')
    work,tokens=verifier.source_input();require(saved['workload']==load(output/'workload.json')==work and saved['token_source']==tokens,'Changed workload')
    modes=['capture'] if saved['kind']=='q8_expanded_capture_v1' else ['validate','timing']
    require(saved['kind'] in ('q8_expanded_capture_v1','q8_expanded_screen_v1') and len(saved['host_preflight'])<=len(modes),'Wrong stage')
    host=load(output/'host-producer.json');require(host['complete'] and host['base_native_fingerprint']==frozen['build'] and
        host['binary_sha256']==sha(host['binary'])==frozen['files'][host['binary']],'Changed host producer')
    for mode,item in zip(modes,saved['host_preflight']):
        p=output/(mode+'-host.json');require(item==dict(source=p.name,sha256=sha(p),observation=observe_host(load(p),frozen['build'])),'Changed host evidence')
    if modes==['capture']:
        if 'capture' in saved:
            require(saved['capture']==exact_capture(output,frozen,work),'Changed capture')
            fixtures.verify(output/'inputs',output/'fixtures',ROOT/'.cache/qwen-mixed-reference',frozen['build'])
        if saved['complete']:require(saved['status']=='captured' and saved['capture']['clean_memory'] and saved['capture']['clean_host'],'Incomplete capture')
    else:
        source=Path(saved['source_proof']['path']);p,manifest=source_proof(source,frozen,work);require(p==saved['source_proof'],'Changed source')
        for mode in modes:
            if mode in saved:require(saved[mode]==analyze(load(output/(mode+'.json')),mode=='validate',frozen,manifest),'Changed result')
        if saved['complete']:
            require(len(saved['host_preflight'])==2 and all(saved[m]['clean_memory'] and saved[m]['clean_host'] for m in modes) and
                saved['status']==saved['timing']['status'] and saved['advance_to_verifier_screen']==saved['timing']['advance_to_verifier_screen'] and
                [c['reference_sha256'] for c in load(output/'validate.json')['cases']]==[c['reference_sha256'] for c in load(output/'timing.json')['cases']],
                'Incomplete expanded screen')
    if not saved['complete']:require(saved['status'] in ('failed','resource_blocked','interrupted','time_budget_exhausted'),'Invalid incomplete status')
    return dict(kind='q8_expanded_audit_v1',complete=True,audit_passed=True,recorded_complete=saved['complete'],recorded_status=saved['status'],
        source_seal_sha256=seal,source_provenance=provenance,production_promoted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['capture','run','verify'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--build-directory',type=Path);p.add_argument('--source',type=Path)
    a=p.parse_args()
    if a.action=='verify':save(a.output,audit(a.source.resolve()))
    elif a.action=='capture':raise SystemExit(0 if capture(a.output,a.build_directory.resolve())['complete'] else 2)
    else:raise SystemExit(0 if run(a.output,a.build_directory.resolve(),a.source.resolve())['complete'] else 2)
