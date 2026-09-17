#!/usr/bin/env python3
"""Profile the updated target path; verified source tokens are numerical evidence only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import block_compute_profile as analysis
import build_mtp_target_profile as builder
from benchmark_host import build_probe
from cache_residency import require
from combined_q4 import freeze
from perfect_draft import configuration
from prepare_mtp import verify as verify_artifact
from qualification_evidence import ResourceBlocked,save,sha,verify_seal
from screen_mtp_continuation import ROOT,BASE,PREPARED,host_check,observe
from screen_mtp_forward import clean
from stage200 import Experiment


def analyze_block(block,profile,build,revision):
    result=analysis.analyze_block(block,profile,build,revision,'commands')
    a,b=block['target_before'],block['target_after']
    expected=Counter({k:b['kernel_dispatches'].get(k,0)-a['kernel_dispatches'].get(k,0)
        for k in set(a['kernel_dispatches'])|set(b['kernel_dispatches'])})
    require(Counter(result['kernels'])==expected and result['dispatches']==b['dispatches']-a['dispatches'] and
        result['command_groups']==b['submissions']-a['submissions'],'Incomplete target-only operation population')
    return result


def observation(directory,raw,reference,build,revision):
    require(raw['kind']=='native_mtp_target_profile_v1' and raw['complete'] is True and raw['validation'] is False and
        raw['performance_measurement'] is False and raw['profile_workspace_bytes']==builder.WORKSPACE and
        reference['kind']=='native_mtp_continuation_v1' and reference['complete'] is True and
        raw['mode']==reference['mode']=='fast-timing' and len(raw['cycles'])==4 and
        raw['generated_tokens']==reference['generated_tokens']==16,'Incomplete or different target profile')
    for k in ('input_sha256','draft_manifest_sha256','prime_logits_sha256','committed_token_ids','row_logits_sha256',
              'final_target_state','final_draft_state','next_id','stop_reason'):
        require(raw[k]==reference[k],'Target profiling changed '+k)
    for k in ('target','draft_slots','draft_bytes','host_checkpoint_logits_bytes'):
        require(raw['admission'][k]==reference['admission'][k],'Changed profile memory control')
    require(raw['before']['expert_cache']['diagnostic_cache_state']==
        reference['before']['expert_cache']['diagnostic_cache_state'],'Changed initial target cache')
    require(raw['admission']['profile_workspace_bytes']==builder.WORKSPACE and
        raw['admission']['combined_bytes']==reference['admission']['combined_bytes']+builder.WORKSPACE<=12*1024**3,
        'Profile workspace not admitted')
    fields=('offset','width','proposals','accepted_proposals','committed_tokens','forced_rejection','next_id')
    require([{k:c[k] for k in fields} for c in raw['cycles']]==[{k:c[k] for k in fields} for c in reference['cycles']],
        'Profiling changed proposals or recovery path')
    blocks=[]
    for i,cycle in enumerate(raw['cycles']):
        name=f'profile-block-{4*i}.profile.json';path=directory/name
        require(cycle['profile_file']==name and cycle['profile_sha256']==sha(path) and path.stat().st_size<=32*1024**2,
            'Changed or oversized profile capture')
        require(cycle['width']==cycle['committed_tokens']==4,'Unqualified profile width or rejection')
        blocks.append(analyze_block(cycle,json.loads(path.read_text()),build,revision))
    return dict(exact_logits_tokens_draft_and_target_state=True,**clean(raw),captured_tokens=16,
        dependency_passes=192,blocks=blocks,
        mean_buckets_ms_per_token={k:sum(b['buckets_ms_per_token'][k] for b in blocks)/4 for k in blocks[0]['buckets_ms_per_token']},
        performance_measurement=False,production_promoted=False,
        limitations=['One short real-draft workload, four complete target calls; draft computation itself is not profiled.',
            'Intervals describe observed overlap, not the cause of a wait or attainable savings.',
            'Instrumented time is not normal request latency. No counter-derived isolated kernel costs are collected.'])


def run(output,directory,reference_run):
    reference_run=reference_run.resolve();verify_seal(reference_run,sha(reference_run/'evidence-files.json'))
    input_path=reference_run/'case-0.json';path=reference_run/'case-0-fast-timing.json'
    work=json.loads(input_path.read_text());reference=json.loads(path.read_text())
    observe(reference,work,sha(input_path),'fast-timing')
    require(len(work['prompt_ids'])==72 and work['max_tokens']==16 and all(c['width']==c['committed_tokens']==4 for c in reference['cycles']),
        'Profile requires a completed compatible short numerical reference')
    exp=Experiment(output,'mtp_real_target_profile_v1',[configuration(4,expert_slots=1460)],work,240)
    with exp:
        cfg,proof=builder.verify(directory);require(proof['base_native_fingerprint']==exp.frozen['build'],'Changed native base')
        reference_producer=json.loads((reference_run/'producer.json').read_text())
        require(reference_producer==builder.base.verify(Path(reference_producer['binary']).parent)[1],
            'Changed numerical reference producer')
        save(exp.out/'producer.json',proof);save(exp.out/'draft-audit.json',verify_artifact(PREPARED))
        host=build_probe(ROOT/'.cache/benchmark-host'/hashlib.sha256(str(exp.out).encode()).hexdigest())
        freeze(exp,[*builder.inputs(cfg),*builder.generated(cfg['output']),*cfg['objects'],cfg['binary'],*host['files'],
            PREPARED/'manifest.json',PREPARED/'dense.bin',PREPARED/'experts.bin',BASE/'protocol.md',BASE/'profile-protocol.md',
            path,input_path,reference_run/'producer.json',reference_run/'evidence-files.json'])
        exp.env['FREELLM_Q8_EXPANDED']='packed';exp.guard.check_resources(initial=True)
        require(sha(exp.out/'workload.json')==sha(input_path),'Changed profile workload bytes')
        exp.report.update(performance_measurement=False,reference_source=str(path),reference_sha256=sha(path),timing_samples_reused=False)
        host_check(exp,host,'profile')
        exp.command([cfg['binary'],exp.model,PREPARED,exp.out/'workload.json',exp.out/'profile.json','fast-timing'],'profile',limit=150)
        raw=json.loads((exp.out/'profile.json').read_text())
        result=observation(exp.out,raw,reference,exp.frozen['build'],exp.frozen['artifact_revision'])
        exp.report['capture']=result;exp.persist()
        if not result['clean_memory'] or not result['clean_host']:raise ResourceBlocked('Updated target profile resource disturbance')
        exp.report['status']='captured'
    return exp.report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('output','build','reference-run'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.output,a.build,a.reference_run)
