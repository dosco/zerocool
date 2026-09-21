#!/usr/bin/env python3
"""Capture current-width target dependencies; instrumented time is never speed evidence."""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path

import block_compute_profile as analysis
import build_mtp_width_profile as builder
from benchmark_host import build_probe
from cache_residency import require
from combined_q4 import freeze
from perfect_draft import configuration
from prepare_mtp import verify as verify_artifact
from q4_request_profile import command_classes
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
from screen_mtp_continuation import PREPARED, host_check
from screen_mtp_widths import observed, compare_widths, reusable_validation, validation_cases
from stage200 import Experiment
from target_recovery_checks import resource_failure

ROOT=builder.ROOT
BASE=ROOT/'docs/benchmarks/2026-09-17-mtp-widths'
REFERENCES={
    'lru_cache': ('long-lru-01','pair-0-width-1.json','case-0.json',1),
    'merge_intervals': ('other-coding-01','case-0-pair-0-width-4.json','case-0.json',4),
}
MODES={
    'commands': ('native_mtp_width_profile_v1','mtp_width_target_profile_v1','mtp_width_profile_producer_v1'),
    'dispatch': ('native_mtp_width_counter_profile_v1','mtp_width_target_counters_v1','mtp_width_counter_profile_producer_v1'),
}


def read(path): return json.loads(Path(path).read_text())


def window(cycles, prompt_tokens, total):
    selected=[];committed=verified=0;first=None
    for i,c in enumerate(cycles):
        output=c['offset']-prompt_tokens
        if output>=builder.START and committed<builder.TOKENS:
            if first is None:first=output
            require(output==first+committed,'Noncontiguous profile window')
            selected.append(i);committed+=c['committed_tokens'];verified+=c['width']
    require(first is not None and builder.START<=first<builder.START+4 and
            builder.TOKENS<=committed<=builder.TOKENS+3 and len(selected)<=builder.TOKENS and
            first+committed<=total,'Incomplete middle-generation window')
    return selected,dict(requested_start=builder.START,requested_committed=builder.TOKENS,
        first_output=first,end_output=first+committed,captured_calls=len(selected),
        captured_committed=committed,captured_verified=verified,omitted_before=first,
        omitted_after=total-first-committed)


def validate_request(raw, reference, work, input_sha, producer_sha, mode='commands'):
    require(mode in MODES, 'Unknown current-width profile mode')
    require(raw.get('kind')==MODES[mode][0] and raw.get('complete') is True and
            raw.get('performance_measurement') is False and raw['producer_binary_sha256']==producer_sha and
            raw['profile_workspace_bytes']==builder.WORKSPACE and
            raw['requested_width'] in (1,4) and raw['requested_tokens']==raw['generated_tokens']==128,
            'Incomplete or incompatible current-width profile')
    control=observed(reference,work,input_sha,False)
    require(control['clean_memory'] and control['clean_host'] and control['completed_requested_length'],
            'Numerical reference is not a complete clean width run')
    expected=dict(reference['admission'],profile_workspace_bytes=builder.WORKSPACE,
                  combined_bytes=reference['admission']['combined_bytes']+builder.WORKSPACE)
    require(raw['admission']==expected and expected['combined_bytes']<=12*1024**3,
            'Profile workspace changed cache capacity or exceeds admission')
    # Reuse the numerical/resource contract only after checking the explicit
    # profiler identity and extra admission. The copied projection is never
    # written or exposed as normal timing evidence.
    normalized=copy.deepcopy(raw)
    normalized['kind']='native_mtp_width_v1';normalized['admission']=reference['admission']
    result=observed(normalized,work,input_sha,False)
    fields=('mode','validation','input_sha256','draft_manifest_sha256','requested_width','prompt_tokens',
            'requested_tokens','eos_ids','prime_logits_sha256','committed_token_ids','row_logits_sha256',
            'final_target_state','final_draft_state','generated_tokens','next_id','stop_reason',
            'host_checkpoint_allocated_bytes','target_recovery_journal')
    require(all(raw[k]==reference[k] for k in fields),'Profiling changed model identity, logits, outputs or persistent state')
    for boundary in ('before','after'):
        a,b=raw[boundary],reference[boundary]
        require(all(a[k]==b[k] for k in ('prepared','artifact_revision','memory_plan','execution',
                    'completion_pipeline','ready_group','chunk_tokens','io_workers','short_append_tokens')) and
                all(a['metal'][k]==b['metal'][k] for k in ('kernels','build_fingerprint','device','physical_bytes')),
                'Profiling changed runtime controls')
    require(raw['before']['expert_cache']['diagnostic_cache_state']==reference['before']['expert_cache']['diagnostic_cache_state'],
            'Profiling changed initial expert cache')
    require(all(raw['draft_before'][k]==reference['draft_before'][k] for k in ('recipe','budget_bytes','context','norm_convention')),
            'Profiling changed draft identity')
    fields=('cycle_id','offset','width','proposals','accepted_proposals','committed_tokens',
            'forced_rejection','next_id','target_recovery_forward_calls')
    require([{k:c[k] for k in fields} for c in raw['cycles']]==
            [{k:c[k] for k in fields} for c in reference['cycles']], 'Profiling changed proposal/recovery path')
    selected,coverage=window(raw['cycles'],raw['prompt_tokens'],raw['generated_tokens'])
    require(raw['profile_window']==coverage,'False profile coverage claim')
    return dict(exact_logits_tokens_draft_and_target_state=True,
                **{k:result[k] for k in ('clean_memory','clean_host','peak_physical_bytes')}),selected,coverage


def analyze_block(cycle,profile,build,revision,mode='commands'):
    require(mode in MODES, 'Unknown current-width profile mode')
    result=analysis.analyze_block(cycle,profile,build,revision,mode,tokens=cycle['width'],
                                  expert_geometry='down_projection',entry_limit=builder.ENTRY_LIMIT)
    a,b=cycle['target_before'],cycle['target_after']
    expected=Counter({k:b['kernel_dispatches'].get(k,0)-a['kernel_dispatches'].get(k,0)
                      for k in set(a['kernel_dispatches'])|set(b['kernel_dispatches'])})
    require(Counter(result['kernels'])==expected and result['dispatches']==b['dispatches']-a['dispatches'] and
            result['command_groups']==b['submissions']-a['submissions'] and
            a['kernels']['profile'] is b['kernels']['profile'] is True and
            a['kernels']['counter_profile'] is b['kernels']['counter_profile'] is (mode=='dispatch') and
            a['live_command_groups']==b['live_command_groups']==0,
            'Target dispatch population or command ownership differs')
    result.update(cycle_id=cycle['cycle_id'],committed_tokens=cycle['committed_tokens'],
                  layer_classes=command_classes(profile['command_groups'],cycle['width'],True))
    return result


def observation(directory,raw,reference,work,input_sha,producer_sha,build,revision,mode='commands'):
    result,selected,coverage=validate_request(raw,reference,work,input_sha,producer_sha,mode)
    names={f'profile-cycle-{i}.profile.json' for i in selected}
    require({p.name for p in directory.glob('*.profile.json')}==names,'Missing or extra profile files')
    blocks=[];sources=[]
    for i,c in enumerate(raw['cycles']):
        fields=('profile_file','profile_sha256','forward_begin_ns','forward_end_ns','forward_ns','target_before','target_after')
        if i not in selected:
            require(not any(k in c for k in fields),'Out-of-window capture');continue
        name=f'profile-cycle-{i}.profile.json';path=directory/name
        require(c.get('profile_file')==name and path.stat().st_size<=32*1024**2 and
                c['profile_sha256']==sha(path) and c['forward_ns']==c['verify_ns'],
                'Changed, oversized or incorrectly timed profile')
        profile=read(path);blocks.append(analyze_block(c,profile,build,revision,mode))
        sources.append(dict(path=str(path),sha256=sha(path)))
    committed=coverage['captured_committed'];totals=defaultdict(float);classes=defaultdict(lambda:[0.0,0,0])
    for b in blocks:
        for k,v in b['buckets_ms_per_token'].items():totals[k]+=v*b['tokens']/committed
        for c in b['classes']:
            value=classes[tuple(c['stages'])]
            value[0]+=c['gpu_command_ms_per_token']*b['tokens']/committed
            value[1]+=c['command_groups'];value[2]+=c['dispatches']
    if mode=='dispatch': result['counter_operations']=counter_operations(blocks,committed)
    return dict(result,coverage=coverage,dependency_passes=48*len(blocks),blocks=blocks,sources=sources,
        target_ms_per_committed_token=sum(totals.values()),buckets_ms_per_committed_token=dict(totals),
        command_classes=[dict(stages=list(k),gpu_command_ms_per_committed_token=v[0],
            command_groups=v[1],dispatches=v[2]) for k,v in sorted(classes.items(),key=lambda x:-x[1][0])],
        performance_measurement=False,production_promoted=False,
        limitations=['Only the declared middle-generation window is traced; the full 128-token request is checked numerically.',
            'Target verification is separate from unprofiled draft, checkpoint and recovery work.',
            'Instrumented intervals describe observed overlap, not a causal stall or recoverable latency.',
            'Mixed command groups retain every stage; no group duration is assigned to each constituent kernel.',
            'Application expert reads are not observed physical storage traffic.',
            'One capture per workload has no normal-throughput or promotion conclusion.'])


def counter_operations(blocks,committed):
    require(type(committed) is int and committed>0, 'Invalid committed-token denominator')
    operations=defaultdict(lambda:[0.0,0])
    for block in blocks:
        for op in block['counter_operations']:
            key=tuple(op[k] for k in ('stage','kernel','K','N','rows'))
            value=operations[key]
            value[0]+=op['gpu_pass_ms_per_token']*block['tokens']/committed
            value[1]+=op['dispatches']
    return [dict(zip(('stage','kernel','K','N','rows'),key),
                 gpu_pass_ms_per_committed_token=value[0],dispatches=value[1])
            for key,value in sorted(operations.items(),key=lambda item:-item[1][0])]


def run(output,directory,case,mode='commands'):
    require(mode in MODES, 'Unknown current-width profile mode')
    producer=builder
    if mode=='dispatch':
        import build_mtp_width_counters as producer
    folder,name,work_name,width=REFERENCES[case];source=BASE/folder
    verify_seal(source,sha(source/'evidence-files.json'))
    reference=read(source/name);work=read(source/work_name);reference_proof=read(source/'producer.json')
    require(read(source/'summary.json')['complete'] is True and
            reference_proof==builder.base.verify(Path(reference_proof['binary']).parent)[1] and
            reference['producer_binary_sha256']==reference_proof['binary_sha256'] and
            reference['requested_width']==width, 'Changed reference producer or width')
    # All ten sealed numerical cases are independently rechecked with current
    # numerical checks. Native prerequisites stay fixed; timings are not reused.
    validation=BASE/'validation-04';usable=reusable_validation(validation,reference_proof)
    require(len(usable)==len(validation_cases()),'Missing clean numerical prerequisite')
    refs={}
    for i,(w,v) in enumerate(validation_cases()):
        raw=read(usable[i][1])
        if w==1:refs[v['name']]=raw
        compare_widths(refs[v['name']],raw)
    exp=Experiment(output,MODES[mode][1],[configuration(4,expert_slots=1460)],work,360)
    with exp:
        cfg,proof=producer.verify(directory);require(proof['base_native_fingerprint']==exp.frozen['build'],'Native base changed')
        save(exp.out/'producer.json',proof);save(exp.out/'reference-producer.json',reference_proof)
        save(exp.out/'draft-audit.json',verify_artifact(PREPARED))
        host=build_probe(ROOT/'.cache/benchmark-host'/hashlib.sha256(str(exp.out).encode()).hexdigest())
        freeze(exp,[*producer.inputs(cfg),*producer.generated(cfg['output']),*cfg['objects'],cfg['binary'],*host['files'],
            PREPARED/'manifest.json',PREPARED/'dense.bin',PREPARED/'experts.bin',Path(__file__),
            BASE/'next-protocol.md',*[p for p in source.iterdir() if p.is_file()],
            *[p for p in validation.iterdir() if p.is_file()],
            exp.model/'tokenizer.json',exp.model/'generation_config.json'])
        if mode=='dispatch':
            freeze(exp,[ROOT/'docs/benchmarks/2026-09-17-mtp-width-profile/counter-protocol.md'])
        require(sha(exp.out/'workload.json')==sha(source/work_name),'Profile workload bytes differ')
        exp.env.update(ZEROCOOL_Q8_EXPANDED='packed',ZEROCOOL_MTP_NGRAM_INIT='lazy',ZEROCOOL_MTP_EXPERT_SCRATCH='off',
            ZEROCOOL_MTP_DIRECT_OUTPUT='on',ZEROCOOL_TARGET_RECOVERY='full-replay',ZEROCOOL_MTP_WIDTH=str(width))
        exp.report.update(performance_measurement=False,reference_source=str(source/name),
            reference_sha256=sha(source/name),timing_samples_reused=False,requested_width=width)
        exp.guard.check_resources(initial=True);host_check(exp,host,'profile')
        exp.command([cfg['binary'],exp.model,PREPARED,exp.out/'workload.json',exp.out/'profile.json','fast-timing'],
                    'profile',limit=240)
        raw=read(exp.out/'profile.json')
        result=observation(exp.out,raw,reference,work,sha(source/work_name),proof['binary_sha256'],
                           exp.frozen['build'],exp.frozen['artifact_revision'],mode)
        exp.report['capture']=result;exp.persist()
        if not result['clean_memory'] or not result['clean_host']:
            raise ResourceBlocked(resource_failure(raw,'Width profile resources disturbed'))
        exp.report.update(status='captured')
    return exp.report


def query(index,selector):
    """Compact source-verified answer; never reinterpret profile time as speed."""
    data,source=index.json(selector);root=Path(source['path']).parent
    modes=[mode for mode,kinds in MODES.items() if data.get('kind')==kinds[1]]
    require(len(modes)==1,'Select a current-width profile stage');mode=modes[0]
    verify_seal(root,sha(root/'evidence-files.json'))
    limits=['Instrumented time is not normal throughput or a speedup estimate.',
            'Overlap buckets do not establish the cause of a stall or recoverable latency.',
            'Profile coverage is bounded; unrelated requests and timings are not pooled.']
    if data.get('complete') is not True or data.get('status')!='captured':
        return dict(status='partial_diagnostic',recorded_status=data.get('status'),error=data.get('error'),
            sources=[source],limitations=limits,performance_measurement=False,production_promoted=False,
            possible_request_benefit=None,missing_evidence=['Complete clean exact target profile'],
            smallest_experiment='Resolve the recorded blocker before one fresh bounded capture; preserve this attempt.')
    raw=read(root/'profile.json');proof=read(root/'producer.json');identity=read(root/'identity.json')
    require(proof.get('kind')==MODES[mode][2] and proof.get('complete') is True and
            identity['files'].get(proof['binary'])==proof['binary_sha256'] and
            proof['base_native_fingerprint']==identity['build']==raw['before']['metal']['build_fingerprint'],
            'Profile producer is not bound to the frozen native build')
    reference_path=Path(data['reference_source']);verify_seal(reference_path.parent,sha(reference_path.parent/'evidence-files.json'))
    require(sha(reference_path)==data['reference_sha256'],'Numerical reference changed')
    reference=read(reference_path);reference_proof=read(root/'reference-producer.json')
    require(reference_proof==read(reference_path.parent/'producer.json') and
            reference['producer_binary_sha256']==reference_proof['binary_sha256'],'Reference producer differs')
    captured=observation(root,raw,reference,read(root/'workload.json'),sha(root/'workload.json'),
        proof['binary_sha256'],identity['build'],identity['artifact_revision'],mode)
    require(captured==data['capture'] and captured['clean_memory'] and captured['clean_host'],
            'Profile summary or resource qualification differs from raw evidence')
    buckets=captured['buckets_ms_per_committed_token'];largest=max(buckets,key=buckets.get)
    if largest=='gpu_active':
        hypothesis='GPU-active time dominates this captured target window; inspect its largest complete command classes.'
        experiment='Use the recorded dispatch shapes to isolate one costly operation, then require a fresh normal-request screen.'
    elif largest=='gpu_idle_pending_read':
        hypothesis='GPU idle time often overlaps pending expert reads; their actual blocking dependencies need confirmation.'
        experiment='Inspect the captured layer/read lifetimes before testing one I/O or scheduling change.'
    elif largest in ('gpu_idle_ready_expert','gpu_idle_submitted'):
        hypothesis='Ready or submitted work overlaps the largest idle bucket in this target window.'
        experiment='Inspect ready-to-submit and GPU admission intervals before another kernel experiment.'
    else:
        hypothesis='The largest target-window bucket is not attributed to GPU execution or a confirmed read dependency.'
        experiment='Add only the missing linkage needed to attribute the recorded target gap.'
    extra={}
    if mode=='dispatch':
        extra['counter_operations']=captured['counter_operations'][:12]
        limits.append('Counter sampling splits compute passes and perturbs execution; costs only rank an operator experiment.')
        limits.append('Per-dispatch intervals can overlap; their sums are not exclusive portions of request latency.')
    return dict(extra,status='instrumented_diagnostic',hypothesis=hypothesis,smallest_experiment=experiment,
        possible_request_benefit=None,coverage=captured['coverage'],dependency_passes=captured['dependency_passes'],
        target_ms_per_committed_token=captured['target_ms_per_committed_token'],
        buckets_ms_per_committed_token=buckets,command_classes=captured['command_classes'][:4],
        peak_physical_bytes=captured['peak_physical_bytes'],performance_measurement=False,production_promoted=False,
        missing_evidence=['A causal mechanism and fresh paired normal-request timing before selecting an optimization'],
        limitations=limits+captured['limitations'],sources=[source,
            *[dict(path=str(p),sha256=sha(p)) for p in (root/'profile.json',root/'producer.json',root/'identity.json',reference_path)],
            *captured['sources']])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('output','build'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--case',choices=tuple(REFERENCES),required=True)
    parser.add_argument('--mode',choices=tuple(MODES),default='commands')
    args=parser.parse_args();result=run(args.output,args.build,args.case,args.mode)
    raise SystemExit(0 if result['complete'] else 2)
