#!/usr/bin/env python3
"""Bounded fixed-width validation and early rejection on the retained recovery baseline."""
import argparse
import hashlib
from pathlib import Path

import build_mtp_widths as builder
from benchmark_host import build_probe
from cache_residency import require
from combined_q4 import freeze
from mtp_evidence import account
from perfect_draft import configuration
from prepare_mtp import verify as verify_artifact
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
from screen_mtp_continuation import ROOT, PREPARED, host_check, workloads
from stage200 import Experiment
from target_recovery_checks import observe, resource_failure
from trial_recovery import read, reference_work


def validation_cases():
    work = dict(reference_work(), max_tokens=7, name='irregular-seven')
    cases = [(width, dict(work, force_prefix=keep)) for width in (1, 2, 4)
             for keep in range(1, width+1)]
    eos = dict(work, name='immediate-eos', eos_ids=[work['expected_prompt_id']], force_prefix=1)
    return cases + [(width, dict(eos)) for width in (1, 2, 4)]


def observed(raw, work, input_sha, validation):
    result = observe(raw, work, input_sha, validation, width_report=True)
    require(raw['target_recovery']=='full-replay', 'Width screening retains full-replay until recovery qualifies')
    kernel_policy(raw)
    totals = account(raw)
    return dict(result, requested_width=raw['requested_width'],
                acceptance_by_position={str(k):v for k,v in totals['acceptance_by_position'].items()},
                target_expert_bytes_per_committed_token=totals['target_expert_bytes_per_committed_token'])


def kernel_policy(raw):
    # MtpDraft::execute selects a row tile from the current row count. A stats
    # snapshot reports the last execution's tile, not a permanent kernel policy.
    before=raw['before']['metal']['kernels'];after=raw['after']['metal']['kernels']
    tile=lambda rows: rows if rows in (2,4) else 1
    require(before['token_tile']==tile((raw['prompt_tokens']-1)%16) and
            after['token_tile']==tile(raw['cycles'][-1]['committed_tokens']),
            'Kernel row tile differs from the actual draft input shape')
    policy=lambda value: {k:v for k,v in value.items() if k!='token_tile'}
    require(policy(before)==policy(after), 'Kernel policy changed during run')
    return policy(before)


def compare_widths(a, b):
    """Width may change proposal grouping, never committed arithmetic or state."""
    require(a['kind']==b['kind']=='native_mtp_width_v1' and a['complete'] is b['complete'] is True,
            'Incomplete width comparison')
    fields = ('producer_binary_sha256','mode','validation','target_recovery','admission',
              'draft_manifest_sha256','prompt_tokens','requested_tokens','eos_ids','prime_logits_sha256',
              'committed_token_ids','row_logits_sha256','final_target_state','final_draft_state',
              'generated_tokens','next_id','stop_reason','host_checkpoint_allocated_bytes')
    require(all(a[k]==b[k] for k in fields), 'Width changed identity, logits, output or persistent state')
    for raw in (a,b):
        account(raw)
        require(raw['validation'] or raw['generated_tokens']==raw['requested_tokens'],
                'Fixed-length width timing stopped before the requested output length')
    require(kernel_policy(a)==kernel_policy(b) and
            a['before']['prepared']==b['before']['prepared'] and
            a['before']['artifact_revision']==b['before']['artifact_revision'] and
            a['before']['expert_cache']['diagnostic_cache_state']==b['before']['expert_cache']['diagnostic_cache_state'],
            'Changed arithmetic, artifact or starting cache')
    for key in ('recipe','budget_bytes','context','norm_convention'):
        require(a['draft_before'][key]==b['draft_before'][key], 'Changed draft configuration')
    for key in ('fixed_allocation_bytes','reserved_bytes','peak_incremental_bound_bytes'):
        require(a['target_recovery_journal'][key]==b['target_recovery_journal'][key], 'Changed journal capacity')
    if a['validation']:
        require(a['requested_width']==1, 'Use serial-width validation as the boundary reference')
        boundaries = {r['tokens']:r for r in a['boundaries']}
        require(len(boundaries)==a['generated_tokens'] and
                len(b['boundaries'])==len(b['cycles']) and
                all(boundaries.get(r['tokens'])==r for r in b['boundaries']),
                'A committed boundary differs from serial-width replay')
    return dict(exact_all_logits_tokens_and_state=True,
                ratio=b['decode_wall_ns']/a['decode_wall_ns'],
                control_tps=a['tokens_per_second'],candidate_tps=b['tokens_per_second'])


def setup(exp, directory):
    cfg, proof = builder.verify(directory)
    require(proof['base_native_fingerprint']==exp.frozen['build'], 'Native base changed')
    save(exp.out/'producer.json', proof)
    save(exp.out/'draft-audit.json', verify_artifact(PREPARED))
    host = build_probe(ROOT/'.cache/benchmark-host'/hashlib.sha256(str(exp.out).encode()).hexdigest())
    freeze(exp, [*builder.inputs(cfg), *builder.generated(cfg['output']), *cfg['objects'], cfg['binary'],
                 *host['files'], PREPARED/'manifest.json', PREPARED/'dense.bin', PREPARED/'experts.bin',
                 Path(__file__), exp.model/'tokenizer.json', exp.model/'generation_config.json'])
    exp.env.update(FREELLM_Q8_EXPANDED='packed', FREELLM_MTP_NGRAM_INIT='lazy',
                   FREELLM_MTP_EXPERT_SCRATCH='off', FREELLM_MTP_DIRECT_OUTPUT='on',
                   FREELLM_TARGET_RECOVERY='full-replay')
    exp.guard.check_resources(initial=True)
    return cfg, host


def sample(exp, cfg, host, width, work, stem, validation):
    input_path = exp.out/(stem+'.input.json'); save(input_path, work); freeze(exp,[input_path])
    host_check(exp,host,stem); exp.env['FREELLM_MTP_WIDTH']=str(width)
    path = exp.out/(stem+'.json')
    exp.command([cfg['binary'],exp.model,PREPARED,input_path,path,'fast-validate' if validation else 'fast-timing'],
                stem,limit=180,validation=validation)
    raw=read(path)
    require(raw['requested_width']==width and raw['producer_binary_sha256']==read(exp.out/'producer.json')['binary_sha256'],
            'Wrong width or producer')
    result=observed(raw,work,sha(input_path),validation)
    exp.report['samples'].append(dict(width=width,source=path.name,input=input_path.name,sha256=sha(path),**result))
    exp.persist()
    require(validation or result['completed_requested_length'],
            'Fixed-length width timing stopped before the requested output length')
    return raw,result


def reusable_validation(source, proof):
    source=Path(source).resolve();verify_seal(source,sha(source/'evidence-files.json'))
    summary=read(source/'summary.json')
    require(summary.get('kind')=='mtp_width_validation_v1' and read(source/'producer.json')==proof,
            'Reuse requires the same native validation producer')
    # Revalidate raw reports with today's checker. Historical Python checker
    # edits do not change the sealed native producer or its exact input bytes.
    require(all(sha(p)==h for key in ('inputs','generated','objects') for p,h in proof[key].items()) and
            sha(proof['binary'])==proof['binary_sha256'], 'Reused native sources changed')
    cases=validation_cases();result={};seen=set()
    for row in summary.get('samples',[]):
        name=row['source'];matches=[i for i in range(len(cases)) if name==f'case-{i}.json']
        require(len(matches)==1 and matches[0] not in seen, 'Unknown or duplicate validation case')
        i=matches[0];seen.add(i);width,work=cases[i];path=source/name;inp=source/f'case-{i}.input.json'
        require(sha(path)==row['sha256'] and row['input']==inp.name and read(inp)==work,
                'Changed reusable report or workload')
        raw=read(path)
        require(raw['producer_binary_sha256']==proof['binary_sha256'] and raw['requested_width']==width,
                'Changed reusable native producer or width')
        check=observed(raw,work,sha(inp),True)
        if check['clean_memory'] and check['clean_host']: result[i]=(source,path,inp)
    return result


def validate(output, directory, diagnostic=False, resume=None):
    cases=validation_cases()
    exp=Experiment(output,'mtp_width_validation_v1',[configuration(4,expert_slots=1460)],cases,1800)
    with exp:
        cfg,host=setup(exp,directory)
        reference=ROOT/'docs/benchmarks/2026-09-17-target-recovery/numerical-diagnostic-01'
        verify_seal(reference,sha(reference/'evidence-files.json'))
        oracle=read(reference/'case-0-full-replay.json')
        require(oracle['complete'] is True and oracle['mode']=='fast-validate', 'Missing independent logit reference')
        freeze(exp,[reference/p for p in ('evidence-files.json','case-0-full-replay.json','case-0.json')])
        exp.report.update(performance_measurement=False,diagnostic=diagnostic,resource_qualified=False,
                          samples=[],comparisons=[],production_promoted=False,reused_validation=[])
        reusable=reusable_validation(resume,read(exp.out/'producer.json')) if resume else {}
        references={}
        for case,(width,work) in enumerate(cases):
            if case in reusable:
                source,path,inp=reusable[case]
                freeze(exp,[path,inp,*[source/p for p in ('producer.json','evidence-files.json','summary.json')]])
                (exp.out/path.name).write_bytes(path.read_bytes());(exp.out/inp.name).write_bytes(inp.read_bytes())
                raw=read(path);result=observed(raw,work,sha(inp),True)
                exp.report['samples'].append(dict(width=width,source=path.name,input=inp.name,sha256=sha(path),**result))
                exp.report['reused_validation'].append(dict(case=case,source=str(path),sha256=sha(path)))
                print(f'case-{case}: reused complete clean numerical validation',flush=True)
            else:
                raw,result=sample(exp,cfg,host,width,work,f'case-{case}',True)
            key=work['name']
            if width==1:
                n=raw['generated_tokens']
                require(raw['prime_logits_sha256']==oracle['prime_logits_sha256'] and
                        raw['row_logits_sha256']==oracle['row_logits_sha256'][:n] and
                        raw['committed_token_ids']==oracle['committed_token_ids'][:n],
                        'Serial-width logits differ from the established independent reference')
                references[key]=raw
            compared=compare_widths(references[key],raw)
            exp.report['comparisons'].append(dict(case=case,width=width,force_prefix=work['force_prefix'],
                exact_all_logits_tokens_and_state=compared['exact_all_logits_tokens_and_state']))
            exp.persist()
            if not result['clean_host']: raise ResourceBlocked(resource_failure(raw,'Width validation host disturbed'))
            if diagnostic:
                memory=[raw['before_load'],raw['before']['process'],
                        *[c[k] for c in raw['cycles'] for k in ('memory_before','memory_after')],
                        raw['after']['process'],raw['after_destroy']]
                require(max(m['compressed_peak_bytes'] for m in memory)<=512*1024**2,
                        'Diagnostic exceeded compression bound')
                if any(b['system_swap_used_bytes']>a['system_swap_used_bytes'] for a,b in zip(memory,memory[1:])):
                    raise ResourceBlocked('Diagnostic observed system swap growth')
            elif not result['clean_memory']:
                raise ResourceBlocked(resource_failure(raw,'Width validation memory disturbed'))
        exp.report.update(status='numerically_exact',resource_qualified=all(s['clean_memory'] for s in exp.report['samples']),
                          performance_measurement=False)
    return exp.report


def validation_prerequisite(source, proof):
    source=Path(source).resolve();verify_seal(source,sha(source/'evidence-files.json'))
    summary=read(source/'summary.json');identity=read(source/'identity.json')
    require(summary.get('kind')=='mtp_width_validation_v1' and summary.get('complete') is True and
            summary.get('status')=='numerically_exact' and read(source/'producer.json')==proof,
            'Complete same-producer numerical validation is required')
    require(all(Path(p).is_file() and sha(p)==h for p,h in identity['files'].items()), 'Validation sources changed')
    cases=validation_cases();require(len(summary['samples'])==len(cases), 'Missing width cases')
    references={};clean=True
    for i,(width,work) in enumerate(cases):
        row=summary['samples'][i];path=source/f'case-{i}.json';input_path=source/f'case-{i}.input.json'
        require(row['source']==path.name and row['input']==input_path.name and row['width']==width and
                sha(path)==row['sha256'] and read(input_path)==work, 'Validation workload or report changed')
        raw=read(path);require(raw['producer_binary_sha256']==proof['binary_sha256'] and raw['requested_width']==width,
                              'Validation producer or width changed')
        result=observed(raw,work,sha(input_path),True)
        require(all(row.get(k)==v for k,v in result.items()), 'Validation summary differs from raw observations')
        if width==1:references[work['name']]=raw
        compare_widths(references[work['name']],raw)
        clean &= result['clean_memory'] and result['clean_host']
    return clean,[source/p for p in ('summary.json','identity.json','producer.json','evidence-files.json')]


def screen(output, directory, validation_source, candidate):
    require(candidate in (1,2), 'Compare width one or two against width four')
    work=next(w for w in workloads(ROOT/'.cache/qwen-mixed-reference',128) if w['name']=='lru_cache')
    work['max_tokens']=64
    exp=Experiment(output,'mtp_width_screen_v1',[configuration(4,expert_slots=1460)],work,720)
    with exp:
        cfg,host=setup(exp,directory)
        clean_validation,files=validation_prerequisite(validation_source,read(exp.out/'producer.json'));freeze(exp,files)
        exp.report.update(performance_measurement=True,preliminary=True,advancement_allowed=False,
            full_clean_correctness=clean_validation,controlled_change='requested_width',recovery='full-replay',
            control_width=4,candidate_width=candidate,samples=[],pairs=[],historical_timing_reused=False,
            first_pair_required_gain=.03,geometric_required_gain=.03,
            limitations=['64 generated tokens after one coding prompt; no 2K/4K/7K or sustained qualification.',
                         'Two fresh alternating pairs are a screen, not a confidence bound.',
                         'Width one includes draft catch-up cost; all widths reserve identical cache and checkpoint capacity.',
                         'A numerical diagnostic prerequisite does not establish clean full-model qualification.'])
        save(exp.out/'case-0.json',work);freeze(exp,[exp.out/'case-0.json'])
        for pair,order in enumerate(((4,candidate),(candidate,4))):
            values={}
            for width in order:
                raw,result=sample(exp,cfg,host,width,work,f'pair-{pair}-width-{width}',False)
                exp.report['samples'][-1].update(case=0,pair=pair,arm=str(width));exp.persist()
                if not result['clean_memory'] or not result['clean_host']:
                    raise ResourceBlocked(resource_failure(raw,'Width timing resources disturbed'))
                values[width]=raw
            compared=compare_widths(values[4],values[candidate])
            require(values[4]['input_sha256']==values[candidate]['input_sha256'], 'Timing workload differs')
            exp.report['pairs'].append(dict(case=0,pair=pair,**compared));exp.persist()
            ratios=[p['ratio'] for p in exp.report['pairs']]
            if ratios[0]>.97 or any(r>=1 for r in ratios) or (len(ratios)==2 and (ratios[0]*ratios[1])**.5>.97):
                exp.report.update(status='insufficient_early_gain');break
        else: exp.report.update(status='promising_early_screen')
    return exp.report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('validate','screen'))
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--diagnostic',action='store_true')
    parser.add_argument('--validation-source',type=Path)
    parser.add_argument('--resume',type=Path,help='Reuse complete clean same-producer validation runs only')
    parser.add_argument('--candidate',type=int,choices=(1,2),default=2)
    args=parser.parse_args()
    if args.stage=='validate':result=validate(args.output,args.build,args.diagnostic,args.resume)
    else:
        parser.error('--resume applies only to correctness; timing samples are always fresh') if args.resume else None
        parser.error('--validation-source is required for screening') if args.validation_source is None else None
        parser.error('--diagnostic applies only to numerical validation') if args.diagnostic else None
        result=screen(args.output,args.build,args.validation_source,args.candidate)
    raise SystemExit(0 if result['complete'] else 2)
