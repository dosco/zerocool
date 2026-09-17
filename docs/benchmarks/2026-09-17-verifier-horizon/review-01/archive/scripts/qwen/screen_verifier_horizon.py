#!/usr/bin/env python3
"""Measure a perfect-proposal ceiling; never label it real speculative speed."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import build_verifier_horizon as builder
from benchmark_host import build_probe
from cache_residency import require
from combined_q4 import freeze
from perfect_draft import configuration
from prepare_mtp import verify as verify_artifact
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
from screen_mtp_continuation import PREPARED, host_check
from screen_mtp_forward import clean
from screen_mtp_widths import observed
from stage200 import Experiment
from target_recovery_checks import resource_failure, replay_resources

ROOT=builder.ROOT
SOURCE=ROOT/'docs/benchmarks/2026-09-17-mtp-widths/other-coding-01'
PROTOCOL=ROOT/'docs/benchmarks/2026-09-17-verifier-horizon/protocol.md'


def read(path): return json.loads(Path(path).read_text())


def reference():
    verify_seal(SOURCE,sha(SOURCE/'evidence-files.json'))
    path=SOURCE/'case-0-pair-0-width-4.json';raw=read(path);work=read(SOURCE/'case-0.json')
    result=observed(raw,work,sha(SOURCE/'case-0.json'),False)
    require(result['clean_memory'] and result['clean_host'] and result['completed_requested_length'],
            'Horizon requires a complete clean numerical reference')
    return dict(work,continuation_ids=raw['committed_token_ids'],
        expected_next_ids=raw['committed_token_ids'][1:]+[raw['next_id']],
        row_logits_sha256=raw['row_logits_sha256'],prime_logits_sha256=raw['prime_logits_sha256'],
        final_target_state=raw['final_target_state'],reference=dict(path=str(path),sha256=sha(path)))


def prefix(work,count):
    require(type(count) is int and 1<=count<=len(work['continuation_ids']),'Invalid horizon prefix')
    result=dict(work,max_tokens=count)
    for key in ('continuation_ids','expected_next_ids','row_logits_sha256'): result[key]=work[key][:count]
    if count!=len(work['continuation_ids']): result.pop('final_target_state',None)
    return result


def validate(raw,work,input_sha,producer_sha,width,validation,streamed=False):
    require(type(width) is int and width in (1,4,8) and
        raw.get('kind')=='native_verifier_horizon_v1' and raw.get('complete') is True and
        raw['perfect_proposals_only'] is True and raw['normal_request_latency_qualified'] is False and
        raw['production_promoted'] is False and raw['validation'] is validation and
        raw['mode']==('fast-validate' if validation else 'fast-timing') and
        raw['producer_binary_sha256']==producer_sha and raw['input_sha256']==input_sha and
        raw['requested_width']==width,'Incomplete or incompatible horizon')
    count=len(work['continuation_ids']);offset=len(work['prompt_ids']);wall=verified=0
    require(raw['requested_tokens']==raw['generated_tokens']==count and raw['prompt_tokens']==offset and
        raw['committed_token_ids']==work['continuation_ids'] and raw['row_logits_sha256']==work['row_logits_sha256'] and
        raw['prime_logits_sha256']==work['prime_logits_sha256'] and raw['next_id']==work['expected_next_ids'][-1],
        'Horizon changed known outputs or full logits')
    for c in raw['cycles']:
        n=width if count-verified>=width else 1
        require(c['offset']==offset+verified and c['width']==c['committed_tokens']==n and
            type(c['wall_ns']) is int and c['wall_ns']>0 and type(c['verify_ns']) is int and c['verify_ns']>0 and
            type(c['checkpoint_save_ns']) is int and c['checkpoint_save_ns']>=0 and
            c['verify_ns']+c['checkpoint_save_ns']<=c['wall_ns'],'Invalid horizon cycle')
        verified+=n;wall+=c['wall_ns']
    require(verified==count and raw['decode_wall_ns']==wall and
        raw['decode_including_reporting_ns']>=wall and
        raw['verify_wall_ns']==sum(c['verify_ns'] for c in raw['cycles']) and
        math.isclose(raw['verified_tokens_per_second'],count*1e9/wall,rel_tol=1e-12), 'Invalid horizon totals')
    require('tokens_per_second' not in raw and raw['excluded_costs']==
        ['proposal_generation','rejection_recovery','draft_state_catchup'],'Ceiling masquerades as real generation')
    require(raw['initial_draft_state']==raw['final_draft_state'] and raw['draft_before']==raw['draft_after'],
        'Horizon performed draft work')
    admission=raw['admission'];plan=admission['target']
    require(plan['limit_bytes']==12*1024**3 and plan['expert_slots']==1460 and admission['draft_slots']==32 and
        admission['checkpoint_rows']==8 and admission['combined_bytes']<=12*1024**3 and
        0<raw['host_checkpoint_allocated_bytes']<=128*1024**2,'Horizon memory controls differ')
    resident=5362515968
    if streamed:
        first,last=(raw['embedding_rows_'+k] for k in ('before','after'))
        for entry in (first,last):
            require(entry['storage']=='exact-packed-rows' and entry['capacity']==256 and
                entry['host_reserve_bytes']==2*1024**2 and 0<entry['fixed_host_bytes']<entry['host_reserve_bytes'] and
                entry['row_bytes']==2720 and entry['removed_resident_allocation_bytes']==675446784,
                'Changed streamed embedding geometry or budget')
        require(all(last[k]>=first[k] for k in ('hits','misses','evictions','application_read_bytes')) and
            last['application_read_bytes']-first['application_read_bytes']==2720*(last['misses']-first['misses']),
            'Incomplete streamed embedding reads')
        resident=resident-first['removed_resident_allocation_bytes']+first['host_reserve_bytes']
    else: require(not any(k.startswith('embedding_rows_') for k in raw),'Undeclared embedding storage change')
    require(plan['resident_bytes']==resident and admission['host_checkpoint_logits_bytes']==128*1024**2+4*8*248320*4+1024**2 and
        admission['draft_bytes']==318234624 and admission['expert_scratch_reserve_bytes']==2*1024**2 and
        admission['target_recovery_reserve_bytes']==16*1024**2 and
        admission['combined_bytes']==plan['planned_bytes']+admission['host_checkpoint_logits_bytes']+
        admission['draft_bytes']+admission['expert_scratch_reserve_bytes']+admission['target_recovery_reserve_bytes'],
        'Unaccounted horizon memory change')
    state=raw['final_target_state'];require(state['valid'] is True and state['tokens']==offset+count and
        state['history']==(work['prompt_ids']+work['continuation_ids'])[-2:] and len(state['layers'])==48 and
        all(l['position']==offset+count for l in state['layers']), 'Incomplete final target state')
    if 'final_target_state' in work: require(state==work['final_target_state'],'Changed full reference state')
    for edge in ('before','after'):
        stats=raw[edge];k=stats['metal']['kernels']
        require(stats['memory_plan']==plan and stats['artifact_revision']=='b2c422f3c643e36f04227a64d61796b44a4b1029' and
            stats['expert_cache']['policy']=='clock' and k['profile'] is False and k['q4_decode']=='reference' and
            k['q8_decode_rows']==2 and stats['metal']['live_command_groups']==0, 'Changed horizon execution controls')
    result=clean(raw)
    return dict(result,exact_full_logits=True,verified_tps=raw['verified_tokens_per_second'],
                normal_request_latency_qualified=False,perfect_proposals_only=True)


def compare(a,b,*,numerical_across_producers=False):
    fields=('kind','complete','mode','validation','input_sha256','producer_binary_sha256','admission',
        'draft_manifest_sha256','prompt_tokens','requested_tokens','prime_logits_sha256','committed_token_ids',
        'row_logits_sha256','final_target_state','initial_draft_state','final_draft_state','generated_tokens',
        'next_id','host_checkpoint_allocated_bytes','target_recovery_journal','excluded_costs')
    if numerical_across_producers:
        require(a['validation'] is b['validation'] is True,'Cross-producer comparison is numerical only')
        fields=tuple(k for k in fields if k!='producer_binary_sha256')
    require(all(a[k]==b[k] for k in fields),'Horizon changed numerical results, identity or allocation')
    require(a['before']['expert_cache']['diagnostic_cache_state']==b['before']['expert_cache']['diagnostic_cache_state'],
            'Different initial expert cache')
    for key in ('prepared','execution','completion_pipeline','ready_group','chunk_tokens','io_workers','short_append_tokens'):
        require(a['before'][key]==b['before'][key], 'Horizon changed '+key)
    if a['validation']:
        require(a['requested_width']==1,'Validation needs serial reference')
        boundaries={x['tokens']:x for x in a['boundaries']}
        require(len(boundaries)==a['generated_tokens'] and len(b['boundaries'])==len(b['cycles']) and
                all(boundaries.get(x['tokens'])==x for x in b['boundaries']),'Horizon persistent boundary differs')
    if numerical_across_producers:
        return dict(exact_logits_and_state=True,numerical_across_producers=True,timing_used=False,
            perfect_proposals_only=True,normal_request_latency_qualified=False)
    return dict(exact_logits_and_state=True,ratio=b['decode_wall_ns']/a['decode_wall_ns'],
        control_verified_tps=a['verified_tokens_per_second'],candidate_verified_tps=b['verified_tokens_per_second'],
        perfect_proposals_only=True,normal_request_latency_qualified=False)


def run(output,directory,streamed=False):
    if streamed:
        import build_streamed_verifier_horizon as producer_builder
    else: producer_builder=builder
    work=reference()
    configs=[dict(configuration(4,expert_slots=1460),token_tile=w,embedding_storage='rows' if streamed else 'resident') for w in (1,4,8)]
    exp=Experiment(output,'verifier_horizon_screen_v1',configs,work,720)
    with exp:
        cfg,proof=producer_builder.verify(directory);save(exp.out/'producer.json',proof)
        require(proof['base_native_fingerprint']==exp.frozen['build'],'Native base changed')
        save(exp.out/'draft-audit.json',verify_artifact(PREPARED))
        host=build_probe(ROOT/'.cache/benchmark-host'/hashlib.sha256(str(exp.out).encode()).hexdigest())
        freeze(exp,[*producer_builder.inputs(cfg),*producer_builder.generated(cfg['output']),*cfg['objects'],cfg['binary'],*host['files'],
            PREPARED/'manifest.json',PREPARED/'dense.bin',PREPARED/'experts.bin',PROTOCOL,
            Path(work['reference']['path']),SOURCE/'evidence-files.json'])
        exp.env.update(FREELLM_Q8_EXPANDED='packed',FREELLM_MTP_NGRAM_INIT='lazy',
            FREELLM_MTP_EXPERT_SCRATCH='off',FREELLM_MTP_DIRECT_OUTPUT='on',FREELLM_TARGET_RECOVERY='full-replay')
        exp.report.update(samples=[],comparisons=[],perfect_proposals_only=True,embedding_storage='rows' if streamed else 'resident')
        exp.guard.check_resources(initial=True)
        if streamed:
            freeze(exp,[PROTOCOL.parent/'streamed-protocol.md'])
            host_check(exp,host,'embedding',full=False)
            exp.command([cfg['binary'],'--streamed-embedding-test',exp.model,exp.out/'embedding.json'],'embedding',limit=60,validation=True)
            check=read(exp.out/'embedding.json');resources=replay_resources(check)
            require(check['kind']=='streamed_embedding_test_v1' and check['complete'] is True and
                check['performance_measurement'] is False and check['owner_released'] is check['invalid_request_atomic'] is True and
                check['failed_read_atomic'] is True and check['evictions_before_gpu_submission']==2 and
                check['peak_gpu_bytes']<=1024**3 and len(check['cases'])==10 and all(x['exact'] is True for x in check['cases']) and
                check['cache']['evictions']>0 and check['cache']['hits']>0, 'Incomplete real embedding checks')
            exp.report['embedding']=dict(resources,source='embedding.json',sha256=sha(exp.out/'embedding.json'));exp.persist()
            if not resources['clean_memory'] or not resources['clean_host']: raise ResourceBlocked(resource_failure(check,'embedding'))
        host_check(exp,host,'kernel',full=False)
        exp.command([cfg['binary'],'--horizon-kernel-test',exp.out/'kernel.json'],'kernel',limit=60,validation=True)
        kernel=read(exp.out/'kernel.json')
        require(kernel['kind']=='verifier_horizon_kernel_test_v1' and kernel['complete'] is True and
            kernel['performance_measurement'] is False and kernel['peak_gpu_bytes']<=128*1024**2 and
            {(x['K'],x['N'],x['float_output']) for x in kernel['cases']}==
                {(k,n,f) for k,n in ((2560,6144),(6144,2560)) for f in (False,True)} and
            len(kernel['cases'])==4 and all(x['exact_reference'] is x['guards_intact'] is True and x['rows']==8 for x in kernel['cases']),
            'Incomplete horizon kernel checks')
        resources=replay_resources(kernel);exp.report['kernel']=dict(resources,source='kernel.json',sha256=sha(exp.out/'kernel.json'));exp.persist()
        if not resources['clean_memory'] or not resources['clean_host']: raise ResourceBlocked(resource_failure(kernel,'kernel'))
        def sample(width,count,stem,validation):
            values=prefix(work,count);path=exp.out/(stem+'.input.json');save(path,values);freeze(exp,[path])
            host_check(exp,host,stem);exp.env['FREELLM_VERIFIER_HORIZON']=str(width)
            raw_path=exp.out/(stem+'.json')
            exp.command([cfg['binary'],exp.model,PREPARED,path,raw_path,'fast-validate' if validation else 'fast-timing'],
                        stem,limit=180,validation=validation)
            raw=read(raw_path);result=validate(raw,values,sha(path),proof['binary_sha256'],width,validation,streamed)
            exp.report['samples'].append(dict(width=width,source=raw_path.name,input=path.name,sha256=sha(raw_path),**result));exp.persist()
            if not result['clean_memory'] or not result['clean_host']: raise ResourceBlocked(resource_failure(raw,stem))
            return raw
        serial=sample(1,9,'validation-serial',True)
        for width in (4,8):
            raw=sample(width,9,'validation-'+str(width),True);exp.report['comparisons'].append(compare(serial,raw));exp.persist()
        a=sample(4,64,'pair-0-width-4',False);b=sample(8,64,'pair-0-width-8',False)
        decision=compare(a,b);exp.report['comparisons'].append(decision)
        # Require a substantial verifier improvement before paying for a larger
        # proposer/recovery implementation. Still no claim about actual yield.
        if decision['ratio']>.85 or decision['candidate_verified_tps']<6.5:
            exp.report.update(status='insufficient_verifier_headroom',advance_to_real_draft=False)
        else:
            b=sample(8,64,'pair-1-width-8',False);a=sample(4,64,'pair-1-width-4',False)
            reverse=compare(a,b);exp.report['comparisons'].append(reverse)
            advance=reverse['ratio']<=.85 and reverse['candidate_verified_tps']>=6.5
            exp.report.update(status='promising_verifier_ceiling' if advance else 'unconfirmed_verifier_headroom',
                advance_to_real_draft=False,advance_to_acceptance_cost_model=advance)
        exp.report['limitations']=['Known-correct token inputs exclude proposal generation, rejection and draft catch-up.',
            'The actual draft remains allocated and primed but does not execute during the measured continuation.',
            'One short coding prompt; neither 5 tokens/s generation nor 2K/4K/7K/session acceptance is established.',
            'The four-row recovery journal is reserved but unused; eight-row rejection recovery is not implemented.',
            'The 6.5 verified-token/s threshold is only an early screen. Measured draft yield and all costs must support 5 generated tokens/s.',
            'Two alternating pairs at most; no confidence-bounded promotion.']
    return exp.report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--stream-embeddings',action='store_true')
    args=p.parse_args();run(args.output,args.build,args.stream_embeddings)
