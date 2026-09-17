#!/usr/bin/env python3
"""Reconstruct MTP screen evidence without running inference or accepting gaps."""
import argparse
import json
from pathlib import Path
from cache_residency import require
from qualification_evidence import sha,save,verify_seal
from screen_mtp_forward import clean,compare_joint


def recovery_sample(raw,mode,workload_sha,draft_sha,admission):
    """Check coverage before using even a completed process from a blocked run."""
    validation=mode in ('validate','fast-validate')
    require(raw['complete'] is True and raw['mode']==mode and raw['validation'] is validation and
        raw['input_sha256']==workload_sha and raw['draft_manifest_sha256']==draft_sha and
        raw['admission']==admission,'incompatible recovery sample')
    cycles=raw['cycles'];count=4 if validation else 16
    require(raw['generated_tokens']==count==sum(c['committed_tokens'] for c in cycles) and
        raw['decode_wall_ns']>0 and raw['decode_wall_ns']==sum(c['wall_ns'] for c in cycles) and
        raw['tokens_per_second']==count*1e9/raw['decode_wall_ns'],'incomplete recovery coverage')
    require(raw['proposed_tokens']==sum(c['width']-1 for c in cycles) and
        raw['accepted_proposals']==sum(c['accepted_proposals'] for c in cycles) and
        all(c['width'] in (1,4) and len(c['proposals'])==c['width'] and
            0<=c['accepted_proposals']<c['width'] and c['committed_tokens']==c['accepted_proposals']+1
            for c in cycles),'incomplete recovery acceptance')
    if validation:
        require(len(raw['boundaries'])==len(cycles) and any(c['forced_rejection'] and
            c['accepted_proposals']==0 for c in cycles),'missing recovery validation')
    else:require(not any(c['forced_rejection'] for c in cycles),'forced rejection in timing')
    observed=clean(raw)
    memory=[raw['before_load'],raw['after_destroy'],*[raw[k]['process'] for k in ('before','after')],
        *[c[k] for c in cycles for k in ('memory_before','memory_after')]]
    return dict(**observed,max_observed_compressed_peak_bytes=max(m['compressed_peak_bytes'] for m in memory),
        decompressions=max(m['decompressions'] for m in memory)-min(m['decompressions'] for m in memory),
        observed_swap_range_bytes=max(m['system_swap_used_bytes'] for m in memory)-min(m['system_swap_used_bytes'] for m in memory))


def verify_recovery(directory,saved,producer,root):
    from screen_mtp_recovery import equivalent
    from screen_residency import paired_log_interval
    def read(base,name):return json.loads((base/name).read_text())
    require(saved['normal_request_latency_qualified'] is False and saved['production_promoted'] is False and
        saved['timing_samples_reused'] is False,'changed recovery qualification scope')
    workload_sha=sha(directory/'workload.json')
    require(read(directory,'workload.json')==saved['workload'],'changed recovery workload')
    draft_sha=read(directory,'draft-audit.json')['manifest_sha256']
    fixture=Path(saved['fixture_source']);fixture=fixture if fixture.is_absolute() else root/fixture
    verify_seal(fixture,sha(fixture/'evidence-files.json'))
    fixture_summary=read(fixture,'summary.json');fixture_native=read(fixture,'native.json')
    require(fixture_summary['complete'] is True and fixture_summary['status']=='forward_fixture_validated' and
        read(fixture,'producer.json')==producer and fixture_native['state_only_catchup_exact'] is True and
        fixture_native['state_only_expert_cache_untouched'] is True,'missing state-only fixture proof')
    numerical=directory
    if 'numerical_source' in saved:
        numerical=Path(saved['numerical_source']['path']);verify_seal(numerical,sha(numerical/'evidence-files.json'))
        source=read(numerical,'summary.json')
        require(source['complete'] is False and source['status']=='resource_blocked' and
            source['workload']==saved['workload'] and read(numerical,'producer.json')==producer and
            sha(numerical/'workload.json')==workload_sha,'changed numerical source binding')
    checks=[];validation=[];admission=None
    for mode in ('validate','fast-validate'):
        raw=read(numerical,mode+'.json')
        if admission is None:admission=raw['admission']
        validation.append(dict(mode=mode,source=str(numerical/(mode+'.json')),sha256=sha(numerical/(mode+'.json')),
            **recovery_sample(raw,mode,workload_sha,draft_sha,admission)))
        checks.append(raw)
    equivalent(*checks)
    require(saved.get('exact_recovery') is True or saved['status']=='resource_blocked','missing exact recovery decision')
    samples=[];by={}
    order=[(p,a) for p in range(5) for a in (('control','candidate') if p%2==0 else ('candidate','control'))]
    recorded=saved.get('samples',[])
    require([(s['pair'],s['arm']) for s in recorded]==order[:len(recorded)],'changed alternating recovery order')
    for sample in recorded:
        pair,arm=sample['pair'],sample['arm'];mode='timing' if arm=='control' else 'fast-timing'
        path=directory/f'pair-{pair}-{arm}.json';raw=read(directory,path.name)
        observed=recovery_sample(raw,mode,workload_sha,draft_sha,admission)
        require(sample['source']==path.name and sample['sha256']==sha(path) and
            sample['tokens_per_second']==raw['tokens_per_second'] and
            all(sample[k]==observed[k] for k in ('clean_memory','clean_host','peak_physical_bytes')),'changed recovery observation')
        samples.append(dict(source=path.name,sha256=sha(path),tokens_per_second=raw['tokens_per_second'],**observed))
        by[pair,arm]=raw
    pairs=[];numerical_pairs=0
    for p in range(5):
        if any((p,a) not in by for a in ('control','candidate')):break
        a,b=(by[p,arm] for arm in ('control','candidate'));equivalent(a,b);numerical_pairs+=1
        if not all(clean(r)[k] for r in (a,b) for k in ('clean_memory','clean_host')):break
        pairs.append(dict(pair=p,ratio=b['decode_wall_ns']/a['decode_wall_ns'],
            control_tps=a['tokens_per_second'],candidate_tps=b['tokens_per_second']))
    require(saved.get('pairs',[])==pairs,'changed usable recovery pairs')
    if saved['complete'] is True:
        if len(pairs)==1:
            require(pairs[0]['candidate_tps']<5 and saved['status']=='below_absolute_target' and
                saved['early_stop'] is True and saved['paired_confidence_qualified'] is False,'invalid early stop')
        else:
            require(len(pairs)==5,'missing five recovery pairs')
            interval=paired_log_interval([p['ratio'] for p in pairs]);floor=min(p['candidate_tps'] for p in pairs)
            status='promising_short_screen' if interval['high']<1 and floor>=5 else 'inconclusive_short_screen'
            require(saved['status']==status and saved['confidence_95']==interval and
                saved['all_candidates_at_least_5_tps']==(floor>=5) and saved['paired_confidence_qualified'] is True,'changed recovery decision')
    else:require(saved['status'] in ('resource_blocked','failed','time_budget_exhausted','interrupted'),'missing unfinished status')
    return dict(passed=True,recorded_status=saved['status'],recorded_complete=saved['complete'],
        numerical_equality_verified=True,validation_sources=validation,timing_samples=samples,
        timing_pairs_numerically_equal=numerical_pairs,usable_timing_pairs=len(pairs),
        timing_qualified=saved['complete'] is True,paired_confidence_qualified=saved['complete'] is True and len(pairs)==5,
        normal_request_latency_qualified=False,production_promoted=False)


def verify(directory,snapshot=None):
    directory=Path(directory).resolve();verify_seal(directory,sha(directory/'evidence-files.json'))
    saved=json.loads((directory/'summary.json').read_text());frozen=json.loads((directory/'identity.json').read_text())
    producer=json.loads((directory/'producer.json').read_text());root=Path(frozen['root'])
    require(producer['complete'] is True and producer['base_native_fingerprint']==frozen['build'],'missing producer identity')
    checked=[]
    for name,digest in {**producer['inputs'],**producer['generated'],**producer['objects'],producer['binary']:producer['binary_sha256']}.items():
        p=Path(name);candidates=[p]
        if snapshot is not None:candidates.append(Path(snapshot)/p.relative_to(root))
        match=next((c for c in candidates if c.is_file() and sha(c)==digest),None)
        require(match is not None,'missing immutable build input: '+name);checked.append(str(match))
    if saved['kind']=='mtp_recovery_screen_v1':
        result=verify_recovery(directory,saved,producer,root);result['build_files']=len(checked);return result
    if saved['complete'] is not True:
        require(saved['status'] in ('resource_blocked','failed','time_budget_exhausted','interrupted'),'unfinished status missing')
        return dict(passed=True,recorded_status=saved['status'],timing_qualified=False,build_files=len(checked))
    if saved['kind']=='mtp_forward_fixture_v1':
        native=json.loads((directory/'native.json').read_text());ref=json.loads((directory/'reference.json').read_text())
        require(native['complete'] is True and native['input_sha256']==sha(directory/'input.json') and
            native['draft_manifest_sha256']==json.loads((directory/'draft-audit.json').read_text())['manifest_sha256'],'fixture identity mismatch')
        require(saved['native']==clean(native) and all(saved['native'][k] for k in ('clean_memory','clean_host')),'fixture memory differs')
        require(ref['passed'] is True and ref['native_sha256']==sha(directory/'native.json') and ref['input_sha256']==sha(directory/'input.json') and
            saved['reference']==ref and len(ref['cases'])==22 and all(c['passed'] for c in ref['cases']),'reference proof mismatch')
        for name,t in native['fixture_tensors'].items():require(sha(directory/'trace'/(name+'.bin'))==t['sha256'],'changed fixture bytes')
        return dict(passed=True,recorded_status=saved['status'],independent_reference_passed=True,build_files=len(checked),timing_qualified=False)
    require(saved['kind']=='mtp_joint_screen_v1','unknown MTP evidence kind')
    serial=json.loads((directory/'serial.json').read_text());candidate=json.loads((directory/'timing.json').read_text())
    ratio=compare_joint(serial,candidate)
    require(saved['candidate_to_serial_latency_ratio']==ratio,'changed comparison ratio')
    for mode in ('validate','serial','timing'):
        raw=json.loads((directory/(mode+'.json')).read_text());observed=clean(raw)
        require(raw['complete'] is True and raw['input_sha256']==sha(directory/'workload.json') and
            all(observed[k] for k in ('clean_memory','clean_host')) and all(saved[mode][k]==v for k,v in observed.items()),'changed joint observation')
        require(saved[mode]['tokens_per_second']==raw['tokens_per_second']==raw['generated_tokens']*1e9/raw['decode_wall_ns'],'changed throughput')
        require(raw['accepted_proposals']==sum(c['accepted_proposals'] for c in raw['cycles']) and
            raw['proposed_tokens']==sum(c['width']-1 for c in raw['cycles']),'changed acceptance coverage')
    raw=json.loads((directory/'validate.json').read_text());require(any(c['forced_rejection'] and c['accepted_proposals']==0 for c in raw['cycles']),'missing forced rejection')
    require(saved['status']==('promising_short_screen' if ratio<1 else 'slower_than_serial'),'changed screen decision')
    ref=json.loads((directory/'real-fixture/reference.json').read_text())
    require(ref['passed'] is True and ref['native_sha256']==sha(directory/'real-fixture/native.json') and
        ref['input_sha256']==sha(directory/'real-input.json'),'missing real-hidden reference')
    return dict(passed=True,recorded_status=saved['status'],ratio=ratio,actual_cost_counted=True,
        build_files=len(checked),paired_confidence_qualified=False,production_promoted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--snapshot',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=verify(a.directory,a.snapshot);save(a.output,r);print(json.dumps(r))
