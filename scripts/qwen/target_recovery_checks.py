"""Revalidate raw target-recovery evidence. Summary claims are never sufficient."""
import math

from cache_residency import require
from mtp_evidence import account
from screen_mtp_forward import clean
from screen_mtp_recovery import equivalent

ARMS = ('full-replay', 'state-only')
MIB = 1024**2


def resource_failure(raw, label):
    """Explain a failed resource gate without changing its acceptance rules."""
    if 'before_load' in raw:
        memory = [raw['before_load'], raw['before']['process']]
        memory += [c[k] for c in raw.get('cycles', []) for k in ('memory_before', 'memory_after')]
        memory += [raw['after']['process'], raw['after_destroy']]
    else:
        memory = [raw[k] for k in ('before', 'after')]
    reasons = []
    compressed = max(max(m['compressed_bytes'], m['compressed_peak_bytes']) for m in memory)
    if compressed:
        reasons.append(f'process compression observed: {compressed} bytes')
    decompressions = [m['decompressions'] for m in memory]
    if len(set(decompressions)) != 1:
        reasons.append(f'process decompressions changed: {decompressions[0]} to {decompressions[-1]}')
    swap = [m['system_swap_used_bytes'] for m in memory]
    if len(set(swap)) != 1:
        direction = 'decreased' if swap[-1] < swap[0] else 'increased' if swap[-1] > swap[0] else 'fluctuated'
        reasons.append(f'system swap {direction}: {swap[0]} to {swap[-1]} bytes (range {min(swap)}..{max(swap)})')
        if not compressed and len(set(decompressions)) == 1:
            reasons.append('no process compression or decompression observed')
    for key in ('host_before', 'host_after'):
        host = raw[key]
        if host['thermal_state'] != 0 or host['low_power_mode'] or host['power_source'] != 'AC Power':
            reasons.append(f'{key}: thermal={host["thermal_state"]}, low_power={host["low_power_mode"]}, power={host["power_source"]}')
    return label + ': ' + '; '.join(reasons or ['resources disturbed; see raw observations'])


def replay_resources(raw):
    memory=[raw[k] for k in ('before','after')]
    fields=('physical_footprint_bytes','physical_footprint_peak_bytes','compressed_bytes',
        'compressed_peak_bytes','decompressions','system_swap_used_bytes')
    require(all(type(m.get(k)) is int and m[k]>=0 for m in memory for k in fields),
        'Missing replay memory observation')
    require(all(0<m['physical_footprint_bytes']<=m['physical_footprint_peak_bytes']<=2*1024**3 for m in memory),
        'Replay exceeded 2GiB process ceiling')
    host=[raw[k] for k in ('host_before','host_after')]
    require(all(type(h.get('thermal_state')) is int and type(h.get('low_power_mode')) is bool and
        isinstance(h.get('power_source'),str) for h in host),'Missing replay host observation')
    return dict(clean_memory=all(m['compressed_bytes']==m['compressed_peak_bytes']==0 for m in memory) and
        len({m['decompressions'] for m in memory})==len({m['system_swap_used_bytes'] for m in memory})==1,
        clean_host=all(h['thermal_state']==0 and h['low_power_mode'] is False and h['power_source']=='AC Power' for h in host),
        peak_physical_bytes=max(m['physical_footprint_peak_bytes'] for m in memory))


def observe(raw, work, input_sha, validation, *, width_report=False):
    require(raw.get('kind') == ('native_mtp_width_v1' if width_report else 'native_mtp_continuation_v2') and raw.get('complete') is True and
            raw['validation'] is validation and raw['mode'] == ('fast-validate' if validation else 'fast-timing') and
            raw['input_sha256'] == input_sha and isinstance(raw['request_id'],str) and raw['request_id'].startswith(input_sha+':') and
            raw['target_recovery'] in ARMS, 'Incomplete or incompatible recovery evidence')
    totals = account(raw)
    count = min(8,work['max_tokens']) if validation and width_report else 8 if validation else work['max_tokens']
    require(raw['requested_tokens'] == count and raw['prompt_tokens'] == len(work['prompt_ids']) and
            raw['eos_ids'] == work['eos_ids'] and raw['decode_including_reporting_ns'] >= raw['decode_wall_ns'],
            'Changed workload or incomplete timing')
    ids = raw['committed_token_ids']; offset = len(work['prompt_ids']) + len(ids)
    require(not any(i in work['eos_ids'] for i in ids[:-1]) and
            ((raw['stop_reason']=='eos' and ids[-1] in work['eos_ids']) or
             (raw['stop_reason']=='length' and len(ids)==count and ids[-1] not in work['eos_ids'])), 'Invalid EOS coverage')
    require(len(raw['row_logits_sha256']) == len(ids) and
            all(isinstance(h,str) and len(h)==64 for h in raw['row_logits_sha256']), 'Missing full logits')
    require(raw['final_target_state']['valid'] is True and raw['final_target_state']['tokens']==offset and
            len(raw['final_target_state']['layers'])==48 and all(l['position']==offset for l in raw['final_target_state']['layers']) and
            raw['final_target_state']['history']==(work['prompt_ids']+ids)[-2:] and
            raw['final_draft_state']['valid'] is True and raw['final_draft_state']['position']==offset-1,
            'Final persistent state positions differ')
    for l,layer in enumerate(raw['final_target_state']['layers']):
        sizes=[122880,3145728,0,0,0,368640 if l==1 else 0] if (l+1)%4 else [0,0,8192*512*4,8192*512*4,8192*128*4,0]
        require(len(layer['buffers'])==6 and all((b is None if not n else isinstance(b,dict) and b.get('bytes')==n and
            isinstance(b.get('sha256'),str) and len(b['sha256'])==64) for b,n in zip(layer['buffers'],sizes)), 'Incomplete persistent buffer hashes')
    require(all(isinstance(raw['final_draft_state'].get(k),str) and len(raw['final_draft_state'][k])==64 for k in ('keys','values','index')),
            'Missing draft state hashes')
    state_only = raw['target_recovery']=='state-only'; rejected = verifies = 0
    for c in raw['cycles']:
        require(c['width'] in ((1,2,4) if width_report else (1,4)), 'Unsupported recovery verifier width')
        verifies += c['width']>1
        partial = c['committed_tokens']<c['width']; rejected += partial
        expected_calls = c['committed_tokens'] if partial and not state_only else 0
        require(c['target_recovery_forward_calls']==expected_calls and
                (not state_only or c['target_recovery_read_bytes']==0), 'Recovery executed unexpected target forwards or reads')
        if not partial:
            require(c['target_restore_ns']==c['target_repair_ns']==c['target_recovery_read_bytes']==0,
                    'Recovery reported on a fully accepted cycle')
        require(validation or c['forced_rejection'] is False, 'Forced rejection in timing sample')
    if validation and ids[0] not in work['eos_ids']:
        first=raw['cycles'][0];keep=work.get('force_prefix',1)
        if first['width']>1:
            require(1<=keep<=first['width'] and first['committed_tokens']==keep and
                    first['forced_rejection'] is (keep<first['width']), 'Missing forced prefix position')
        else:
            require(width_report and first['committed_tokens']==1 and first['forced_rejection'] is False,
                    'Invalid width-one validation cycle')
    j=raw['target_recovery_journal']; admission=raw['admission']
    require(j['reserved_bytes']==admission['target_recovery_reserve_bytes']==16*MIB and
            0<j['fixed_allocation_bytes']<=j['peak_incremental_bound_bytes']<=16*MIB and
            j['captures']==(verifies if state_only else 0) and j['repairs']==(rejected if state_only else 0) and
            j['active'] is False and j['complete'] is (state_only and verifies>0) and
            0<=j['peak_gpu_groups']<=2, 'Unbounded or incomplete recovery journal')
    require(admission['draft_slots']==32 and admission['expert_scratch_reserve_bytes']==2*MIB and
            admission['combined_bytes']<=12*1024**3 and admission['target']['expert_slots']==1460 and
            admission['target']['limit_bytes']==12*1024**3 and raw['ngram_initialization']=='lazy', 'Changed memory/configuration')
    for p in ('before','after'):
        d=raw['direct_output_'+p]; scratch=raw['expert_scratch_'+p]; metal=raw[p]['metal']
        require(metal['device']=='Apple M1 Pro' and metal['physical_bytes']==32*1024**3 and
                metal['kernels']['q4_decode']=='reference' and metal['kernels']['q8_decode_rows']==2 and
                metal['kernels']['gdn']=='original' and metal['kernels']['route_selection']=='simd', 'Changed machine or arithmetic policy')
        require(d['enabled'] is True and d['active'] is False and scratch['enabled'] is False and
                scratch['forwards']==scratch['groups']==0 and metal['active_scratch_slot']==-1 and
                metal['live_command_groups']==0 and not metal['kernels']['profile'] and
                not metal['kernels']['counter_profile'], 'Uncontrolled mode or outstanding GPU users')
    return dict(**clean(raw), tokens_per_second=raw['tokens_per_second'], generated_tokens=len(ids),
                rejected_cycles=rejected, component_ms_per_token=totals['component_ms_per_token'],
                recovery_children_ms_per_token=totals['recovery_children_ms_per_token'],
                repeated_target_rows=totals['repeated_target_rows'],
                target_recovery_read_bytes=sum(c['target_recovery_read_bytes'] for c in raw['cycles']),
                completed_requested_length=len(ids)==count, stop_reason=raw['stop_reason'])


def comparison(a,b):
    require((a['target_recovery'],b['target_recovery'])==ARMS, 'Wrong recovery arms')
    equivalent(a,b)
    fields=('mode','validation','committed_token_ids','row_logits_sha256','final_draft_state','next_id','stop_reason')
    require(all(a[k]==b[k] for k in fields), 'Recovery changed logits, committed tokens or draft state')
    require(a['before']['expert_cache']['diagnostic_cache_state']==b['before']['expert_cache']['diagnostic_cache_state'], 'Changed initial target cache')
    require(all(a['draft_before'][k]==b['draft_before'][k] for k in ('recipe','budget_bytes','context','norm_convention')),
            'Changed draft configuration')
    # Post-run expert/ngram caches may legitimately differ: eliminated replays no
    # longer touch them. Persistent inference state must still match exactly.
    for key in ('fixed_allocation_bytes','reserved_bytes','peak_incremental_bound_bytes'):
        require(a['target_recovery_journal'][key]==b['target_recovery_journal'][key], 'Unequal journal capacity')
    return dict(exact_all_logits_tokens_and_state=True,ratio=b['decode_wall_ns']/a['decode_wall_ns'],
        control_tps=a['tokens_per_second'],candidate_tps=b['tokens_per_second'],
        eliminated_target_forward_calls=sum(c['target_recovery_forward_calls'] for c in a['cycles']),
        eliminated_target_read_bytes=sum(c['target_recovery_read_bytes'] for c in a['cycles']),
        candidate_target_recovery_forward_calls=sum(c['target_recovery_forward_calls'] for c in b['cycles']),
        candidate_target_recovery_read_bytes=sum(c['target_recovery_read_bytes'] for c in b['cycles']))


def short_gate(pairs):
    require(1<=len(pairs)<=2, 'Short gate requires one or two pairs')
    rs=[p['ratio'] for p in pairs]
    require(all(type(r) in (int,float) and math.isfinite(r) and r>0 for r in rs), 'Invalid pair ratio')
    return rs[0]<=.95 and all(r<1 for r in rs) and math.prod(rs)**(1/len(rs))<=.95


def long_gate(pairs):
    require(len(pairs)==6 and {(p['case'],p['pair']) for p in pairs}=={(c,p) for c in range(3) for p in range(2)},
            'Long screen requires two pairs per case')
    by={c:[p['ratio'] for p in pairs if p['case']==c] for c in range(3)}
    require(all(type(r) in (int,float) and math.isfinite(r) and r>0 for rs in by.values() for r in rs),'Invalid long ratio')
    cases={c:math.sqrt(math.prod(rs)) for c,rs in by.items()}
    return dict(passed=max(cases.values())<=1.02 and math.prod(cases.values())**(1/3)<=.95,
        per_case_geometric_ratio=cases,across_case_geometric_ratio=math.prod(cases.values())**(1/3),
        confidence_qualified=False)
