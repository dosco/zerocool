"""Offline MTP cycle accounting and source-verified paired comparisons."""
import math
from pathlib import Path

from evidence_index import read, parse, digest, terminal_result

WIDTH_KIND = 'native_mtp_width_v1'
NATIVE = {'native_mtp_continuation_v1', 'native_mtp_continuation_v2', WIDTH_KIND}
SCREENS = {'mtp_direct_output_screen_v1', 'mtp_target_recovery_screen_v1', 'mtp_width_screen_v1'}
ROOT_PHASES = ('draft', 'verify', 'recovery')
RECOVERY_PARTS = ('target_restore', 'target_repair', 'draft_restore', 'draft_catchup')


def need(ok, message):
    if not ok: raise ValueError(message)


def integer(value, minimum=0):
    need(type(value) is int and value >= minimum, 'Invalid integer measurement')
    return value


def original(path, expected=None):
    path = Path(path).resolve(); raw = read(path); sha = digest(raw)
    need(expected is None or sha == expected, 'Changed raw evidence: ' + str(path))
    return parse(raw), dict(path=str(path), sha256=sha)


def child(root, name):
    root = Path(root).resolve(); path = (root/name).resolve()
    need(path.is_relative_to(root) and path != root, 'Evidence path escapes experiment')
    return path


def screening_context(data):
    if data.get('kind')=='mtp_width_screen_v1':
        need(data.get('preliminary') is True and data.get('advancement_allowed') is False and
             data.get('production_promoted') is False and type(data.get('full_clean_correctness')) is bool,
             'Width screening cannot claim adoption')
        return dict(preliminary=True, full_correctness_stage_passed=data['full_clean_correctness'], advancement_allowed=False)
    early = data.get('stage') == 'early' or data.get('preliminary') is True
    if not early:
        return dict(preliminary=False)
    need(data.get('kind') == 'mtp_target_recovery_screen_v1' and data.get('stage') == 'early' and
         data.get('preliminary') is True and data.get('full_correctness_stage_passed') is False and
         data.get('advancement_allowed') is False and data.get('production_promoted') is False,
         'Early screening cannot claim correctness qualification or adoption')
    return dict(preliminary=True, full_correctness_stage_passed=False, advancement_allowed=False)


def account(raw):
    need(isinstance(raw, dict) and raw.get('kind') in NATIVE, 'Unsupported MTP native report')
    cs = raw.get('cycles', []); need(isinstance(cs, list), 'Missing cycle array')
    totals = dict.fromkeys((*ROOT_PHASES, 'wall', 'checkpoint_save'), 0)
    width_report = raw['kind'] == WIDTH_KIND
    requested_width = integer(raw.get('requested_width'), 1) if width_report else None
    if width_report:
        need(requested_width in (1, 2, 4), 'Unsupported requested MTP width')
        requested = integer(raw.get('requested_tokens'), 1)
    children = dict.fromkeys(RECOVERY_PARTS, 0) if raw['kind'].endswith('_v2') or width_report else None
    committed = []; offset = integer(raw.get('prompt_tokens'), 1); proposed = accepted = discarded = replayed = rejected = 0
    positions = {p: dict(proposed=0, accepted=0) for p in range(1, 4)}
    for number, c in enumerate(cs):
        width = integer(c.get('width'), 1); keep = integer(c.get('committed_tokens'), 1)
        need(width in ((1, 2, 4) if width_report else (1, 4)) and keep <= width and c.get('offset') == offset, 'Invalid cycle coverage')
        ids = c.get('proposals'); need(isinstance(ids, list) and len(ids) == width and
            all(type(t) is int and 0 <= t < 248320 for t in ids), 'Invalid proposal IDs')
        if width_report:
            remaining = requested - len(committed)
            expected_width = requested_width if remaining >= requested_width and ids[0] not in raw.get('eos_ids', []) else 1
            need(remaining > 0 and width == expected_width and keep <= remaining,
                 'Cycle differs from fixed width or output-tail schedule')
        need(c.get('accepted_proposals') == keep-1, 'Accepted length differs')
        for p in range(1, width):
            positions[p]['proposed'] += 1; positions[p]['accepted'] += p < keep
        wall = integer(c.get('wall_ns'), 1)
        parts = {p: integer(c.get(p+'_ns')) for p in ROOT_PHASES}
        save = integer(c.get('checkpoint_save_ns')) if children is not None else 0
        need(sum(parts.values()) + save <= wall, 'Overlapping or invalid top-level timing')
        if children is not None:
            need(c.get('cycle_id') == number and c.get('request_id') == raw.get('request_id') and
                 isinstance(raw.get('request_id'), str) and bool(raw['request_id']), 'Missing cycle identity')
            sub = {p: integer(c.get(p+'_ns')) for p in RECOVERY_PARTS}
            need(sum(sub.values()) <= parts['recovery'], 'Recovery children exceed parent')
            for p, value in sub.items(): children[p] += value
            calls = integer(c.get('target_recovery_forward_calls'))
            integer(c.get('target_recovery_read_bytes'))
            replayed += calls
        else:
            replayed += keep if keep < width else 0
        for p, value in parts.items(): totals[p] += value
        totals['wall'] += wall; totals['checkpoint_save'] += save
        committed += ids[:keep]; offset += keep; proposed += width-1; accepted += keep-1
        discarded += width-keep; rejected += keep < width
    n = len(committed); complete = terminal_result(raw)
    if complete:
        need(n > 0 and raw.get('generated_tokens') == n and raw.get('committed_token_ids') == committed and
             raw.get('decode_wall_ns') == totals['wall'] and raw.get('proposed_tokens') == proposed and
             raw.get('accepted_proposals') == accepted, 'Incomplete native cycle totals')
        need(raw.get('tokens_per_second') == n*1e9/totals['wall'], 'Native throughput differs')
        requested = integer(raw.get('requested_tokens'), 1)
        need(n <= requested and ((raw.get('stop_reason') == 'length' and n == requested) or
             (raw.get('stop_reason') == 'eos' and committed[-1] in raw.get('eos_ids', []))), 'Invalid stop coverage')
    timed = complete and raw.get('validation') is False and raw.get('mode') in ('serial', 'fast-timing', 'timing')
    totals['other'] = totals['wall'] - sum(totals[p] for p in ROOT_PHASES) - totals['checkpoint_save']
    expert_bytes = None
    if complete:
        a = raw.get('before', {}).get('expert_cache', {}).get('application_read_bytes')
        b = raw.get('after', {}).get('expert_cache', {}).get('application_read_bytes')
        if a is not None and b is not None:
            expert_bytes = integer(b) - integer(a); need(expert_bytes >= 0, 'Expert counter moved backwards')
    return dict(complete=complete, timing_mode=timed, captured_cycles=len(cs), captured_committed_tokens=n,
        component_ms_per_token={p: value/n/1e6 if n and (p!='checkpoint_save' or children is not None) else None for p, value in totals.items()},
        recovery_children_ms_per_token={p: value/n/1e6 if n else None for p, value in children.items()} if children is not None else None,
        checkpoint_save_measured=children is not None, proposed_tokens=proposed, accepted_proposals=accepted,
        acceptance_by_position=positions, discarded_verification_rows=discarded, repeated_target_rows=replayed,
        repeated_target_rows_basis='measured forward calls' if children is not None else 'known v1 full-replay schedule',
        rejected_cycles=rejected, mean_committed_per_cycle=n/len(cs) if cs else None,
        target_expert_application_bytes=expert_bytes, target_expert_bytes_per_committed_token=expert_bytes/n if n and expert_bytes is not None else None,
        measured_tps=n*1e9/totals['wall'] if timed else None,
        stop_reason=raw.get('stop_reason'), phase_ns=dict(totals,checkpoint_save=totals['checkpoint_save'] if children is not None else None),
        recovery_part_ns=children)


def records(index, selector):
    data, source = index.json(selector)
    need(isinstance(data, dict), 'Select a native MTP report or supported MTP summary')
    if data.get('kind') in NATIVE: return [(data, source, {})], source, data
    need(data.get('kind') in SCREENS, 'Unsupported MTP experiment kind')
    root = Path(source['path']).parent; result = []; seen = set()
    need(len(data.get('samples', [])) <= 128, 'Too many MTP samples')
    for s in data.get('samples', []):
        sha = s.get('sha256'); need(isinstance(sha, str) and len(sha) == 64 and sha not in seen, 'Duplicate or missing raw run hash')
        seen.add(sha); raw, ref = original(child(root, s['source']), sha)
        result.append((raw, ref, {k: s[k] for k in ('case', 'pair', 'arm')}))
    return result, source, data


def cycles(index, selector):
    runs, source, data = records(index, selector)
    def identity(raw):
        before=raw.get('before',{});metal=before.get('metal',{})
        return dict(workload_sha256=raw.get('input_sha256'),draft_manifest_sha256=raw.get('draft_manifest_sha256'),
            artifact_revision=before.get('artifact_revision'),prepared=before.get('prepared'),
            native_build=metal.get('build_fingerprint'),producer_binary_sha256=raw.get('producer_binary_sha256'),
            admission=raw.get('admission'),device=metal.get('device'),kernel_policy=metal.get('kernels'),
            validation=raw.get('validation'),mode=raw.get('mode'),target_recovery=raw.get('target_recovery'),
            direct_output=raw.get('direct_output_after',{}).get('enabled'), requested_width=raw.get('requested_width'))
    context = screening_context(data)
    result = dict(status='measured' if terminal_result(data) else 'partial_diagnostic', **context,
        runs=[dict(**labels, source=ref, identity=identity(raw), **account(raw)) for raw, ref, labels in runs], sources=[source],
        limitations=['Milliseconds are amortized per committed token, not individual token-delivery latencies.',
            'Recovery children are included in recovery and must not be added again.',
            'Legacy checkpoint saving is inside other overhead; separate subdivisions are unavailable.',
            'Expert bytes are application reads, not observed device traffic.',
            'Incomplete cycles and runs do not establish a request throughput result.'], production_promoted=False)
    if context['preliminary']:
        result['limitations'].append('Early screen only; full correctness eligibility is reported separately and production remains unqualified.')
    return result


def opportunity(index, selector, target_tps=5, phase='recovery'):
    need(type(target_tps) in (int, float) and math.isfinite(target_tps) and target_tps > 0, 'Invalid target throughput')
    need(phase in (*ROOT_PHASES, 'checkpoint_save', *RECOVERY_PARTS), 'Unsupported removable phase')
    out = cycles(index, selector); parent_complete = out['status'] == 'measured'
    for r in out['runs']:
        n = r['captured_committed_tokens']; wall = r['phase_ns']['wall']
        available = r['recovery_part_ns'] if phase in RECOVERY_PARTS else r['phase_ns']
        remove = available.get(phase) if available is not None else None
        if phase == 'checkpoint_save' and not r['checkpoint_save_measured']: remove = None
        eligible = parent_complete and r['complete'] and r['timing_mode'] and remove is not None
        r['opportunity'] = dict(target_tps=target_tps, budget_ms_per_token=1000/target_tps, removable_phase=phase,
            gap_ms_per_token=max(0, wall/n/1e6-1000/target_tps) if eligible else None,
            optimistic_zero_phase_tps=n*1e9/(wall-remove) if eligible and wall > remove else None,
            phase_measured=remove is not None)
    out['limitations'].append('Zero-phase throughput subtracts that phase at unchanged other costs; it is an optimistic arithmetic bound, not a forecast.')
    return out


def compare(index, selector, control, candidate, changes, case=None):
    from qualification_evidence import verify_seal, sha
    from screen_mtp_continuation import observe
    from screen_mtp_direct_output import comparison as direct_comparison
    from screen_residency import paired_log_interval
    runs, source, data = records(index, selector)
    need(terminal_result(data), 'Unfinished experiment cannot establish a paired result')
    direct = data['kind'] == 'mtp_direct_output_screen_v1'
    widths = data['kind'] == 'mtp_width_screen_v1'
    if widths:
        need(data.get('control_width')==4 and data.get('candidate_width') in (1,2), 'Invalid width experiment arms')
    expected_arms = ('4',str(data['candidate_width'])) if widths else ('off', 'on') if direct else ('full-replay', 'state-only')
    axis = 'requested_width' if widths else 'direct_output' if direct else 'target_recovery'
    need((control, candidate) == expected_arms and changes == [axis], 'Declare exactly the recorded MTP arms and change')
    root = Path(source['path']).parent; verify_seal(root, sha(root/'evidence-files.json'))
    producer,pref=original(root/'producer.json');frozen,fref=original(root/'identity.json')
    producer_kind='mtp_width_producer_v1' if widths else 'mtp_direct_output_producer_v1' if direct else 'target_recovery_producer_v1'
    need(producer.get('complete') is True and producer.get('kind')==producer_kind and
         frozen.get('files',{}).get(producer.get('binary'))==producer.get('binary_sha256'), 'Missing frozen producer identity')
    need(bool(runs), 'Missing raw samples'); keyed = {}; sources = [source,pref,fref]; identity = None
    for raw, ref, labels in runs:
        c, p, arm = labels['case'], labels['pair'], labels['arm']
        integer(c); integer(p); need(arm in expected_arms and (c, p, arm) not in keyed, 'Duplicate or unexpected sample')
        work, wref = original(child(root, f'case-{c}.json'))
        if widths:
            from screen_mtp_widths import observed as width_observe
            need(str(raw.get('requested_width'))==arm, 'Width arm differs from native setting')
            observed=width_observe(raw,work,wref['sha256'],False)
        elif direct: observed = observe(raw, work, wref['sha256'], 'fast-timing')
        else:
            from target_recovery_checks import observe as recovery_observe
            observed = recovery_observe(raw, work, wref['sha256'], False)
        need(observed['clean_host'] and observed['clean_memory'], 'Resource-disturbed sample')
        claim=next(s for s in data['samples'] if s['sha256']==ref['sha256'])
        need(all(claim.get(k)==v for k,v in observed.items()),'Sample summary differs from raw observations')
        account(raw)
        info = raw['before']; current = (raw['draft_manifest_sha256'], info['artifact_revision'],
            info['prepared']['manifest_sha256'], info['metal']['build_fingerprint'],
            info['metal']['device'], info['metal']['physical_bytes'], raw['admission'])
        if identity is None: identity = current
        need(identity == current, 'MTP machine, artifact, build or budget differs')
        need(producer['base_native_fingerprint']==info['metal']['build_fingerprint'] and
            (direct or raw.get('producer_binary_sha256')==producer['binary_sha256']),'Run producer differs')
        if widths:
            from screen_mtp_widths import kernel_policy
            kernel_policy(raw)
        need((widths or info['metal']['kernels'] == raw['after']['metal']['kernels']) and
            not info['metal']['kernels']['profile'] and not info['metal']['kernels']['counter_profile'], 'Instrumented or changed kernels')
        keyed[c, p, arm] = raw; sources.extend((ref, wref))
    pairs = []; grouped = {}
    for c in sorted({k[0] for k in keyed}):
        ns = sorted({k[1] for k in keyed if k[0] == c}); need(ns == list(range(len(ns))), 'Noncontiguous pairs')
        work, _ = original(root/f'case-{c}.json'); name = work.get('name', f'case-{c}')
        for p in ns:
            need(all((c, p, a) in keyed for a in expected_arms), 'Incomplete pair')
            order = [x[2]['arm'] for x in runs if x[2]['case'] == c and x[2]['pair'] == p]
            need(order == list(expected_arms if (c+p)%2 == 0 else expected_arms[::-1]), 'Nonalternating recorded order')
            a, b = (keyed[c, p, arm] for arm in expected_arms)
            if widths:
                from screen_mtp_widths import compare_widths
                comparison=compare_widths(a,b)
            elif direct: comparison = direct_comparison(a, b)
            else:
                from target_recovery_checks import comparison as recovery_comparison
                comparison = recovery_comparison(a, b)
            claims=[v for v in data.get('pairs',[]) if v.get('case')==c and v.get('pair')==p]
            need(len(claims)==1 and all(claims[0].get(k)==v for k,v in comparison.items()), 'Summary pair differs from raw validation')
            if case is None or case == name:
                pairs.append(dict(case=name, pair=p, **comparison)); grouped.setdefault(name, []).append(comparison['ratio'])
    need(bool(pairs), 'No selected cases')
    need(len(data.get('pairs',[]))*2==len(runs),'Missing or duplicate summary pairs')
    context = screening_context(data)
    result = dict(comparable=True, status='measured', **context, controlled_change={axis: list(expected_arms)}, pairs=pairs,
        cases=[dict(case=n, pairs=len(rs), geometric_mean_ratio=math.exp(sum(map(math.log, rs))/len(rs)),
                    confidence_95=paired_log_interval(rs) if len(rs) >= 5 else None) for n, rs in grouped.items()],
        sources=sources, limitations=['Each workload has its own repetitions; different cases are not repeated samples.',
            'Fewer than five paired runs have no confidence bound.', 'These short screens do not qualify long-context or sustained inference.'],
        normal_request_latency_qualified=False, production_promoted=False)
    if context['preliminary']:
        result['limitations'].append('Preliminary timing does not allow advancement. Numerical qualification is reported separately; promotion requires fresh timing evidence.')
    return result


def next_experiment(index, selector):
    out = opportunity(index, selector)
    out.update(hypothesis='Avoid repeated full target forwards during partial draft rejection.',
        smallest_experiment='Replay captured target state for every accepted prefix; then screen a fresh 64-token LRU pair.',
        missing_evidence=['target restore versus target repair versus draft catch-up costs on legacy reports',
                          'fresh complete-cycle paired outcome for state-only target recovery'],
        related_experiments=related_history(index))
    if out['status'] != 'measured':
        out['smallest_experiment'] = 'Resolve the recorded blocker and collect a complete screen; this partial report selects no winner.'
    else:
        _,_,data=records(index,selector)
        if data.get('kind')=='mtp_width_screen_v1':
            checked=compare(index,selector,'4',str(data['candidate_width']),['requested_width'])
            clean=data['full_clean_correctness']
            remaining=['Longer fresh paired requests on the other coding workloads',
                       'Measured workload crossover before any adaptive policy',
                       'Five fresh pairs, long-context and sustained qualification']
            if not clean: remaining.insert(0,'Clean full-model correctness')
            out.update(validated_comparison=checked,
                hypothesis='A fixed draft length must improve complete requests at the same cache and checkpoint capacity.',
                smallest_experiment='Do not repeat the unchanged rejected width.' if data.get('status')=='insufficient_early_gain' else
                    ('Compare the surviving width on the remaining 128-token coding workloads; profile target verification if no width approaches 200ms/token.'
                     if clean else 'Finish clean correctness and compare the surviving width on three fresh 128-token coding workloads.'),
                missing_evidence=remaining)
        elif data.get('kind')==WIDTH_KIND:
            out.update(hypothesis='Measure draft length using complete-cycle cost per committed token.',
                smallest_experiment='Compare a fresh 64-token width-four/width-two pair after numerical validation.',
                missing_evidence=['Compatible fresh width comparison', 'Clean full-model qualification'])
        elif data.get('kind')=='mtp_target_recovery_screen_v1':
            checked=compare(index,selector,'full-replay','state-only',['target_recovery'])
            out['validated_comparison']=checked
            if data.get('status') in ('insufficient_early_gain','insufficient_short_gain','all_accepted_regression','insufficient_long_gain'):
                out.update(hypothesis='Target recovery failed its recorded screening gate; do not repeat the unchanged candidate.',
                    smallest_experiment='Use draft/verify/acceptance costs to design a fixed-width 1/2/4 screen on the retained baseline.',
                    missing_evidence=['A changed mechanism and explicit rationale are required before revisiting the rejected recovery candidate.'])
            elif data.get('status')=='promising_early_screen':
                out.update(hypothesis='Recovery passed preliminary timing; full correctness still blocks advancement.',
                    smallest_experiment='Finish the clean forced-prefix/EOS stage, then rerun the registered timing gates with fresh samples.',
                    missing_evidence=['Clean complete forced-prefix/EOS stage', 'Fresh registered timing gates',
                                      'Five fresh pairs, long-context and sustained qualification'])
            elif data.get('status')=='promising_long_screen':
                out.update(hypothesis='Recovery passed screening; acceptance and verification cost now bound further gain.',
                    smallest_experiment='Plan fixed draft widths 1/2/4 with real MTP proposals and the selected recovery path.',
                    missing_evidence=['Five fresh pairs, long-context and sustained qualification remain required for production.'])
            else:
                out['smallest_experiment']='Complete the next declared recovery recipe gate using fresh samples.'
    return out


def related_history(index):
    """Keep recall small and specific; do not dump raw pair arrays into context."""
    found={}
    for phrase in ('mtp-width','target-state-recovery','mtp-forward/recovery','mtp-expert-scratch','mtp-direct-output'):
        for row in index.history(phrase,limit=20)['results']:found[row['id']]=row
    rows=sorted(found.values(),key=lambda r:({'rejected':0,'blocked':1,'inconclusive':2,'promising':3}.get(r.get('decision'),4),r['id']))
    return [{k:r[k] for k in ('id','kind','status','complete','decision','hypothesis','outcome','error','tags','rerun_rationale','sources','stale') if k in r}
        for r in rows[:10]]
