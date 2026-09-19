#!/usr/bin/env python3
"""Exact embedding storage with real MTP: numerical gate, then paired costs."""
import argparse
import hashlib
import math
from pathlib import Path

import build_streamed_mtp as builder
from benchmark_host import build_probe
from cache_residency import require
from combined_q4 import freeze
from perfect_draft import configuration
from prepare_mtp import verify as verify_artifact
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
from screen_mtp_continuation import ROOT, PREPARED, host_check, workloads
from screen_mtp_widths import observed
from screen_horizon_early import diagnostic_resources
from streamed_mtp_capsule import experiment, verify as verify_capsule
from target_recovery_checks import replay_resources, resource_failure
from trial_recovery import read, reference_work

BASE = ROOT/'docs/benchmarks/2026-09-18-streamed-mtp'
SAVING = 675446784-2*1024**2
RESIDENT = 5362515968
PLANNED = 11809357824
ARMS = ('resident', 'rows')


def checked_diagnostic(result, raw):
    try:
        diagnostic_resources(result, raw)
    except ValueError as error:
        raise ResourceBlocked(resource_failure(raw, 'Numerical diagnostic resource bound')) from error


def cases():
    work = dict(reference_work(), max_tokens=7, name='irregular-seven')
    return [(1, dict(work, force_prefix=1))] + [
        (4, dict(work, force_prefix=keep)) for keep in range(1, 5)] + [
        (4, dict(work, name='immediate-eos', eos_ids=[work['expected_prompt_id']], force_prefix=1))]


def observe(raw, work, input_sha, producer_sha, width, arm, validation, priming_policy='completion'):
    require(type(width) is int and width in (1, 4) and arm in ARMS and
        raw['producer_binary_sha256'] == producer_sha and raw['requested_width'] == width and
        raw['embedding_storage'] == arm and raw['embedding_owner_released'] is True,
        'Changed producer, width, embedding storage or retained owner')
    require(raw.get('draft_priming_policy', 'completion') == priming_policy, 'Changed draft priming policy')
    if priming_policy != 'completion':
        require(priming_policy == 'fixed-lease-batches-32', 'Unknown draft priming policy')
        priming = raw['draft_priming_before']
        require(priming == raw['draft_priming_after'] and priming['policy'] == priming_policy and
            priming['active'] is False and all(type(priming[k]) is int and priming[k] > 0
                for k in ('calls', 'batches', 'experts', 'peak_leases')) and
            priming['calls'] <= priming['batches'] <= priming['experts'] and priming['peak_leases'] <= 32,
            'Incomplete priming or fixed scheduling continued during decode')
    result = observed(raw, work, input_sha, validation)
    require(raw['production_promoted'] is False and raw['normal_request_latency_qualified'] is False,
        'Developer trial cannot qualify production')
    plan = raw['admission']['target']
    saving = SAVING if arm == 'rows' else 0
    require(plan['resident_bytes'] == RESIDENT-saving and plan['planned_bytes'] == PLANNED-saving and
        raw['admission']['draft_bytes'] == 318234624 and
        raw['admission']['host_checkpoint_logits_bytes'] == 151158784 and
        raw['admission']['combined_bytes'] == plan['planned_bytes']+318234624+151158784+18*1024**2,
        'Unexpected joint memory accounting')
    for edge in ('before', 'after'):
        require(raw[edge]['memory_plan'] == plan and
            raw[edge]['artifact_revision'] == 'b2c422f3c643e36f04227a64d61796b44a4b1029' and
            raw[edge]['expert_cache']['policy'] == 'clock', 'Changed artifact or memory/cache controls')
        entry = raw['embedding_rows_'+edge]
        if arm == 'resident':
            require(entry is None, 'Resident arm unexpectedly uses row storage')
        else:
            require(entry['storage'] == 'exact-packed-rows' and entry['capacity'] == 256 and
                entry['host_reserve_bytes'] == 2*1024**2 and 0 < entry['fixed_host_bytes'] < 2*1024**2 and
                entry['row_bytes'] == 2720 and entry['removed_resident_allocation_bytes'] == 675446784 and
                all(type(entry[k]) is int and entry[k] >= 0 for k in ('hits', 'misses', 'evictions', 'application_read_bytes')) and
                entry['application_read_bytes'] == 2720*entry['misses'], 'Changed row storage or incomplete reads')
    if arm == 'rows':
        a, b = (raw['embedding_rows_'+k] for k in ('before', 'after'))
        require(all(b[k] >= a[k] for k in ('hits', 'misses', 'evictions', 'application_read_bytes')),
            'Embedding counters reset during request')
    for c in raw['cycles']:
        require(len(c['unforced_proposals']) == c['width'] and c['unforced_proposals'][0] == c['proposals'][0] and
            all(type(t) is int and 0 <= t < 248320 for t in c['unforced_proposals']), 'Missing actual proposals')
        require(c['forced_rejection'] or c['unforced_proposals'] == c['proposals'], 'Unexplained proposal change')
        for key, count in (('draft_row_logits_sha256', c['width']-1), ('verified_row_logits_sha256', c['width'])):
            require(len(c[key]) == (count if validation else 0) and
                all(isinstance(h, str) and len(h) == 64 and set(h) <= set('0123456789abcdef') for h in c[key]),
                'Incomplete draft/verification logit coverage')
    if validation:
        require(raw['initial_target_state']['tokens'] == len(work['prompt_ids']) and
            raw['initial_draft_state']['position'] == len(work['prompt_ids'])-1, 'Missing primed state')
    return dict(result, planned_saving_bytes=saving, embedding_storage=arm)


def compare(a, b):
    require((a['embedding_storage'], b['embedding_storage']) == ARMS, 'Wrong storage arms')
    require(a.get('draft_priming_policy', 'completion') == b.get('draft_priming_policy', 'completion'),
        'Changed draft priming schedule')
    if a.get('draft_priming_policy') == 'fixed-lease-batches-32':
        require(a['draft_priming_before'] == b['draft_priming_before'], 'Draft priming work differs')
    fields = ('kind', 'complete', 'producer_binary_sha256', 'input_sha256', 'requested_width', 'mode', 'validation',
        'target_recovery', 'draft_manifest_sha256', 'prompt_tokens', 'requested_tokens', 'eos_ids',
        'prime_logits_sha256', 'committed_token_ids', 'row_logits_sha256', 'final_target_state', 'final_draft_state',
        'boundaries', 'generated_tokens', 'next_id', 'stop_reason', 'host_checkpoint_allocated_bytes',
        'target_recovery_journal', 'ngram_before_decode', 'ngram_after_decode')
    require(all(a[k] == b[k] for k in fields), 'Storage changed identity, logits, output or state')
    require(len(a['cycles']) == len(b['cycles']), 'Storage changed cycle count')
    keys = ('offset', 'width', 'proposals', 'unforced_proposals', 'draft_row_logits_sha256',
        'verified_row_logits_sha256', 'accepted_proposals', 'committed_tokens', 'forced_rejection',
        'next_id', 'target_recovery_forward_calls')
    require(all(x[k] == y[k] for x, y in zip(a['cycles'], b['cycles']) for k in keys),
        'Storage changed draft predictions, verifier logits or rejection recovery')
    for edge in ('before', 'after'):
        for key in ('prepared', 'execution', 'completion_pipeline', 'ready_group', 'chunk_tokens',
                    'io_workers', 'short_append_tokens'):
            require(a[edge][key] == b[edge][key], 'Storage changed execution/cache control '+key)
        require(a[edge]['metal']['kernels'] == b[edge]['metal']['kernels'], 'Storage changed arithmetic policy')
        for key in ('recipe', 'budget_bytes', 'context', 'norm_convention'):
            require(a['draft_'+edge][key] == b['draft_'+edge][key], 'Storage changed draft configuration')
        for scope in (edge, 'draft_'+edge):
            for key in ('policy', 'entry_metadata_bytes'):
                require(a[scope]['expert_cache'][key] == b[scope]['expert_cache'][key], 'Storage changed cache allocation/policy')
    # I/O completion timing may change ready hits, joins and the resulting cache
    # order. Those are measured outcomes, not arithmetic/configuration changes.
    initial_cache_equal = {scope: a[scope]['expert_cache']['diagnostic_cache_state'] ==
        b[scope]['expert_cache']['diagnostic_cache_state'] for scope in ('before', 'draft_before')}
    # Completion order may alter priming's eviction state. Correctness must be
    # invariant to that state; timing still needs identical starting caches.
    if not a['validation'] or a.get('draft_priming_policy') == 'fixed-lease-batches-32':
        require(all(initial_cache_equal.values()), 'Changed initial cache state')
    expected = dict(a['admission'], target=dict(a['admission']['target']))
    for key in ('resident_bytes', 'planned_bytes'): expected['target'][key] -= SAVING
    expected['combined_bytes'] -= SAVING
    require(b['admission'] == expected, 'Unaccounted admission difference')
    if a['validation']:
        require(a['initial_target_state'] == b['initial_target_state'] and
            a['initial_draft_state'] == b['initial_draft_state'], 'Storage changed priming state')
        return dict(exact_proposals_logits_and_state=True, timing_used=False,
                    initial_cache_equal=initial_cache_equal)
    require(a['generated_tokens'] == a['requested_tokens'], 'Incomplete timing length')
    return dict(exact_proposals_logits_and_state=True, timing_used=True,
        latency_ratio=b['decode_wall_ns']/a['decode_wall_ns'], control_tps=a['tokens_per_second'],
        candidate_tps=b['tokens_per_second'],
        physical_peak_saved_bytes=a['after_destroy']['physical_footprint_peak_bytes']-b['after_destroy']['physical_footprint_peak_bytes'],
        planned_saving_bytes=SAVING, normal_request_latency_qualified=False)


def setup(exp, directory):
    isolated = (Path(directory)/'capsule.json').is_file()
    if isolated:
        cfg, proof, host, _ = verify_capsule(directory)
        files = [p for p in Path(directory).iterdir() if p.is_file()]
    else:
        selected_builder = builder
        kind = read(Path(directory)/'producer.json')['kind']
        if kind == 'streamed_mtp_fixed_priming_producer_v1':
            import build_streamed_mtp_fixed_priming as selected_builder
        else: require(kind == 'streamed_mtp_producer_v1', 'Unknown storage producer')
        cfg, proof = selected_builder.verify(directory)
        host = build_probe(ROOT/'.cache/benchmark-host'/hashlib.sha256(str(exp.out).encode()).hexdigest())
        files = [*selected_builder.inputs(cfg), *selected_builder.generated(cfg['output']), *cfg['objects'], cfg['binary'], *host['files']]
    require(proof['base_native_fingerprint'] == exp.frozen['build'], 'Native base changed')
    save(exp.out/'producer.json', proof)
    save(exp.out/'draft-audit.json', verify_artifact(PREPARED))
    freeze(exp, [*files,
        PREPARED/'manifest.json', PREPARED/'dense.bin', PREPARED/'experts.bin', BASE/'protocol.md',
        BASE/'cache-state-protocol.md'])
    if proof['kind'] == 'streamed_mtp_fixed_priming_producer_v1': freeze(exp, [BASE/'fixed-priming-protocol.md'])
    if isolated: freeze(exp, [BASE/'capsule-protocol.md'])
    exp.env.update(FREELLM_Q8_EXPANDED='packed', FREELLM_MTP_NGRAM_INIT='lazy', FREELLM_MTP_EXPERT_SCRATCH='off',
        FREELLM_MTP_DIRECT_OUTPUT='on', FREELLM_TARGET_RECOVERY='full-replay')
    exp.report.update(samples=[], comparisons=[], controlled_change='embedding_storage', real_mtp=True,
        historical_timing_reused=False, advancement_allowed=False)
    exp.guard.check_resources(initial=True)
    return cfg, proof, host


def sample(exp, cfg, proof, host, width, arm, work, stem, validation, diagnostic=False):
    require(not diagnostic or validation, 'Diagnostic allowance is numerical only')
    inp = exp.out/(stem+'.input.json')
    save(inp, work)
    freeze(exp, [inp])
    host_check(exp, host, stem)
    exp.env.update(FREELLM_MTP_WIDTH=str(width), FREELLM_MTP_EMBEDDINGS=arm)
    path = exp.out/(stem+'.json')
    exp.command([cfg['binary'], exp.model, PREPARED, inp, path, 'fast-validate' if validation else 'fast-timing'],
        stem, limit=180, validation=validation)
    raw = read(path)
    result = observe(raw, work, sha(inp), proof['binary_sha256'], width, arm, validation,
                     proof.get('draft_priming_policy', 'completion'))
    exp.report['samples'].append(dict(source=path.name, input=inp.name, sha256=sha(path), width=width, **result))
    exp.persist()
    if diagnostic: checked_diagnostic(result, raw)
    elif not result['clean_memory'] or not result['clean_host']: raise ResourceBlocked(resource_failure(raw, stem))
    return raw


def embedding_fixture(exp, cfg, host):
    host_check(exp, host, 'embedding', full=False)
    exp.command([cfg['binary'], '--streamed-embedding-test', exp.model, exp.out/'embedding.json'],
        'embedding', limit=60, validation=True)
    fixture = read(exp.out/'embedding.json')
    resources = replay_resources(fixture)
    require(fixture['complete'] is True and len(fixture['cases']) == 10 and all(c['exact'] is True for c in fixture['cases']) and
        fixture['owner_released'] is fixture['invalid_request_atomic'] is fixture['failed_read_atomic'] is True and
        fixture['evictions_before_gpu_submission'] == 2, 'Incomplete real embedding checks')
    exp.report['embedding'] = dict(resources, source='embedding.json', sha256=sha(exp.out/'embedding.json'))
    exp.persist()
    if not resources['clean_memory'] or not resources['clean_host']: raise ResourceBlocked(resource_failure(fixture, 'embedding'))


def fixture(output, directory):
    exp = experiment(directory)(output, 'streamed_mtp_embedding_fixture_v1', [], dict(kind='real_embedding_rows'), 180)
    with exp:
        cfg, _, host = setup(exp, directory)
        exp.report.update(real_mtp=False, performance_measurement=False, full_model_loaded=False)
        embedding_fixture(exp, cfg, host)
        exp.report.update(status='embedding_fixture_exact', resource_qualified=True)
    return exp.report


def reusable_samples(source, proof, diagnostic):
    source = Path(source).resolve()
    verify_seal(source, sha(source/'evidence-files.json'))
    summary = read(source/'summary.json')
    require(summary['kind'] == 'streamed_mtp_validation_v1' and
        summary['status'] in ('numerically_exact', 'resource_blocked', 'failed', 'interrupted', 'time_budget_exhausted') and
        read(source/'producer.json') == proof, 'Resume needs a sealed same-producer validation attempt')
    allowed = {f'case-{i}-{arm}.json': (i, width, arm, work) for i, (width, work) in enumerate(cases()) for arm in ARMS}
    result = {}
    seen = set()
    for row in summary.get('samples', []):
        name = row['source']
        require(name in allowed and name not in seen, 'Unexpected or duplicate reusable sample')
        seen.add(name)
        case, width, arm, work = allowed[name]
        raw_path = source/name
        inp = source/(Path(name).stem+'.input.json')
        require(row['input'] == inp.name and row['sha256'] == sha(raw_path) and read(inp) == work,
            'Changed reusable sample/input')
        raw = read(raw_path)
        observation = observe(raw, work, sha(inp), proof['binary_sha256'], width, arm, True,
                              proof.get('draft_priming_policy', 'completion'))
        if not (observation['clean_memory'] and observation['clean_host']):
            if not diagnostic: continue
            try: checked_diagnostic(observation, raw)
            except ResourceBlocked: continue
        result[case, arm] = (raw, raw_path, inp, observation)
    return result


def validate(output, directory, diagnostic_source=None, resume=None):
    work = cases()
    configs = [dict(configuration(4, expert_slots=1460), embedding_storage=arm) for arm in ARMS]
    exp = experiment(directory)(output, 'streamed_mtp_validation_v1', configs, work, 1800)
    with exp:
        cfg, proof, host = setup(exp, directory)
        diagnostic = diagnostic_source is not None
        exp.report.update(diagnostic_only=diagnostic, resource_qualified=False, performance_measurement=False)
        reused = None
        if diagnostic:
            source = Path(diagnostic_source).resolve()
            verify_seal(source, sha(source/'evidence-files.json'))
            parent = read(source/'summary.json')
            require(parent['kind'] == 'streamed_mtp_validation_v1' and parent['status'] == 'resource_blocked' and
                read(source/'producer.json') == proof, 'Diagnostic reference must be the preserved same-producer attempt')
            fixture_raw = read(source/'embedding.json')
            resources = replay_resources(fixture_raw)
            require(resources['clean_memory'] and resources['clean_host'] and fixture_raw['complete'] is True and
                len(fixture_raw['cases']) == 10 and all(c['exact'] is True for c in fixture_raw['cases']) and
                fixture_raw['owner_released'] is fixture_raw['invalid_request_atomic'] is fixture_raw['failed_read_atomic'] is True and
                fixture_raw['evictions_before_gpu_submission'] == 2,
                'Diagnostic requires clean operator prerequisites')
            raw_path = source/'case-0-resident.json'
            inp = source/'case-0-resident.input.json'
            reused = read(raw_path)
            require(read(inp) == work[0][1], 'Diagnostic reference workload changed')
            result = observe(reused, work[0][1], sha(inp), proof['binary_sha256'], 1, 'resident', True,
                             proof.get('draft_priming_policy', 'completion'))
            checked_diagnostic(result, reused)
            for p in (raw_path, inp): (exp.out/p.name).write_bytes(p.read_bytes())
            freeze(exp, [*[p for p in source.iterdir() if p.is_file()], BASE/'diagnostic-protocol.md'])
            exp.report['samples'].append(dict(source=raw_path.name, input=inp.name, sha256=sha(raw_path), width=1, **result))
            exp.report['numerical_parent'] = dict(path=str(source), seal_sha256=sha(source/'evidence-files.json'),
                source_sha256=sha(raw_path), timing_used=False)
            exp.persist()
        else: embedding_fixture(exp, cfg, host)
        reusable = reusable_samples(resume, proof, diagnostic) if resume else {}
        exp.report['reused_validation'] = []
        if resume: freeze(exp, [p for p in Path(resume).iterdir() if p.is_file()])
        # The historical oracle supplies numerical evidence only, not timings
        # or clean-resource qualification for this new producer.
        oracle_dir = ROOT/'docs/benchmarks/2026-09-17-target-recovery/numerical-diagnostic-01'
        verify_seal(oracle_dir, sha(oracle_dir/'evidence-files.json'))
        oracle = read(oracle_dir/'case-0-full-replay.json')
        freeze(exp, [oracle_dir/'evidence-files.json', oracle_dir/'case-0-full-replay.json'])
        for case, (width, values) in enumerate(work):
            pair = []
            for arm in ARMS:
                if case == 0 and arm == 'resident' and reused is not None:
                    raw = reused
                elif (case, arm) in reusable:
                    raw, path, inp, observation = reusable[case, arm]
                    for p in (path, inp): (exp.out/p.name).write_bytes(p.read_bytes())
                    exp.report['samples'].append(dict(source=path.name, input=inp.name, sha256=sha(path), width=width, **observation))
                    exp.report['reused_validation'].append(dict(case=case, arm=arm, source=str(path), sha256=sha(path), timing_used=False))
                    print(f'case-{case}-{arm}: rechecked sealed numerical sample', flush=True)
                    exp.persist()
                else: raw = sample(exp, cfg, proof, host, width, arm, values, f'case-{case}-{arm}', True, diagnostic)
                pair.append(raw)
            check = compare(*pair)
            for raw in pair:
                n = raw['generated_tokens']
                require(raw['prime_logits_sha256'] == oracle['prime_logits_sha256'] and
                    raw['row_logits_sha256'] == oracle['row_logits_sha256'][:n] and
                    raw['committed_token_ids'] == oracle['committed_token_ids'][:n], 'Established target reference changed')
            exp.report['comparisons'].append(dict(case=case, width=width, **check))
            exp.persist()
        exp.report.update(status='numerically_exact',
            resource_qualified=all(s['clean_memory'] and s['clean_host'] for s in exp.report['samples']),
            performance_measurement=False)
    return exp.report


def prerequisite(source, proof, preliminary=False):
    source = Path(source).resolve()
    verify_seal(source, sha(source/'evidence-files.json'))
    summary = read(source/'summary.json')
    require(summary['kind'] == 'streamed_mtp_validation_v1' and summary['complete'] is True and
        summary['status'] == 'numerically_exact' and read(source/'producer.json') == proof and
        len(summary['samples']) == 2*len(cases()), 'Complete same-producer numerical validation required')
    clean = True
    for case, (width, work) in enumerate(cases()):
        pair = []
        for arm in ARMS:
            path = source/f'case-{case}-{arm}.json'
            inp = source/f'case-{case}-{arm}.input.json'
            require(read(inp) == work, 'Changed validation workload')
            raw = read(path)
            result = observe(raw, work, sha(inp), proof['binary_sha256'], width, arm, True,
                             proof.get('draft_priming_policy', 'completion'))
            clean &= result['clean_memory'] and result['clean_host']
            if preliminary: checked_diagnostic(result, raw)
            else: require(result['clean_memory'] and result['clean_host'], 'Validation resources disturbed')
            pair.append(raw)
        compare(*pair)
    require(summary['resource_qualified'] is clean, 'Validation summary changed resource qualification')
    return [p for p in source.iterdir() if p.is_file()], clean


def screen(output, directory, validation_source, preliminary=False):
    work = workloads(ROOT/'.cache/qwen-mixed-reference', 128)[0]
    work['max_tokens'] = 64
    configs = [dict(configuration(4, expert_slots=1460), embedding_storage=arm) for arm in ARMS]
    exp = experiment(directory)(output, 'streamed_mtp_screen_v1', configs, work, 1200)
    with exp:
        cfg, proof, host = setup(exp, directory)
        files, clean_correctness = prerequisite(validation_source, proof, preliminary)
        freeze(exp, files)
        if preliminary: freeze(exp, [BASE/'diagnostic-protocol.md'])
        exp.report.update(performance_measurement=True, preliminary=True, pairs=[],
            full_clean_correctness=clean_correctness,
            numerical_validation=dict(path=str(Path(validation_source).resolve()),
                seal_sha256=sha(Path(validation_source)/'evidence-files.json')))
        for width in (4, 1):
            rows = []
            for pair, order in enumerate((ARMS, ARMS[::-1])):
                raw = {arm: sample(exp, cfg, proof, host, width, arm, work, f'width-{width}-pair-{pair}-{arm}', False)
                       for arm in order}
                compared = compare(raw['resident'], raw['rows'])
                rows.append(compared)
                exp.report['pairs'].append(dict(width=width, pair=pair, **compared))
                exp.persist()
                if compared['latency_ratio'] > 1.10:
                    exp.report.update(status='early_cost_regression', stopped_width=width)
                    break
            exp.report.setdefault('by_width', {})[str(width)] = dict(pairs=len(rows),
                geometric_latency_ratio=math.prod(r['latency_ratio'] for r in rows)**(1/len(rows)),
                measured_peak_savings_bytes=[r['physical_peak_saved_bytes'] for r in rows])
        if 'stopped_width' not in exp.report: exp.report['status'] = 'storage_cost_measured'
        exp.report['limitations'] = ['Two short alternating pairs at most per width, one coding workload, no confidence-bounded promotion.',
            'Width-one/four results are separate comparisons; no widths or historical samples are pooled.',
            'Exact row storage may free memory without improving latency at unchanged expert capacity.',
            'Long context, retained conversation, cancellation and sustained coding remain unqualified.']
    return exp.report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('fixture', 'validate', 'screen'))
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--validation', type=Path)
    p.add_argument('--diagnostic-source', type=Path)
    p.add_argument('--preliminary', action='store_true')
    p.add_argument('--resume', type=Path)
    a = p.parse_args(argv)
    if a.diagnostic_source and a.phase != 'validate': p.error('--diagnostic-source requires validate')
    if a.preliminary and a.phase != 'screen': p.error('--preliminary requires screen')
    if a.resume and a.phase != 'validate': p.error('--resume requires validate')
    if a.phase == 'screen':
        if a.validation is None: p.error('--validation is required for screen')
        result = screen(a.output, a.build, a.validation, a.preliminary)
    elif a.phase == 'fixture': result = fixture(a.output, a.build)
    else: result = validate(a.output, a.build, a.diagnostic_source, a.resume)
    return 0 if result.get('complete') is True else 2


if __name__ == '__main__': raise SystemExit(main())
