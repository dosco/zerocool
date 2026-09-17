#!/usr/bin/env python3
"""Bounded perfect-continuation block verification; never a draft performance claim."""
import argparse
import hashlib
import math
from pathlib import Path
import shutil
import statistics

from cache_residency import require
from capture_routes import load
from combined_q4 import configs, freeze
from native_q4_replay import uint, valid_sha256
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
from qualify_exact_sessions import check_configuration
import q4_request_context as request_context
from selector_qualification import check_machine
from stage200 import Experiment
from verify_stage200 import verify_sources

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'docs/benchmarks/2026-09-15-perfect-draft'
PROTOCOL = ROOT/'docs/benchmarks/2026-09-16-perfect-draft/protocol-stable-prime.md'
SOURCE = request_context.BASE/'capture-02'
PRIME_REFERENCE=ROOT/'docs/benchmarks/2026-09-16-perfect-draft/screen-03/validate-1.json'
BUDGET = 12*1024**3
VOCAB = 248320
ORDER = [(0, 1), (0, 2), (0, 4), (1, 4), (1, 2), (1, 1)]
VALIDATION_WIDTHS = [1, 2, 4]
KERNEL_POLICY = 'decode token_tile=width; q8r2 atT1'
CRITERIA = dict(budget_bytes=BUDGET, expert_slots=1072, context=8192,
    validation_widths=VALIDATION_WIDTHS, validation_tokens=4, timing_order=ORDER,
    timing_tokens=16, pairs=2, target_verified_tokens_per_second=5,
    every_paired_wall_ratio_below=1, all_candidate_runs_reach_target=True,
    process_seconds=90, validation_process_seconds=150, stage_seconds=600, observed_compression_allowed=False,
    draft_cost_included=False, assumed_acceptance=1.0,prime_expert_schedule='fixed-lease-batches-32')
FLAGS = dict(normal_request_latency_qualified=False, production_promoted=False,
             speculative_decoding_qualified=False, actual_draft_measured=False)
LIMITATIONS = [
    'Continuation tokens are known in advance and accepted at 100 percent with no draft model cost. This optimistic verifier screen is not speculative generation throughput.',
    'Only sixteen continuation inputs after the fixed 72-token prompt are timed. Initial model load and prompt processing are reported separately; no TTFT, 2K/4K/7K or sustained-use qualification follows.',
    'Two fresh alternating rounds are an early screening gate, not confidence-bounded performance evidence. Every run remains included and prior timing is not pooled.',
    'All-vocabulary logits and persistent state are compared at matching prefix boundaries. Hashing and evidence serialization are outside continuation timing; required checkpoint, forward and greedy acceptance work are inside.',
    'Every arm records exact router selections during inference. This route-audit overhead is part of the measured diagnostic and is not removed or extrapolated into an uninstrumented result.',
    'Memory is checked at native lifecycle boundaries and cumulative peaks. Zero observed compression and stable swap do not provide continuous attribution or prove the absence of every transient.',
    'The sealed source supplies only fixed tokens and artifact identity. Its resource-blocked stage and disturbed timing retain their original status.']
LIMITATIONS.append('Priming uses existing fixed lease batches solely to make starting CLOCK contents and flags reproducible. Measured decode restores completion-driven execution; this is not a normal-prefill performance measurement.')


def configuration(width, *, expert_slots=1072):
    require(width in (1, 2, 4), 'Unsupported verification width')
    require(type(expert_slots) is int and expert_slots in (1072, 1460), 'Unsupported fixed expert capacity')
    return dict(memory_bytes=BUDGET, context=8192, expert_slots=expert_slots, artifact='mixed-4_8bit',
        q4_decode='reference', q8_decode_rows=2, route_selection='simd', residency='core-cache',
        decode_scratch='reuse', token_tile=width, cache_policy='clock', ready_group=4,
        io_workers=8, panel=512, chunk=128, decode_submission='immediate')


def source_input():
    verify_seal(SOURCE, sha(SOURCE/'evidence-files.json'))
    summary, frozen, raw = (load(SOURCE/name) for name in
                            ('summary.json', 'identity.json', 'normal-A.json'))
    require(summary['measurements'][0]['source'] == 'normal-A.json' and
        summary['measurements'][0]['sha256'] == sha(SOURCE/'normal-A.json'),
        'Sealed fixed-token source differs')
    request_context.observe(raw, frozen, configs()[0], summary['workload'], {}, False)
    row = raw['runs'][0]; prompt = raw['workloads'][0]['tokens']; outputs = row['output_token_ids']
    require(len(prompt) == 72 and len(outputs) == 17 and raw['complete'] is True,
            'Fixed-token source must contain the complete requested short continuation')
    work = dict(prompt_ids=prompt, continuation_ids=outputs[:16], expected_next_ids=outputs[1:17],
                expected_prompt_id=outputs[0])
    source = dict(path=str(SOURCE), source_seal_sha256=sha(SOURCE/'evidence-files.json'),
        raw_sha256=sha(SOURCE/'normal-A.json'), source_build=frozen['build'],
        artifact_revision=frozen['artifact_revision'], prepared_manifest_sha256=frozen['prepared_manifest_sha256'],
        recorded_stage_complete=summary['complete'], recorded_stage_status=summary['status'],
        use='fixed tokens and artifact identity only; no timing reused')
    return work, source


def valid_work(work):
    require(set(work) == {'prompt_ids', 'continuation_ids', 'expected_next_ids', 'expected_prompt_id'} and
        len(work['prompt_ids']) == 72 and len(work['continuation_ids']) == len(work['expected_next_ids']) == 16 and
        all(type(v) is int and 0 <= v < VOCAB for k in ('prompt_ids', 'continuation_ids', 'expected_next_ids')
            for v in work[k]) and work['expected_prompt_id'] == work['continuation_ids'][0] and
        work['continuation_ids'][1:] == work['expected_next_ids'][:-1],
        'Invalid or inconsistent fixed continuation')


def check_prime_reference(prime,reference):
    # Cache layout intentionally changes with deterministic priming. Arithmetic,
    # recurrent state and real routes must still match the original clean prime.
    require(all(prime.get(k)==reference.get(k) and prime.get(k) is not None
        for k in ('logits_sha256','state','routes')),'Stable priming changed model outputs, state or routes')


def memory_observation(raw):
    values = [raw['before']['process']]
    for block in raw['blocks']:
        values.extend(block[k] for k in ('memory_before', 'memory_after'))
    values.extend((raw['after']['process'], raw['process_after_destroy']))
    fields = ('physical_footprint_bytes', 'physical_footprint_peak_bytes', 'compressed_bytes',
              'compressed_peak_bytes', 'decompressions', 'system_swap_used_bytes')
    require(values and all(uint(v.get(k)) for v in values for k in fields), 'Missing memory observations')
    require(all(0 < v['physical_footprint_bytes'] <= v['physical_footprint_peak_bytes'] <= BUDGET and
                v['compressed_bytes'] <= v['compressed_peak_bytes'] for v in values),
            'Invalid or excessive process footprint')
    clean = all(v['compressed_bytes'] == v['compressed_peak_bytes'] == 0 for v in values)
    clean &= len({v['decompressions'] for v in values}) == len({v['system_swap_used_bytes'] for v in values}) == 1
    hosts = [raw.get('host_before', {}), raw.get('host_after', {})]
    clean_host = all(h.get('thermal_state') == 0 and h.get('low_power_mode') is False and
        isinstance(h.get('power_source'), str) and bool(h['power_source']) and
        uint(h.get('monotonic_ns')) for h in hosts)
    clean_host &= (hosts[0].get('power_source') == hosts[1].get('power_source') and
                   hosts[0].get('monotonic_ns', 0) <= hosts[1].get('monotonic_ns', -1))
    return dict(clean_memory=bool(clean), peak_physical_bytes=max(v['physical_footprint_peak_bytes'] for v in values),
                peak_compressed_bytes=max(v['compressed_peak_bytes'] for v in values),
                clean_host=bool(clean_host),
                swap_observations_bytes=[v['system_swap_used_bytes'] for v in values])


def check_state(state, work, consumed, revision):
    history = work['prompt_ids']+work['continuation_ids'][:consumed]
    require(isinstance(state, dict) and state.get('artifact_revision') == revision and
            state.get('valid') is True and state.get('tokens') == 72+consumed and
            state.get('history') == history[-2:] and len(state.get('layers', [])) == 48,
            'Incomplete persistent state or changed token/ngram history')
    for index, layer in enumerate(state['layers']):
        buffers = layer.get('buffers', [])
        require(layer.get('position') == 72+consumed and len(buffers) == 6 and any(buffers) and
                all(b is None or (isinstance(b, dict) and uint(b.get('bytes')) and b['bytes'] > 0 and
                                  valid_sha256(b.get('sha256'))) for b in buffers),
                'Missing layer state or invalid buffer identity')
        expected = ([122880, 3145728, None, None, None, None] if (index+1)%4 else
                    [None, None, 16777216, 16777216, 4194304, None])
        if index == 1: expected[5] = 368640
        require([b['bytes'] if b else None for b in buffers] == expected,
                'Missing or changed persistent state buffer geometry')


def check_routes(routes, consumed):
    require(isinstance(routes, list) and len(routes) == 48 and
            [r.get('layer') for r in routes] == list(range(48)) and
            all(r.get('tokens') == 72+consumed and valid_sha256(r.get('sha256')) for r in routes),
            'Missing full-layer routed-expert identity')


def observe(raw, frozen, work, width, validation, input_sha256, *, expert_slots=1072):
    valid_work(work); total = 4 if validation else 16
    require(raw.get('kind') == 'perfect_draft_probe_v1' and raw.get('complete') is True and
            raw.get('prime_expert_schedule')==CRITERIA['prime_expert_schedule'] and
            raw.get('validation') is validation and raw.get('width') == width and
            raw.get('verified_tokens') == total and raw.get('configuration') == configuration(width, expert_slots=expert_slots) and
            raw.get('runtime_kernel_policy') == KERNEL_POLICY and raw.get('input_sha256') == input_sha256 and
            raw.get('artifact_revision') == frozen['artifact_revision'] and
            raw.get('prepared_manifest_sha256') == frozen['prepared_manifest_sha256'] and
            raw.get('route_audit_enabled') is True and raw.get('optimistic_upper_bound_only') is True and
            raw.get('normal_request_latency_qualified') is False and raw.get('production_promoted') is False and
            uint(raw.get('setup_ns')) and raw['setup_ns'] > 0,
            'Incomplete verifier or changed scope/configuration/input')
    plan = raw['memory_plan']
    require(plan.get('limit_bytes') == BUDGET and plan.get('expert_slots') == expert_slots and
            plan.get('panel_tokens') == 512 and
            uint(raw.get('snapshot_allocated_bytes')) and 0 < raw['snapshot_allocated_bytes'] <= 128*1024**2 and
            uint(raw.get('host_logits_bound_bytes')) and raw['host_logits_bound_bytes'] >= 4*VOCAB*4 and
            plan['planned_bytes']+raw['snapshot_allocated_bytes']+raw['host_logits_bound_bytes'] <= BUDGET,
            'Snapshot/logit workspace is unaccounted or outside the fixed budget')
    for state, tile in ((raw['before'], 1), (raw['after'], width)):
        check_machine(state, frozen); check_configuration(state, dict(configs()[0], token_tile=tile))
        metal = state['metal']; residency = metal['residency']; kernels = metal['kernels']
        require(state['memory_plan'] == plan and state.get('diagnostic_stream_trunk') is False and
                metal['live_command_groups'] == 0 and metal['peak_command_groups'] <= 2 and
                state.get('completion_pipeline') is True and
                kernels.get('profile') is False and kernels.get('counter_profile') is False and
                kernels.get('profile_decode_only') is False and residency['mode'] == 'core-cache' and
                residency['pending_retirements'] == 0 and
                residency['bytes_by_class']['expert'] == plan['expert_bytes'],
                'Outstanding users, changed workspace, residency or instrumentation')
    prime = raw.get('prime', {})
    require(valid_sha256(prime.get('logits_sha256')) and valid_sha256(prime.get('cache_state')) and
            prime.get('state') and prime.get('routes'),
            'Missing prompt logits, state or route identity')
    check_state(prime['state'], work, 0, frozen['artifact_revision']); check_routes(prime['routes'], 0)
    blocks = raw.get('blocks', [])
    require(len(blocks) == total//width, 'Missing or duplicated verification block')
    hashes = []; endpoints = []; wall = 0
    for i, block in enumerate(blocks):
        consumed = i*width
        require(block.get('offset') == 72+consumed and
                block.get('input_tokens') == work['continuation_ids'][consumed:consumed+width] and
                block.get('next_ids') == work['expected_next_ids'][consumed:consumed+width] and
                isinstance(block.get('logits_sha256'), list) and len(block['logits_sha256']) == width and
                all(valid_sha256(v) for v in block['logits_sha256']),
                'Incomplete token/logit/state/route coverage or greedy mismatch')
        checkpointed = validation or (consumed+width)%4 == 0
        if checkpointed:
            check_state(block['state'], work, consumed+width, frozen['artifact_revision'])
            check_routes(block['routes'], consumed+width)
        else:
            require(block.get('state') is None and block.get('routes') is None,
                    'Unplanned state/route scan between common timing boundaries')
        times = [block.get(k) for k in ('checkpoint_ns', 'forward_ns', 'accept_ns', 'wall_ns')]
        require(all(uint(v) for v in times) and times[1] > 0 and times[2] > 0 and
                times[3] >= sum(times[:3]) and (times[0] == 0 if width == 1 else times[0] > 0),
                'Invalid or incomplete checkpoint/forward/acceptance timing')
        hashes.extend(block['logits_sha256']); wall += block['wall_ns']
        if checkpointed:
            endpoints.append(dict(consumed=consumed+width, state=block['state'], routes=block['routes']))
    require(raw.get('final_state') == blocks[-1]['state'] and raw.get('decode_wall_ns') == wall and
            type(raw.get('verified_tokens_per_second')) in (int, float) and
            math.isfinite(raw['verified_tokens_per_second']) and
            math.isclose(raw['verified_tokens_per_second'], total*1e9/wall, rel_tol=1e-10),
            'Final state or measured throughput does not reconcile')
    expected_checks = (['causal_prefix_unchanged', 'zero_accept_restores_state']+
        [f'accepted_prefix_replay_{k}' for k in range(1, width)]) if validation and width > 1 else []
    require(isinstance(raw.get('rollback_checks'), list) and
            [c.get('name') for c in raw['rollback_checks']] == expected_checks and
            all(c.get('passed') is True for c in raw['rollback_checks']),
            'Missing, duplicate or failed rollback checks')
    if not validation:
        before, after = raw['before'], raw['after']
        a, b = before['metal'], after['metal']
        require(all(uint(s.get(k)) for s in (a, b) for k in ('dispatches', 'submissions', 'gpu_command_ns')) and
                all(b[k] > a[k] for k in ('dispatches', 'submissions', 'gpu_command_ns')),
                'Missing native execution counters')
        delta = {k: b['kernel_dispatches'].get(k, 0)-a['kernel_dispatches'].get(k, 0)
                 for k in set(a['kernel_dispatches']) | set(b['kernel_dispatches'])}
        require(all(uint(v) for v in delta.values()) and sum(delta.values()) == b['dispatches']-a['dispatches'],
                'Native dispatch population does not reconcile')
        require(len(before.get('layer_expert_passes', [])) == len(after.get('layer_expert_passes', [])) == 48 and
                all(b-a == total//width for a, b in zip(before['layer_expert_passes'], after['layer_expert_passes'])) and
                all(after['passes'][name]-before['passes'][name] ==
                    (total//width if name == ('decode' if width == 1 else 'short_append') else 0)
                    for name in ('decode', 'short_append', 'prefill', 'panel')),
                'Missing full-layer forward coverage or unplanned schedule work')
    memory = memory_observation(raw)
    return dict(width=width, validation=validation, verified_tokens=total, prime=prime,
        row_logits_sha256=hashes, endpoints=endpoints,
        snapshot_allocated_bytes=raw['snapshot_allocated_bytes'], host_logits_bound_bytes=raw['host_logits_bound_bytes'],
        decode_wall_ns=wall, verified_tokens_per_second=total*1e9/wall,
        checkpoint_ns=sum(b['checkpoint_ns'] for b in blocks), forward_ns=sum(b['forward_ns'] for b in blocks),
        accept_ns=sum(b['accept_ns'] for b in blocks), setup_ns=raw['setup_ns'],
        rollback_checks=raw['rollback_checks'], **memory)


def compare(reference, candidate):
    require(reference['width'] == 1 and candidate['width'] in (2, 4) and
            reference['validation'] == candidate['validation'] and
            reference['verified_tokens'] == candidate['verified_tokens'], 'Incompatible verifier comparison')
    require(reference['prime'] == candidate['prime'] and
            reference['row_logits_sha256'] == candidate['row_logits_sha256'] and
            reference['snapshot_allocated_bytes'] == candidate['snapshot_allocated_bytes'] and
            reference['host_logits_bound_bytes'] == candidate['host_logits_bound_bytes'],
            'Block verification changed prompt, row logits or reserved allocation')
    by_prefix = {p['consumed']: p for p in reference['endpoints']}
    require(all(p == by_prefix.get(p['consumed']) for p in candidate['endpoints']),
            'Block verification changed persistent state or exact routes')
    if candidate['validation']:
        checks = {c['name']: c for c in candidate['rollback_checks']}
        require(checks['zero_accept_restores_state'].get('recovery_state') == reference['prime']['state'] and
            checks['causal_prefix_unchanged'].get('logits_sha256') ==
                reference['row_logits_sha256'][:candidate['width']-1],
            'Zero acceptance or causal-prefix validation differs from serial reference')
        for count in range(1, candidate['width']):
            check = checks[f'accepted_prefix_replay_{count}']
            require(check.get('recovery_state') == by_prefix[count]['state'] and
                    check.get('logits_sha256') == reference['row_logits_sha256'][count-1],
                    'Accepted-prefix recovery differs from serial reference')
    return dict(exact_row_logits=True, exact_matching_prefix_state=True, exact_routes=True,
                width=candidate['width'], matched_prefixes=len(candidate['endpoints']))


def decide(measurements):
    require([(m['pair'], m['observation']['width']) for m in measurements] == ORDER,
            'Missing, duplicated or reordered alternating timing rounds')
    by = {(m['pair'], m['observation']['width']): m['observation'] for m in measurements}
    require(all(not v['validation'] and v['verified_tokens'] == 16 for v in by.values()),
            'Validation or short work cannot qualify the timing screen')
    require(all(uint(v['decode_wall_ns']) and v['decode_wall_ns'] > 0 and
                type(v['verified_tokens_per_second']) in (int, float) and
                math.isfinite(v['verified_tokens_per_second']) and
                math.isclose(v['verified_tokens_per_second'], 16e9/v['decode_wall_ns'], rel_tol=1e-10)
                for v in by.values()), 'Invalid or inconsistent verifier throughput')
    require(all(v['prime'] == by[0, 1]['prime'] and
                v['row_logits_sha256'] == by[0, 1]['row_logits_sha256'] for v in by.values()),
                'Initial cache state or exact outputs changed across fresh rounds')
    clean = all(v['clean_memory'] and v['clean_host'] for v in by.values()); candidates = []
    for width in (2, 4):
        proof = [compare(by[p, 1], by[p, width]) for p in (0, 1)]
        ratios = [by[p, width]['decode_wall_ns']/by[p, 1]['decode_wall_ns'] for p in (0, 1)]
        speeds = [by[p, width]['verified_tokens_per_second'] for p in (0, 1)]
        passed = clean and all(r < 1 for r in ratios) and min(speeds) >= 5
        candidates.append(dict(width=width, paired_wall_ratios=ratios,
            median_wall_ratio=statistics.median(ratios), verified_tokens_per_second=speeds,
            median_verified_tokens_per_second=statistics.median(speeds), correctness=proof,
            optimistic_screen_passed=passed, confidence_95=None))
    promising = [c['width'] for c in candidates if c['optimistic_screen_passed']]
    return dict(status='memory_disturbed' if not clean else 'perfect_verifier_promising' if promising else
        'perfect_verifier_below_gate', clean_memory=clean, candidates=candidates,
        promising_widths=promising, actual_draft_tokens_per_second=None,
        accepted_fraction_assumed=1.0, draft_cost_included=False, **FLAGS)


def run(output, binary):
    from build_perfect_draft import verify_producer
    from benchmark_host import build_probe, preflight
    work, source = source_input(); valid_work(work)
    exp = Experiment(output, 'perfect_draft_screen_v1', [configuration(w) for w in (1, 2, 4)], work, 600)
    with exp:
        require(all(source[k] == exp.frozen[k] for k in ('artifact_revision', 'prepared_manifest_sha256')),
                'Fixed-token source artifact differs')
        exp.report.update(criteria=CRITERIA, limitations=LIMITATIONS, token_source=source,
                          validation=[], binary=str(binary), **FLAGS)
        proof = verify_producer(binary, exp.frozen['build'])
        verify_seal(PRIME_REFERENCE.parent,sha(PRIME_REFERENCE.parent/'evidence-files.json'))
        prime_reference=load(PRIME_REFERENCE)
        require(prime_reference['complete'] is True and prime_reference['width']==1 and prime_reference['validation'] is True and
            prime_reference['input_sha256']==sha(exp.out/'workload.json') and
            prime_reference['before']['metal']['build_fingerprint']==exp.frozen['build'] and
            prime_reference['artifact_revision']==exp.frozen['artifact_revision'] and
            prime_reference['prepared_manifest_sha256']==exp.frozen['prepared_manifest_sha256'],
            'Changed original prime reference identity')
        exp.report['prime_reference']=dict(path=str(PRIME_REFERENCE),sha256=sha(PRIME_REFERENCE))
        save(exp.out/'producer.json', proof['producer'])
        exp.report['producer'] = dict(source='producer.json', sha256=sha(exp.out/'producer.json'),
            binary_sha256=proof['producer']['binary_sha256'],
            base_native_fingerprint=proof['producer']['base_native_fingerprint'],
            actual_binary_is_base_native_build=False)
        shutil.copyfile(PROTOCOL, exp.out/'protocol.md'); exp.report['protocol_sha256'] = sha(PROTOCOL)
        host_probe=build_probe(ROOT/'.cache/benchmark-host'/hashlib.sha256(str(exp.out).encode()).hexdigest())
        save(exp.out/'host-producer.json',host_probe['producer'])
        exp.report['host_preflight']=[]
        exp.report['host_producer']=dict(source='host-producer.json',sha256=sha(exp.out/'host-producer.json'))
        freeze(exp, [*proof['files'], *host_probe['files'], PRIME_REFERENCE,exp.out/'host-producer.json',exp.out/'producer.json', PROTOCOL, exp.out/'protocol.md',
                     *[p for p in SOURCE.iterdir() if p.is_file()]])
        exp.persist(); exp.guard.check_resources(initial=True)
        input_hash = sha(exp.out/'workload.json'); validated = {}; observed = {}
        for mode, pair, width in [('validate', None, w) for w in VALIDATION_WIDTHS]+[
                                  ('timing', p, w) for p, w in ORDER]:
            stem = f'validate-{width}' if mode == 'validate' else f'pair-{pair}-width-{width}'
            preflight(exp,host_probe,stem)
            exp.command([binary, exp.model, exp.prepared, exp.out/'workload.json',
                         exp.out/(stem+'.json'), str(width), mode], stem, limit=150 if mode == 'validate' else 90,
                        validation=mode == 'validate')
            result = observe(load(exp.out/(stem+'.json')), exp.frozen, work, width,
                             mode == 'validate', input_hash)
            check_prime_reference(result['prime'],prime_reference['prime'])
            item = dict(source=stem+'.json', sha256=sha(exp.out/(stem+'.json')), observation=result)
            if mode == 'validate':
                if width != 1: item['correctness'] = compare(validated[1], result)
                validated[width] = result; exp.report['validation'].append(item)
            else:
                item['pair'] = pair; observed[pair, width] = result
                exp.report['measurements'].append(item)
                if (pair, 1) in observed:
                    for candidate in (2, 4):
                        if (pair, candidate) in observed: compare(observed[pair, 1], observed[pair, candidate])
            exp.persist()
            if not result['clean_memory'] or not result['clean_host']:
                raise ResourceBlocked('Perfect-verifier memory or host observation is disturbed')
        exp.report.update(decide(exp.report['measurements']))
    return exp.report


def audit(output):
    from build_perfect_draft import verify_producer
    digest = sha(output/'evidence-files.json'); verify_seal(output, digest)
    saved, frozen = load(output/'summary.json'), load(output/'identity.json')
    require(saved['kind'] == 'perfect_draft_screen_v1' and saved['criteria'] == load_json(CRITERIA) and
            saved['limitations'] == LIMITATIONS and saved['configurations'] == [configuration(w) for w in (1, 2, 4)] and
            saved['identity'] == {k: frozen[k] for k in saved['identity']} and
            all(saved.get(k) is False for k in FLAGS), 'Changed perfect-verifier protocol')
    provenance = verify_sources(output.parent, frozen)
    proof = verify_producer(Path(saved['binary']), frozen['build'])
    receipt = saved['producer']; producer = load(output/'producer.json')
    require(receipt['source'] == 'producer.json' and sha(output/'producer.json') == receipt['sha256'] and
            producer == proof['producer'] and receipt['binary_sha256'] == producer['binary_sha256'] and
            receipt['base_native_fingerprint'] == producer['base_native_fingerprint'] and
            receipt['actual_binary_is_base_native_build'] is False and
            all(frozen['files'][str(Path(p).resolve())] == sha(p) for p in proof['files']),
            'Source-copy producer or actual executable changed')
    work, source = source_input()
    require(saved['token_source'] == source and saved['workload'] == work == load(output/'workload.json') and
            sha(output/'protocol.md') == saved['protocol_sha256'] == frozen['files'][str(PROTOCOL)],
            'Changed source input or protocol')
    if 'host_preflight' in saved:
        from benchmark_host import observe as observe_host
        host_receipt=saved['host_producer'];host_producer=load(output/'host-producer.json')
        require(host_receipt['source']=='host-producer.json' and host_receipt['sha256']==sha(output/'host-producer.json') and
            host_producer['kind']=='benchmark_host_producer_v1' and host_producer['complete'] is True and
            host_producer['base_native_fingerprint']==frozen['build'] and
            host_producer['binary_sha256']==frozen['files'][host_producer['binary']] and
            all(frozen['files'].get(p)==h for p,h in host_producer['files'].items()),'Changed host probe producer')
        stems=[f'validate-{w}' for w in VALIDATION_WIDTHS]+[f'pair-{p}-width-{w}' for p,w in ORDER]
        checks=saved['host_preflight']
        require(len(checks)<=len(stems) and (not saved['complete'] or len(checks)==len(stems)),
            'Missing or extra host preflights')
        for i,item in enumerate(checks):
            path=output/(stems[i]+'-host.json')
            result=observe_host(load(path),frozen['build'])
            require(item['source']==path.name and item['sha256']==sha(path) and item['observation']==result and
                (result['clean_host'] or (i==len(checks)-1 and saved['status']=='resource_blocked' and
                    not (output/(stems[i]+'.json')).exists())), 'Changed host preflight or work ran after blocked host')
    validated = {}; measurements = []; input_hash = sha(output/'workload.json')
    require(saved['prime_reference']==dict(path=str(PRIME_REFERENCE),sha256=sha(PRIME_REFERENCE)) and
        frozen['files'][str(PRIME_REFERENCE)]==sha(PRIME_REFERENCE),'Changed original prime anchor')
    prime_reference=load(PRIME_REFERENCE)['prime']
    require(len(saved['validation']) <= 3 and len(saved['measurements']) <= 6, 'Extra native processes')
    for i, item in enumerate(saved['validation']):
        width = VALIDATION_WIDTHS[i]; path = output/f'validate-{width}.json'
        require(item['source'] == path.name and sha(path) == item['sha256'], 'Validation source changed')
        result = observe(load(path), frozen, work, width, True, input_hash)
        check_prime_reference(result['prime'],prime_reference)
        require(item['observation'] == result, 'Validation observation changed')
        validated[width] = result
        if width != 1: require(item.get('correctness') == compare(validated[1], result), 'Validation proof changed')
    for i, item in enumerate(saved['measurements']):
        pair, width = ORDER[i]; path = output/f'pair-{pair}-width-{width}.json'
        require(item['pair'] == pair and item['source'] == path.name and sha(path) == item['sha256'],
                'Timing source changed')
        result = observe(load(path), frozen, work, width, False, input_hash)
        check_prime_reference(result['prime'],prime_reference)
        require(item['observation'] == result, 'Timing observation changed')
        measurements.append(dict(item, observation=result))
    if saved['complete']:
        require(len(validated) == 3 and all(v['clean_memory'] and v['clean_host'] for v in validated.values()),
                'Timing screen lacks clean validation')
        result = decide(measurements)
        require(all(saved.get(k) == v for k, v in result.items()), 'Saved decision differs from evidence')
    else:
        require(saved['status'] in ('failed', 'resource_blocked', 'interrupted', 'time_budget_exhausted'),
                'Incomplete stage lacks terminal disposition')
    return dict(kind='perfect_draft_audit_v1', complete=True, audit_passed=True,
        recorded_complete=saved['complete'], status=saved['status'], source_seal_sha256=digest,
        source_provenance=provenance, **FLAGS)


def load_json(value):
    # JSON stores ordered tuple pairs as arrays; use identical persisted shape.
    import json
    return json.loads(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'verify'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    if args.action == 'verify':
        if args.source is None: parser.error('--source is required')
        save(args.output, audit(args.source.resolve()))
    else:
        if args.binary is None: parser.error('--binary is required')
        raise SystemExit(0 if run(args.output, args.binary.resolve())['complete'] else 2)
