import copy
import hashlib
import unittest

from native_q4_replay import FIXED_KERNELS
from perfect_draft import (BUDGET, FLAGS, KERNEL_POLICY, ORDER, VOCAB,
                           compare, configuration, decide, observe)


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def fixture(width=1, validation=False):
    work = dict(prompt_ids=list(range(72)), continuation_ids=list(range(100, 116)),
                expected_next_ids=list(range(101, 117)), expected_prompt_id=100)
    frozen = dict(build='build', device='Apple M1 Pro', physical_bytes=32*1024**3,
                  artifact_revision='revision', prepared_manifest_sha256='manifest')
    memory = dict(physical_footprint_bytes=8*1024**3, physical_footprint_peak_bytes=9*1024**3,
                  compressed_bytes=0, compressed_peak_bytes=0, decompressions=0, system_swap_used_bytes=0)
    host = dict(thermal_state=0, low_power_mode=False, power_source='AC Power', monotonic_ns=100)
    plan = dict(limit_bytes=BUDGET, expert_slots=1072, expert_bytes=1072*2768896,
                panel_tokens=512, planned_bytes=10*1024**3)
    before = dict(memory_plan=plan, artifact_revision='revision', process=memory,
        prepared=dict(manifest_sha256='manifest'), diagnostic_stream_trunk=False,
        execution=dict(cache_policy='clock', sparse_selection='cpu', residency='core-cache',
            decode_path='reference', expert_tail='wait', decode_scratch='reuse', prefill_pipeline='serial',
            phase_memory='fixed', cached_token_replay=False), ready_group=4, chunk_tokens=128, io_workers=8,
        metal=dict(build_fingerprint='build', device='Apple M1 Pro', physical_bytes=32*1024**3,
            peak_buffer_bytes=9*1024**3, live_command_groups=0, peak_command_groups=2,
            kernels=dict(FIXED_KERNELS, q4_decode='reference'),
            residency=dict(mode='core-cache', pending_retirements=0,
                bytes_by_class=dict(expert=plan['expert_bytes'])),
            dispatches=1, submissions=1, gpu_command_ns=1, kernel_dispatches=dict(kernel=1)),
        layer_expert_passes=[1]*48, passes=dict(decode=0, short_append=0, prefill=1, panel=0))
    count = 4 if validation else 16
    after = copy.deepcopy(before)
    after['metal']['kernels']['token_tile'] = width
    after['metal'].update(dispatches=1+count//width, submissions=1+count//width,
                          gpu_command_ns=1+count//width, kernel_dispatches=dict(kernel=1+count//width))
    after['layer_expert_passes'] = [1+count//width]*48
    after['passes']['decode' if width == 1 else 'short_append'] = count//width

    def state(consumed):
        ids = work['prompt_ids']+work['continuation_ids'][:consumed]
        layers = []
        for layer in range(48):
            sizes = ([122880, 3145728, None, None, None, None] if (layer+1)%4 else
                     [None, None, 16777216, 16777216, 4194304, None])
            if layer == 1: sizes[5] = 368640
            layers.append(dict(position=72+consumed, buffers=[
                dict(bytes=size, sha256=digest((layer, consumed, i))) if size else None
                for i, size in enumerate(sizes)]))
        return dict(artifact_revision='revision', valid=True, tokens=72+consumed, history=ids[-2:],
                    layers=layers)

    def routes(consumed):
        return [dict(layer=l, tokens=72+consumed, sha256=digest((l, consumed))) for l in range(48)]

    raw = dict(kind='perfect_draft_probe_v1', complete=True, validation=validation, width=width,
        verified_tokens=count, configuration=configuration(width), memory_plan=plan,
        input_sha256='input', artifact_revision='revision', prepared_manifest_sha256='manifest',
        runtime_kernel_policy=KERNEL_POLICY, route_audit_enabled=True,
        optimistic_upper_bound_only=True, normal_request_latency_qualified=False, production_promoted=False,
        setup_ns=1000000000,
        before=before, after=after, host_before=host, host_after=dict(host, monotonic_ns=101),
        process_after_destroy=dict(memory, physical_footprint_bytes=1024**3),
        snapshot_allocated_bytes=120*1024**2, host_logits_bound_bytes=16*VOCAB*4+1024**2,
        prime=dict(logits_sha256=digest('prime'), state=state(0), routes=routes(0), cache_state=digest('cache')),
        blocks=[], rollback_checks=[])
    for at in range(0, count, width):
        endpoint = validation or (at+width)%4 == 0
        wall = {1:250000000, 2:300000000, 4:400000000}[width]
        checkpoint = 0 if width == 1 else 10000
        raw['blocks'].append(dict(offset=72+at, input_tokens=work['continuation_ids'][at:at+width],
            next_ids=work['expected_next_ids'][at:at+width],
            logits_sha256=[digest(k) for k in range(at, at+width)],
            state=state(at+width) if endpoint else None, routes=routes(at+width) if endpoint else None,
            checkpoint_ns=checkpoint, forward_ns=wall-checkpoint-2000, accept_ns=1000, wall_ns=wall,
            memory_before=memory, memory_after=memory))
    raw['final_state'] = state(count)
    raw['decode_wall_ns'] = sum(b['wall_ns'] for b in raw['blocks'])
    raw['verified_tokens_per_second'] = count*1e9/raw['decode_wall_ns']
    if validation and width > 1:
        raw['rollback_checks'] = [dict(name='causal_prefix_unchanged', passed=True,
            logits_sha256=[digest(k) for k in range(width-1)]),
            dict(name='zero_accept_restores_state', passed=True, recovery_state=state(0))]+[
            dict(name=f'accepted_prefix_replay_{k}', passed=True, recovery_state=state(k),
                 logits_sha256=digest(k-1)) for k in range(1, width)]
    return raw, frozen, work


def observation(width, validation=False):
    raw, frozen, work = fixture(width, validation)
    return observe(raw, frozen, work, width, validation, 'input')


class PerfectDraftTest(unittest.TestCase):
    def test_complete_exact_blocks_use_common_timing_state_boundaries(self):
        serial = observation(1)
        for width in (2, 4):
            candidate = observation(width)
            result = compare(serial, candidate)
            self.assertTrue(result['exact_row_logits'])
            self.assertEqual(result['matched_prefixes'], 4)
            self.assertEqual([e['consumed'] for e in candidate['endpoints']], [4, 8, 12, 16])

    def test_validation_checks_recovery_against_serial_prefixes(self):
        serial = observation(1, True)
        for width in (2, 4):
            candidate = observation(width, True)
            self.assertTrue(compare(serial, candidate)['exact_matching_prefix_state'])
            for field in ('recovery_state', 'logits_sha256'):
                bad = copy.deepcopy(candidate)
                bad['rollback_checks'][-1][field] = 'incorrect'
                with self.assertRaises(ValueError): compare(serial, bad)

    def test_mismatched_logits_state_routes_or_initial_cache_fail(self):
        a, b = observation(1), observation(4)
        changes = [lambda r:r['row_logits_sha256'].__setitem__(0, digest('different')),
            lambda r:r['prime'].update(cache_state=digest('different')),
            lambda r:r['endpoints'][0]['state']['layers'][0]['buffers'][0].update(sha256=digest('different')),
            lambda r:r['endpoints'][0]['routes'][0].update(sha256=digest('different'))]
        for change in changes:
            bad = copy.deepcopy(b); change(bad)
            with self.assertRaises(ValueError): compare(a, bad)

    def test_missing_memory_work_identity_and_counts_fail_closed(self):
        mutations = [lambda r:r.update(input_sha256='changed'),
            lambda r:r.update(complete=False), lambda r:r['blocks'].pop(),
            lambda r:r['blocks'][0]['logits_sha256'].pop(),
            lambda r:r['blocks'][0]['next_ids'].__setitem__(0, 99),
            lambda r:r['blocks'][0]['state']['layers'].pop(),
            lambda r:r['blocks'][0]['memory_before'].pop('compressed_peak_bytes'),
            lambda r:r['before']['metal']['kernels'].update(token_tile=4),
            lambda r:r['after']['layer_expert_passes'].pop(),
            lambda r:r.update(snapshot_allocated_bytes=129*1024**2),
            lambda r:r['after']['metal'].update(dispatches=999),
            lambda r:r['blocks'][0].update(wall_ns=1)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                raw, frozen, work = fixture(4); mutation(raw)
                with self.assertRaises((ValueError, KeyError)): observe(raw, frozen, work, 4, False, 'input')

    def test_short_width_cannot_do_extra_state_scans_between_common_boundaries(self):
        raw, frozen, work = fixture(1)
        raw['blocks'][0]['state'] = raw['blocks'][3]['state']
        with self.assertRaises(ValueError): observe(raw, frozen, work, 1, False, 'input')

    def test_optimistic_gate_never_qualifies_actual_draft_or_production(self):
        rows = [dict(pair=p, observation=observation(w)) for p, w in ORDER]
        result = decide(rows)
        self.assertEqual(result['promising_widths'], [2, 4])
        self.assertIsNone(result['actual_draft_tokens_per_second'])
        self.assertEqual(result['accepted_fraction_assumed'], 1)
        self.assertTrue(all(result[k] is False for k in FLAGS))
        self.assertTrue(all(c['confidence_95'] is None for c in result['candidates']))

    def test_any_dirty_run_or_slow_pair_prevents_candidate_advancement(self):
        for flag in ('clean_memory', 'clean_host'):
            rows = [dict(pair=p, observation=observation(w)) for p, w in ORDER]
            rows[1]['observation'][flag] = False
            self.assertEqual(decide(rows)['promising_widths'], [])
        rows = [dict(pair=p, observation=observation(w)) for p, w in ORDER]
        slow = rows[1]['observation']; slow['decode_wall_ns'] = 4_100_000_000
        slow['verified_tokens_per_second'] = 16e9/slow['decode_wall_ns']
        self.assertEqual(decide(rows)['promising_widths'], [4])
        with self.assertRaises(ValueError): decide(rows[:-1])

    def test_lifetime_compression_and_host_conditions_are_not_clean(self):
        for mutate in (lambda r:r['process_after_destroy'].update(compressed_peak_bytes=1),
                       lambda r:r['host_after'].update(thermal_state=1),
                       lambda r:r['host_after'].update(power_source='Battery Power')):
            raw, frozen, work = fixture(4); mutate(raw)
            observed = observe(raw, frozen, work, 4, False, 'input')
            self.assertFalse(observed['clean_memory'] and observed['clean_host'])


if __name__ == '__main__':
    unittest.main()
