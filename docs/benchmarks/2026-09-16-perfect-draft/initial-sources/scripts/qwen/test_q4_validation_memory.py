import copy
import unittest

from native_q4_replay import BUDGET, COUNTERS, fixture_records
from q4_validation_memory import DISPATCHES, FLAGS, ORDER, PHASES, analyze, decision
from test_q4_scratch_lifetime import fixture as lifetime_fixture


RETAINED = 48*27*16384


def fixture(validation='off'):
    """A complete process with fixed warm/observed work and 57 lifecycle samples."""
    raw, manifest = lifetime_fixture('check', 'forward')
    raw.update(kind='native_q4_scratch_memory_v1', mode='memory-'+validation,
               validation=validation == 'on', profiling=False, counters=False, **FLAGS)
    pair = raw['pairs'][0]; pair['arms'] = pair['arms'][:1]; raw['pairs'] = [pair]
    arm = pair['arms'][0]; before = arm['metal_before']; after = arm['metal_after']
    before.update(allocation_count=2000, scratch_reuses=1296,
                  kernel_dispatches=copy.deepcopy(DISPATCHES))
    before['scratch_pools'][0].update(reuses=1296)
    for key in COUNTERS:
        after[key] = before[key]+arm['sample'][key]
    after['kernel_dispatches'] = {k: 2*v for k, v in DISPATCHES.items()}
    after['scratch_pools'][0]['reuses'] = 2592
    initial = copy.deepcopy(before)
    initial['scratch_pools'] = [copy.deepcopy(before['scratch_pools'][1]) for _ in range(2)]
    initial['live_buffer_bytes'] -= RETAINED
    final = copy.deepcopy(after)
    final['scratch_pools'] = copy.deepcopy(initial['scratch_pools'])
    final['live_buffer_bytes'] -= RETAINED
    raw.update(initial_metal=initial, final_metal=final)
    phases = []
    for index, phase in enumerate(PHASES):
        state = copy.deepcopy(initial)
        pass_index = None
        if phase == 'startup': state = None
        elif phase == 'post_warmup': state = copy.deepcopy(before)
        elif phase == 'pass_complete':
            pass_index = index-8; count = pass_index+1
            state = copy.deepcopy(before)
            state['active_scratch_slot'] = 0 if count < 48 else -1
            for key in COUNTERS:
                state[key] += arm['sample'][key]*count//48
            state['scratch_pools'][0].update(unused_retained_bytes=(48-count)*27*16384,
                reuses=1296+27*count, wait_ns=101)
            state['kernel_dispatches'] = {k: v+v*count//48 for k, v in DISPATCHES.items()}
            if count == 48: state = copy.deepcopy(after)
        elif phase == 'scratch_released': state = copy.deepcopy(final)
        counters = None if state is None else dict(live_command_groups=0, encoded_buffer_references=0,
            live_buffer_bytes=state['live_buffer_bytes'],
            scratch_bytes=sum(p['allocated_bytes'] for p in state['scratch_pools']),
            active_scratch_slot=state['active_scratch_slot'], device_allocated_bytes=state['live_buffer_bytes'])
        memory = dict(arm['memory_before'], compressed_peak_bytes=0)
        host = dict(arm['host_before'], monotonic_ns=index+1)
        entry = dict(phase=phase, at_ns=index+1, memory=memory, host=host,
                     metal=state, memory_counters=counters)
        if pass_index is not None: entry['pass_index'] = pass_index
        phases.append(entry)
    raw['memory_phases'] = phases
    frozen = dict(build=before['build_fingerprint'], artifact_revision=raw['fixture_manifest']['artifact_revision'],
                  prepared_manifest_sha256=raw['prepared_manifest_sha256'])
    return raw, frozen, fixture_records(raw['fixture_manifest']), manifest, None


def measured(validation='off'):
    return analyze(fixture(validation)[0], validation, *fixture(validation)[1:])


def dirty_from(raw, index, current=True):
    for phase in raw['memory_phases'][index:]:
        phase['memory']['compressed_peak_bytes'] = 16384
        phase['memory']['compressed_bytes'] = 16384 if current else 0


class Q4ValidationMemoryTest(unittest.TestCase):
    def test_same_native_work_in_both_modes_has_all_57_boundaries(self):
        results = []
        for mode in ('off', 'on'):
            args = fixture(mode); result = analyze(args[0], mode, *args[1:]); results.append(result)
            self.assertEqual(result['status'], 'memory_clean')
            self.assertTrue(result['clean_memory'])
            self.assertTrue(result['clean_host'])
            self.assertTrue(result['hard_guards_passed'])
            self.assertEqual([p['phase'] for p in result['phases']], PHASES)
            self.assertEqual(len(result['phases']), 57)
            self.assertEqual(result['lifetime']['retained_scratch_bytes'], RETAINED)
            self.assertEqual(result['lifetime']['reused_buffers_per_arm'], 1296)
            self.assertIsNone(result['first_dirty_phase'])
            for key in FLAGS: self.assertIs(result[key], False)
        self.assertEqual(results[0]['native_work_signature'], results[1]['native_work_signature'])

    def test_missing_duplicated_or_reordered_phases_are_rejected(self):
        for change in ('missing', 'duplicate', 'reorder', 'pass', 'stamp', 'startup_metal'):
            args = fixture(); phases = args[0]['memory_phases']
            if change == 'missing': phases.pop(12)
            elif change == 'duplicate': phases.insert(12, copy.deepcopy(phases[12]))
            elif change == 'reorder': phases[2:4] = reversed(phases[2:4])
            elif change == 'pass': phases[12]['pass_index'] += 1
            elif change == 'stamp': phases[12]['at_ns'] = phases[11]['at_ns']
            else: phases[0]['metal'] = copy.deepcopy(phases[1]['metal'])
            with self.subTest(change=change), self.assertRaises(ValueError):
                analyze(args[0], 'off', *args[1:])

    def test_scratch_prefix_reuse_allocation_and_outstanding_users_cannot_change(self):
        for change in ('prefix', 'reuse', 'global_reuse', 'pool_allocation', 'allocation', 'active', 'gpu', 'encoded', 'retained'):
            args = fixture(); phase = args[0]['memory_phases'][20]; state = phase['metal']
            if change == 'prefix': state['scratch_pools'][0]['unused_retained_bytes'] += 16384
            elif change == 'reuse': state['scratch_pools'][0]['reuses'] += 1
            elif change == 'global_reuse': state['scratch_reuses'] += 1
            elif change == 'pool_allocation': state['scratch_pools'][0]['allocation_count'] += 1
            elif change == 'allocation': state['allocation_count'] += 1
            elif change == 'active': state['active_scratch_slot'] = phase['memory_counters']['active_scratch_slot'] = -1
            elif change == 'gpu': state['live_command_groups'] = phase['memory_counters']['live_command_groups'] = 1
            elif change == 'encoded': phase['memory_counters']['encoded_buffer_references'] = 1
            else:
                state['scratch_pools'][0]['allocated_bytes'] += 16384
                phase['memory_counters']['scratch_bytes'] += 16384
            with self.subTest(change=change), self.assertRaises(ValueError):
                analyze(args[0], 'off', *args[1:])

    def test_hard_budget_host_swap_and_missing_gauges_are_rejected(self):
        for change in ('peak', 'thermal', 'low_power', 'power_source', 'swap', 'missing_memory', 'counter_backwards'):
            args = fixture(); phase = args[0]['memory_phases'][20]
            if change == 'peak': phase['memory']['physical_footprint_peak_bytes'] = BUDGET+1
            elif change == 'thermal': phase['host']['thermal_state'] = 1
            elif change == 'low_power': phase['host']['low_power_mode'] = True
            elif change == 'power_source': phase['host']['power_source'] = 'Battery'
            elif change == 'swap': phase['memory']['system_swap_used_bytes'] += 16384
            elif change == 'missing_memory': phase['memory'].pop('compressed_peak_bytes')
            else:
                args[0]['memory_phases'][19]['memory']['decompressions'] = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                analyze(args[0], 'off', *args[1:])

    def test_initial_dirty_and_peak_only_compression_never_become_clean(self):
        for index, current in ((0, True), (20, True), (0, False), (20, False)):
            args = fixture('on'); dirty_from(args[0], index, current)
            result = analyze(args[0], 'on', *args[1:])
            with self.subTest(index=index, current=current):
                self.assertEqual(result['status'], 'memory_dirty')
                self.assertFalse(result['clean_memory'])
                self.assertTrue(result['hard_guards_passed'])
                self.assertEqual(result['first_dirty_phase']['index'], index)
                self.assertEqual(result['peak_compressed_bytes'], 16384)
                for key in FLAGS: self.assertIs(result[key], False)

    def test_native_mode_arithmetic_and_profiling_mismatches_are_rejected(self):
        for change in ('mode', 'validation', 'profiling', 'counters', 'phase_profile', 'phase_counters', 'kernel', 'candidate', 'extra_arm', 'group'):
            args = fixture(); raw = args[0]; arm = raw['pairs'][0]['arms'][0]
            if change == 'mode': raw['mode'] = 'memory-on'
            elif change == 'validation': raw['validation'] = True
            elif change in ('profiling', 'counters'): raw[change] = True
            elif change in ('phase_profile', 'phase_counters'):
                raw['memory_phases'][20]['metal']['kernels']['profile' if change == 'phase_profile' else 'counter_profile'] = True
            elif change == 'kernel': arm['metal_before']['kernels']['q4_decode'] = 'packed-r2'
            elif change == 'candidate': arm['variant'] = 'packed-r2'
            elif change == 'extra_arm': raw['pairs'][0]['arms'].append(copy.deepcopy(arm))
            else: raw['pairs'][0]['group'] = 4
            with self.subTest(change=change), self.assertRaises(ValueError):
                analyze(raw, 'off', *args[1:])

    def test_work_bytes_dispatches_and_complete_release_are_required(self):
        for change in ('reads', 'dispatch', 'gpu_counter', 'release', 'boundary', 'storage'):
            args = fixture(); raw = args[0]; arm = raw['pairs'][0]['arms'][0]
            if change == 'reads': arm['sample']['read_bytes'] -= 1
            elif change == 'dispatch': arm['sample']['kernel_dispatches']['q4_mm'] -= 1
            elif change == 'gpu_counter': arm['sample']['gpu_command_ns'] += 1
            elif change == 'release': raw['final_metal']['live_buffer_bytes'] += 1
            elif change == 'boundary': raw['memory_phases'][7]['metal']['allocation_count'] += 1
            else: arm['disk_after'] = copy.deepcopy(arm['disk_before'])
            with self.subTest(change=change), self.assertRaises(ValueError):
                analyze(raw, 'off', *args[1:])

    def test_decision_requires_off_on_on_off_with_identical_work_and_outputs(self):
        results = [measured(mode) for mode in ORDER]
        expected = decision(results)
        self.assertEqual(expected['status'], 'memory_captured')
        self.assertTrue(expected['timing_observation_admissible'])
        self.assertFalse(expected['validation_phase_association'])
        for key in FLAGS: self.assertIs(expected[key], False)
        for change in ('missing', 'order', 'hard_guard', 'native_work', 'output', 'shared'):
            changed = copy.deepcopy(results)
            if change == 'missing': changed.pop()
            elif change == 'order': changed[0], changed[1] = changed[1], changed[0]
            elif change == 'hard_guard': changed[2]['hard_guards_passed'] = False
            elif change == 'native_work': changed[2]['native_work_signature'][10]['scratch_reuses'] += 1
            elif change == 'output': changed[2]['output_sha256'] = '0'*64
            else: changed[2]['shared_reference']['expected_native_sha256'] = '0'*64
            with self.subTest(change=change), self.assertRaises(ValueError): decision(changed)

    def test_consistent_validation_dirty_phase_is_an_association_not_qualification(self):
        results = []
        for mode in ORDER:
            args = fixture(mode)
            if mode == 'on': dirty_from(args[0], 20, current=False)
            results.append(analyze(args[0], mode, *args[1:]))
        verdict = decision(results)
        self.assertEqual(verdict['status'], 'validation_memory_association')
        self.assertTrue(verdict['validation_phase_association'])
        self.assertEqual(verdict['first_dirty_validation_phase']['index'], 20)
        self.assertFalse(verdict['compressed_pages_identified'])
        self.assertFalse(verdict['statistical_causal_estimate'])
        for key in FLAGS: self.assertIs(verdict[key], False)
        results[2]['first_dirty_phase']['index'] = 21
        self.assertFalse(decision(results)['validation_phase_association'])
        results[0]['clean_memory'] = False
        self.assertFalse(decision(results)['timing_observation_admissible'])


if __name__ == '__main__': unittest.main()
