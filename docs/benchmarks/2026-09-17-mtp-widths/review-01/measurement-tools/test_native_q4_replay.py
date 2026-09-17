import copy
import unittest

from native_q4_replay import BUDGET, COUNTERS, FIXED_KERNELS, PACKED, POSITIONS, analyze, validate_replay


def fixture(condition='scatter'):
    entry = lambda name, size: dict(file=name, bytes=size, sha256='a'*64)
    manifest = dict(complete=True, origin='existing-Q4-experts', artifact_revision='b'*40,
        layers={str(layer): dict(offsets=list(range(72, 80)), inputs=entry(f'input-{layer}.bin', 8*2560*4),
            experts=[dict(expert=e, record=entry(f'expert-{layer}-{e}.bin', 2764800)) for e in (0, 1)])
            for layer in (0, 16, 32, 47)})
    resident = 1024**3 if condition in ('resident', 'coordinator') else 0
    state = dict(build_fingerprint='c'*64, device='Apple M1 Pro', physical_bytes=32*1024**3,
        live_command_groups=0, peak_command_groups=2, active_scratch_slot=-1,
        live_buffer_bytes=resident+32*1024**2, peak_buffer_bytes=resident+32*1024**2,
        residency=dict(mode='core-cache', pending_retirements=0,
            bytes_by_class={'resident': resident}, registered_bytes=resident, set_overhead_bytes=0),
        kernels=dict(FIXED_KERNELS, q4_decode='reference'), kernel_dispatches={}, **dict.fromkeys(COUNTERS, 0))
    memory = dict(compressed_bytes=0, decompressions=0, system_swap_used_bytes=0,
                  physical_footprint_bytes=resident+64*1024**2, physical_footprint_peak_bytes=resident+64*1024**2)
    rows = []; stamp = 1; initial = copy.deepcopy(state)
    for pair in range(5):
        for group in (1, 4):
            arms = []
            for variant in (['packed-r2', 'reference'] if pair % 2 else ['reference', 'packed-r2']):
                state['kernels']['q4_decode'] = variant
                before = copy.deepcopy(state)
                packed = variant == 'packed-r2'
                names = PACKED if packed else ('q4_gate_up', 'q4_mm')
                counts = dict.fromkeys(names, 256)
                if condition != 'direct': counts['scatter_experts'] = 256
                sample = dict(expert_executions=256, wall_ns=140000 if packed else 200000,
                    cpu_encode_ns=20000 if packed else 30000, cpu_gpu_wait_ns=20000 if packed else 40000,
                    gpu_command_ns=50000 if packed else 100000, submissions=256//group,
                    allocation_count=0, scratch_reuses=256 if condition == 'direct' else 768,
                    kernel_dispatches=counts)
                if condition == 'coordinator':
                    sample.update(coordinator_wait_ns=25000 if packed else 50000,
                                  ready_hits=256, new_misses=0, loading_joins=0, read_calls=0, read_bytes=0)
                for key in COUNTERS: state[key] += sample[key]
                for key, value in counts.items(): state['kernel_dispatches'][key] = state['kernel_dispatches'].get(key, 0)+value
                arms.append(dict(variant=variant, sample=sample, metal_before=before, metal_after=copy.deepcopy(state),
                    memory_before=dict(memory), memory_after=dict(memory),
                    host_before=dict(thermal_state=0, low_power_mode=False, power_source='AC Power', monotonic_ns=stamp),
                    host_after=dict(thermal_state=0, low_power_mode=False, power_source='AC Power', monotonic_ns=stamp+1)))
                stamp += 2
            rows.append(dict(pair=pair, group=group, arms=arms))
    raw = dict(kind='native_q4_replay_v1', complete=True, validation=False, exact=True, condition=condition,
        cycles=4, cases_per_cycle=64, max_live_groups=2, scratch_scope='eight-expert-batch',
        budget_bytes=BUDGET, resident_bytes=resident, expected_resident_bytes=resident,
        positions=list(POSITIONS), output_sha256='f'*64, timed_final_outputs_checked=True,
        untouched_destinations_checked=True,
        fixture_manifest=manifest, initial_metal=initial, final_metal=copy.deepcopy(state), pairs=rows)
    if condition == 'coordinator': raw.update(execution_schedule='coordinator-all-hit', cache_slots=8, io_workers=8)
    return raw


def change_counter(arm, key, value):
    arm['sample'][key] = value
    arm['metal_after'][key] = arm['metal_before'][key]+value


class NativeQ4ReplayEvidenceTest(unittest.TestCase):
    def test_each_condition_is_separate_and_never_promotes(self):
        for condition in ('direct', 'scatter', 'resident', 'coordinator'):
            with self.subTest(condition=condition):
                result = analyze(fixture(condition))
                self.assertEqual(result['status'], 'diagnostic_gain')
                self.assertEqual(result['condition'], condition)
                self.assertEqual([row['group'] for row in result['results']], [1, 4])
                self.assertFalse(result['normal_request_latency_qualified'])
                self.assertFalse(result['production_promoted'])
                self.assertFalse(result['next_condition_automatically_admitted'])

    def test_gain_requires_gpu_and_wall_at_both_group_sizes(self):
        for key, value in [('gpu_command_ns', 100000), ('wall_ns', 220000)]:
            raw = fixture()
            for row in raw['pairs']:
                if row['group'] == 4:
                    arm = next(a for a in row['arms'] if a['variant'] == 'packed-r2')
                    if key == 'wall_ns': arm['sample'][key] = value
                    else: change_counter(arm, key, value)
            self.assertEqual(analyze(raw)['status'], 'no_clear_native_gain')

    def test_missing_reordered_and_duplicate_pairs_cannot_pass(self):
        mutations = [lambda r: r['pairs'].pop(), lambda r: r['pairs'][0]['arms'].reverse(),
                     lambda r: r['pairs'][1].update(group=1), lambda r: r.update(exact=False),
                     lambda r: r.update(validation=True), lambda r: r.update(cycles=3),
                     lambda r: r.update(scratch_scope='command-group'), lambda r: r.update(budget_bytes=22*1024**3)]
        for mutate in mutations:
            raw = fixture(); mutate(raw)
            with self.assertRaises(ValueError): analyze(raw)

    def test_dispatch_counters_and_submissions_are_reconstructed(self):
        for mutation in ('sample', 'missing_scatter', 'extra_dispatch', 'group', 'fallback', 'counter_reset'):
            raw = fixture(); arm = raw['pairs'][0]['arms'][1]
            if mutation == 'sample': arm['sample']['gpu_command_ns'] += 1
            elif mutation == 'missing_scatter':
                arm['sample']['kernel_dispatches'].pop('scatter_experts')
                arm['metal_after']['kernel_dispatches']['scatter_experts'] -= 256
            elif mutation == 'extra_dispatch':
                arm['sample']['kernel_dispatches']['extra'] = arm['metal_after']['kernel_dispatches']['extra'] = 1
            elif mutation == 'group': change_counter(arm, 'submissions', 64)
            elif mutation == 'fallback': arm['metal_after']['kernels']['q4_decode'] = 'reference'
            else: arm['metal_after']['gpu_command_ns'] = arm['metal_before']['gpu_command_ns']-1
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError): analyze(raw)

    def test_direct_and_scatter_paths_cannot_be_relabeled(self):
        raw = fixture('direct'); raw['condition'] = 'scatter'
        with self.assertRaises(ValueError): analyze(raw)
        raw = fixture('scatter'); raw['condition'] = 'direct'
        with self.assertRaises(ValueError): analyze(raw)

    def test_resident_allocation_must_be_real_and_constant(self):
        for mutation in ('zero', 'expected', 'class', 'peak', 'pending', 'groups', 'active_scratch'):
            raw = fixture('resident'); state = raw['pairs'][0]['arms'][0]['metal_after']
            if mutation == 'zero': raw['resident_bytes'] = raw['expected_resident_bytes'] = 0
            elif mutation == 'expected': raw['expected_resident_bytes'] += 1
            elif mutation == 'class': state['residency']['bytes_by_class']['resident'] += 1
            elif mutation == 'peak': state['peak_buffer_bytes'] = BUDGET+1
            elif mutation == 'pending': state['residency']['pending_retirements'] = 1
            elif mutation == 'groups': state['live_command_groups'] = 1
            else: state['active_scratch_slot'] = 0
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError): analyze(raw)

    def test_missing_memory_host_and_between_arm_changes_remain_disturbed(self):
        for mutation in ('missing_memory', 'missing_host', 'compressed', 'swap', 'decompress', 'thermal', 'power', 'footprint'):
            raw = fixture(); arm = raw['pairs'][0]['arms'][1]
            if mutation == 'missing_memory': arm['memory_before']['physical_footprint_bytes'] = None
            elif mutation == 'missing_host': arm['host_before']['thermal_state'] = None
            elif mutation == 'compressed': arm['memory_after']['compressed_bytes'] = 1
            elif mutation in ('swap', 'decompress'):
                key = 'system_swap_used_bytes' if mutation == 'swap' else 'decompressions'
                # Both boundaries of this arm agree; the between-arm change must still be noticed.
                arm['memory_before'][key] = arm['memory_after'][key] = 1
            elif mutation == 'thermal': arm['host_after']['thermal_state'] = 1
            elif mutation == 'power': arm['host_after']['power_source'] = 'Battery Power'
            else: arm['memory_after']['physical_footprint_peak_bytes'] = BUDGET+1
            with self.subTest(mutation=mutation):
                self.assertEqual(analyze(raw)['status'], 'disturbed')

    def test_zero_wait_is_not_imputed_or_used_to_reject_gpu_gain(self):
        raw = fixture()
        for row in raw['pairs']:
            for arm in row['arms']: change_counter(arm, 'cpu_gpu_wait_ns', 0)
        result = analyze(raw)
        self.assertEqual(result['status'], 'diagnostic_gain')
        for row in result['results']:
            wait = row['metrics']['cpu_gpu_wait_ns']
            self.assertIsNone(wait['confidence_95'])
            self.assertEqual(wait['ratios'], [None]*5)
            self.assertEqual(wait['reference_ns'], [0]*5)

    def test_nonfinite_missing_or_boolean_timing_is_invalid(self):
        for value in (float('nan'), float('inf'), None, True, -1, 0):
            raw = fixture(); raw['pairs'][0]['arms'][0]['sample']['wall_ns'] = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError): analyze(raw)

    def test_fixture_identity_and_build_are_bound(self):
        raw = fixture()
        frozen = dict(build='c'*64, artifact_revision='b'*40, budget_bytes=BUDGET)
        result = analyze(raw, frozen)
        self.assertEqual(validate_replay(raw, dict(fixture=raw['fixture_manifest']), frozen, result['records']), result)
        changed = copy.deepcopy(raw['fixture_manifest']); changed['layers']['0']['inputs']['sha256'] = 'd'*64
        with self.assertRaises(ValueError): validate_replay(raw, dict(fixture=changed), frozen, result['records'])
        with self.assertRaises(ValueError): analyze(raw, dict(frozen, build='e'*64))
        raw['fixture_manifest']['layers']['0']['experts'][1]['expert'] = 0
        with self.assertRaises(ValueError): analyze(raw)

    def test_timed_outputs_and_untouched_destinations_must_be_checked(self):
        mutations = [lambda r: r.update(timed_final_outputs_checked=False),
                     lambda r: r.pop('untouched_destinations_checked'),
                     lambda r: r.update(output_sha256='not-a-hash'),
                     lambda r: r.update(output_sha256='a'*63),
                     lambda r: r['positions'].reverse(),
                     lambda r: r['positions'].__setitem__(5, False)]
        for mutate in mutations:
            raw = fixture(); mutate(raw)
            with self.assertRaises(ValueError): analyze(raw)

    def test_only_q4_selector_may_vary_within_a_replay(self):
        for key, value in [('policy', 'reference'), ('q8_decode_rows', 0), ('route_selection', 'serial'),
                           ('gate_pair', True), ('token_tile', 2), ('profile', True),
                           ('counter_profile', True), ('operator_capture', '/tmp/capture'),
                           ('shape_table', {}), ('affine_rows', 2)]:
            raw = fixture(); raw['pairs'][0]['arms'][0]['metal_after']['kernels'][key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError): analyze(raw)

    def test_coordinator_requires_all_hits_without_reads_or_allocations(self):
        for key, value in [('ready_hits', 255), ('new_misses', 1), ('loading_joins', 1),
                           ('read_calls', 1), ('read_bytes', 2764800), ('coordinator_wait_ns', None),
                           ('allocation_count', 1)]:
            raw = fixture('coordinator'); arm = raw['pairs'][0]['arms'][0]
            if key == 'allocation_count': change_counter(arm, key, value)
            else: arm['sample'][key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError): analyze(raw)
        raw = fixture('coordinator'); result = analyze(raw)
        self.assertTrue(result['all_hit_expert_work'])
        self.assertEqual(result['timed_read_bytes'], 0)
        self.assertEqual(result['execution_schedule'], 'coordinator-all-hit')
        for row in result['results']:
            self.assertEqual(row['metrics']['coordinator_wait_ns']['ratios'], [.5]*5)

    def test_coordinator_geometry_and_zero_wait_are_explicit(self):
        for key, value in [('execution_schedule', 'fixed-groups'), ('cache_slots', 16), ('io_workers', 4),
                           ('resident_bytes', 0)]:
            raw = fixture('coordinator'); raw[key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError): analyze(raw)
        raw = fixture('coordinator')
        for row in raw['pairs']:
            for arm in row['arms']: arm['sample']['coordinator_wait_ns'] = 0
        result = analyze(raw)
        self.assertEqual(result['status'], 'diagnostic_gain')
        for row in result['results']:
            self.assertIsNone(row['metrics']['coordinator_wait_ns']['confidence_95'])

    def test_original_conditions_do_not_gain_coordinator_fields_or_metrics(self):
        for condition in ('direct', 'scatter', 'resident'):
            raw = fixture(condition); result = analyze(raw)
            self.assertNotIn('execution_schedule', result)
            self.assertNotIn('all_hit_expert_work', result)
            for row in result['results']:
                self.assertNotIn('coordinator_wait_ns', row['metrics'])


if __name__ == '__main__':
    unittest.main()
