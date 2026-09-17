import copy
import unittest

from q4_read_arrivals import (EXPERT_BYTES, INVALIDATE_FILES, OUTPUT_SHA, RETAIN_CACHE,
                              analyze, disk_observation)
from native_q4_replay import COUNTERS, PACKED, POSITIONS
from test_native_q4_replay import fixture as native_fixture


def trace_sample(sample, raw, group, arm_index):
    hits = raw['hits_per_batch']; fixtures = [(int(layer), e['expert'])
        for layer, item in raw['fixture_manifest']['layers'].items() for e in item['experts']]
    batches = []; submission_count = 0; gpu = 0; duration = 0
    for batch in range(8):
        base = 1000000+arm_index*10000000+batch*500000
        hit_indices = [(batch+j) % 8 for j in range(hits)]
        sizes = [1]*8 if group == 1 else [4, 4] if hits == 8 else [2, 2, 4]
        events = []; at = 0
        for command, size in enumerate(sizes):
            submitted = base+5000+command*5000
            for i in range(at, at+size):
                layer, expert = fixtures[i]
                queued, started, completed = ((base+1, base+2, base+3) if i in hit_indices else
                                               (base+50, base+60+i, base+80+i))
                events.append(dict(layer=layer, expert=expert, position=POSITIONS[i],
                    acquisition='ready_hit' if i in hit_indices else 'new_miss',
                    admitted_ns=base+100+i*10, read_queued_ns=queued, read_started_ns=started,
                    read_completed_ns=completed, encoded_ns=base+1000+i*10, submitted_ns=submitted,
                    gpu_start_ns=submitted+1000, gpu_end_ns=submitted+2000, released_ns=submitted+2500+i))
            at += size
        detail = dict(records=events, ready_hits=hits, new_misses=8-hits, loading_joins=0,
            peak_leases=8, peak_gpu_groups=2, duration_ns=max(r['released_ns'] for r in events)-base+1000,
            read_queue_sum_ns=sum(r['read_started_ns']-r['read_queued_ns'] for r in events if r['acquisition']=='new_miss'),
            read_service_sum_ns=sum(r['read_completed_ns']-r['read_started_ns'] for r in events if r['acquisition']=='new_miss'),
            ready_to_encode_sum_ns=sum(r['encoded_ns']-max(r['admitted_ns'],r['read_completed_ns']) for r in events),
            ready_to_gpu_sum_ns=sum(r['gpu_start_ns']-max(r['admitted_ns'],r['read_completed_ns']) for r in events),
            gpu_execution_sum_ns=len(sizes)*1000, coordinator_wait_ns=500)
        batches.append(dict(cycle=0, batch=batch, hit_indices=hit_indices, timing=detail))
        submission_count += len(sizes); gpu += detail['gpu_execution_sum_ns']; duration += detail['duration_ns']
    sample.update(batches=batches, submissions=submission_count, gpu_command_ns=gpu,
                  coordinator_wait_ns=4000, wall_ns=duration+8000)


def fixture(mode='timing', hits=2, invalidate=False):
    raw = native_fixture('resident')
    raw.update(kind='native_q4_arrivals_v1', mode=mode, hits_per_batch=hits,
        cycles=4 if mode == 'timing' else 1, validation=mode == 'check',
        prepared_manifest_sha256='d'*64, cache_slots=8, io_workers=8, output_sha256=OUTPUT_SHA,
        disk_counter_scope='entire-arm-including-preparation-and-other-processes',
        wall_scope='sum-of-coordinator-windows-excludes-preparation')
    if invalidate: raw['cache_preparation'] = INVALIDATE_FILES
    for n, item in enumerate(raw['fixture_manifest']['layers'].values()):
        for i, e in enumerate(item['experts']): e['expert'] = n*2+i
    if mode != 'timing': raw['pairs'] = raw['pairs'][:2]
    executions = raw['cycles']*64; batches = raw['cycles']*8
    arm_index = 0
    for row in raw['pairs']:
        for arm in row['arms']:
            sample = arm['sample']; before, after = arm['metal_before'], arm['metal_after']
            sample.update(expert_executions=executions, coordinator_wait_ns=5000, preparation_ns=10000,
                preparation_read_bytes=batches*hits*EXPERT_BYTES, preparation_load_calls=batches*hits,
                ready_hits=batches*hits, new_misses=batches*(8-hits), loading_joins=0,
                read_bytes=batches*(8-hits)*EXPERT_BYTES, expert_load_calls=batches*(8-hits),
                scratch_reuses=executions*3, submissions=executions//row['group'], cpu_gpu_wait_ns=0, batches=[])
            if mode == 'trace': trace_sample(sample, raw, row['group'], arm_index)
            sample['arm_elapsed_ns'] = sample['wall_ns']+sample['preparation_ns']+1000
            if invalidate: sample.update(invalidation_ns=5000, invalidation_calls=8*batches)
            for key in COUNTERS: after[key] = before[key]+sample[key]
            names = PACKED if arm['variant'] == 'packed-r2' else ('q4_gate_up', 'q4_mm')
            sample['kernel_dispatches'] = dict.fromkeys((*names, 'scatter_experts'), executions)
            after['kernel_dispatches'] = dict(before['kernel_dispatches'])
            for key, count in sample['kernel_dispatches'].items():
                after['kernel_dispatches'][key] = before['kernel_dispatches'].get(key, 0)+count
            arm['disk_before'] = dict(devices=[dict(registry_id=1, read_bytes=0, write_bytes=0)])
            arm['disk_after'] = dict(devices=[dict(registry_id=1,
                read_bytes=sample['read_bytes']+sample['preparation_read_bytes'], write_bytes=0)])
            arm_index += 1
    return raw


class Q4ArrivalEvidenceTest(unittest.TestCase):
    def test_explicit_invalidation_keeps_native_gate_separate(self):
        for hits in (2, 8):
            for mode in ('check', 'timing', 'trace'):
                result = analyze(fixture(mode, hits, invalidate=True))
                control = analyze(fixture(mode, hits))
                self.assertEqual(result['status'], control['status'])
                self.assertEqual(result['results'], control['results'])
                self.assertEqual(result['device_reads_exercised'], True if mode == 'timing' else None)
                self.assertEqual(result['cache_preparation'], INVALIDATE_FILES)
                self.assertFalse(result['production_promoted'])
                self.assertTrue(all(s['within_declared_range'] for s in result['storage_observations']))

    def test_invalidation_requires_every_selected_range_and_valid_time(self):
        for mode in ('check', 'timing', 'trace'):
            for key, value in (('invalidation_ns', None), ('invalidation_ns', 0),
                               ('invalidation_ns', 10001), ('invalidation_calls', None),
                               ('invalidation_calls', 0), ('invalidation_calls', 8)):
                raw = fixture(mode, invalidate=True)
                raw['pairs'][0]['arms'][0]['sample'][key] = value
                with self.subTest(mode=mode, key=key, value=value):
                    with self.assertRaises(ValueError): analyze(raw)
        raw = fixture(invalidate=True); raw.pop('cache_preparation')
        with self.assertRaises(ValueError): analyze(raw)
        raw = fixture(); raw['cache_preparation'] = 'purge-system-cache'
        with self.assertRaises(ValueError): analyze(raw)

    def test_storage_gate_is_per_arm_inclusive_and_missing_is_not_zero(self):
        for ratio in (0.9, 1.1):
            raw = fixture(invalidate=True)
            for pair in raw['pairs']:
                for arm in pair['arms']:
                    amount = arm['sample']['read_bytes']+arm['sample']['preparation_read_bytes']
                    arm['disk_after']['devices'][0]['read_bytes'] = int(amount*ratio)
            self.assertTrue(analyze(raw)['device_reads_exercised'])
        for pair_index in range(10):
            for arm_index in range(2):
                raw = fixture(invalidate=True)
                # Plenty of reads in other arms cannot compensate for this unexercised arm.
                raw['pairs'][pair_index]['arms'][arm_index]['disk_after']['devices'][0]['read_bytes'] = 0
                result = analyze(raw)
                self.assertEqual(result['status'], 'unqualified_storage')
                self.assertEqual(result['native_performance_status'], 'diagnostic_gain')
                self.assertFalse(result['device_reads_exercised'])
        for ratio in (0.899, 1.101, None):
            raw = fixture(invalidate=True); arm = raw['pairs'][0]['arms'][0]
            if ratio is None: arm['disk_after'] = None
            else:
                amount = arm['sample']['read_bytes']+arm['sample']['preparation_read_bytes']
                arm['disk_after']['devices'][0]['read_bytes'] = int(amount*ratio)
            result = analyze(raw)
            self.assertEqual(result['status'], 'unqualified_storage')
            self.assertFalse(result['device_reads_exercised'])
            if ratio is None:
                self.assertIsNone(result['storage_observations'][0]['observed_device_read_bytes'])
                self.assertIsNone(result['storage_observations'][0]['device_to_application_ratio'])
            raw['pairs'][0]['arms'][0]['host_after']['thermal_state'] = 2
            self.assertEqual(analyze(raw)['status'], 'disturbed')

    def test_retained_cache_control_keeps_original_schema_and_status(self):
        raw = fixture(); original = analyze(raw)
        raw['cache_preparation'] = RETAIN_CACHE
        self.assertEqual(analyze(raw), original)
        raw['pairs'][0]['arms'][0]['disk_after']['devices'][0]['read_bytes'] = 0
        result = analyze(raw)
        self.assertEqual(result['status'], 'diagnostic_gain')
        for key in ('cache_preparation', 'native_performance_status', 'storage_criteria',
                    'storage_observations', 'device_reads_exercised'):
            self.assertNotIn(key, result)

    def test_each_population_has_separate_check_timing_and_trace(self):
        for hits in (8, 2):
            for mode in ('check', 'timing', 'trace'):
                result = analyze(fixture(mode, hits))
                self.assertEqual(result['status'], {'check':'checked','timing':'diagnostic_gain','trace':'captured'}[mode])
                self.assertFalse(result['normal_request_latency_qualified'])
                self.assertFalse(result['production_promoted'])
                self.assertEqual(result['coverage']['detailed_experts'], 256 if mode == 'trace' else 0)

    def test_read_counts_and_preparation_are_not_interchangeable(self):
        for key in ('read_bytes', 'expert_load_calls', 'preparation_read_bytes', 'preparation_load_calls',
                    'ready_hits', 'new_misses', 'loading_joins'):
            raw = fixture(); raw['pairs'][0]['arms'][0]['sample'][key] += 1
            with self.subTest(key=key):
                with self.assertRaises(ValueError): analyze(raw)
        raw = fixture(); arm = raw['pairs'][0]['arms'][0]
        arm['sample']['arm_elapsed_ns'] = arm['sample']['wall_ns']
        with self.assertRaises(ValueError): analyze(raw)

    def test_modes_cannot_share_instrumentation_or_pair_coverage(self):
        for change in ('validation', 'pairs', 'order', 'capture', 'scope'):
            raw = fixture()
            if change == 'validation': raw['validation'] = True
            elif change == 'pairs': raw['pairs'].pop()
            elif change == 'order': raw['pairs'][0]['arms'].reverse()
            elif change == 'capture': raw['pairs'][0]['arms'][0]['sample']['batches'] = [dict(cycle=0,batch=0)]
            else: raw['wall_scope'] = 'includes-preparation'
            with self.assertRaises(ValueError): analyze(raw)

    def test_dynamic_mixed_submissions_are_allowed_within_cap(self):
        raw = fixture()
        arm = raw['pairs'][1]['arms'][0]
        arm['sample']['submissions'] = 100
        arm['metal_after']['submissions'] = arm['metal_before']['submissions']+100
        self.assertEqual(analyze(raw)['status'], 'diagnostic_gain')
        for value in (63, 257):
            arm['sample']['submissions'] = value
            arm['metal_after']['submissions'] = arm['metal_before']['submissions']+value
            with self.assertRaises(ValueError): analyze(raw)
        raw = fixture(hits=8); arm = raw['pairs'][1]['arms'][0]
        arm['sample']['submissions'] = 100
        arm['metal_after']['submissions'] = arm['metal_before']['submissions']+100
        with self.assertRaises(ValueError): analyze(raw)

    def test_trace_requires_exact_keys_positions_hits_and_batch_order(self):
        for change in ('missing', 'duplicate', 'position', 'hit', 'batch', 'cycle'):
            raw = fixture('trace'); batches = raw['pairs'][0]['arms'][0]['sample']['batches']
            records = batches[0]['timing']['records']
            if change == 'missing': records.pop()
            elif change == 'duplicate': records[1] = copy.deepcopy(records[0])
            elif change == 'position': records[0]['position'] = 2
            elif change == 'hit': batches[0]['hit_indices'] = [1,2]
            elif change == 'batch': batches.reverse()
            else: batches[0]['cycle'] = 1
            with self.subTest(change=change):
                with self.assertRaises(ValueError): analyze(raw)

    def test_trace_timestamp_reversal_or_command_conflict_fails(self):
        for change in ('read', 'gpu', 'release', 'missing', 'command', 'cap'):
            raw = fixture('trace'); detail = raw['pairs'][1]['arms'][0]['sample']['batches'][0]['timing']
            r = detail['records'][0]
            if change == 'read': r['read_started_ns'] = r['read_completed_ns']+1
            elif change == 'gpu': r['gpu_end_ns'] = r['gpu_start_ns']-1
            elif change == 'release': r['released_ns'] = r['gpu_end_ns']-1
            elif change == 'missing': r['encoded_ns'] = None
            elif change == 'command': detail['records'][1]['gpu_end_ns'] += 1
            else:
                # Rename a second command as the first; five records would exceed cap four.
                for record in detail['records'][2:6]: record['submitted_ns'] = r['submitted_ns']
            with self.subTest(change=change):
                with self.assertRaises(ValueError): analyze(raw)

    def test_trace_reconstructs_native_sums_and_occupancy(self):
        result = analyze(fixture('trace', 2))
        self.assertEqual(result['captures'][0]['occupancy_histogram'], {'1':64})
        self.assertEqual(result['captures'][2]['occupancy_histogram'], {'2':16,'4':8})
        self.assertEqual(len(result['captures'][0]['readiness_by_acquisition']['ready_hit']['ready_to_submission_ns']), 16)
        self.assertEqual(len(result['captures'][0]['readiness_by_acquisition']['new_miss']['ready_to_submission_ns']), 48)
        for key in ('gpu_execution_sum_ns', 'read_queue_sum_ns', 'read_service_sum_ns',
                    'ready_to_encode_sum_ns', 'ready_to_gpu_sum_ns', 'coordinator_wait_ns'):
            raw = fixture('trace'); raw['pairs'][0]['arms'][0]['sample']['batches'][0]['timing'][key] += 10
            with self.subTest(key=key):
                with self.assertRaises(ValueError): analyze(raw)

    def test_overlapping_command_intervals_are_reported_without_exclusive_cost_claim(self):
        raw = fixture('trace'); sample = raw['pairs'][0]['arms'][0]['sample']
        detail = sample['batches'][0]['timing']; first, second = detail['records'][:2]
        increase = second['gpu_start_ns']-first['gpu_end_ns']+500
        first['gpu_end_ns'] += increase; first['released_ns'] = first['gpu_end_ns']+500
        detail['gpu_execution_sum_ns'] += increase
        sample['gpu_command_ns'] += increase
        raw['pairs'][0]['arms'][0]['metal_after']['gpu_command_ns'] += increase
        result = analyze(raw)
        self.assertEqual(result['captures'][0]['gpu_interval_overlap_pairs'], 1)

    def test_missing_device_counters_stay_missing_zero_is_real(self):
        arm = fixture(hits=8)['pairs'][0]['arms'][0]
        self.assertGreater(disk_observation(arm)['read_bytes'], 0)  # Preparation is inside device observations.
        arm['disk_after'] = copy.deepcopy(arm['disk_before'])
        self.assertEqual(disk_observation(arm)['read_bytes'], 0)
        arm['disk_after'] = None
        self.assertIsNone(disk_observation(arm)['read_bytes'])
        self.assertFalse(disk_observation(arm)['available'])
        raw = fixture(); raw['pairs'][0]['arms'][0]['disk_before'] = None
        self.assertFalse(analyze(raw)['device_observations_complete'])

    def test_memory_unknown_is_disturbed_and_zero_wait_ratios_stay_null(self):
        raw = fixture(); result = analyze(raw)
        self.assertIsNone(result['results'][0]['metrics']['cpu_gpu_wait_ns']['confidence_95'])
        raw['pairs'][0]['arms'][0]['memory_after']['decompressions'] = None
        self.assertEqual(analyze(raw)['status'], 'disturbed')
        raw = fixture(); raw['pairs'][0]['arms'][0]['host_after']['thermal_state'] = 2
        self.assertEqual(analyze(raw)['status'], 'disturbed')


if __name__ == '__main__':
    unittest.main()
