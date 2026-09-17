import copy
import unittest

from cache_residency import configs, decide, memory_observation, curve_difference, SLOTS
from stage200 import ORDER


class CacheResidencyTest(unittest.TestCase):
    def test_capacity_is_the_only_configuration_difference(self):
        a, b = configs()
        self.assertEqual(a['residency'], 'core-cache')
        self.assertEqual(a['decode_submission'], 'immediate')
        self.assertEqual((a['expert_slots'], b['expert_slots']), SLOTS)
        self.assertEqual({k: v for k, v in a.items() if k not in ('name', 'expert_slots')},
                         {k: v for k, v in b.items() if k not in ('name', 'expert_slots')})

    def memory(self):
        return dict(runs=[dict(phases={phase: {point: dict(process=dict(
            compressed_bytes=0, decompressions=10, system_swap_used_bytes=100))
            for point in ('before', 'after')} for phase in ('ingest', 'decode')})])

    def test_missing_memory_growth_and_counter_change_cannot_pass(self):
        raw = self.memory()
        self.assertTrue(memory_observation(raw)['memory_screen_passed'])
        for key, value in [('compressed_bytes', 1), ('decompressions', 11),
                           ('system_swap_used_bytes', 101), ('system_swap_used_bytes', None)]:
            bad = copy.deepcopy(raw)
            bad['runs'][0]['phases']['decode']['after']['process'][key] = value
            with self.subTest(key=key):
                self.assertFalse(memory_observation(bad)['memory_screen_passed'])
        # A later fall must not hide an earlier observed rise.
        raw['runs'][0]['phases']['ingest']['after']['process']['system_swap_used_bytes'] = 200
        self.assertFalse(memory_observation(raw)['memory_screen_passed'])

    def rows(self, saving=1000):
        return [dict(pair=p, configuration=arm, clean_memory=True, memory_screen_passed=True,
            requests=[dict(request_ms=20000 if arm == 'control' else 20000-saving,
                time_to_first_token_ms=10000,
                decode_wall_ms=10000 if arm == 'control' else 10000-saving)]*2)
            for p, arm in ORDER]

    def test_full_pair_and_stage_gates_cannot_be_bypassed(self):
        rows = self.rows()
        self.assertTrue(decide(rows)['advance_to_confirmation'])
        with self.assertRaises(ValueError): decide(rows[:-1])
        rows[1]['memory_screen_passed'] = False
        self.assertFalse(decide(rows)['advance_to_confirmation'])
        small = decide(self.rows(450))
        self.assertEqual(small['status'], 'directional_gain_below_gate')
        self.assertFalse(small['advance_to_confirmation'])
        self.assertFalse(small['request_gain_qualified'])

    def test_saved_demand_cannot_be_reported_as_latency(self):
        curve = dict(status='simulated', coverage=dict(whole_request_covered=True, cold_starts=1),
            curves=[dict(slots=s, policies=[dict(policy='clock', request_phases=[dict(
                request_id=1, phase='decode', demands=200, misses=m,
                application_miss_bytes=m*2764800)])]) for s, m in zip(SLOTS, (100, 90))])
        result = curve_difference(curve)
        self.assertAlmostEqual(result['phases'][0]['fewer_read_fraction'], .1)
        self.assertIsNone(result['predicted_latency_saving_ms'])
        self.assertIsNone(result['predicted_tokens_per_second'])
        curve['coverage']['whole_request_covered'] = None
        with self.assertRaises(ValueError): curve_difference(curve)


if __name__ == '__main__': unittest.main()
