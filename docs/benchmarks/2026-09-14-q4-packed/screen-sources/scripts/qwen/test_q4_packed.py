import copy
import unittest

from screen_q4_packed import analyze


class PackedQ4EvidenceTest(unittest.TestCase):
    def fixture(self):
        memory = dict(compressed_bytes=0, decompressions=0, system_swap_used_bytes=0)
        return dict(complete=True, validation=False, exact=True, device='Apple M1 Pro', cycles=4,
            expert_input_cases=64, output_modes=['BF16', 'FP32'], max_shared_buffer_bytes=30*1024**2,
            pairs=[dict(pair=p, group=g, arms=[dict(variant=v,
                sample=dict(gpu_us_per_expert=100 if v == 0 else 40, wall_us_per_expert=200 if v == 0 else 140),
                memory_before=dict(memory), memory_after=dict(memory)) for v in ([2, 0, 1] if p%2 else [0, 2, 1])])
                for p in range(5) for g in (1, 4)])

    def test_primary_candidate_needs_both_group_sizes(self):
        raw = self.fixture()
        self.assertTrue(analyze(raw)['advance_to_request_screen'])
        for p in raw['pairs']:
            if p['group'] == 4:
                next(a for a in p['arms'] if a['variant'] == 2)['sample']['gpu_us_per_expert'] = 95
        self.assertFalse(analyze(raw)['advance_to_request_screen'])
        self.assertFalse(analyze(raw)['normal_request_latency_qualified'])

    def test_missing_pairs_changed_order_and_memory_fail_closed(self):
        raw = self.fixture()
        for mutate in (lambda r: r['pairs'].pop(), lambda r: r.update(exact=False),
                       lambda r: r['pairs'][0]['arms'].reverse()):
            broken = copy.deepcopy(raw); mutate(broken)
            with self.assertRaises(ValueError): analyze(broken)
        raw['pairs'][0]['arms'][0]['memory_after']['decompressions'] = 1
        self.assertEqual(analyze(raw)['status'], 'memory_disturbed')


if __name__ == '__main__':
    unittest.main()
