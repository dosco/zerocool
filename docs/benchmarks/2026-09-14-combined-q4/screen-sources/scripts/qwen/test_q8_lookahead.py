import copy
import unittest

from screen_q8_lookahead import SHAPES, analyze


class LookaheadEvidenceTest(unittest.TestCase):
    def fixture(self):
        memory = dict(compressed_bytes=0, decompressions=0, system_swap_used_bytes=0)
        return dict(complete=True, validation=False, device='Apple M1 Pro', dispatch_repeats=32,
            max_shared_buffer_bytes=120*1024**2,
            cases=[dict(K=k, N=n, exact=True, width=8) for k, n in SHAPES],
            pairs=[dict(pair=p, case=c, arms=[dict(candidate=bool(v), sample=dict(
                wall_us=1000 if not v else 500, gpu_us=800 if not v else 400),
                memory_before=dict(memory), memory_after=dict(memory)) for v in (p%2, 1-p%2)])
                for p in range(5) for c in range(7)])

    def test_isolated_screen_never_qualifies_request_speed(self):
        result = analyze(self.fixture())
        self.assertTrue(result['advance_to_request_screen'])
        self.assertFalse(result['normal_request_latency_qualified'])
        self.assertFalse(result['production_promoted'])

    def test_pressure_and_missing_evidence_cannot_pass(self):
        raw = self.fixture()
        raw['pairs'][0]['arms'][0]['memory_after']['decompressions'] = 1
        self.assertEqual(analyze(raw)['status'], 'memory_disturbed')
        raw = self.fixture()
        for mutate in (lambda r: r['pairs'].pop(),
                       lambda r: r.update(validation=True),
                       lambda r: r['cases'][0].update(exact=False),
                       lambda r: r['pairs'][0]['arms'][0]['sample'].update(gpu_us=float('nan'))):
            broken = copy.deepcopy(raw); mutate(broken)
            with self.assertRaises(ValueError): analyze(broken)


if __name__ == '__main__':
    unittest.main()
