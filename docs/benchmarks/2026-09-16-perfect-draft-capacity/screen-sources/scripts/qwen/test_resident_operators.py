import copy
import unittest

from profile_resident_operators import split_operations


class ResidentOperatorTest(unittest.TestCase):
    def fixture(self):
        raw = dict(runs=[dict(name='initial', prompt_tokens=72, token_latency_ms=[250],
            phases=dict(decode=dict(before=dict(metal=dict(kernel_dispatches={})),
                after=dict(metal=dict(kernel_dispatches={'route_simd': 48})))) )])
        profile = dict(truncated=False, coverage='all-dispatches',
            timing_kind='instrumented per-dispatch compute passes; submission boundaries preserved',
            command_groups=[dict(operations=[dict(request_phase='decode', offset=72, tokens=1,
                layer=l, stage='router', kernel='route_simd', counter_index=0,
                gpu_pass_ns=1000, gpu_begin_ticks=10, gpu_end_ticks=1010)]) for l in range(48)])
        return raw, profile

    def test_complete_coverage_and_no_latency_prediction(self):
        raw, profile = self.fixture()
        result = split_operations(raw, profile)[0]
        self.assertEqual(result['dispatches'], 48)
        self.assertEqual(result['offsets'], [72])
        self.assertEqual(result['operations'][0]['gpu_pass_ms_per_token'], .048)
        self.assertTrue(result['operations'][0]['before_expert_reads'])

    def test_incomplete_duplicate_or_misattributed_counters_fail(self):
        raw, profile = self.fixture()
        for mutate in (
            lambda p: p.update(truncated=True),
            lambda p: p['command_groups'].pop(),
            lambda p: p['command_groups'][0]['operations'][0].update(layer=1),
            lambda p: p['command_groups'][0]['operations'][0].update(counter_index=2),
            lambda p: p['command_groups'][0]['operations'][0].update(gpu_pass_ns=0),
            lambda p: p['command_groups'][0]['operations'][0].update(request_phase='append'),
        ):
            broken = copy.deepcopy(profile)
            mutate(broken)
            with self.assertRaises(ValueError):
                split_operations(raw, broken)


if __name__ == '__main__':
    unittest.main()
