import copy
import unittest
from benchmark_panels import validate


class PanelComparisonTest(unittest.TestCase):
    def fixture(self):
        state = dict(diagnostic_stream_trunk=False, artifact_revision='revision',
                     memory_plan=dict(limit_bytes=8*1024**3, planned_bytes=7*1024**3, panel_tokens=256, expert_slots=500),
                     metal=dict(device='Apple M1 Pro', physical_bytes=32*1024**3, build_fingerprint='native', peak_buffer_bytes=6*1024**3),
                     prepared=dict(manifest_sha256='artifact', application_read_bytes=100), checkpoint_application_read_bytes=200,
                     process=dict(physical_footprint_bytes=6*1024**3))
        return dict(complete=True, model_revision='revision', workloads=[dict(name='prompt_2k', tokens=[1, 2])], runs=[dict(name='prompt_2k', before=state, after=copy.deepcopy(state),
            output_tokens=256, output_token_ids=list(range(256)), prompt_tokens=2048, reused_tokens=0,
            time_to_first_token_ms=1000, tokens_per_second=10)])

    def test_equal_outputs_and_budget(self):
        r = self.fixture()
        self.assertEqual(len(validate(r, dict(build='native', revision='revision'), 256, 8*1024**3)), 1)

    def test_changed_budget_output_build_and_diagnostics_fail(self):
        original = self.fixture()
        expected = dict(build='native', revision='revision')
        validate(original, expected, 256, 8*1024**3)
        for mutate in (
            lambda r: r.update(model_revision='another-artifact'),
            lambda r: r['runs'][0]['after'].update(artifact_revision='another-artifact'),
            lambda r: r['runs'][0]['after']['memory_plan'].update(limit_bytes=7*1024**3),
            lambda r: r['runs'][0]['after']['memory_plan'].update(panel_tokens=0),
            lambda r: r['runs'][0]['after']['memory_plan'].update(expert_slots=400),
            lambda r: r['workloads'][0].update(tokens=[3, 4]),
            lambda r: r['runs'][0]['after']['metal'].update(build_fingerprint='changed'),
            lambda r: r['runs'][0].update(output_token_ids=[0]*256),
            lambda r: r['runs'][0].update(output_tokens=10),
            lambda r: r['runs'][0]['after'].update(diagnostic_stream_trunk=True),
        ):
            r = copy.deepcopy(original);mutate(r)
            with self.assertRaises(ValueError):
                validate(r, expected, 256, 8*1024**3)


if __name__ == '__main__':
    unittest.main()
