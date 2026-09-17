import copy
import unittest
from benchmark_dependencies import measurements


class DependencySummaryTest(unittest.TestCase):
    def fixture(self):
        return dict(machine=dict(build_fingerprint='build'), artifact_revision='artifact', runs=[
            dict(recorded_pass=p, repetition=0, expert_dependency_ns=p+100,
                 layers=[dict(layer=l, routes=list(range(10)), output_sha256=f'{p*48+l:064x}',
                              application_read_bytes=100) for l in range(48)]) for p in range(2)])

    def test_all_passes_retained(self):
        result = measurements(self.fixture(), {})
        self.assertEqual([r['recorded_pass'] for r in result], [0, 1])
        self.assertEqual([r['nanoseconds'] for r in result], [100, 101])
        self.assertEqual([r['application_bytes'] for r in result], [4800, 4800])

    def test_changed_outputs_build_and_pass_count_fail(self):
        original = self.fixture()
        expected = {}
        measurements(original, expected)
        for mutate in (
            lambda r: r['runs'][1]['layers'][47].update(output_sha256='f'*64),
            lambda r: r['machine'].update(build_fingerprint='changed'),
            lambda r: r['runs'].pop(),
            lambda r: r['runs'][1]['layers'].pop(),
        ):
            changed = copy.deepcopy(original)
            mutate(changed)
            with self.assertRaises(ValueError):
                measurements(changed, expected)


if __name__ == '__main__':
    unittest.main()
