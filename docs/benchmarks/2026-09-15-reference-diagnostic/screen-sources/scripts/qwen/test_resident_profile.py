import unittest

from profile_resident_decode import dependency_details


class ResidentProfileTest(unittest.TestCase):
    def fixture(self):
        row = dict(name='decode', prompt_tokens=10, decode_diagnostics=dict(samples=[{}, {}]))
        deps = [dict(offset=t, layer=l, tokens=1, routes=list(range(10)), records=[dict(
            expert=e, acquisition='ready_hit') for e in range(10)])
            for t in (10, 11) for l in range(48)]
        return row, deps

    def test_repeated_selection_is_not_a_useful_miss_prediction(self):
        row, deps = self.fixture()
        r = dependency_details(row, deps, [])
        self.assertIsNone(r['queue_ms'])
        self.assertEqual(r['previous_top_two']['proposed'], 94)
        self.assertEqual(r['previous_top_two']['selected_again'], 94)
        self.assertEqual(r['previous_top_two']['observed_new_misses'], 0)
        self.assertIsNone(r['previous_top_two']['predicted_latency_saving_ms'])
        deps[49]['records'][0].update(acquisition='new_miss', read_queued_ns=1000,
            read_started_ns=2000, read_completed_ns=1002000)
        r = dependency_details(row, deps, [])
        self.assertEqual(r['previous_top_two']['observed_new_misses'], 1)
        self.assertEqual(r['service_ms']['median'], 1)

    def test_missing_layers_and_reversed_reads_cannot_produce_a_recommendation(self):
        row, deps = self.fixture()
        with self.assertRaises(ValueError): dependency_details(row, deps[:-1], [])
        with self.assertRaises(ValueError): dependency_details(row, deps+[deps[0]], [])
        deps[49]['records'][0].update(acquisition='new_miss', read_queued_ns=1000,
            read_started_ns=0, read_completed_ns=100)
        with self.assertRaises(ValueError): dependency_details(row, deps, [])


if __name__ == '__main__': unittest.main()
