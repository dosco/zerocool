import copy
import unittest

from measure_buffer_costs import CLASSES,FIELDS,analyze,delta
from probe_buffer_reuse import analyze as analyze_probe


def state(n):
    counts={k:0 for k in FIELDS};counts.update(allocations=n,allocated_bytes=n*16384,allocation_ns=n*1000,
        owner_releases=n,owner_released_bytes=n*16384,outside_release_ns=0)
    return dict(allocation_count=6*n,submissions=n,live_command_groups=0,
        buffer_costs=dict(kind='buffer_costs_v1',classes={c:copy.deepcopy(counts) for c in CLASSES},retired_groups=n,group_retirement_ns=100*n))


class BufferCostsTest(unittest.TestCase):
    def test_empty_or_one_phase_cannot_approve_reuse(self):
        for rows in ([],[{'name':'coding_routes_32'}],[{'name':'append_128'}]):
            with self.assertRaises(ValueError):analyze({'runs':rows})

    def test_class_deltas_preserve_counts_and_separate_retirement(self):
        result=delta(state(2),state(7));self.assertEqual(result['retired_groups'],5)
        self.assertEqual(result['group_retirement_ns'],500)
        self.assertEqual(sum(c['allocations'] for c in result['classes'].values()),30)
        self.assertTrue(all(c['allocation_ns']==5000 for c in result['classes'].values()))

    def test_missing_or_disabled_costs_are_not_zero(self):
        for value in (None,{},dict(kind='old',classes={})):
            a=state(0);a['buffer_costs']=value
            with self.assertRaises(ValueError):delta(a,state(1))

    def test_rejects_partial_counters_reset_and_uncompleted_users(self):
        changes=[lambda s:s['buffer_costs']['classes'].pop('state'),
            lambda s:s['buffer_costs']['classes']['temporary'].pop('allocation_ns'),
            lambda s:s['buffer_costs']['classes']['temporary'].update(allocations=-1),
            lambda s:s.update(allocation_count=7),lambda s:s.update(submissions=7),
            lambda s:s.update(live_command_groups=1),
            lambda s:s['buffer_costs'].update(group_retirement_ns=None)]
        for change in changes:
            b=state(2);change(b)
            with self.assertRaises(ValueError):delta(state(1),b)


class BufferReuseProbeTest(unittest.TestCase):
    def fixture(self):
        def arm(pool):
            sample=dict(elapsed_ns=30_000_000 if pool else 60_000_000,validated_elements=15138816,
                allocation_count=0 if pool else 3072,scratch_reuses=3072 if pool else 0,
                process_before=dict(compressed_bytes=0,decompressions=0),process_after=dict(compressed_bytes=0,decompressions=0))
            return dict(cold=copy.deepcopy(sample),warmup=copy.deepcopy(sample),warm=[copy.deepcopy(sample) for _ in range(8)])
        return dict(kind='buffer_reuse_probe_v1',complete=True,synthetic=True,
            pairs=[dict(pair=i,order=['pool','control'] if i%2 else ['control','pool'],control=arm(False),pool=arm(True)) for i in range(5)])

    def test_isolated_win_never_qualifies_inference(self):
        result=analyze_probe(self.fixture())
        self.assertTrue(result['material_isolated_opportunity'])
        self.assertEqual(result['median_saved_ms_per_synthetic_iteration'],30)
        self.assertFalse(result['normal_request_latency_qualified']);self.assertFalse(result['production_promoted'])

    def test_missing_or_disturbed_memory_keeps_probe_inconclusive(self):
        for memory in ({},dict(compressed_bytes=1,decompressions=0),dict(compressed_bytes=0,decompressions=1)):
            raw=self.fixture();raw['pairs'][0]['pool']['warm'][0]['process_after']=memory
            self.assertFalse(analyze_probe(raw)['material_isolated_opportunity'])

    def test_partial_or_wrong_count_cannot_pass(self):
        for change in (lambda r:r['pairs'].pop(),lambda r:r['pairs'][0]['pool']['warm'].pop(),
                       lambda r:r['pairs'][0]['pool']['warm'][0].update(allocation_count=1),
                       lambda r:r['pairs'][0].update(order=['pool','control'])):
            raw=self.fixture();change(raw)
            with self.assertRaises(ValueError):analyze_probe(raw)


if __name__=='__main__':unittest.main()
