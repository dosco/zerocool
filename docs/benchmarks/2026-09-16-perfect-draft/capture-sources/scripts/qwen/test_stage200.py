import copy
import unittest
from stage200 import ORDER, capacity_decision, clean_memory, q3_decision
import test_decode_timeline as fixtures
from decode_timeline import summarize
from screen_cache import validate_pressure_state


class Stage200Test(unittest.TestCase):
    def rows(self):
        return [dict(pair=p,configuration=c,clean_memory=c=='candidate',requests=[dict(request_ms=101 if c=='candidate' else 100)]) for p,c in ORDER]
    def test_capacity_selects_stability_with_bounded_cost(self):
        rows=self.rows();self.assertEqual(capacity_decision(rows)['selected_slots'],1460)
        for r in rows:r['clean_memory']=True
        self.assertEqual(capacity_decision(rows)['selected_slots'],1848)
        for r in rows:r['clean_memory']=False
        self.assertIsNone(capacity_decision(rows)['selected_slots'])
        with self.assertRaises(ValueError):capacity_decision(rows[:-1])
    def test_capacity_rejects_large_regression(self):
        rows=self.rows()
        for r in rows:
            if r['configuration']=='candidate':r['requests'][0]['request_ms']=110
        self.assertIsNone(capacity_decision(rows)['selected_slots'])
    def test_missing_and_reset_memory_never_clean(self):
        r=dict(runs=[dict(phases=dict(decode=dict(before=dict(process=dict(compressed_bytes=0,decompressions=5)),after=dict(process=dict(compressed_bytes=0,decompressions=5)))))])
        self.assertTrue(clean_memory(r))
        for value in (None,4,6):
            r['runs'][0]['phases']['decode']['after']['process']['decompressions']=value
            self.assertFalse(clean_memory(r))
    def fixture(self):
        timing=lambda ns:dict(ns_per_chain=ns,memory_before=dict(compressed_bytes=0,decompressions=0),memory_after=dict(compressed_bytes=0,decompressions=0))
        return dict(complete=True,validation=False,cases=[dict(layer=l,expert=e,rows=t,q4_aligned_bytes=100,q3_aligned_bytes=85,
            exact_expanded_reference=True,pairs=[dict(pair=p,q3=timing(95),q4=timing(100)) for p in range(5)])
            for l in (0,16,32,47) for e in (0,1) for t in (1,2,4,8)])
    def test_q3_pair_is_unit_and_incomplete_evidence_rejected(self):
        r=self.fixture();self.assertTrue(q3_decision(r)['advance_to_quality_stage'])
        for mutate in (lambda r:r['cases'].pop(),lambda r:r.update(complete=False),lambda r:r['cases'][0]['pairs'].pop(),
                       lambda r:r['cases'][0].update(exact_expanded_reference=False),lambda r:r['cases'][0].update(q3_aligned_bytes=90),
                       lambda r:r['cases'][0]['pairs'][0]['q3'].update(ns_per_chain=float('nan')),
                       lambda r:r['cases'][3]['pairs'][0]['q3'].update(ns_per_chain=-1),lambda r:r['cases'][0].update(layer=1)):
            bad=copy.deepcopy(r);mutate(bad)
            with self.assertRaises(ValueError):q3_decision(bad)
    def test_q3_missing_or_disturbed_memory_cannot_advance(self):
        for mutate in (lambda arm:arm.pop('memory_before'),lambda arm:arm['memory_after'].update(decompressions=1),
                       lambda arm:arm['memory_after'].update(compressed_bytes=1)):
            r=self.fixture();mutate(r['cases'][1]['pairs'][1]['q3'])
            self.assertEqual(q3_decision(r)['status'],'memory_disturbed')
            self.assertFalse(q3_decision(r)['advance_to_quality_stage'])
    def test_q3_noninferiority_failure_is_not_proof_of_regression(self):
        r=self.fixture()
        for c in r['cases']:
            for p in c['pairs']:p['q3']['ns_per_chain']=[118,120,98,97,99][p['pair']]
        d=q3_decision(r)
        self.assertEqual(d['status'],'operator_latency_inconclusive')
        self.assertLess(d['one_token_lower_95'],1)
        self.assertGreater(d['one_token_upper_95'],1.03)
        self.assertFalse(d['advance_to_quality_stage'])
        for c in r['cases']:
            for p in c['pairs']:p['q3']['ns_per_chain']=120
        self.assertEqual(q3_decision(r)['status'],'operator_regression')
    def test_decode_only_coverage_uses_phase_dispatch_count(self):
        native,profile,deps=fixtures.DecodeTimelineTest().fixture();row=native['runs'][0]
        count=row['after']['metal']['dispatches'];row['after']['metal']['dispatches']+=100
        row['phases']={'decode':{'before':{'metal':{'dispatches':100}},'after':{'metal':{'dispatches':100+count}}}}
        profile['coverage']='decode-only';self.assertEqual(summarize(native,profile,deps)['captured_tokens'],1)
        profile['coverage']='all-dispatches'
        with self.assertRaises(ValueError):summarize(native,profile,deps)
    def test_pressure_changes_require_matching_ownership_events(self):
        initial=dict(expert_slots=1848,expert_bytes=1848*2768896,planned_bytes=12*1024**3)
        event=dict(sequence=1,level=1,before_slots=1848,requested_slots=1460,achieved_slots=1460,released_bytes=388*2768896,drain_duration_ns=1)
        state=dict(memory_pressure=dict(policy='shrink',source='dispatch-memorypressure',event_count=1,events=[event]),
            phase_memory=dict(pressure_resizes=1),memory_plan=dict(initial,expert_slots=1460,expert_bytes=1460*2768896,planned_bytes=initial['planned_bytes']-388*2768896))
        validate_pressure_state(state,initial,'shrink')
        for mutation in (lambda s:s['phase_memory'].update(pressure_resizes=2),lambda s:s['memory_pressure'].update(event_count=2),
                         lambda s:s['memory_pressure']['events'][0].update(achieved_slots=1072),lambda s:s['memory_plan'].update(planned_bytes=1)):
            bad=copy.deepcopy(state);mutation(bad)
            with self.assertRaises(ValueError):validate_pressure_state(bad,initial,'shrink')


if __name__=='__main__':unittest.main()
