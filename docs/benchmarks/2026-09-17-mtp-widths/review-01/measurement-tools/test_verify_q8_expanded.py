import unittest
from verify_q8_expanded import decide, ORDER, CRITERIA


def rows(speeds=(4.6,5.2,5.1,4.5)):
    return [dict(pair=p,arm=a,source=f'{p}-{a}.json',observation=dict(width=4,validation=False,verified_tokens=16,
        clean_memory=True,clean_host=True,decode_wall_ns=16e9/speed,verified_tokens_per_second=16e9/(16e9/speed)))
        for (p,a),speed in zip(ORDER,speeds)]


class ExpandedVerifierTest(unittest.TestCase):
    def test_requires_complete_alternating_pairs(self):
        self.assertEqual(decide(rows()[:2])['status'],'running')
        r=decide(rows());self.assertTrue(r['expanded_q8_promising']);self.assertTrue(r['paired_comparison_complete'])
        self.assertTrue(all(x<1 for x in r['paired_wall_ratios']))
        r=rows();r.reverse()
        with self.assertRaises(ValueError):decide(r)

    def test_weak_candidate_stops_without_claiming_pairs(self):
        r=decide(rows((4.5,4.99))[:2]);self.assertTrue(r['early_stop']);self.assertFalse(r['paired_comparison_complete'])
        with self.assertRaises(ValueError):decide(rows((4.5,4.99,5.2,4.4)))

    def test_pressure_is_not_a_performance_rejection(self):
        r=rows();r[0]['observation']['clean_memory']=False
        with self.assertRaises(ValueError):decide(r)
        r=rows();r[0]['observation']['verified_tokens']=4
        with self.assertRaises(ValueError):decide(r)

    def test_no_promotion_or_actual_draft_claim(self):
        r=decide(rows())
        self.assertFalse(r['actual_draft_measured']);self.assertFalse(r['production_promoted'])
        self.assertIsNone(r['confidence_95']);self.assertEqual(CRITERIA['packed_dispatches_per_block'],133)


if __name__=='__main__':unittest.main()
