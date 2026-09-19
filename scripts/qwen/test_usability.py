import copy
import tempfile
import unittest
from pathlib import Path
from check_aider import verify_add
from check_api_performance import compare, clean_memory
from api_check import admission_error, cancellation_observed


class UsabilityEvidenceTests(unittest.TestCase):
    def test_cancellation_evidence_must_match_the_current_request(self):
        old=dict(phase='generating',active_request_id='new',last_request_id='old',last_request_cancelled=True)
        self.assertFalse(cancellation_observed(old,'new'))
        self.assertFalse(cancellation_observed(dict(old,phase='draining',active_request_id='different'),'new'))
        self.assertTrue(cancellation_observed(dict(old,phase='cancelling'),'new'))
        self.assertTrue(cancellation_observed(dict(old,phase='ready',last_request_id='new'),'new'))
        self.assertFalse(cancellation_observed(old,None))

    def test_real_checker_rejects_fake_or_compressed_engines(self):
        status=dict(phase='ready',process=dict(physical_footprint_bytes=100,compressed_bytes=0,compressed_peak_bytes=0))
        self.assertIsNone(admission_error(status))
        self.assertIsNotNone(admission_error(dict(status,test_executor=True)))
        status['process']['compressed_peak_bytes']=1
        self.assertIsNotNone(admission_error(status))

    def test_coding_fixture_requires_correct_output_and_rejects_other_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'maths.py'
            for code,expected in [('return a-b',False),('return a+b',True),('return a+b+1',False)]:
                path.write_text('def add(a, b):\n    '+code+'\n');self.assertEqual(verify_add(path)['passed'],expected)
            path.write_text('import os\ndef add(a,b): return a+b')
            with self.assertRaises(ValueError):verify_add(path)

    def test_comparison_rejects_missing_or_incompatible_evidence(self):
        row=dict(complete=True,memory_plan={'limit_bytes':12},elapsed_ms=10,
            response=dict(freellm=dict(output_token_ids=[1,2]),usage=dict(prompt_tokens=5)))
        rows=[dict(copy.deepcopy(row),pair=i,arm=arm) for i in range(2) for arm in (('old','new') if i==0 else ('new','old'))]
        self.assertEqual(compare(rows,2)['decision'],'short_screen_within_3_percent')
        self.assertFalse(compare(rows,2)['qualified'])
        for change in [lambda r:r.pop(),lambda r:r[1].update(complete=False),lambda r:r[1].update(memory_plan={}),
                       lambda r:r[1]['response']['freellm'].update(output_token_ids=[3]),
                       lambda r:r[1].update(elapsed_ms=float('inf'))]:
            broken=copy.deepcopy(rows);change(broken)
            with self.assertRaises(ValueError):compare(broken,2)
        rows[1]['elapsed_ms']=11
        self.assertEqual(compare(rows,2)['decision'],'investigate_or_extend')

    def test_memory_missing_and_compression_do_not_pass(self):
        m=dict(physical_footprint_bytes=100,physical_footprint_peak_bytes=200,compressed_bytes=0,compressed_peak_bytes=0,decompressions=3,system_swap_used_bytes=40)
        self.assertTrue(clean_memory([m,m]))
        self.assertFalse(clean_memory([m,dict(m,compressed_peak_bytes=1)]))
        self.assertFalse(clean_memory([m,dict(m,system_swap_used_bytes=41)]))
        self.assertFalse(clean_memory([dict(m,physical_footprint_bytes=0)]))
        self.assertFalse(clean_memory([dict(m,physical_footprint_peak_bytes=99)]))
        with self.assertRaises(ValueError):clean_memory([{}])


if __name__=='__main__':unittest.main()
