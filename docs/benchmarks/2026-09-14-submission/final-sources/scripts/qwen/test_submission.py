import copy
import unittest

from submission_timeline import split
from screen_submission import analyze
from screen_coalesced import decide
from stage200 import ORDER


class SubmissionTest(unittest.TestCase):
    def group(self):
        return dict(submitted_ns=1000000,commit_returned_ns=2000000,driver_start_seconds=.0015,
                    driver_end_seconds=.003,gpu_start_seconds=.004,gpu_end_seconds=.006,completed_ns=7000000)

    def test_idle_partition_and_overlapping_commit(self):
        g=self.group();r=split(1,8000000,[g])
        self.assertEqual([r[k] for k in ('before_driver','in_driver','after_driver','commit_api_overlap')],[.5,1.5,1,1])
        # Completion can race ahead of the returning commit call.
        g['commit_returned_ns']=8000000;r=split(1,9000000,[g]);self.assertEqual(r['commit_api_overlap'],3)

    def test_later_group_is_not_double_counted_or_counted_during_gpu_work(self):
        first=self.group();later=dict(first,submitted_ns=2000000,commit_returned_ns=3000000,
            driver_start_seconds=.003,driver_end_seconds=.005,gpu_start_seconds=.007,gpu_end_seconds=.008,completed_ns=9000000)
        r=split(1,10000000,[later,first])
        self.assertEqual([r[k] for k in ('before_driver','in_driver','after_driver')],[.5,1.5,2])

    def test_missing_or_reversed_driver_data_rejected(self):
        for key,value in [('driver_start_seconds',0),('driver_start_seconds',float('nan')),
                          ('driver_end_seconds',.001),('commit_returned_ns',1)]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):split(1,10000000,[dict(self.group(),**{key:value})])

    def timing(self):
        return dict(complete=True,validation=False,pairs=[dict(pair=p,experts_per_group=size,arms=[dict(
            retained_references=retained,samples=[dict(wall_ns=400000 if retained else 200000)]*64,
            memory_before=dict(compressed_bytes=0,decompressions=0),memory_after=dict(compressed_bytes=0,decompressions=0))
            for retained in (p%2==0,p%2!=0)]) for p in range(5) for size in (1,2,4)])

    def test_small_probe_requires_full_clean_positive_paired_evidence(self):
        raw=self.timing();self.assertTrue(analyze(raw)['advance_to_request_screen'])
        for change in (lambda r:r['pairs'].pop(),lambda r:r['pairs'][0]['arms'][0]['samples'].pop(),
                       lambda r:r.update(complete=False)):
            bad=copy.deepcopy(raw);change(bad)
            with self.assertRaises(ValueError):analyze(bad)
        raw['pairs'][0]['arms'][0]['memory_after']['compressed_bytes']=1
        self.assertFalse(analyze(raw)['advance_to_request_screen'])

    def test_request_screen_requires_both_phases_material_gain_and_clean_memory(self):
        rows=[dict(pair=p,configuration=arm,clean_memory=True,requests=[dict(request_ms=20000 if arm=='control' else 19000,
            time_to_first_token_ms=10000,decode_wall_ms=10000 if arm=='control' else 9000)]*2) for p,arm in ORDER]
        self.assertTrue(decide(rows)['advance_to_confirmation'])
        rows[0]['clean_memory']=False;self.assertFalse(decide(rows)['advance_to_confirmation'])
        with self.assertRaises(ValueError):decide(rows[:-1])


if __name__=='__main__':unittest.main()
