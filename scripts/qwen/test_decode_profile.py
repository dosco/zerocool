import copy
import unittest
from benchmark_cached_tokens import validate
from summarize_decode_profile import summarize


class DecodeProfileTest(unittest.TestCase):
    def fixture(self):
        before=dict(metal=dict(build_fingerprint='build',kernels=dict(counter_profile=True),dispatches=0),
                    artifact_revision='artifact',memory_plan=dict(limit_bytes=12*1024**3))
        after=copy.deepcopy(before);after['metal']['dispatches']=2
        report=dict(kind='cached_full_token_replay',passed=True,profiling_enabled=True,
                    runs=[dict(exact=True,ready_hits=480,application_read_bytes=0,before=before,after=after,forward_ns=100000)])
        op=dict(request_phase='cached_replay',kernel='q4_mm',stage='expert',layer=0,gpu_pass_ns=1000)
        profile=dict(truncated=False,command_groups=[dict(operations=[op,op])])
        return report,profile

    def test_complete_and_incomplete_dispatches(self):
        report,profile=self.fixture();result=summarize(report,profile)
        self.assertEqual(result['mean_dispatches'],2);self.assertEqual(result['mean_submissions'],1)
        self.assertFalse(result['normal_request_latency_qualified'])
        profile['command_groups'][0]['operations'].pop()
        with self.assertRaises(ValueError):summarize(report,profile)

    def test_profile_cannot_qualify_uninstrumented_timing(self):
        report,profile=self.fixture()
        with self.assertRaisesRegex(ValueError,'Instrumented'):validate(report,'build','artifact',2048,1)
        report['runs'][0]['application_read_bytes']=1
        with self.assertRaises(ValueError):summarize(report,profile)


class CachedComparisonTest(unittest.TestCase):
    def fixture(self):
        rows=[]
        for rep in range(5):
            for variant in (('control','candidate') if rep%2==0 else ('candidate','control')):
                state=dict(metal=dict(build_fingerprint='build',kernels=dict(q8_decode_rows=2 if variant=='candidate' else 0)),
                           diagnostic_stream_trunk=False,ngram_misses=0,
                           execution=dict(decode_path='grouped'),ready_group=2,chunk_tokens=128,io_workers=8,short_append_tokens=128,
                           memory_plan=dict(limit_bytes=12*1024**3,expert_slots=480,snapshot_bytes=128))
                rows.append(dict(repetition=rep,variant=variant,forward_ns=100000 if variant=='candidate' else 200000,
                                 exact=True,ready_hits=480,application_read_bytes=0,before=state,after=copy.deepcopy(state)))
        return dict(kind='cached_full_token_replay',passed=True,normal_request_latency_qualified=False,
                    paired_comparison=True,profiling_enabled=False,artifact_revision='artifact',prompt_tokens=2048,
                    snapshot_bytes=128,runs=rows)

    def test_exact_alternating_comparison(self):
        from summarize_cached_comparison import summarize
        result=summarize(self.fixture())
        self.assertEqual(result['latency_ratio']['high'],0.5)
        self.assertFalse(result['normal_request_latency_qualified'])

    def test_reject_changed_or_incomplete_comparisons(self):
        from summarize_cached_comparison import summarize
        changes=[lambda r:r['runs'].pop(),lambda r:r.update(profiling_enabled=True),
                 lambda r:r['runs'][1].update(application_read_bytes=1),
                 lambda r:r['runs'][2].update(variant='control'),
                 lambda r:r['runs'][1]['after']['execution'].update(decode_path='reference'),
                 lambda r:r['runs'][1]['before'].update(ready_group=4),
                 lambda r:r['runs'][1]['after']['metal']['kernels'].update(q8_decode_rows=4)]
        for change in changes:
            report=self.fixture();change(report)
            with self.assertRaises(ValueError):summarize(report)


class Q8OperatorComparisonTest(unittest.TestCase):
    def fixture(self):
        return dict(kind='captured_q8_decode_screen',exact=True,measurements=[
            dict(matrix=dict(K=2560,N=512,rows=1,bits=8),case='activation',
                 q8_decode_rows=rows,repetition=rep,wall_ns=100 if rows else 200,exact=True)
            for rep in range(5) for rows in (0,2)])

    def test_complete_q8_operator_pairs(self):
        from benchmark_q8_decode import summarize
        result=summarize(self.fixture(),2,5)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['bounds']['high'],0.5)

    def test_reject_unpaired_duplicate_or_inexact_samples(self):
        from benchmark_q8_decode import summarize
        changes=[lambda r:r['measurements'].pop(),
                 lambda r:r['measurements'].append(copy.deepcopy(r['measurements'][0])),
                 lambda r:r['measurements'][1].update(repetition=8),
                 lambda r:r['measurements'][1].update(q8_decode_rows=4),
                 lambda r:r['measurements'][1].update(exact=False)]
        for change in changes:
            report=self.fixture();change(report)
            with self.assertRaises(ValueError):summarize(report,2,5)

if __name__=='__main__':unittest.main()
