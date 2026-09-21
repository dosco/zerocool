import unittest
import subprocess
import hashlib
import struct
from pathlib import Path
from benchmark_sparse_attention import acceptance,choose,check_cached
import test_decode_profile

class SparseStageTest(unittest.TestCase):
    def test_cached_stage_binds_machine_build_and_input(self):
        report=test_decode_profile.CachedComparisonTest().fixture();tokens=[1]*2048+[760]
        report.update(comparison_axis='q8_decode_rows',continuation_token=760,
                      input_sha256=hashlib.sha256(struct.pack('<'+'i'*len(tokens),*tokens)).hexdigest())
        for row in report['runs']:
            for state in (row['before'],row['after']):
                state['metal'].update(device='Apple M1 Pro',physical_bytes=32*1024**3)
                state['memory_plan']['planned_bytes']=10*1024**3
                state['process']=dict(physical_footprint_bytes=10*1024**3)
        self.assertTrue(check_cached(report,'build','artifact',tokens,'q8_decode_rows')['exact'])
        for changed in ([2]+tokens[1:],tokens[:-1]+[761]):
            with self.assertRaises(ValueError):check_cached(report,'build','artifact',changed,'q8_decode_rows')
        for key,value in [('build_fingerprint','stale'),('device','Apple M5 Max'),('physical_bytes',128*1024**3)]:
            state=report['runs'][0]['before']['metal'];saved=state[key];state[key]=value
            with self.assertRaises(ValueError):check_cached(report,'build','artifact',tokens,'q8_decode_rows')
            state[key]=saved
        report['runs'][0]['after']['process']['physical_footprint_bytes']=13*1024**3
        with self.assertRaises(ValueError):check_cached(report,'build','artifact',tokens,'q8_decode_rows')

    def test_native_cli_rejects_invalid_and_production_candidates(self):
        binary=Path(__file__).resolve().parents[2]/'build/qwen/bin/zerocool'
        for args,message in [(['inspect','--sparse-selection','wrong'],'sparse selection'),
                             (['inspect','--attention-score-tiles','wrong'],'attention score tiles'),
                             (['serve','--sparse-selection','gpu'],'execution experiments'),
                             (['run','--attention-score-tiles','skip-masked'],'execution experiments'),
                             (['bench','--cached-compare-axis','wrong'],'comparison axis')]:
            result=subprocess.run([str(binary),*args],capture_output=True,text=True,timeout=30)
            self.assertNotEqual(result.returncode,0)
            self.assertIn(message,result.stderr)

    def evidence(self):
        bounds=[dict(case=c,metric=m,pairs=5,median=.95,low=.94,high=.96)
                for c in ('prompt_2k','prompt_4k','append_128') for m in ('ttft_ms','decode_ms_per_token','request_ms')]
        return dict(complete=True,mode='paired',comparison_purpose='experiment',comparisons=[dict(confidence_bounds=bounds)])
    def test_complete_request_gate(self):
        r=self.evidence();self.assertTrue(acceptance(r)['stage_latency_passed'])
        for b in r['comparisons'][0]['confidence_bounds']:b.update(median=.99,high=.999)
        self.assertFalse(acceptance(r)['stage_latency_passed'])
        r=self.evidence();r['comparisons'][0]['confidence_bounds'][0]['high']=1.04
        self.assertFalse(acceptance(r)['stage_latency_passed'])
    def test_missing_or_inconclusive_evidence(self):
        for mutate in (lambda r:r.update(complete=False),lambda r:r['comparisons'][0]['confidence_bounds'].pop(),
                       lambda r:r['comparisons'][0]['confidence_bounds'][0].update(pairs=2),
                       lambda r:r['comparisons'][0]['confidence_bounds'][0].update(high=float('nan'))):
            r=self.evidence();mutate(r)
            with self.assertRaises(ValueError):acceptance(r)
    def test_choose_prefers_simpler_within_one_percent(self):
        rows=[dict(name=case,pair=p,configuration=name,request_ms=value) for p in range(2)
              for case in ('prompt_2k','prompt_4k','append_128','prompt_7k') for name,value in [('cpu-selection',100),('gpu-selection',90),('gpu-selection-skip',89.5)]]
        report=dict(complete=True,mode='screen',measurements=rows)
        self.assertEqual(choose(report),'gpu-selection')
        bad=dict(report,measurements=rows[:-1]+[rows[0]])
        with self.assertRaises(ValueError):choose(bad)
        bad=dict(report,measurements=[r for r in rows if r['name']!='prompt_7k'])
        with self.assertRaises(ValueError):choose(bad)
        report['measurements'].pop()
        with self.assertRaises(ValueError):choose(report)

if __name__=='__main__':unittest.main()
