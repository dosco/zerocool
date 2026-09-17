import argparse
import contextlib
import copy
import io
import json
import math
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark_exact import config_args,inspect_admission
from screen_residency import ORDER,check_residency,configs,decide,pair_order,paired_log_interval,run,validate_screen


class ResidencyScreenTest(unittest.TestCase):
    def test_metadata_retry_respects_remaining_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            stem=Path(d)/'admit'
            def denied(command,**kwargs):
                self.assertEqual(command[1],'inspect')
                self.assertEqual(kwargs['timeout'],1)
                Path(command[-1]).write_text(json.dumps(dict(current_admission=dict(error='busy'))))
            with patch('benchmark_exact.subprocess.run',side_effect=denied) as calls,patch('benchmark_exact.time.sleep') as sleep:
                with self.assertRaises(subprocess.TimeoutExpired):inspect_admission('engine',[],stem,12,512,None,remaining=lambda:1)
                self.assertEqual(calls.call_count,1);sleep.assert_not_called()
                self.assertTrue(stem.with_suffix('.admission.json').exists())

    def test_metadata_retry_preserves_rejection_and_never_runs_inference(self):
        with tempfile.TemporaryDirectory() as d:
            stem=Path(d)/'admit'
            def inspect(command,**kwargs):
                self.assertEqual(command[1],'inspect');self.assertEqual(kwargs['timeout'],20)
                Path(command[-1]).write_text(json.dumps(dict(current_admission=(dict(error='busy') if command[-1].endswith('.admission.json') else dict(limit_bytes=12,panel_tokens=512)))))
            with patch('benchmark_exact.subprocess.run',side_effect=inspect) as calls,patch('benchmark_exact.time.sleep') as sleep:
                result=inspect_admission('engine',[],stem,12,512,None,remaining=lambda:20)
                self.assertEqual(calls.call_count,2);sleep.assert_called_once_with(2)
                self.assertEqual(result,stem.with_suffix('.admission-retry-1.json'))
                self.assertIn('error',json.loads(stem.with_suffix('.admission.json').read_text())['current_admission'])

    def test_five_pair_interval_against_symmetric_log_fixture(self):
        values=[.9*math.exp(v) for v in (-.04,-.02,0,.02,.04)]
        interval=paired_log_interval(values)
        self.assertAlmostEqual(interval['geometric_mean'],.9)
        half=2.7764451051977987*math.sqrt(.0002)
        self.assertAlmostEqual(interval['low'],.9*math.exp(-half))
        self.assertAlmostEqual(interval['high'],.9*math.exp(half))
        for bad in ([.9]*4,[.9]*6,[.9,.9,.9,.9,0],[True]*5,[float('nan')]*5):
            with self.assertRaises(ValueError):paired_log_interval(bad)

    def test_five_pair_gate_requires_uncertainty_and_catches_regression(self):
        rows=[dict(pair=i,residency=p,requests=[dict(request_ms=95 if p=='core' else 100,
            time_to_first_token_ms=47.5 if p=='core' else 50,decode_wall_ms=47.5 if p=='core' else 50) for _ in range(2)]) for i,p in pair_order(5)]
        self.assertEqual(pair_order(2),ORDER)
        result=decide(rows,5);self.assertTrue(result['advance_to_long_validation'])
        self.assertFalse(result['advance_to_five_pairs']);self.assertFalse(result['production_promoted'])
        self.assertEqual(result['confidence_95']['pairs'],5)
        changed=copy.deepcopy(rows)
        for m in changed:
            if m['residency']=='core':m['requests'][0]['request_ms']=80 if m['pair']%2 else 115
        self.assertFalse(decide(changed,5)['advance_to_long_validation'])
        changed=copy.deepcopy(rows)
        for m in changed:
            if m['residency']=='core':m['requests'][1]['decode_wall_ms']=52
        self.assertFalse(decide(changed,5)['advance_to_long_validation'])
        for bad in (rows[:-2],rows[::-1],rows+rows[:2]):
            with self.assertRaises(ValueError):decide(bad,5)
        for count in (True,0,3,6):
            with self.assertRaises(ValueError):pair_order(count)

    def test_five_pair_prerequisite_must_be_complete_and_matching(self):
        with patch('screen_residency.load',return_value=dict(complete=False)):
            with self.assertRaises(ValueError):validate_screen({},[],{})
        with patch('screen_residency.load',return_value=dict(complete=True,status='promising',build='old')):
            with self.assertRaises(ValueError):validate_screen({'build':'new'},[],{})

    def test_only_residency_changes(self):
        a,b=configs()
        self.assertEqual({k for k in a if a[k]!=b[k]},{'name','residency'})
        self.assertEqual(a['cache_policy'],'clock')
        args=config_args(b);self.assertEqual(args[args.index('--residency')+1],'core')
        self.assertNotIn('--decode-diagnostics',args)

    def test_rejects_append_regression_and_inconsistent_pair(self):
        rows=[dict(pair=i,residency=p,requests=[dict(request_ms=98 if p=='core' else 100,
            time_to_first_token_ms=49 if p=='core' else 50,decode_wall_ms=49 if p=='core' else 50) for _ in range(2)]) for i,p in ORDER]
        result=decide(rows);self.assertTrue(result['advance_to_five_pairs']);self.assertIsNone(result['confidence_95'])
        self.assertFalse(result['production_promoted']);self.assertFalse(result['normal_request_latency_qualified'])
        regression=copy.deepcopy(rows)
        for m in regression:
            if m['residency']=='core':m['requests'][1]['decode_wall_ms']=55
        self.assertFalse(decide(regression)['advance_to_five_pairs'])
        rows[2]['requests'][0]['request_ms']=103
        self.assertFalse(decide(rows)['advance_to_five_pairs'])
        for invalid in (rows[:-1],rows[::-1]):
            with self.assertRaises(ValueError):decide(invalid)

    def test_invalid_timing_and_missing_request(self):
        rows=[dict(pair=i,residency=p,requests=[dict(request_ms=100,time_to_first_token_ms=50,decode_wall_ms=50) for _ in range(2)]) for i,p in ORDER]
        for value in (0,-1,True,float('nan'),float('inf'),None):
            changed=copy.deepcopy(rows);changed[0]['requests'][0]['request_ms']=value
            with self.assertRaises(ValueError):decide(changed)
        rows[0]['requests'].pop()
        with self.assertRaises(ValueError):decide(rows)

    def test_checks_phase_residency_and_rejects_extra_enrollment_or_profiling(self):
        state=dict(memory_plan=dict(resident_bytes=100,session_bytes=20,runtime_control_bytes=1),
            metal=dict(kernels={},residency=dict(mode='core',registered_bytes=121,allocations=3,
                bytes_by_class=dict(resident=100,state=21),set_overhead_bytes=0,pending_retirements=0)))
        raw=dict(runs=[dict(before=state,after=copy.deepcopy(state),
            phases={'ingest':dict(before=copy.deepcopy(state),after=copy.deepcopy(state))})])
        check_residency(raw,'core')
        for change in (lambda s:s['metal']['residency']['bytes_by_class'].update(expert=1),
            lambda s:s['metal']['residency'].update(pending_retirements=1),
            lambda s:s['metal']['residency'].update(registered_bytes=120),
            lambda s:s['metal']['residency'].update(set_overhead_bytes=65*1024**2),
            lambda s:s['metal']['residency']['bytes_by_class'].update(state=22),
            lambda s:s['metal']['kernels'].update(profile=True)):
            altered=copy.deepcopy(raw);change(altered['runs'][0]['phases']['ingest']['after'])
            with self.assertRaises(ValueError):check_residency(altered,'core')
        with self.assertRaises(ValueError):check_residency(raw,'off')

    def test_missing_prerequisite_is_unfinished_and_sealed(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'attempt'
            with patch('screen_residency.verify_seal',side_effect=ValueError('changed source')),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(argparse.Namespace(output=out,time_limit=1)),2)
            self.assertFalse(json.loads((out/'summary.json').read_text())['complete'])
            self.assertTrue((out/'evidence-files.json').exists())
