import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark_exact import config_args
from screen_residency import ORDER,check_residency,configs,decide,run


class ResidencyScreenTest(unittest.TestCase):
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
