import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from diagnose_decode_startup import analyze_steps,delta,run
from screen_cache import validate_request
from test_cache_screen import request_fixture


def fixture(steps=40):
    samples=[]
    for i in range(min(32,steps)):
        def counters(n):
            return dict(sample_ns=100,process=dict(decompressions=n*10,page_faults=n*3,compressed_bytes=1000),
                        metal=dict(live_command_groups=0,cpu_gpu_wait_ns=n*100),expert_cache=dict(misses=n),expert_dependencies={})
        samples.append(dict(step=i,offset=72+i,input_token_id=10,forward_ms=1.0,
            begin_ns=100+i*2000000,end_ns=1000100+i*2000000,before=counters(i),after=counters(i+1)))
    return dict(profiling_enabled=True,prompt_tokens=72,output_token_ids=[10]*(steps+1),token_latency_ms=[1.0]*steps,
                decode_diagnostics=dict(kind='decode_step_diagnostics_v1',max_steps=32,total_decode_steps=steps,
                    captured_steps=len(samples),omitted_steps=steps-len(samples),samples=samples))


class DecodeStartupTest(unittest.TestCase):
    def test_coverage_windows_and_missing_counters(self):
        r=analyze_steps(fixture())
        self.assertEqual((r['captured_steps'],r['omitted_steps']),(32,8))
        self.assertEqual([w['steps'] for w in r['windows']],[4,28])
        self.assertEqual([w['process']['decompressions'] for w in r['windows']],[40,280])
        self.assertEqual(r['observation_ns'],6400)
        self.assertIsNone(r['windows'][0]['process']['pageins'])
        self.assertEqual(analyze_steps(fixture(0))['windows'],[])
        for values in [(None,1),(1,None),(True,2),(5,4),(-1,0)]:self.assertIsNone(delta(*values))

    def test_rejects_uncommitted_or_misidentified_step_measurements(self):
        for change in [lambda r:r.update(profiling_enabled=False),
            lambda r:r['decode_diagnostics'].update(captured_steps=31),
            lambda r:r['decode_diagnostics'].update(omitted_steps=0),
            lambda r:r['decode_diagnostics']['samples'][1].update(offset=99),
            lambda r:r['decode_diagnostics']['samples'][1].update(input_token_id=11),
            lambda r:r['decode_diagnostics']['samples'][1].update(begin_ns=0),
            lambda r:r['decode_diagnostics']['samples'][1].update(forward_ms=2),
            lambda r:r['decode_diagnostics']['samples'][1]['after']['metal'].update(live_command_groups=1),
            lambda r:r['decode_diagnostics']['samples'][1]['after'].pop('sample_ns')]:
            row=fixture();change(row)
            with self.assertRaises(ValueError):analyze_steps(row)

    def test_observation_mode_requires_explicit_instrumentation_and_normal_timing_rejects_it(self):
        raw,evidence,config,work=request_fixture()
        raw['runs']=raw['runs'][:1];work=work[:1];raw['workloads']=work
        row=raw['runs'][0];row.update(profiling_enabled=True,decode_diagnostics=fixture(32)['decode_diagnostics'])
        with patch('screen_cache.check_machine'),patch('screen_cache.check_configuration'):
            self.assertEqual(len(validate_request(raw,evidence,config,work,{},instrumented=True)),1)
            with self.assertRaises(ValueError):validate_request(raw,evidence,config,work,{})
            row['profiling_enabled']=False
            with self.assertRaises(ValueError):validate_request(raw,evidence,config,work,{})

    def test_missing_source_is_unfinished_and_sealed(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'attempt'
            with patch('diagnose_decode_startup.verify_seal',side_effect=ValueError('changed source')),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(argparse.Namespace(output=out,time_limit=1)),2)
            self.assertFalse(json.loads((out/'summary.json').read_text())['complete'])
            self.assertTrue((out/'evidence-files.json').exists())
