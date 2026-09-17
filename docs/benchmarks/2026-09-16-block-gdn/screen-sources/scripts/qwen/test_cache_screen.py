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
from screen_cache import configs,decide,run,validate_request,validate_correctness,correctness_case,CHECKS,ORDER
from test_route_trace import capture_fixture


def request_fixture(policy='clock'):
    raw,trace,evidence,work,config=capture_fixture(32,True)
    for i,row in enumerate(raw['runs']):
        row.update(profiling_enabled=False,prefill_tokens=row['prompt_tokens']-row['reused_tokens'],
                   request_ms=100,time_to_first_token_ms=60,decode_wall_ms=40,token_latency_ms=[1.25]*32)
        state=row['before'];state.update(completion_pipeline=True,process={},
            expert_cache=dict(policy=policy,hits=0,misses=0,application_read_bytes=0))
        row['after']=copy.deepcopy(state);row['after']['expert_cache'].update(hits=100,misses=200,application_read_bytes=600)
        row['phases']={p:dict(before=copy.deepcopy(state),after=copy.deepcopy(row['after'])) for p in ('ingest','decode')}
    return raw,evidence,next(c for c in configs() if c['cache_policy']==policy),work


class CacheScreenTest(unittest.TestCase):
    def test_only_policy_differs_and_option_is_forwarded(self):
        a,b=configs()
        self.assertEqual({k for k in a if a[k]!=b[k]},{'name','cache_policy'})
        args=config_args(b);self.assertEqual(args[args.index('--cache-policy')+1],'slru')

    def test_request_rejects_wrong_policy_memory_outputs_reuse_or_instrumentation(self):
        args=request_fixture()
        with patch('screen_cache.check_machine'),patch('screen_cache.check_configuration'):
            expected={};rows=validate_request(*args,expected)
            self.assertEqual(rows[1]['reused_tokens'],33)
            validate_request(*request_fixture('slru'),expected)
            changes=[lambda a:a[0]['runs'][0].update(profiling_enabled=True),
                lambda a:a[0]['runs'][1].update(reused_tokens=34),
                lambda a:a[0]['runs'][1].update(pending_tokens_ingested=0),
                lambda a:a[0]['runs'][1].update(output_token_ids=[12]*33),
                lambda a:a[0]['runs'][1].update(phases={}),
                lambda a:a[0]['runs'][1]['after']['expert_cache'].update(policy='slru'),
                lambda a:a[0]['runs'][0]['after']['memory_plan'].update(panel_tokens=256),
                lambda a:a[0]['runs'][1]['phases']['ingest']['after']['phase_memory'].update(pressure_resizes=1),
                lambda a:a[0]['runs'][0].update(token_latency_ms=[0]*32),
                lambda a:a[0].update(complete=False)]
            for change in changes:
                altered=copy.deepcopy(args);change(altered)
                with self.assertRaises(ValueError):validate_request(*altered,copy.deepcopy(expected))

    def test_state_gate_requires_exact_stages_and_all_failure_checks(self):
        reports=[]
        for policy in ('clock','slru'):
            state=dict(memory_plan=dict(expert_slots=32),diagnostic_stream_trunk=False,
                       expert_cache=dict(policy=policy,evictions=1))
            reports.append(dict(passed=True,case=correctness_case(policy),full_model=True,layers=48,
                checks=[dict(name=c,passed=True) for c in CHECKS],runs=[dict(panel=0,
                    stages=[dict(layers=[{}]*48,routes=[[]]*48,logits_sha256='a'*64)]*3,
                    continued_statistics=copy.deepcopy(state),after_fresh=copy.deepcopy(state))]))
        with patch('screen_cache.check_machine'),patch('screen_cache.check_configuration'):
            self.assertTrue(validate_correctness(reports,{})['passed'])
            for mutation in (lambda r:r[1]['runs'][0]['stages'][0].update(logits_sha256='b'*64),
                lambda r:r[1]['checks'].pop(),lambda r:r[1]['case'].update(cache_policy='clock'),
                lambda r:r[1]['runs'][0]['after_fresh']['expert_cache'].update(evictions=0)):
                altered=copy.deepcopy(reports);mutation(altered)
                with self.assertRaises(ValueError):validate_correctness(altered,{})

    def test_decision_needs_alternating_pairs_and_catches_append_tradeoff(self):
        rows=[dict(pair=i,policy=p,requests=[dict(request_ms=98 if p=='slru' else 100,
            time_to_first_token_ms=49 if p=='slru' else 50,decode_wall_ms=49 if p=='slru' else 50) for _ in range(2)]) for i,p in ORDER]
        result=decide(rows);self.assertTrue(result['advance_to_five_pairs']);self.assertIsNone(result['confidence_95'])
        for m in rows:
            if m['policy']=='slru':m['requests'][1]['decode_wall_ms']=55
        self.assertFalse(decide(rows)['advance_to_five_pairs'])
        with self.assertRaises(ValueError):decide(rows[:-1])
        with self.assertRaises(ValueError):decide(rows[::-1])

    def test_blocked_prerequisite_remains_unfinished_and_sealed(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'attempt'
            with patch('screen_cache.verify_seal',side_effect=ValueError('changed source')),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(argparse.Namespace(output=output,time_limit=1)),2)
            self.assertFalse(json.loads((output/'summary.json').read_text())['complete'])
            self.assertTrue((output/'evidence-files.json').exists())
