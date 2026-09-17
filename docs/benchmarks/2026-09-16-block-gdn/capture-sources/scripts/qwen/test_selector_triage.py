import argparse
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from qualification_evidence import ResourceBlocked
from triage_selector import CONFIG, decide, prerequisites, run


class TriageTest(unittest.TestCase):
    def test_timeout_summary_retains_last_native_phase_without_a_performance_decision(self):
        from test_cached_progress import CachedProgressTest
        with tempfile.TemporaryDirectory() as d:
            args=argparse.Namespace(prerequisites=Path(d)/'prior',output=Path(d)/'run',time_limit=1)
            def timeout(command,**kwargs):
                path=args.output/'cached-run/cached-4096/progress.jsonl';path.parent.mkdir(parents=True)
                path.write_text(''.join(json.dumps(row)+'\n' for row in CachedProgressTest().events()))
                raise subprocess.TimeoutExpired(command,1)
            with patch('triage_selector.prerequisites',return_value={'build':'build'}),patch('triage_selector.EvidenceGuard') as guard:
                guard.return_value.run.side_effect=timeout
                self.assertEqual(run(args),2)
            report=json.loads((args.output/'summary.json').read_text())
            self.assertEqual(report['status'],'time_budget_exhausted')
            self.assertFalse(report['complete']);self.assertFalse(report['advance_to_normal_screen'])
            self.assertEqual(report['phase_progress']['unfinished_phase'],'reference_priming')
            self.assertEqual(report['phase_progress']['latest_details']['completed_tokens'],512)

    def result(self, low=.97, median=.98, high=.99):
        return dict(exact=True, pairs=5, application_read_bytes=0, prompt_tokens=4096,
                    comparison_axis='sparse_selection',
                    latency_ratio=dict(low=low, median=median, high=high, pairs=5))

    def test_fixed_gate_never_qualifies_normal_latency_or_production(self):
        result = decide(self.result())
        self.assertTrue(result['advance_to_normal_screen'])
        for key in ('normal_request_latency_qualified', 'full_validation', 'production_promoted'):
            self.assertFalse(result[key])
        self.assertEqual(decide(self.result(.97, .98, 1.01))['status'], 'inconclusive')
        self.assertFalse(decide(self.result(.994, .995, .996))['advance_to_normal_screen'])
        self.assertFalse(decide(self.result(1.01, 1.02, 1.03))['advance_to_normal_screen'])

    def test_rejects_incomplete_inexact_or_invalid_measurements(self):
        for key, value in [('exact', False), ('pairs', 4), ('application_read_bytes', 1),
                           ('prompt_tokens', 2048), ('comparison_axis', 'other')]:
            result = self.result(); result[key] = value
            with self.assertRaises(ValueError): decide(result)
        for value in (0, -1, True, float('nan'), float('inf')):
            result = self.result(); result['latency_ratio']['median'] = value
            with self.assertRaises(ValueError): decide(result)
        result = self.result(); result['latency_ratio']['low'] = 2
        with self.assertRaises(ValueError): decide(result)

    def test_failure_outcomes_are_unfinished_and_never_launch_full_validation(self):
        for error, status in [(ResourceBlocked('memory'), 'resource_blocked'),
                              (subprocess.TimeoutExpired(['cached'], 1), 'time_budget_exhausted'),
                              (KeyboardInterrupt(), 'interrupted')]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as d:
                args = argparse.Namespace(prerequisites=Path(d)/'prior', output=Path(d)/'run', time_limit=1)
                with patch('triage_selector.prerequisites', return_value={'build':'test'}), \
                     patch('triage_selector.EvidenceGuard') as guard:
                    guard.return_value.run.side_effect = error
                    self.assertEqual(run(args), 2)
                    command = guard.return_value.run.call_args.args[0]
                    self.assertEqual(command[command.index('--phase')+1], 'cached')
                    self.assertEqual(command[command.index('--contexts')+1], '4096')
                    self.assertLessEqual(guard.return_value.run.call_args.kwargs['timeout'], 1)
                report = json.loads((args.output/'summary.json').read_text())
                self.assertEqual(report['status'], status)
                for key in ('complete', 'advance_to_normal_screen', 'full_validation', 'production_promoted'):
                    self.assertFalse(report[key])

    def test_deadline_and_output_directory_are_not_silently_extended_or_reused(self):
        with tempfile.TemporaryDirectory() as d:
            args = argparse.Namespace(prerequisites=Path(d), output=Path(d)/'run', time_limit=601)
            with self.assertRaises(ValueError): run(args)
            args.time_limit = 600; args.output.mkdir()
            with self.assertRaises(FileExistsError): run(args)

    def test_reuses_original_checks_only_when_all_original_dependencies_match(self):
        for changed in (False, True):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as d:
                source=Path(d)/'prior'; source.mkdir()
                output=Path(d)/'out'; output.mkdir()
                previous=dict(build='native', files={'old':'original',str(CONFIG.resolve()):'cfg'})
                current=copy.deepcopy(previous); current['files']['new-tool']='new'
                if changed: current['files']['old']='changed'
                (source/'identity.json').write_text(json.dumps(previous))
                (source/'summary.json').write_text(json.dumps(dict(phases={
                    p:dict(status='passed',files_sha256='sealed') for p in ('recovery','capture')})))
                with patch('triage_selector.identity',return_value=current), \
                     patch('triage_selector.EvidenceGuard') as guard, \
                     patch('triage_selector.sha',return_value='cfg'), \
                     patch('triage_selector.verify_seal') as verify:
                    if changed:
                        with self.assertRaises(ValueError): prerequisites(source,output)
                        verify.assert_not_called()
                    else:
                        self.assertEqual(prerequisites(source,output),current)
                        self.assertEqual(verify.call_count,2)
                    guard.return_value.check_identity.assert_called_once()

    def test_completed_cached_result_can_only_request_a_normal_screen(self):
        with tempfile.TemporaryDirectory() as d:
            args=argparse.Namespace(prerequisites=Path(d)/'prior',output=Path(d)/'run',time_limit=10)
            result=self.result()
            def complete(command, **kwargs):
                path=args.output/'cached-run/cached-4096'; path.mkdir(parents=True)
                (path/'report.json').write_text('{}'); (path/'tokens.json').write_text('[]')
                (path.parent/'summary.json').write_text(json.dumps(dict(phases={
                    'cached-4096':dict(status='passed',result=result,files_sha256='sealed')})))
            with patch('triage_selector.prerequisites',return_value={'build':'test'}), \
                 patch('triage_selector.EvidenceGuard') as guard, \
                 patch('triage_selector.verify_seal'), \
                 patch('triage_selector.check_cached',return_value=result):
                guard.return_value.run.side_effect=complete
                self.assertEqual(run(args),0)
                guard.return_value.run.assert_called_once()
            report=json.loads((args.output/'summary.json').read_text())
            self.assertTrue(report['complete']); self.assertTrue(report['advance_to_normal_screen'])
            self.assertFalse(report['full_validation']); self.assertFalse(report['production_promoted'])


if __name__ == '__main__': unittest.main()
