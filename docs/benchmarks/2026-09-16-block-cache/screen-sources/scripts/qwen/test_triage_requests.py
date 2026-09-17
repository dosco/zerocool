import argparse
import copy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import triage_requests as short
from qualification_evidence import ResourceBlocked


class ShortRequestTriageTest(unittest.TestCase):
    def rows(self):
        return [dict(pair=0,configuration=name,name='append_128',output_token_ids=list(range(8)),
                     request_ms=100*factor,ttft_ms=80*factor,decode_ms_per_token=2*factor,
                     report=name+'.json')
                for name,factor in [('cpu-selection',1),('gpu-selection',.98)]]

    def test_gate_is_exploratory_and_stops_regressions(self):
        result=short.decide(self.rows())
        self.assertTrue(result['advance_to_full_validation'])
        self.assertIsNone(result['confidence_95'])
        self.assertFalse(result['normal_request_latency_qualified']);self.assertFalse(result['production_promoted'])
        for metric,value in [('request_ms',100),('ttft_ms',84),('decode_ms_per_token',2.1)]:
            rows=self.rows();rows[1][metric]=value
            self.assertFalse(short.decide(rows)['advance_to_full_validation'])

    def test_missing_duplicate_changed_tokens_and_invalid_timings_cannot_pass(self):
        rows=self.rows()
        for change in (lambda r:r.pop(),lambda r:r.append(copy.deepcopy(r[0])),
                       lambda r:r[1].update(configuration='cpu-selection'),
                       lambda r:r[1].update(output_token_ids=[999]*8),
                       lambda r:r[1].update(request_ms=float('nan')),
                       lambda r:r[1].update(ttft_ms=0)):
            data=copy.deepcopy(rows);change(data)
            with self.assertRaises(ValueError):short.decide(data)

    def test_prerequisite_rejects_blocked_negative_or_incomplete_cached_runs(self):
        for flags in (dict(status='not_promising'),dict(complete=False),dict(advance_to_normal_screen=False),dict(kind='other')):
            with tempfile.TemporaryDirectory() as temp:
                source=Path(temp);report=dict(kind='selector_cached_triage',complete=True,status='promising',advance_to_normal_screen=True)
                report.update(flags);(source/'summary.json').write_text(json.dumps(report))
                with patch.object(short,'EvidenceGuard') as guard:
                    with self.assertRaises(ValueError):short.prerequisites(source,source)
                    guard.assert_not_called()

    def test_positive_prerequisite_is_rederived_and_original_dependencies_remain_bound(self):
        cached=dict(exact=True,pairs=5,application_read_bytes=0,prompt_tokens=4096,comparison_axis='sparse_selection',
                    latency_ratio=dict(low=.97,median=.98,high=.99,pairs=5))
        for tampered in (False,True):
            with self.subTest(tampered=tampered),tempfile.TemporaryDirectory() as temp:
                source=Path(temp)/'cached';directory=source/'cached-run/cached-4096';directory.mkdir(parents=True)
                output=Path(temp)/'normal';output.mkdir()
                old=dict(build='native',files={'native-source':'unchanged'})
                for p,data in [(source/'identity.json',old),(directory/'report.json',{}),(directory/'tokens.json',[]),
                               (directory/'evidence-files.json',{}),
                               (source/'cached-run/summary.json',dict(phases={'cached-4096':dict(status='passed',files_sha256='seal',result=cached)}))]:
                    p.write_text(json.dumps(data))
                claimed=copy.deepcopy(cached)
                if tampered:claimed['latency_ratio']['median']=.90
                (source/'summary.json').write_text(json.dumps(dict(kind='selector_cached_triage',complete=True,
                    status='promising',advance_to_normal_screen=True,cached_result=claimed)))
                current=copy.deepcopy(old);current['files']['new-tool']='added'
                with patch.object(short,'identity',return_value=current),patch.object(short,'EvidenceGuard') as guard, \
                     patch.object(short,'verify_seal') as verified,patch.object(short,'check_cached',return_value=cached):
                    if tampered:
                        with self.assertRaises(ValueError):short.prerequisites(source,output)
                    else:
                        evidence=short.prerequisites(source,output)
                        bound=short.load(output/'prerequisites.json')['files_sha256']
                        for name,digest in bound.items():self.assertEqual(evidence['files'][name],digest)
                        self.assertIn(str((source/'summary.json').resolve()),bound)
                    guard.return_value.check_identity.assert_called_once();verified.assert_called_once()

    def test_total_deadline_and_failures_never_launch_full_validation(self):
        for error,status in [(ResourceBlocked('memory'),'resource_blocked'),
                             (subprocess.TimeoutExpired('normal',1),'time_budget_exhausted'),
                             (KeyboardInterrupt(),'interrupted')]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp:
                root=Path(temp);(root/'.cache').mkdir()
                args=argparse.Namespace(output=root/'out',cached_triage=root/'cached',time_limit=1)
                with patch.object(short,'ROOT',root),patch.object(short,'prerequisites',return_value={'build':'test'}), \
                     patch.object(short,'EvidenceGuard') as guard,contextlib.redirect_stdout(io.StringIO()):
                    guard.return_value.run.side_effect=error
                    self.assertEqual(short.run(args),2)
                    command=guard.return_value.run.call_args.args[0]
                    self.assertIn('--worker',command)
                    self.assertNotIn('--phase',command)
                    self.assertLessEqual(guard.return_value.run.call_args.kwargs['timeout'],1)
                report=short.load(args.output/'summary.json')
                self.assertEqual(report['status'],status);self.assertFalse(report['complete'])
                self.assertFalse(report['advance_to_full_validation'])

    def test_complete_pair_returns_decision_without_launching_more_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'.cache').mkdir()
            args=argparse.Namespace(output=root/'out',cached_triage=root/'cached',time_limit=10)
            with patch.object(short,'ROOT',root),patch.object(short,'prerequisites',return_value={'build':'test'}), \
                 patch.object(short,'EvidenceGuard') as guard,patch.object(short,'check_requests',return_value=(self.rows(),short.decide(self.rows()))), \
                 patch.object(short,'seal',return_value='sealed'),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(short.run(args),0)
                guard.return_value.run.assert_called_once()
            report=short.load(args.output/'summary.json')
            self.assertTrue(report['complete']);self.assertTrue(report['advance_to_full_validation'])
            self.assertFalse(report['production_promoted'])

    def test_deadline_and_existing_output_are_not_silently_changed(self):
        with tempfile.TemporaryDirectory() as temp:
            args=argparse.Namespace(output=Path(temp),cached_triage=Path(temp),time_limit=901)
            with self.assertRaises(ValueError):short.run(args)
            args.time_limit=900
            with self.assertRaises(FileExistsError):short.run(args)

    def test_revalidation_binds_workload_instrumentation_and_both_state_boundaries(self):
        rows=self.rows();evidence=dict(build='build',artifact_revision='artifact',configurations=[],budget_bytes=12*1024**3)
        workload=short.normal.workloads([1,2],8)['append_128']
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            report=dict(complete=True,measurements=rows,instrumentation=dict(profile=False,metal_api_validation=False,metal_shader_validation=False))
            for row in rows:
                (directory/row['report']).write_text(json.dumps(dict(workloads=workload,runs=[dict(before={},after={}),dict(before={},after={})])))
            def check(report):
                (directory/'summary.json').write_text(json.dumps(report))
                with patch.object(short.normal,'import_pairs',return_value=rows) as imported, \
                     patch.object(short,'check_machine') as machine, \
                     patch.object(short,'normal_identity',return_value={}), \
                     patch.object(short,'WORKLOAD',directory/'seed.json'):
                    (directory/'seed.json').write_text(json.dumps([dict(tokens=[1,2])]))
                    result=short.check_requests(directory,evidence)
                    self.assertEqual(machine.call_count,8)
                    self.assertEqual(imported.call_args.args[7],8)
                    return result
            self.assertTrue(check(report)[1]['advance_to_full_validation'])
            report['instrumentation']['metal_api_validation']=True
            with self.assertRaises(ValueError):check(report)
            report['instrumentation']['metal_api_validation']=False
            p=directory/rows[0]['report'];raw=json.loads(p.read_text());raw['workloads'][1]['tokens'].pop();p.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):check(report)


if __name__=='__main__':unittest.main()
