import copy
import json
from pathlib import Path
import tempfile
import unittest
import shutil

from evidence_index import Index
from mtp_evidence import account, compare, opportunity, next_experiment
from qualification_evidence import seal
import evidence_fixture


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'docs/benchmarks/2026-09-16-mtp-direct-output/long-01'


class MtpEvidenceTests(unittest.TestCase):
    def raw(self):
        return json.loads((BASE/'case-1-pair-0-on.json').read_text())

    def test_real_reports_reproduce_comparisons_and_ceilings(self):
        evidence_fixture.require(
            'docs/benchmarks/2026-09-16-mtp-direct-output/long-01/case-0-pair-0-off.json',
            'docs/benchmarks/2026-09-16-mtp-direct-output/long-01/case-1-pair-0-on.json',
        )
        with tempfile.TemporaryDirectory() as d:
            index = Index(Path(d)/'index.sqlite')
            index.import_paths([BASE/'summary.json'])
            result = compare(index,str(BASE/'summary.json'),'off','on',['direct_output'])
            self.assertEqual(len(result['cases']),3)
            self.assertTrue(all(c['confidence_95'] is None for c in result['cases']))
            out = opportunity(index,str(BASE/'summary.json'))
            actual = [r['opportunity']['optimistic_zero_phase_tps'] for r in out['runs'] if r['arm']=='on']
            for a,b in zip(actual,[4.757218903055051,3.8080019828980327,4.341916604248082]):self.assertAlmostEqual(a,b)
            index.close()

    def test_legacy_missing_components_and_known_replay_count(self):
        evidence_fixture.require(
            'docs/benchmarks/2026-09-16-mtp-direct-output/long-01/case-0-pair-0-off.json',
            'docs/benchmarks/2026-09-16-mtp-direct-output/long-01/case-1-pair-0-on.json',
        )
        a=account(self.raw());self.assertEqual(a['repeated_target_rows'],46)
        self.assertEqual(a['discarded_verification_rows'],42)
        self.assertIsNone(a['recovery_children_ms_per_token'])
        self.assertFalse(a['checkpoint_save_measured'])
        self.assertIsNone(a['component_ms_per_token']['checkpoint_save'])
        self.assertIsNone(a['phase_ns']['checkpoint_save'])

    def test_invalid_coverage_and_timing_rejected(self):
        evidence_fixture.require(
            'docs/benchmarks/2026-09-16-mtp-direct-output/long-01/case-0-pair-0-off.json',
            'docs/benchmarks/2026-09-16-mtp-direct-output/long-01/case-1-pair-0-on.json',
        )
        for kind in ('overlap','offset','negative','sum','throughput','stop','kind','bool'):
            r=self.raw()
            if kind=='overlap':r['cycles'][0]['draft_ns']=r['cycles'][0]['wall_ns']
            elif kind=='offset':r['cycles'][0]['offset']+=1
            elif kind=='negative':r['cycles'][0]['verify_ns']=-1
            elif kind=='sum':r['generated_tokens']-=1
            elif kind=='throughput':r['tokens_per_second']+=1
            elif kind=='stop':r['stop_reason']='eos'
            elif kind=='kind':r['kind']='unknown'
            else:r['cycles'][0]['width']=True
            with self.subTest(kind=kind),self.assertRaises(ValueError):account(r)

    def test_partial_runs_remain_diagnostic(self):
        evidence_fixture.require(
            'docs/benchmarks/2026-09-16-mtp-direct-output/long-01/case-0-pair-0-off.json',
            'docs/benchmarks/2026-09-16-mtp-direct-output/long-01/case-1-pair-0-on.json',
        )
        r=self.raw();r['complete']=False;r['cycles']=r['cycles'][:2]
        a=account(r);self.assertFalse(a['complete']);self.assertIsNone(a['measured_tps'])

    def test_disturbed_or_missing_resources_cannot_supply_a_raw_run_ceiling(self):
        evidence_fixture.require(
            'docs/benchmarks/2026-09-16-mtp-direct-output/long-01/case-0-pair-0-off.json',
            'docs/benchmarks/2026-09-16-mtp-direct-output/long-01/case-1-pair-0-on.json',
        )
        for mode in ('compression', 'power', 'missing'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                raw = self.raw()
                if mode == 'compression': raw['after_destroy']['compressed_peak_bytes'] = 16384
                elif mode == 'power': raw['host_after']['power_source'] = 'Battery Power'
                else: raw['after_destroy'].pop('decompressions')
                path = Path(d)/'raw.json';path.write_text(json.dumps(raw))
                index = Index(Path(d)/'index.sqlite');index.import_paths([path])
                try:
                    result = opportunity(index, str(path))['runs'][0]
                    self.assertIsNot(result['resource_qualified'], True)
                    self.assertIsNone(result['measured_tps'])
                    self.assertIsNone(result['opportunity']['optimistic_zero_phase_tps'])
                    self.assertIsNone(result['opportunity']['gap_ms_per_token'])
                    self.assertIn('fresh clean sample', next_experiment(index, str(path))['smallest_experiment'])
                finally: index.close()

    def test_eos_shortened_run(self):
        evidence_fixture.require(
            'docs/benchmarks/2026-09-16-mtp-direct-output/long-01/case-0-pair-0-off.json',
            'docs/benchmarks/2026-09-16-mtp-direct-output/long-01/case-1-pair-0-on.json',
        )
        r=self.raw();r['requested_tokens']=256;r['stop_reason']='eos';r['eos_ids']=[r['committed_token_ids'][-1]]
        self.assertTrue(account(r)['complete'])

    def test_nested_parts_not_double_counted(self):
        evidence_fixture.require(
            'docs/benchmarks/2026-09-16-mtp-direct-output/long-01/case-0-pair-0-off.json',
            'docs/benchmarks/2026-09-16-mtp-direct-output/long-01/case-1-pair-0-on.json',
        )
        r=self.raw();r.update(kind='native_mtp_continuation_v2',request_id='test')
        for i,c in enumerate(r['cycles']):
            c.update(cycle_id=i,request_id='test',checkpoint_save_ns=0,target_restore_ns=0,
                     target_repair_ns=c['recovery_ns'],draft_restore_ns=0,draft_catchup_ns=0,
                     target_recovery_forward_calls=c['committed_tokens'] if c['committed_tokens']<c['width'] else 0,
                     target_recovery_read_bytes=0)
        a=account(r);self.assertEqual(a['recovery_part_ns']['target_repair'],a['phase_ns']['recovery'])
        missing=copy.deepcopy(r);missing['cycles'][0].pop('checkpoint_save_ns')
        with self.assertRaises(ValueError):account(missing)
        r['cycles'][0]['draft_restore_ns']=1
        with self.assertRaises(ValueError):account(r)

    def test_changed_raw_duplicate_pair_missing_phase_and_false_summary_fail(self):
        evidence_fixture.require(
            'docs/benchmarks/2026-09-16-mtp-direct-output/long-01/case-0-pair-0-off.json',
            'docs/benchmarks/2026-09-16-mtp-direct-output/long-01/case-1-pair-0-on.json',
        )
        for mode in ('changed_raw','duplicate','missing_phase','false_summary','instrumentation','unfinished','wrong_axis'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                root=Path(d)/'evidence';shutil.copytree(BASE,root)
                path=root/'summary.json';summary=json.loads(path.read_text())
                if mode=='duplicate':summary['samples'].append(summary['samples'][0])
                if mode=='false_summary':summary['pairs'][0]['ratio']=.1
                if mode=='unfinished':summary['complete']=False
                if mode in ('changed_raw','missing_phase','instrumentation'):
                    sample=root/summary['samples'][0]['source'];raw=json.loads(sample.read_text())
                    if mode=='changed_raw':raw['tokens_per_second']+=1
                    elif mode=='missing_phase':raw['cycles'][0].pop('verify_ns')
                    else:raw['before']['metal']['kernels']['profile']=True
                    sample.write_text(json.dumps(raw))
                    if mode!='changed_raw':
                        from qualification_evidence import sha
                        summary['samples'][0]['sha256']=sha(sample)
                path.write_text(json.dumps(summary));seal(root);index=Index(Path(d)/'index.sqlite');index.import_paths([path])
                try:
                    with self.assertRaises((ValueError,KeyError)):
                        compare(index,str(path),'off','on',['wrong'] if mode=='wrong_axis' else ['direct_output'])
                finally:index.close()


if __name__=='__main__':unittest.main()
