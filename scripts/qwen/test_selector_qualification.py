import copy
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import benchmark_exact as normal
from qualification_evidence import (EvidenceGuard,ResourceBlocked,GiB,sha,save,seal,verify_seal,import_sealed)
from selector_qualification import configurations,screen_gate,stage_complete,check_capture,run
import test_exact_benchmark as exact_fixture


class SelectorGateTest(unittest.TestCase):
    def screen(self,ratio=.98):
        rows=[dict(pair=p,configuration=c,name=n,ttft_ms=100,decode_ms_per_token=10,
                   request_ms=100*(ratio if c=='gpu-selection' else 1),output_token_ids=list(range(64)))
              for p in range(2) for c in ('cpu-selection','gpu-selection')
              for n in ('prompt_2k','prompt_4k','append_128','prompt_7k')]
        return dict(complete=True,mode='screen',comparison_purpose='experiment',measurements=rows)

    def test_only_selection_changes_and_tiles_stay_full(self):
        a,b=configurations();self.assertEqual(a['sparse_selection'],'cpu');self.assertEqual(b['sparse_selection'],'gpu')
        for c in (a,b):self.assertEqual(c['attention_score_tiles'],'full')
        for c in (a,b):c.pop('sparse_selection');c.pop('name')
        self.assertEqual(a,b)

    def test_screen_gate_is_triage_not_qualification(self):
        r=screen_gate(self.screen());self.assertTrue(r['advance_to_paired']);self.assertFalse(r['confidence_claim'])
        self.assertFalse(screen_gate(self.screen(.995))['advance_to_paired'])
        self.assertFalse(screen_gate(self.screen(1.01))['advance_to_paired'])
        r=self.screen()
        for row in r['measurements']:
            if row['configuration']=='gpu-selection' and row['pair']==1:row['request_ms']=100.1
        self.assertFalse(screen_gate(r)['advance_to_paired'])
        r=self.screen()
        for row in r['measurements']:
            if row['configuration']=='gpu-selection' and row['name']=='prompt_7k':row['ttft_ms']=104
        self.assertFalse(screen_gate(r)['advance_to_paired'])

    def test_rejects_incomplete_duplicate_changed_and_invalid_evidence(self):
        changes=[lambda r:r.update(complete=False),lambda r:r['measurements'].pop(),
                 lambda r:r['measurements'].__setitem__(0,r['measurements'][1]),
                 lambda r:r['measurements'][0].update(output_token_ids=[7]*64)]
        for metric in ('request_ms','ttft_ms','decode_ms_per_token'):
            for value in (0,-1,True,float('nan'),float('inf')):
                changes.append(lambda r,m=metric,v=value:r['measurements'][0].update(**{m:v}))
        for change in changes:
            r=self.screen();change(r)
            with self.assertRaises(ValueError):screen_gate(r)

    def test_negative_screen_can_finish_only_with_correctness_and_profile(self):
        phases={name:dict(status='passed') for name in ('recovery','capture','qualify','cached-4096','cached-2048','cached-7168','screen','profile')}
        phases['screen']['result']=dict(advance_to_paired=False)
        self.assertTrue(stage_complete(phases))
        phases['screen']['result']['advance_to_paired']=True;self.assertFalse(stage_complete(phases))
        phases['paired']=dict(status='passed',result=dict(stage_latency_passed=False));self.assertTrue(stage_complete(phases))
        phases['recovery']['status']='resource_blocked';self.assertFalse(stage_complete(phases))


class EvidenceGuardTest(unittest.TestCase):
    def test_disk_reserve_and_output_cap(self):
        with tempfile.TemporaryDirectory() as d:
            guard=EvidenceGuard({},Path(d))
            with patch('qualification_evidence.shutil.disk_usage') as usage:
                usage.return_value.free=7*GiB-1
                with self.assertRaises(ResourceBlocked):guard.check_resources(initial=True)
                usage.return_value.free=7*GiB;guard.check_resources(initial=True)
                usage.return_value.free=5*GiB-1
                with self.assertRaises(ResourceBlocked):guard.check_resources()
                usage.return_value.free=7*GiB
                with patch('qualification_evidence.tree_bytes',return_value=2*GiB):
                    with self.assertRaises(ResourceBlocked):guard.check_resources()

    def test_post_subprocess_source_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'input';source.write_text('before')
            evidence=dict(root=d,build='build',files={str(source):sha(source)},assets={})
            guard=EvidenceGuard(evidence,root)
            with patch('qualification_evidence.build_fingerprint',return_value='build'),patch.object(guard,'check_resources'):
                with (root/'log').open('w') as log:
                    with self.assertRaisesRegex(ValueError,'Frozen'):
                        guard.run([sys.executable,'-c','from pathlib import Path; import sys; Path(sys.argv[1]).write_text("changed")',source],stdout=log,timeout=10)

    def test_wrong_build_is_rejected_before_launch(self):
        guard=EvidenceGuard(dict(root='.',build='old',files={},assets={}),Path('.'))
        with patch('qualification_evidence.build_fingerprint',return_value='new'),patch('qualification_evidence.subprocess.Popen') as launch:
            with self.assertRaises(ValueError):guard.run(['unused'],stdout=None,timeout=1)
            launch.assert_not_called()

    def test_timeout_drains_subprocess(self):
        with tempfile.TemporaryDirectory() as d:
            guard=EvidenceGuard(dict(root=d,build='build',files={},assets={}),Path(d))
            with patch('qualification_evidence.build_fingerprint',return_value='build'),patch.object(guard,'check_resources'):
                with (Path(d)/'log').open('w') as log:
                    with self.assertRaises(subprocess.TimeoutExpired):guard.run([sys.executable,'-c','import time; time.sleep(10)'],stdout=log,timeout=.05)

    def test_nested_admission_error_preserves_resource_blocked_status(self):
        with tempfile.TemporaryDirectory() as d:
            guard=EvidenceGuard(dict(root=d,build='build',files={},assets={}),Path(d))
            with patch('qualification_evidence.build_fingerprint',return_value='build'),patch.object(guard,'check_resources'):
                for message,error in [('qualification_evidence.ResourceBlocked: Native memory/storage admission failed; see candidate.log',ResourceBlocked),
                                      ('ValueError: changed model output',subprocess.CalledProcessError)]:
                    with (Path(d)/'log').open('w') as log:
                        with self.assertRaises(error):
                            guard.run([sys.executable,'-c','import sys; print(sys.argv[1]); sys.exit(1)',message],stdout=log,timeout=10)

    def test_sealed_evidence_rejects_corruption_and_escape(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.mkdir();(source/'raw').write_text('correct')
            digest=seal(source);verify_seal(source,digest)
            import_sealed(source,root/'copy',digest);verify_seal(root/'copy',digest)
            (source/'extra').write_text('unlisted')
            with self.assertRaisesRegex(ValueError,'inventory'):verify_seal(source,digest)
            (source/'extra').unlink()
            (source/'raw').write_text('corrupt')
            with self.assertRaises(ValueError):verify_seal(source,digest)
            (source/'link').symlink_to(root/'copy/raw')
            with self.assertRaises(ValueError):seal(source)

    def test_blocked_stage_writes_unfinished_report_without_launching_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'experiment'
            args=argparse.Namespace(output=output,resume_from=None,screen=None,contexts=None,phase='all')
            with patch('selector_qualification.identity',return_value=dict(build='test',files={})), \
                 patch('selector_qualification.ROOT',Path(d)), \
                 patch.object(EvidenceGuard,'check_identity'), \
                 patch.object(EvidenceGuard,'check_resources',side_effect=ResourceBlocked('disk reserve')), \
                 patch('qualification_evidence.subprocess.Popen') as launch:
                with self.assertRaises(ResourceBlocked):run(args)
            report=json.loads((output/'summary.json').read_text())
            self.assertFalse(report['complete']);self.assertFalse(report['production_promoted'])
            self.assertEqual(report['status'],'resource_blocked');self.assertFalse(report['phases'])
            launch.assert_not_called()


class CaptureEvidenceTest(unittest.TestCase):
    def test_exact_geometry_hashes_and_all_arms_are_required(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);evidence=dict(build='test',artifact_revision='artifact');cases=[]
            sizes=dict(q=6144*4,keys=4097*512*4,values=4097*512*4,qg=12288*4,index_scores=1024*4)
            for layer in (3,47):
                tensors={}
                for name,size in sizes.items():
                    path=root/f'{layer}-{name}'
                    with path.open('wb') as stream:stream.truncate(size)
                    tensors[name]=dict(file=path.name,bytes=size,sha256=sha(path))
                cases.append(dict(layer=layer,phase='decode',tokens=1,offset=4096,length=4097,tensors=tensors))
            manifest=dict(kind='sparse_attention_fixture_v1',build_fingerprint='test',artifact_revision='artifact',
                          bytes=2*sum(sizes.values()),cases=cases)
            replay=dict(manifest,kind='sparse_attention_replay_v1',passed=True,
                cases=[dict(c,exact=True,arms=[dict(arm=a,hashes=['score','probability','value']) for a in range(3)]) for c in cases])
            self.assertEqual(check_capture(manifest,replay,'prompt_4k',evidence,root),manifest['bytes'])
            changes=[lambda m,r:m['cases'][0].update(layer=47),lambda m,r:m['cases'][0].update(offset=4095),
                     lambda m,r:r['cases'][0]['arms'].pop(),lambda m,r:r['cases'][0]['arms'][1].update(hashes=['changed']),
                     lambda m,r:m['cases'][0]['tensors']['q'].update(sha256='changed')]
            for change in changes:
                m,r=copy.deepcopy(manifest),copy.deepcopy(replay);change(m,r)
                with self.assertRaises(ValueError):check_capture(m,r,'prompt_4k',evidence,root)


class PairResumeTest(unittest.TestCase):
    def fixture(self,root):
        source=root/'source';source.mkdir();destination=root/'destination';destination.mkdir()
        config=normal.configurations()[0];candidate=dict(config,name='candidate')
        configs=[config,candidate];cases={'prompt_2k':[dict(name='prompt_2k',tokens=[1,2],max_tokens=256)]}
        identity=dict(build='native',budget_bytes=8*GiB,mode='screen',configurations=configs)
        observations=[]
        for pair in (0,1):
            for c in (configs if pair==0 else configs[::-1]):
                raw=exact_fixture.ExactBenchmarkTest().fixture();name=f"{pair}-{c['name']}";path=source/(name+'.json')
                save(path,raw);save(source/(name+'-admission.json'),dict(current_admission=dict(limit_bytes=8*GiB,panel_tokens=512)))
                save(source/(name+'-workload.json'),cases['prompt_2k'])
                row=normal.validate(raw,c,dict(build='native',revision='revision'),8*GiB,256)
                row.update(pair=pair,configuration=c['name'],report=path.name,report_sha256=sha(path),
                    admission_report=name+'-admission.json',admission_report_sha256=sha(source/(name+'-admission.json')),
                    workload_report=name+'-workload.json',workload_report_sha256=sha(source/(name+'-workload.json')))
                observations.append(row)
        save(source/'summary.json',dict(identity,complete=False,measurements=observations))
        return source,destination,identity,configs,cases,observations

    def resume(self,fixture):
        source,dest,identity,configs,cases,_=fixture
        return normal.import_pairs(source,dest,identity,configs,cases,2,dict(build='native',revision='revision'),256)

    def test_partial_pair_is_excluded_and_complete_pair_preserves_order(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));source,dest,identity,configs,cases,rows=f
            save(source/'summary.json',dict(identity,complete=False,measurements=rows[:-1]))
            resumed=self.resume(f);self.assertEqual([r['pair'] for r in resumed],[0,0])
            self.assertEqual([r['configuration'] for r in resumed],[c['name'] for c in configs])
            self.assertFalse(any(p.name.startswith('1-') for p in dest.iterdir()))
            self.assertEqual(sha(source/rows[0]['report']),sha(dest/rows[0]['report']))

    def test_resume_rejects_wrong_identity_corruption_duplicate_and_changed_measurement(self):
        for change in ('identity','corrupt','duplicate','timing','workload','escape'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                f=self.fixture(Path(d));source,dest,identity,configs,cases,rows=f
                saved=json.loads((source/'summary.json').read_text())
                if change=='identity':saved['build']='stale'
                if change=='corrupt':(source/rows[0]['report']).write_text('{}')
                if change=='duplicate':saved['measurements'].append(rows[0])
                if change=='timing':saved['measurements'][0]['request_ms']=1
                if change=='workload':
                    path=source/rows[0]['workload_report'];save(path,[]);saved['measurements'][0]['workload_report_sha256']=sha(path)
                if change=='escape':saved['measurements'][0]['report']='../outside'
                save(source/'summary.json',saved)
                with self.assertRaises((ValueError,FileNotFoundError)):self.resume(f)
                self.assertFalse(list(dest.iterdir()))

    def test_validation_only_does_not_write_or_relabel(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));source,dest,identity,configs,cases,rows=f
            result=normal.import_pairs(source,dest,identity,configs,cases,2,dict(build='native',revision='revision'),256,validation_only=True)
            self.assertEqual(result,rows);self.assertFalse(list(dest.iterdir()))

    def test_abrupt_exit_can_resume_complete_pairs_from_progress(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));source=f[0]
            (source/'summary.json').rename(source/'progress.json')
            self.assertEqual(self.resume(f),f[-1])

if __name__=='__main__':unittest.main()
