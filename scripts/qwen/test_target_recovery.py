import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import build_target_recovery as builder
from target_recovery_checks import observe, comparison, short_gate, long_gate, replay_resources, resource_failure
from trial_recovery import recipe, run, correctness_cases, fixture_identity, verified_stage, resumed_stages, capture_checks, external_capture
from qualification_evidence import save, seal
from qualification_evidence import sha
from evidence_index import Index
from mtp_evidence import compare as evidence_compare
import evidence_fixture


def setUpModule():
    evidence_fixture.require(
        'docs/benchmarks/2026-09-15-q4-request-context/capture-02/evidence-files.json',
        'docs/benchmarks/2026-09-16-mtp-direct-output/long-01/case-1-pair-0-on.json',
        'docs/benchmarks/2026-09-16-mtp-direct-output/long-01/case-1.json',
    )


ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'docs/benchmarks/2026-09-16-mtp-direct-output/long-01'


class TargetRecoveryTests(unittest.TestCase):
    def test_resource_message_distinguishes_swap_cleanup_without_relaxing_gate(self):
        memory=dict(physical_footprint_bytes=1024,physical_footprint_peak_bytes=2048,
            compressed_bytes=0,compressed_peak_bytes=0,decompressions=0,system_swap_used_bytes=16*1024**2)
        host=dict(thermal_state=0,low_power_mode=False,power_source='AC Power')
        raw=dict(before=memory,after=dict(memory,system_swap_used_bytes=8*1024**2),
                 host_before=host,host_after=host.copy())
        self.assertFalse(replay_resources(raw)['clean_memory'])
        message=resource_failure(raw,'Replay')
        self.assertIn('system swap decreased: 16777216 to 8388608 bytes',message)
        self.assertIn('no process compression or decompression observed',message)
        raw['after'].update(compressed_peak_bytes=16384,decompressions=2)
        message=resource_failure(raw,'Replay')
        self.assertIn('process compression observed: 16384 bytes',message)
        self.assertIn('process decompressions changed: 0 to 2',message)
        self.assertNotIn('no process compression',message)

    def test_resource_message_covers_native_cycles_and_host_conditions(self):
        memory=dict(compressed_bytes=0,compressed_peak_bytes=0,decompressions=0,system_swap_used_bytes=0)
        host=dict(thermal_state=0,low_power_mode=False,power_source='AC Power')
        raw=dict(before_load=memory,before={'process':memory},after={'process':memory},after_destroy=memory,
                 cycles=[dict(memory_before=memory,memory_after=dict(memory,compressed_peak_bytes=32768))],
                 host_before=host,host_after=dict(host,power_source='Battery Power'))
        message=resource_failure(raw,'Sample')
        self.assertIn('process compression observed: 32768 bytes',message)
        self.assertIn('power=Battery Power',message)
        self.assertNotIn('system swap',message)

    def test_external_fixture_retains_original_capture_resources_and_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();bundle=root/'bundle';bundle.mkdir();manifest=bundle/'manifest.json'
            work=json.loads((BASE/'case-1.json').read_text());save(root/'workload.json',work)
            raw=self.fixture();raw.update(kind='target_recovery_capture_v1',mode='fast-validate',validation=True,
                performance_measurement=False,input_sha256=sha(root/'workload.json'),producer_binary_sha256='b'*64)
            source={k:raw[k] for k in ('request_id','input_sha256','producer_binary_sha256','draft_manifest_sha256')}
            source.update(target_prepared_sha256=raw['before']['prepared']['manifest_sha256'],
                native_build_fingerprint=raw['before']['metal']['build_fingerprint'],kernel_policy=raw['before']['metal']['kernels'])
            save(manifest,dict(source=source,payload_bytes=1234))
            raw['capture']=dict(manifest=str(manifest),sha256=sha(manifest),payload_bytes=1234,
                independent_full_target_prefixes=[1,2,3,4],fixture_limit_bytes=2*1024**3)
            self.assertTrue(capture_checks(raw,manifest)['clean_memory'])
            # Replay may be clean, but the original capture still disqualifies it.
            raw['before']['process']['compressed_peak_bytes']=16384
            save(root/'capture.json',raw)
            ref=dict(kind='target_recovery_capture_source_v1',report=str(root/'capture.json'),
                sha256=sha(root/'capture.json'),manifest_sha256=sha(manifest))
            save(bundle/'capture-source.json',ref);seal(root)
            loaded,_=external_capture(bundle)
            self.assertFalse(capture_checks(loaded,manifest)['clean_memory'])
            save(root/'capture.json',dict(raw,complete=False));seal(root)
            with self.assertRaises(ValueError):external_capture(bundle)
            (bundle/'capture-source.json').unlink()
            with self.assertRaises(FileNotFoundError):external_capture(bundle)

    def test_replay_resources_require_complete_clean_observations(self):
        memory=dict(physical_footprint_bytes=1024,physical_footprint_peak_bytes=2048,
            compressed_bytes=0,compressed_peak_bytes=0,decompressions=4,system_swap_used_bytes=4096)
        host=dict(thermal_state=0,low_power_mode=False,power_source='AC Power')
        raw=dict(before=memory,after=memory.copy(),host_before=host,host_after=host.copy())
        self.assertTrue(replay_resources(raw)['clean_memory'])
        for field in ('compressed_bytes','compressed_peak_bytes','decompressions','system_swap_used_bytes'):
            bad=copy.deepcopy(raw);bad['after'][field]+=1
            self.assertFalse(replay_resources(bad)['clean_memory'])
        for field,value in [('thermal_state',1),('low_power_mode',True),('power_source','Battery Power')]:
            bad=copy.deepcopy(raw);bad['host_after'][field]=value
            self.assertFalse(replay_resources(bad)['clean_host'])
        for field in memory:
            bad=copy.deepcopy(raw);del bad['after'][field]
            with self.assertRaises(ValueError):replay_resources(bad)
        bad=copy.deepcopy(raw);bad['after']['physical_footprint_peak_bytes']=2*1024**3+1
        with self.assertRaises(ValueError):replay_resources(bad)

    def test_fixture_identity_rejects_synthetic_and_changed_sources(self):
        proof=dict(binary_sha256='b'*64,base_native_fingerprint='n'*64)
        policy=dict(q4_decode='reference',q8_decode_rows=2,gdn='original',route_selection='simd',
            token_tile=1,affine_rows=1,gate_pair=False,profile=False,counter_profile=False)
        data=dict(kind='target_recovery_fixture_v1',complete=True,reference_origin='full-target-forward',
            artifact_revision='b2c422f3c643e36f04227a64d61796b44a4b1029',context=8192,layers=48,
            source=dict(producer_binary_sha256='b'*64,native_build_fingerprint='n'*64,target_prepared_sha256='t'*64,
                draft_manifest_sha256='d'*64,input_sha256='i'*64,request_id='i'*64+':test',kernel_policy=policy),
            expected=[dict(keep=k,row_logits_sha256=['h'*64]*k) for k in range(1,5)])
        fixture_identity(data,proof,'t'*64,'d'*64)
        edits=[lambda d:d.update(reference_origin='synthetic-operators'),lambda d:d['expected'].pop(),
            lambda d:d['source']['kernel_policy'].update(profile=True)]
        for field in ('producer_binary_sha256','native_build_fingerprint','target_prepared_sha256','draft_manifest_sha256','input_sha256'):
            edits.append(lambda d,f=field:d['source'].update({f:'changed'}))
        for edit in edits:
            bad=copy.deepcopy(data);edit(bad)
            with self.assertRaises(ValueError):fixture_identity(bad,proof,'t'*64,'d'*64)

    def test_resume_validates_external_stage_reference_before_reuse(self):
        with tempfile.TemporaryDirectory() as d,patch('trial_recovery.verified_stage') as verify:
            path=Path(d).resolve();save(path/'summary.json',{});seal(path)
            item=dict(name='fixtures',directory=d,complete=True,status='fixture_validated',
                summary_sha256=sha(path/'summary.json'),seal_sha256=sha(path/'evidence-files.json'))
            self.assertEqual(resumed_stages(dict(stages=[item]),{}),dict(fixtures=path))
            verify.assert_called_once_with(path,'fixture_validated',{})
            for field in ('summary_sha256','seal_sha256','status'):
                wrong=dict(item,**{field:'changed'})
                with self.assertRaises(ValueError):resumed_stages(dict(stages=[wrong]),{})
            with self.assertRaises(ValueError):resumed_stages(dict(stages=[item,item]),{})

    def test_correctness_prerequisite_requires_all_distinct_cases_and_raw_samples(self):
        # Stub numerical calculations only: exercise the real inventory, source,
        # workload, seal and producer checks without running the model.
        with tempfile.TemporaryDirectory() as d,patch('trial_recovery.observe',return_value=dict(clean_host=True,clean_memory=True)),\
                patch('trial_recovery.comparison',return_value=dict(exact=True)):
            root=Path(d);proof=dict(binary_sha256='b'*64)
            save(root/'producer.json',proof);save(root/'identity.json',dict(files={}))
            pairs=[];samples=[]
            for i,work in enumerate(correctness_cases()):
                save(root/f'case-{i}.json',work)
                for arm in (('full-replay','state-only') if i%2==0 else ('state-only','full-replay')):
                    path=root/f'case-{i}-pair-0-{arm}.json'
                    save(path,dict(target_recovery=arm,producer_binary_sha256='b'*64))
                    samples.append(dict(case=i,pair=0,arm=arm,source=path.name,sha256=sha(path),clean_host=True,clean_memory=True))
                pairs.append(dict(case=i,pair=0,name=work['name'],exact=True))
            raw=dict(kind='mtp_target_recovery_validation_v1',complete=True,status='numerically_validated',
                checked_prefixes=[1,2,3,4],eos_checked=True,pairs=pairs,samples=samples)
            save(root/'summary.json',raw);seal(root)
            verified_stage(root,'numerically_validated',proof)
            for case in (1,4):
                save(root/f'case-{case}.json',correctness_cases()[0]);seal(root)
                with self.assertRaises(ValueError):verified_stage(root,'numerically_validated',proof)
                save(root/f'case-{case}.json',correctness_cases()[case])
            for key in ('pairs','samples'):
                wrong=copy.deepcopy(raw);wrong[key][-1]=wrong[key][0];save(root/'summary.json',wrong);seal(root)
                with self.assertRaises(ValueError):verified_stage(root,'numerically_validated',proof)
            save(root/'summary.json',raw)
            path=root/'case-0-pair-0-full-replay.json';save(path,dict(target_recovery='full-replay',producer_binary_sha256='changed'));seal(root)
            with self.assertRaises(ValueError):verified_stage(root,'numerically_validated',proof)
            path.unlink();seal(root)
            with self.assertRaises(FileNotFoundError):verified_stage(root,'numerically_validated',proof)

    def fixture(self,arm='full-replay'):
        r=json.loads((BASE/'case-1-pair-0-on.json').read_text())
        r.update(kind='native_mtp_continuation_v2',target_recovery=arm,request_id=r['input_sha256']+':test')
        state=arm=='state-only'; captures=rejected=0
        for i,c in enumerate(r['cycles']):
            partial=c['committed_tokens']<c['width'];captures+=c['width']==4;rejected+=partial
            c.update(cycle_id=i,request_id=r['request_id'],checkpoint_save_ns=0,
                target_restore_ns=0,target_repair_ns=c['recovery_ns'] if partial else 0,
                draft_restore_ns=0,draft_catchup_ns=0 if partial else c['recovery_ns'],
                target_recovery_forward_calls=c['committed_tokens'] if partial and not state else 0,
                target_recovery_read_bytes=1024 if partial and not state else 0)
        r['admission']['target_recovery_reserve_bytes']=16*1024**2;r['admission']['combined_bytes']+=16*1024**2
        r['target_recovery_journal']=dict(reserved_bytes=16*1024**2,fixed_allocation_bytes=13*1024**2,
            peak_incremental_bound_bytes=15*1024**2,captures=captures if state else 0,repairs=rejected if state else 0,
            peak_gpu_groups=2 if state else 0,active=False,complete=state)
        return r

    def test_raw_validator_and_comparison(self):
        work=json.loads((BASE/'case-1.json').read_text());a,b=(self.fixture(arm) for arm in ('full-replay','state-only'))
        for r in (a,b):self.assertTrue(observe(r,work,r['input_sha256'],False)['clean_memory'])
        result=comparison(a,b);self.assertEqual(result['eliminated_target_forward_calls'],46)
        self.assertEqual(result['candidate_target_recovery_forward_calls'],0)
        self.assertTrue(result['exact_all_logits_tokens_and_state'])

    def test_new_recovery_summary_revalidates_from_raw_sources(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);work=json.loads((BASE/'case-1.json').read_text());save(root/'case-0.json',work)
            values=[];samples=[];binary_hash='a'*64
            for arm in ('full-replay','state-only'):
                r=self.fixture(arm);r.update(input_sha256=sha(root/'case-0.json'),producer_binary_sha256=binary_hash)
                r['request_id']=r['input_sha256']+':'+arm
                for c in r['cycles']:c['request_id']=r['request_id']
                path=root/(arm+'.json');save(path,r);observed=observe(r,work,r['input_sha256'],False)
                values.append(r);samples.append(dict(case=0,pair=0,arm=arm,source=path.name,sha256=sha(path),**observed))
            save(root/'producer.json',dict(complete=True,kind='target_recovery_producer_v1',binary='fixture-binary',binary_sha256=binary_hash,
                base_native_fingerprint=values[0]['before']['metal']['build_fingerprint']))
            save(root/'identity.json',dict(files={'fixture-binary':binary_hash}))
            save(root/'summary.json',dict(kind='mtp_target_recovery_screen_v1',complete=True,samples=samples,
                pairs=[dict(case=0,pair=0,**comparison(*values))]))
            seal(root);index=Index(':memory:');index.import_paths([root/'summary.json'])
            try:
                result=evidence_compare(index,str(root/'summary.json'),'full-replay','state-only',['target_recovery'])
                self.assertTrue(result['comparable']);self.assertIsNone(result['cases'][0]['confidence_95'])
                self.assertEqual(result['pairs'][0]['candidate_target_recovery_forward_calls'],0)
                summary=json.loads((root/'summary.json').read_text())
                summary.update(stage='early',preliminary=True,full_correctness_stage_passed=False,
                               advancement_allowed=False,production_promoted=False)
                save(root/'summary.json',summary);seal(root);index.import_paths([root/'summary.json'])
                early=evidence_compare(index,str(root/'summary.json'),'full-replay','state-only',['target_recovery'])
                self.assertTrue(early['preliminary']);self.assertFalse(early['advancement_allowed'])
                self.assertFalse(early['full_correctness_stage_passed'])
                for key in ('full_correctness_stage_passed','advancement_allowed','production_promoted'):
                    bad=dict(summary,**{key:True});save(root/'summary.json',bad);seal(root)
                    index.import_paths([root/'summary.json'])
                    with self.assertRaisesRegex(ValueError,'Early screening cannot claim'):
                        evidence_compare(index,str(root/'summary.json'),'full-replay','state-only',['target_recovery'])
            finally:index.close()

    def test_reject_reads_forwards_unbounded_journal_missing_modes_and_dirty_state(self):
        work=json.loads((BASE/'case-1.json').read_text())
        edits=[lambda r:r['cycles'][0].update(target_recovery_forward_calls=1),
            lambda r:r['cycles'][0].update(target_recovery_read_bytes=1),
            lambda r:r['target_recovery_journal'].update(peak_incremental_bound_bytes=17*1024**2),
            lambda r:r['target_recovery_journal'].update(captures=0),
            lambda r:r['direct_output_after'].update(enabled=False),
            lambda r:r['final_target_state'].update(valid=False),
            lambda r:r.update(complete=False),lambda r:r['cycles'][0].pop('draft_catchup_ns')]
        for edit in edits:
            r=self.fixture('state-only');edit(r)
            with self.subTest(edit=edit),self.assertRaises((ValueError,KeyError)):
                observe(r,work,r['input_sha256'],False)

    def test_outputs_and_capacity_must_match(self):
        for key in ('row_logits_sha256','committed_token_ids'):
            a,b=self.fixture(),self.fixture('state-only');b[key][0]='changed'
            with self.assertRaises(ValueError):comparison(a,b)
        a,b=self.fixture(),self.fixture('state-only');b['target_recovery_journal']['fixed_allocation_bytes']-=1
        with self.assertRaises(ValueError):comparison(a,b)

    def test_screen_stops_and_does_not_treat_cases_as_repetitions(self):
        self.assertFalse(short_gate([dict(ratio=.951)]))
        self.assertTrue(short_gate([dict(ratio=.94)]))
        self.assertFalse(short_gate([dict(ratio=.90),dict(ratio=1.01)]))
        self.assertFalse(short_gate([dict(ratio=.949),dict(ratio=.96)]))
        self.assertTrue(short_gate([dict(ratio=.94),dict(ratio=.95)]))
        pairs=[dict(case=c,pair=p,ratio=.94) for c in range(3) for p in range(2)]
        self.assertTrue(long_gate(pairs)['passed'])
        with self.assertRaises(ValueError):long_gate(pairs[:3])
        for p in pairs:
            if p['case']==2:p['ratio']=1.03
        self.assertFalse(long_gate(pairs)['passed'])

    def test_recipe_cannot_silently_weaken_limits(self):
        path=ROOT/'scripts/qwen/recipes/target_recovery.json';r=recipe(path)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'recipe.json'
            for key,value in [('memory_bytes',22*1024**3),('target_slots',1600),('thresholds',{})]:
                wrong=copy.deepcopy(r);wrong[key]=value;p.write_text(json.dumps(wrong))
                with self.assertRaises(ValueError):recipe(p)

    def test_generated_runtime_counts_actual_forward_entry_and_isolates_production(self):
        output=Path('/tmp/target-recovery-source-test');sources=builder.generated(output)
        model=sources[output/'model.cpp'];probe=sources[output/'probe.cpp']
        self.assertEqual(model.count('++recovery_target_forward_calls;'),1)
        self.assertIn('target_recovery_forward_calls=recovery_target_forward_calls-forwards_before',probe)
        self.assertIn('checkpoint.commit_prefix',probe)
        self.assertIn('replay_recovery_fixture(argv[2],hash_file(argv[0]))',probe)
        self.assertNotIn('recovery_target_forward_calls',(ROOT/'src/engine/model.cpp').read_text())

    def test_trial_preserves_blocker_and_stops_before_later_stages(self):
        def stage(output,*args,**kwargs):
            output=Path(output);output.mkdir();value=dict(complete=False,status='resource_blocked',error='memory gate')
            save(output/'summary.json',value);seal(output);return value
        with tempfile.TemporaryDirectory() as d,patch('trial_recovery.builder.build'),patch('trial_recovery.fixture_stage',side_effect=stage),\
                patch('trial_recovery.correctness_stage') as correctness,patch('trial_recovery.screen_stage') as screen:
            result=run(ROOT/'scripts/qwen/recipes/target_recovery.json',Path(d)/'trial')
            self.assertEqual(result['status'],'resource_blocked');self.assertFalse(result['complete'])
            self.assertFalse(result['historical_timing_reused']);correctness.assert_not_called();screen.assert_not_called()
            self.assertTrue((Path(d)/'trial/evidence-files.json').is_file())

    def test_trial_rejection_stops_long_screen_without_promotion(self):
        def stage(output,status):
            output=Path(output);output.mkdir();value=dict(complete=True,status=status)
            save(output/'summary.json',value);seal(output);return value
        with tempfile.TemporaryDirectory() as d,patch('trial_recovery.builder.build'),\
                patch('trial_recovery.fixture_stage',side_effect=lambda o,*a:stage(o,'fixture_validated')),\
                patch('trial_recovery.correctness_stage',side_effect=lambda o,*a:stage(o,'numerically_validated')),\
                patch('trial_recovery.screen_stage',side_effect=lambda o,*a:stage(o,'insufficient_short_gain')) as screen:
            result=run(ROOT/'scripts/qwen/recipes/target_recovery.json',Path(d)/'trial')
            self.assertEqual(result['status'],'rejected');self.assertTrue(result['complete']);self.assertFalse(result['production_promoted'])
            self.assertEqual(screen.call_count,1);self.assertEqual([s['name'] for s in result['stages']],['fixtures','correctness','short'])


if __name__=='__main__':unittest.main()
