import copy
import unittest
from unittest.mock import patch

from benchmark_exact import config_args
from qualify_exact_sessions import check_configuration
from screen_decode_scratch import configs,decide,observations,correctness,state_cases,SCRATCH_CHECKS,check_state_workspace
from screen_cache import CHECKS


class DecodeScratchTest(unittest.TestCase):
    def test_only_tail_changes_and_configuration_is_checked(self):
        a,b=configs()
        self.assertEqual({k for k in a if a[k]!=b[k]},{'name','decode_scratch'})
        self.assertIn('--decode-scratch',config_args(b))
        state=dict(execution=dict(decode_scratch='reuse'),metal=dict(kernels={}),ready_group=4,chunk_tokens=128,io_workers=8)
        with self.assertRaisesRegex(ValueError,'execution'):check_configuration(state,{})
        check_configuration(state,dict(decode_scratch='reuse'))

    def test_rejects_gain_that_does_not_help_decode_or_regresses_ttft(self):
        rows=[]
        for p,name in ((0,'control'),(0,'candidate'),(1,'candidate'),(1,'control')):
            c=name=='candidate'
            rows.append(dict(pair=p,configuration=name,requests=[dict(request_ms=100 if not c else 90,
                time_to_first_token_ms=50,decode_wall_ms=50 if not c else 40,decode_memory_disturbance=False)]*2))
        self.assertTrue(decide(rows)['advance_to_confirmation'])
        for mutate in (lambda r:r[1]['requests'][0].update(decode_wall_ms=55),
                       lambda r:r[1]['requests'][0].update(time_to_first_token_ms=55)):
            changed=copy.deepcopy(rows);mutate(changed)
            self.assertFalse(decide(changed)['advance_to_confirmation'])
        with self.assertRaises(ValueError):decide(rows[:-1])
        changed=copy.deepcopy(rows);changed[1]['requests'][0]['decode_memory_disturbance']=True
        self.assertFalse(decide(changed)['advance_to_confirmation'])
        self.assertEqual(decide(changed)['status'],'memory_disturbed')

    def test_request_requires_bounded_pool_execution_release_and_allocation_reduction(self):
        def state(n):return dict(expert_tail_pending=False,decode_scratch_passes=n,
            process=dict(compressed_bytes=0,decompressions=0),
            metal=dict(live_command_groups=0,peak_command_groups=2,kernel_dispatches={'q4_mm':12},active_scratch_slot=-1,
                allocation_count=0 if not n else 4500,scratch_reuses=0 if not n else 99000,
                scratch_pools=[dict(allocated_bytes=80*1024**2 if n else 0),dict(allocated_bytes=0)]))
        a,b=state(0),state(32)
        raw=dict(runs=[dict(name='decode',before=a,after=b,phases={'ingest':dict(before=a,after=a),'decode':dict(before=a,after=b)})])
        with patch('screen_decode_scratch.validate_request',side_effect=lambda *args:[{}]):
            self.assertEqual(observations(raw,{},configs()[1],[],{})[0]['allocations_per_token'],4500/32)
            for mutation in (lambda s:s.update(decode_scratch_passes=0),lambda s:s['metal'].update(active_scratch_slot=0),
                             lambda s:s['metal'].update(live_command_groups=1),lambda s:s['metal'].update(allocation_count=103616),
                             lambda s:s['metal']['scratch_pools'][0].update(allocated_bytes=129*1024**2)):
                changed=copy.deepcopy(raw);mutation(changed['runs'][0]['phases']['decode']['after'])
                with self.assertRaises(ValueError):observations(changed,{},configs()[1],[],{})

    def test_query_enforces_axis_and_does_not_promote_negative_evidence(self):
        from evidence_queries import compare
        class Index:
            def json(self,selector):return dict(kind='decode_scratch_screen_v1',complete=True,status='improvement_not_demonstrated'),{'sha256':'a'*64}
        with patch('screen_decode_scratch.revalidate',return_value=dict(status='improvement_not_demonstrated',advance_to_confirmation=False)):
            result=compare(Index(),'summary','control','candidate',['decode_scratch'])
            self.assertTrue(result['comparable']);self.assertFalse(result['production_promoted'])
            self.assertEqual(result['comparison']['status'],'improvement_not_demonstrated')
            self.assertFalse(compare(Index(),'summary','control','candidate',['ready_group'])['comparable'])

    def test_correctness_requires_exact_state_and_decode_failure_evidence(self):
        reports=[]
        for c in state_cases():
            overlap=c['decode_scratch']=='reuse'
            s=dict(diagnostic_stream_trunk=False,memory_plan=dict(expert_slots=32,panel_tokens=0),chunk_tokens=2,expert_cache=dict(evictions=1),expert_tail_pending=False,
                   decode_scratch_passes=48 if overlap else 0,metal=dict(live_command_groups=0,active_scratch_slot=-1,scratch_pools=[dict(allocated_bytes=8*1024**2 if overlap else 0),dict(allocated_bytes=0)]))
            checks=CHECKS|SCRATCH_CHECKS if overlap else CHECKS
            reports.append(dict(case=c,passed=True,layers=48,full_model=True,
                checks=[dict(name=k,passed=True) for k in checks],
                runs=[dict(continued_statistics=s,after_fresh=s,stages=[{'logits_sha256':'a','layers':list(range(48)),'routes':list(range(48))}]*3)]))
        with patch('screen_decode_scratch.check_machine'),patch('screen_decode_scratch.check_configuration'):
            self.assertTrue(correctness(reports,{})['decode_scratch_cancellation_failure'])
            altered=copy.deepcopy(reports);altered[1]['runs'][0]['stages']=[]
            with self.assertRaises(ValueError):correctness(altered,{})
            altered=copy.deepcopy(reports);altered[1]['checks'].pop()
            with self.assertRaises(ValueError):correctness(altered,{})
            altered=copy.deepcopy(reports);altered[1]['runs'][0]['stages'][0]['logits_sha256']='b'
            with self.assertRaises(ValueError):correctness(altered,{})

    def test_fresh_replay_workspace_follows_actual_last_chunk(self):
        case=state_cases()[1] # nine tokens, two-token microchunks: last forward is one token
        state=dict(memory_plan=dict(panel_tokens=0),chunk_tokens=2,
            metal=dict(active_scratch_slot=-1,scratch_pools=[dict(allocated_bytes=8*1024**2),dict(allocated_bytes=0)]))
        check_state_workspace(state,case,'after_fresh')
        empty=copy.deepcopy(state);empty['metal']['scratch_pools'][0]['allocated_bytes']=0
        with self.assertRaisesRegex(ValueError,'last replay chunk'):check_state_workspace(empty,case,'after_fresh')
        even=copy.deepcopy(case);even['continuation'].append(1) # ten tokens: last chunk has two
        check_state_workspace(empty,even,'after_fresh')
        with self.assertRaisesRegex(ValueError,'last replay chunk'):check_state_workspace(state,even,'after_fresh')
        panel=copy.deepcopy(empty);panel['memory_plan']['panel_tokens']=256
        check_state_workspace(panel,case,'after_fresh')
        for changed in (dict(active_scratch_slot=0),dict(scratch_pools=[dict(allocated_bytes=129*1024**2),dict(allocated_bytes=0)]),
                        dict(scratch_pools=[dict(allocated_bytes=8*1024**2),dict(allocated_bytes=16384)])):
            bad=copy.deepcopy(state);bad['metal'].update(changed)
            with self.assertRaises(ValueError):check_state_workspace(bad,case,'after_fresh')


if __name__=='__main__':unittest.main()
