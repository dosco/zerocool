import copy
import unittest
from unittest.mock import patch

from benchmark_exact import config_args
from qualify_exact_sessions import check_configuration
from screen_expert_tail import configs,decide,observations,correctness,state_cases,TAIL_CHECKS
from screen_cache import CHECKS


class ExpertTailTest(unittest.TestCase):
    def test_only_tail_changes_and_configuration_is_checked(self):
        a,b=configs()
        self.assertEqual({k for k in a if a[k]!=b[k]},{'name','expert_tail'})
        self.assertIn('--expert-tail',config_args(b))
        state=dict(execution=dict(expert_tail='overlap'),metal=dict(kernels={}),ready_group=4,chunk_tokens=128,io_workers=8)
        with self.assertRaisesRegex(ValueError,'execution'):check_configuration(state,{})
        check_configuration(state,dict(expert_tail='overlap'))

    def test_rejects_gain_that_does_not_help_decode_or_regresses_ttft(self):
        rows=[]
        for p,name in ((0,'control'),(0,'candidate'),(1,'candidate'),(1,'control')):
            c=name=='candidate'
            rows.append(dict(pair=p,configuration=name,requests=[dict(request_ms=100 if not c else 90,
                time_to_first_token_ms=50,decode_wall_ms=50 if not c else 40)]*2))
        self.assertTrue(decide(rows)['advance_to_confirmation'])
        for mutate in (lambda r:r[1]['requests'][0].update(decode_wall_ms=55),
                       lambda r:r[1]['requests'][0].update(time_to_first_token_ms=55)):
            changed=copy.deepcopy(rows);mutate(changed)
            self.assertFalse(decide(changed)['advance_to_confirmation'])
        with self.assertRaises(ValueError):decide(rows[:-1])

    def test_request_requires_real_tail_execution_and_bounded_completion(self):
        def state(n):return dict(expert_tail_pending=False,expert_tail_deferrals=n,
            metal=dict(live_command_groups=0,peak_command_groups=2,kernel_dispatches={'q4_mm':12}))
        a,b=state(0),state(48*32)
        raw=dict(runs=[dict(name='decode',before=a,after=b,phases={'decode':dict(before=a,after=b)})])
        with patch('screen_expert_tail.validate_request',return_value=[]):
            observations(raw,{},configs()[1],[],{})
            for key,value in (('expert_tail_pending',True),('expert_tail_deferrals',0)):
                altered=copy.deepcopy(raw);altered['runs'][0]['after'][key]=value
                with self.assertRaises(ValueError):observations(altered,{},configs()[1],[],{})
            altered=copy.deepcopy(raw);altered['runs'][0]['after']['metal']['peak_command_groups']=3
            with self.assertRaises(ValueError):observations(altered,{},configs()[1],[],{})

    def test_query_enforces_axis_and_does_not_promote_negative_evidence(self):
        from evidence_queries import compare
        class Index:
            def json(self,selector):return dict(kind='expert_tail_screen_v1',complete=True,status='improvement_not_demonstrated'),{'sha256':'a'*64}
        with patch('screen_expert_tail.revalidate',return_value=dict(status='improvement_not_demonstrated',advance_to_confirmation=False)):
            result=compare(Index(),'summary','control','candidate',['expert_tail'])
            self.assertTrue(result['comparable']);self.assertFalse(result['production_promoted'])
            self.assertEqual(result['comparison']['status'],'improvement_not_demonstrated')
            self.assertFalse(compare(Index(),'summary','control','candidate',['ready_group'])['comparable'])

    def test_correctness_requires_exact_state_and_decode_failure_evidence(self):
        reports=[]
        for c in state_cases():
            overlap=c['expert_tail']=='overlap'
            s=dict(diagnostic_stream_trunk=False,memory_plan=dict(expert_slots=32),expert_cache=dict(evictions=1),expert_tail_pending=False,
                   expert_tail_deferrals=48 if overlap else 0,metal=dict(live_command_groups=0))
            checks=CHECKS|TAIL_CHECKS if overlap else CHECKS
            reports.append(dict(case=c,passed=True,layers=48,full_model=True,
                checks=[dict(name=k,passed=True) for k in checks],
                runs=[dict(continued_statistics=s,after_fresh=s,stages=[{'logits_sha256':'a','layers':list(range(48)),'routes':list(range(48))}]*3)]))
        with patch('screen_expert_tail.check_machine'),patch('screen_expert_tail.check_configuration'):
            self.assertTrue(correctness(reports,{})['decode_tail_cancellation_failure'])
            altered=copy.deepcopy(reports);altered[1]['runs'][0]['stages']=[]
            with self.assertRaises(ValueError):correctness(altered,{})
            altered=copy.deepcopy(reports);altered[1]['checks'].pop()
            with self.assertRaises(ValueError):correctness(altered,{})
            altered=copy.deepcopy(reports);altered[1]['runs'][0]['stages'][0]['logits_sha256']='b'
            with self.assertRaises(ValueError):correctness(altered,{})


if __name__=='__main__':unittest.main()
