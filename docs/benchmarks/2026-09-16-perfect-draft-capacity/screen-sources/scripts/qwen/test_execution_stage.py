import copy
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import benchmark_exact
import benchmark_execution
from benchmark_exact import configurations,validate,config_args,is_original_control
import test_exact_benchmark as exact_tests
from select_shape_rules import select


class ExecutionStageTest(unittest.TestCase):
    def test_rejected_experiment_setup_cannot_start_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(output=Path(directory)/'experiment',base_config=None,
                experiment='residency',binary='engine',model='model',prepared='prepared',
                artifact='mixed-4_8bit',memory_gb=18)
            with patch.object(benchmark_exact,'inspect_admission',side_effect=ValueError('budget unavailable')), \
                 patch.object(benchmark_exact,'run') as inference:
                with self.assertRaises(ValueError):benchmark_execution.run(args)
                inference.assert_not_called()
            report=json.loads((args.output/'setup-error.json').read_text())
            self.assertFalse(report['complete']);self.assertEqual(report['configuration'],'off')
            self.assertFalse((args.output/'normal').exists())

    def test_candidate_identity_and_cached_replay_rejected(self):
        fixture=exact_tests.ExactBenchmarkTest().fixture();config=configurations()[0]
        row=fixture['runs'][0]
        row['after']['execution']=dict(residency='core',decode_path='direct',prefill_pipeline='double',cached_token_replay=False)
        config=dict(config,residency='core',decode_path='direct',prefill_pipeline='double')
        validate(fixture,config,dict(build='native',revision='revision'),8*1024**3,256)
        for key,value in [('residency','off'),('decode_path','grouped'),('prefill_pipeline','serial'),('cached_token_replay',True)]:
            bad=copy.deepcopy(fixture);bad['runs'][0]['after']['execution'][key]=value
            with self.assertRaises(ValueError):validate(bad,config,dict(build='native',revision='revision'),8*1024**3,256)
        self.assertIn('--decode-path',config_args(config))

    def test_admission_retries_preserve_rejections_and_never_reduce_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            stem=Path(directory)/"case";calls=[]
            def inspect(command,**kwargs):
                calls.append(command)
                destination=Path(command[command.index("--json")+1])
                destination.write_text(json.dumps(dict(current_admission=dict(limit_bytes=12 if len(calls)>1 else 11,panel_tokens=512))))
            with patch.object(benchmark_exact.subprocess,"run",side_effect=inspect),patch.object(benchmark_exact.time,"sleep"):
                result=benchmark_exact.inspect_admission("engine",["--memory-gb","12"],stem,12,512,None)
            self.assertEqual(result.name,"case.admission-retry-1.json")
            self.assertEqual(json.loads(stem.with_suffix(".admission.json").read_text())["current_admission"]["limit_bytes"],11)
            self.assertTrue(all(command[1]=="inspect" and command[3]=="12" for command in calls))
            def rejected(command,**kwargs):
                Path(command[-1]).write_text(json.dumps(dict(current_admission=dict(error="unavailable"))))
            with patch.object(benchmark_exact.subprocess,"run",side_effect=rejected) as run,patch.object(benchmark_exact.time,"sleep"):
                with self.assertRaises(ValueError):benchmark_exact.inspect_admission("engine",["--memory-gb","12"],Path(directory)/"reject",12,512,None)
                self.assertEqual(run.call_count,3)

    def test_promotion_control_cannot_hide_execution_candidates(self):
        reference=configurations()[0]
        self.assertTrue(is_original_control(reference))
        for key,value in [('residency','core'),('decode_path','direct'),('prefill_pipeline','double'),('gdn_path','precompute'),('shape_policy','rules.json')]:
            self.assertFalse(is_original_control(dict(reference,**{key:value})))

    def test_selection_requires_complete_exact_paired_cases(self):
        matrix=dict(K=640,N=2560,rows=8,bits=4,group=64,fused=False,gathered=False)
        report=dict(kind='captured_operator_screen',exact=True,build_fingerprint='build',artifact_revision='revision',measurements=[])
        for rep in range(5):
            for tile,ns in [(1,100),(2,80),(4,90),(8,120)]:
                report['measurements'].append(dict(matrix=matrix,case='input',tile=tile,repetition=rep,wall_ns=ns,exact=True))
        self.assertEqual(select(report)['rules'],[dict(matrix,tile=2)])
        bad=copy.deepcopy(report);bad['measurements'].pop()
        with self.assertRaises(ValueError):select(bad)
        bad=copy.deepcopy(report);bad['measurements'][0]['exact']=False
        with self.assertRaises(ValueError):select(bad)
        bad=copy.deepcopy(report);bad['measurements'].append(bad['measurements'][0])
        with self.assertRaises(ValueError):select(bad)
