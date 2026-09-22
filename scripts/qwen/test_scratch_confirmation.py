import argparse
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from capacity_experiment import order
from confirm_decode_scratch import KIND,configs,decide,revalidate,run,screen_files
from evidence_queries import compare
from check_scratch_slice import validate as validate_slice
from screen_decode_scratch import correctness as validate_full_state
import evidence_fixture


ROOT=Path(__file__).resolve().parents[2]
DISTURBED=ROOT/'docs/benchmarks/2026-09-13-decode-scratch/raw'


def synthetic_rows():
    # Deliberately synthetic timing fixtures, never written into benchmark evidence.
    return [dict(pair=p,configuration=name,requests=[dict(request_ms=90 if name=='candidate' else 100,
        time_to_first_token_ms=45 if name=='candidate' else 50,decode_wall_ms=40 if name=='candidate' else 50,
        decode_memory_disturbance=False) for _ in range(2)]) for p,name in order(5)]


class ScratchConfirmationTest(unittest.TestCase):
    def test_real_slice_revalidates_remainder_but_cannot_qualify_full_state(self):
        evidence_fixture.require(
            'docs/benchmarks/2026-09-13-decode-scratch/raw/prior-output-control.json',
            'docs/benchmarks/2026-09-13-scratch-confirmation/state-slice/control.json',
        )
        root=ROOT/'docs/benchmarks/2026-09-13-scratch-confirmation/state-slice'
        evidence=json.loads((root/'summary.json').read_text())['identity']
        reports=[json.loads((root/(n+'.json')).read_text()) for n in ('control','candidate')]
        proof=validate_slice(reports,evidence)
        self.assertTrue(proof['exact_four_layer_state_and_routes'])
        self.assertFalse(proof['full_model']);self.assertFalse(proof['logits_compared'])
        pool=reports[1]['runs'][0]['after_fresh']['metal']['scratch_pools'][0]['allocated_bytes']
        self.assertGreater(pool,0) # Original checker would falsely reject this actual native report.
        with self.assertRaises(ValueError):validate_full_state(reports,evidence)
        altered=copy.deepcopy(reports);altered[1]['checks'].pop()
        with self.assertRaises(ValueError):validate_slice(altered,evidence)
        altered=copy.deepcopy(reports);altered[1]['full_model']=True
        with self.assertRaises(ValueError):validate_slice(altered,evidence)

    def test_five_pairs_require_both_decode_confidence_and_no_memory_disturbance(self):
        rows=synthetic_rows();result=decide(rows)
        self.assertTrue(result['candidate_for_later_qualification'])
        self.assertEqual(result['confidence_95']['pairs'],5)
        self.assertTrue(all(r['confidence_95']['high']<1 for r in result['ratios'] if r['metric']=='decode_wall_ms'))
        self.assertFalse(result['production_promoted']);self.assertFalse(result['normal_request_latency_qualified'])
        with self.assertRaises(ValueError):decide(rows[:-1])
        with self.assertRaises(ValueError):decide(rows,2)
        for missing in (True,None):
            changed=copy.deepcopy(rows);changed[1]['requests'][0]['decode_memory_disturbance']=missing
            self.assertEqual(decide(changed)['status'],'memory_disturbed')
            self.assertFalse(decide(changed)['candidate_for_later_qualification'])
        for field,value in (('time_to_first_token_ms',80),('decode_wall_ms',51)):
            changed=copy.deepcopy(rows);changed[1]['requests'][0][field]=value
            self.assertFalse(decide(changed)['candidate_for_later_qualification'])

    def fixture(self):
        originals={}
        def add(value):
            digest=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
            originals[digest]=value;return digest
        rows=synthetic_rows()
        for i,row in enumerate(rows):row['sha256']=add(dict(synthetic_iteration=i,requests=copy.deepcopy(row['requests'])))
        old=add(dict(synthetic_iteration='screen',requests=copy.deepcopy(rows[0]['requests'])))
        prior=dict(identity={'build':'synthetic'},workload=['synthetic'],measurements=[dict(sha256=old)])
        prior_digest=add(prior)
        summary=dict(kind=KIND,complete=True,configurations=configs(),gpu_reference_mode='off',metal_validation=False,
            prior_pairs_pooled=False,early_success_stopping=False,identity=prior['identity'],workload=prior['workload'],
            screen_source=dict(sha256=prior_digest),measurements=rows,**decide(rows))
        return summary,originals,old

    def test_reconstruction_rejects_pooled_truncated_changed_or_unproven_results(self):
        summary,originals,old=self.fixture()
        with patch('confirm_decode_scratch.observations',side_effect=lambda raw,*args:raw['requests']), \
             patch('confirm_decode_scratch.revalidate_screen',return_value={'advance_to_confirmation':True}) as screen:
            self.assertTrue(revalidate(summary,originals.__getitem__)['candidate_for_later_qualification'])
            for change in (lambda r:r.update(complete=False),lambda r:r.update(prior_pairs_pooled=True),
                lambda r:r.update(early_success_stopping=True),lambda r:r['measurements'].pop(),
                lambda r:r['measurements'][0].update(sha256=old),
                lambda r:r['measurements'][2].update(sha256=r['measurements'][0]['sha256']),
                lambda r:r.update(identity={'build':'changed'}),
                lambda r:r['measurements'][0]['requests'][0].update(decode_wall_ms=1),
                lambda r:r['confidence_95'].update(high=0),lambda r:r.update(clean_decode_memory=False)):
                altered=copy.deepcopy(summary);change(altered)
                with self.assertRaises(ValueError):revalidate(altered,originals.__getitem__)
            screen.return_value={'advance_to_confirmation':False}
            with self.assertRaisesRegex(ValueError,'passing short screen'):revalidate(summary,originals.__getitem__)

    def test_actual_disturbed_screen_cannot_launch_inference(self):
        evidence_fixture.require(
            'docs/benchmarks/2026-09-13-decode-scratch/raw/prior-output-control.json',
            'docs/benchmarks/2026-09-13-scratch-confirmation/state-slice/control.json',
        )
        prior,files=screen_files(DISTURBED)
        self.assertEqual(prior['status'],'memory_disturbed')
        self.assertIn(prior['prior_output_control']['sha256'],files)
        with tempfile.TemporaryDirectory() as directory,patch('confirm_q8_steady.identity') as identify, \
             patch('confirm_q8_steady.EvidenceGuard.run') as execute,contextlib.redirect_stdout(io.StringIO()):
            out=Path(directory)/'output'
            self.assertEqual(run(argparse.Namespace(screen=DISTURBED,output=out)),2)
            report=json.loads((out/'summary.json').read_text())
            self.assertFalse(report['complete']);self.assertEqual(report['measurements'],[])
            self.assertIn('passing short screen',report['error'])
            identify.assert_not_called();execute.assert_not_called()

    def test_query_reconstructs_confirmation_with_only_the_declared_axis(self):
        summary,originals,_=self.fixture()
        class Index:
            def json(self,key):return (summary if key=='summary' else originals[key]),dict(sha256=key)
        with patch('confirm_decode_scratch.observations',side_effect=lambda raw,*args:raw['requests']), \
             patch('confirm_decode_scratch.revalidate_screen',return_value={'advance_to_confirmation':True}):
            result=compare(Index(),'summary','control','candidate',['decode_scratch'])
            self.assertTrue(result['comparable'],result)
            self.assertEqual(result['scope'],'decode_scratch_confirmation')
            self.assertEqual(result['comparison']['confidence_95']['pairs'],5)
            self.assertFalse(result['production_promoted'])
            self.assertFalse(compare(Index(),'summary','control','candidate',['expert_tail'])['comparable'])


if __name__=='__main__':unittest.main()
