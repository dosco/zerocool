import copy
import hashlib
import json
import unittest

from capacity_experiment import decide, order
from confirm_q8_steady import KIND, SCREEN, revalidate, screen_files
from screen_q8_steady import configs, observations


class Q8ConfirmationTest(unittest.TestCase):
    def setUp(self):
        prior,files=screen_files(SCREEN)
        self.originals={h:json.loads(p.read_text()) for h,p in files.items()}
        self.prior=prior
        expected={};rows=[]
        for pair,name in order(5):
            old=next(r for r in prior['measurements'] if r['configuration']==name)
            raw=copy.deepcopy(self.originals[old['sha256']]);raw['synthetic_confirmation_pair']=pair
            h=hashlib.sha256(json.dumps(raw).encode()).hexdigest();self.originals[h]=raw
            config=next(c for c in configs() if c['name']==name)
            rows.append(dict(pair=pair,configuration=name,sha256=h,
                requests=observations(raw,prior['identity'],config,prior['workload'],expected)))
        self.summary=dict(kind=KIND,complete=True,configurations=configs(),gpu_reference_mode='off',metal_validation=False,
            prior_pairs_pooled=False,early_success_stopping=False,identity=prior['identity'],workload=prior['workload'],
            screen_source=dict(sha256=hashlib.sha256((SCREEN/'summary.json').read_bytes()).hexdigest()),measurements=rows,**decide(rows,5))

    def test_five_pair_reconstruction_and_interval(self):
        result=revalidate(self.summary,self.originals.__getitem__)
        self.assertTrue(result['candidate_for_later_qualification'])
        self.assertEqual(result['confidence_95']['pairs'],5)
        self.assertFalse(result['production_promoted'])

    def test_reject_pooling_duplicates_edits_and_early_completion(self):
        for change in (lambda r:r.update(complete=False),lambda r:r.update(prior_pairs_pooled=True),
                       lambda r:r.update(early_success_stopping=True),lambda r:r['measurements'].pop(),
                       lambda r:r['measurements'][0].update(sha256=self.prior['measurements'][0]['sha256']),
                       lambda r:r['measurements'][2].update(sha256=r['measurements'][0]['sha256']),
                       lambda r:r['identity'].update(build='changed'),
                       lambda r:r['measurements'][0]['requests'][0].update(request_ms=1),
                       lambda r:r['confidence_95'].update(high=0)):
            summary=copy.deepcopy(self.summary);change(summary)
            with self.assertRaises(ValueError):revalidate(summary,self.originals.__getitem__)


if __name__=='__main__':unittest.main()
