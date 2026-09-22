import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from capacity_experiment import decide,order
from confirm_route_selection import KIND,SCREEN,configs,observations,revalidate,screen_files
from evidence_index import Index
from evidence_queries import compare
import evidence_fixture


def setUpModule():
    evidence_fixture.require('docs/benchmarks/2026-09-11-route-selection/raw/operators.json')


class RouteConfirmationTest(unittest.TestCase):
    def setUp(self):
        self.prior,paths=screen_files(SCREEN)
        self.originals={h:json.loads(p.read_text()) for h,p in paths.items()}
        rows=[];expected={}
        for pair,name in order(5):
            old=next(r for r in self.prior['measurements'] if r['configuration']==name)
            raw=copy.deepcopy(self.originals[old['sha256']]);raw['synthetic_router_confirmation_pair']=pair
            h=hashlib.sha256(json.dumps(raw).encode()).hexdigest();self.originals[h]=raw
            config=next(c for c in configs() if c['name']==name)
            rows.append(dict(pair=pair,configuration=name,sha256=h,
                requests=observations(raw,self.prior['identity'],config,self.prior['workload'],expected)))
        self.summary=dict(kind=KIND,complete=True,configurations=configs(),gpu_reference_mode='off',metal_validation=False,
            prior_pairs_pooled=False,early_success_stopping=False,identity=self.prior['identity'],workload=self.prior['workload'],
            screen_source=dict(sha256=hashlib.sha256((SCREEN/'summary.json').read_bytes()).hexdigest()),measurements=rows,**decide(rows,5))

    def test_full_fresh_confirmation(self):
        r=revalidate(self.summary,self.originals.__getitem__)
        self.assertTrue(r['candidate_for_later_qualification']);self.assertEqual(r['confidence_95']['pairs'],5)
        self.assertFalse(r['production_promoted'])

    def test_reject_incomplete_pooled_or_changed_evidence(self):
        for change in (lambda r:r.update(complete=False),lambda r:r.update(prior_pairs_pooled=True),
                       lambda r:r.update(early_success_stopping=True),lambda r:r['measurements'].pop(),
                       lambda r:r['measurements'][0].update(sha256=self.prior['measurements'][0]['sha256']),
                       lambda r:r['measurements'][2].update(sha256=r['measurements'][0]['sha256']),
                       lambda r:r['identity'].update(build='changed'),
                       lambda r:r['measurements'][1]['requests'][0].update(request_ms=1),
                       lambda r:r['confidence_95'].update(high=0),lambda r:r.update(kind='q8_steady_confirmation_v1')):
            bad=copy.deepcopy(self.summary);change(bad)
            with self.assertRaises(ValueError):revalidate(bad,self.originals.__getitem__)
        bad=copy.deepcopy(self.originals)
        candidate=self.summary['measurements'][1]['sha256']
        bad[candidate]['runs'][0]['after']['metal']['kernels']['route_selection']='serial'
        with self.assertRaises(ValueError):revalidate(self.summary,bad.__getitem__)

    def test_query_requires_declared_axis_and_completed_five_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);index=Index(root/'index.sqlite')
            try:
                _,paths=screen_files(SCREEN)
                index.import_paths(list(paths.values()))
                for digest,value in self.originals.items():
                    if digest in paths:continue
                    path=root/(digest+'.json');path.write_text(json.dumps(value))
                    index.import_paths([path])
                path=root/'summary.json';path.write_text(json.dumps(self.summary));index.import_paths([path])
                result=compare(index,str(path),'control','candidate',['route_selection'])
                self.assertTrue(result['comparable'],result)
                self.assertEqual(result['scope'],'router_selection_confirmation')
                self.assertEqual(result['comparison']['confidence_95']['pairs'],5)
                self.assertFalse(result['production_promoted'])
                self.assertFalse(compare(index,str(path),'control','candidate',['q8_decode_rows'])['comparable'])
                unfinished=dict(self.summary,complete=False,status='resource_blocked')
                path.write_text(json.dumps(unfinished));index.import_paths([path])
                self.assertFalse(compare(index,str(path),'control','candidate',['route_selection'])['comparable'])
            finally:index.close()


if __name__=='__main__':unittest.main()
