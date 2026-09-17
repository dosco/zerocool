import copy
import hashlib
import json
import unittest

from capacity_experiment import decide,order
from screen_memory_budget import SOURCE,source_files
from screen_route_selection import KIND,configs,operators,observations,revalidate
from qualify_exact_sessions import check_configuration


class RouteSelectionTest(unittest.TestCase):
    def setUp(self):
        prior,files=source_files(SOURCE)
        self.identity=prior['identity'];self.workload=prior['workload']
        r=next(r for r in prior['measurements'] if r['configuration']=='candidate')
        self.base=json.loads(files[r['sha256']].read_text());self.originals={}

    def fixture(self):
        cases=[dict(phase=phase,layer=l,tokens=t,sha256=str(i)) for i,(phase,l,t) in enumerate(
            [('decode',l,1) for l in range(48)]+[('prefill',l,72) for l in (0,47)])]
        measurements=[dict(case=c['sha256'],tokens=c['tokens'],repetition=r,variant=v,dispatches=8,exact=True,
                           wall_ns=100 if v=='serial' else 20,gpu_ns=80 if v=='serial' else 10)
                      for c in cases for r in range(10) for v in (('serial','simd') if r%2==0 else ('simd','serial'))]
        return dict(kind='captured_router_operator_check_v1',complete=True,exact=True,
            source=dict(kind='captured_router_logits_v1',build=self.identity['build'],artifact_revision=self.identity['artifact_revision'],cases=cases),
            machine=dict(build_fingerprint=self.identity['build']),checks=[dict(case=c['sha256'],cpu_ranking_exact=True) for c in cases],measurements=measurements)

    def raw(self,name,pair=0):
        raw=copy.deepcopy(self.base);raw['synthetic_route_pair']=pair
        for r in raw['runs']:
            if name=='candidate':
                for k in ('request_ms','time_to_first_token_ms','decode_wall_ms'):r[k]*=1.1
            for s in (r['before'],r['after'],*[v[k] for v in r['phases'].values() for k in ('before','after')]):
                s['metal']['kernels']['route_selection']='simd' if name=='candidate' else 'serial'
                if name=='candidate' and 'route' in s['metal']['kernel_dispatches']:
                    s['metal']['kernel_dispatches']['route_simd']=s['metal']['kernel_dispatches'].pop('route')
        return raw

    def keep(self,raw):
        h=hashlib.sha256(json.dumps(raw).encode()).hexdigest();self.originals[h]=raw;return h

    def test_operator_gate_requires_complete_exact_coverage(self):
        raw=self.fixture();self.assertTrue(operators(raw,self.identity)['screen_passed'])
        for change in (lambda r:r['source']['cases'].pop(),lambda r:r['checks'].pop(),
                       lambda r:r.update(exact=False),lambda r:r['measurements'][0].update(gpu_ns=0),
                       lambda r:r['measurements'][0].update(variant='simd'),lambda r:r['measurements'][1].update(exact=False)):
            bad=copy.deepcopy(raw);change(bad)
            with self.assertRaises(ValueError):operators(bad,self.identity)

    def test_route_option_and_dispatches_are_bound(self):
        for c in configs():observations(self.raw(c['name']),self.identity,c,self.workload,{})
        with self.assertRaises(ValueError):check_configuration(self.raw('candidate')['runs'][0]['after'],dict(configs()[0]))
        for change in (lambda r:r['runs'][0]['after']['metal']['kernel_dispatches'].update(route=1),
                       lambda r:r['runs'][0]['after']['metal']['kernel_dispatches'].update(norm=1),
                       lambda r:r['runs'][0]['after']['metal']['kernels'].update(route_selection='serial'),
                       lambda r:r['runs'][0]['after']['metal']['kernels'].update(profile=True),
                       lambda r:r['runs'][1].update(reused_tokens=105)):
            expected={};observations(self.raw('control'),self.identity,configs()[0],self.workload,expected)
            raw=self.raw('candidate');change(raw)
            with self.assertRaises(ValueError):observations(raw,self.identity,configs()[1],self.workload,expected)

    def test_failed_screen_is_terminal_without_expensive_state_checks(self):
        rows=[];expected={}
        for pair,name in order(2):
            raw=self.raw(name,pair);c=next(c for c in configs() if c['name']==name)
            rows.append(dict(pair=pair,configuration=name,sha256=self.keep(raw),requests=observations(raw,self.identity,c,self.workload,expected)))
        op=self.fixture()
        s=dict(kind=KIND,complete=True,configurations=configs(),identity=self.identity,workload=self.workload,
            gpu_reference_mode='off',metal_validation=False,operator_source=dict(sha256=self.keep(op)),operators=operators(op,self.identity),
            measurements=rows,correctness_sources=[],**decide(rows,2))
        self.assertFalse(revalidate(s,self.originals.__getitem__)['advance_to_confirmation'])
        for change in (lambda r:r.update(complete=False),lambda r:r['measurements'].pop(),
                       lambda r:r['measurements'][1].update(sha256=r['measurements'][0]['sha256']),
                       lambda r:r.update(advance_to_confirmation=True),lambda r:r['measurements'][0]['requests'][0].update(request_ms=1)):
            bad=copy.deepcopy(s);change(bad)
            with self.assertRaises(ValueError):revalidate(bad,self.originals.__getitem__)


if __name__=='__main__':unittest.main()
