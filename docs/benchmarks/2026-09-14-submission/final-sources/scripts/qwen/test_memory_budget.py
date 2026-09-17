import copy
import hashlib
import json
import unittest

from capacity_experiment import order,decide
from screen_cache import validate_request
from screen_memory_budget import KIND,BUDGETS,SLOTS,SOURCE,configs,observations,revalidate,source_files


class MemoryBudgetTest(unittest.TestCase):
    def setUp(self):
        self.prior,paths=source_files(SOURCE)
        self.originals={h:json.loads(p.read_text()) for h,p in paths.items()}
        self.identity={k:v for k,v in self.prior['identity'].items() if k!='budget_bytes'}
        row=next(r for r in self.prior['measurements'] if r['configuration']=='candidate')
        self.base=self.originals[row['sha256']]

    def raw(self,name,pair=0):
        raw=copy.deepcopy(self.base);raw['synthetic_budget_pair']=pair
        for row in raw['runs']:
            if name=='candidate':
                for k in ('request_ms','time_to_first_token_ms','decode_wall_ms'):row[k]*=.9
            for state in [row['before'],row['after'],*[v[k] for v in row['phases'].values() for k in ('before','after')]]:
                p=state['memory_plan'];extra=(SLOTS[name]-p['expert_slots'])*2768896
                p.update(limit_bytes=BUDGETS[name],expert_slots=SLOTS[name],expert_bytes=p['expert_bytes']+extra,planned_bytes=p['planned_bytes']+extra)
        return raw

    def test_explicit_budget_only_and_legacy_remains_fixed(self):
        expected={}
        for config in configs():observations(self.raw(config['name']),self.identity,config,self.prior['workload'],expected)
        with self.assertRaises(ValueError):validate_request(self.raw('candidate'),self.identity,configs()[1],self.prior['workload'],{})
        for change in (lambda r:r['runs'][0]['before']['memory_plan'].update(limit_bytes=17*1024**3),
                       lambda r:r['runs'][0]['before']['memory_plan'].update(resident_bytes=1),
                       lambda r:r['runs'][0]['before']['memory_plan'].update(expert_slots=4174),
                       lambda r:r['runs'][0]['after']['process'].update(physical_footprint_bytes=19*1024**3),
                       lambda r:r['runs'][1].update(reused_tokens=105),
                       lambda r:r['runs'][0]['after']['metal']['kernel_dispatches'].update(norm=1)):
            raw=self.raw('candidate');change(raw)
            with self.assertRaises(ValueError):observations(raw,self.identity,configs()[1],self.prior['workload'],expected)

    def test_reconstruct_fresh_budget_pairs(self):
        rows=[];expected={}
        for pair,name in order(2):
            raw=self.raw(name,pair);h=hashlib.sha256(json.dumps(raw).encode()).hexdigest();self.originals[h]=raw
            config=next(c for c in configs() if c['name']==name)
            rows.append(dict(pair=pair,configuration=name,sha256=h,requests=observations(raw,self.identity,config,self.prior['workload'],expected)))
        summary=dict(kind=KIND,complete=True,configurations=configs(),budgets_bytes=BUDGETS,gpu_reference_mode='off',metal_validation=False,
            q8_source=dict(sha256=hashlib.sha256((SOURCE/'summary.json').read_bytes()).hexdigest()),identity=self.identity,
            workload=self.prior['workload'],measurements=rows,**decide(rows,2))
        self.assertTrue(revalidate(summary,self.originals.__getitem__)['advance_to_confirmation'])
        for change in (lambda r:r.update(complete=False),lambda r:r['measurements'].pop(),
                       lambda r:r['measurements'][1].update(sha256=r['measurements'][0]['sha256']),
                       lambda r:r['measurements'][1]['requests'][0].update(request_ms=1),
                       lambda r:r.update(advance_to_confirmation=False)):
            s=copy.deepcopy(summary);change(s)
            with self.assertRaises(ValueError):revalidate(s,self.originals.__getitem__)


if __name__=='__main__':unittest.main()
