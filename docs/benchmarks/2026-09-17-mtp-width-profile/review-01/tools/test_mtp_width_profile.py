import copy
from collections import Counter
import json
from pathlib import Path
import unittest
import tempfile

from build_mtp_width_profile import WORKSPACE, ENTRY_LIMIT
from capture_mtp_width_profile import window, validate_request, analyze_block
from qualification_evidence import sha
from test_block_compute import fixture
from evidence_index import Index
from evidence_queries import next_experiment

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'docs/benchmarks/2026-09-17-mtp-widths/long-lru-01'


def profile_fixture(width):
    block,profile=fixture();block.update(width=width,committed_tokens=width,cycle_id=32)
    profile['entry_limit']=ENTRY_LIMIT
    routes=list(range(10)) if width==1 else sum([list(range(5))+list(range(5+5*t,10+5*t)) for t in range(4)],[])
    for g,d in zip(profile['command_groups'],profile['expert_dependencies']):
        layer=d['layer'];counts=Counter(routes)
        ops=[dict(request_phase='decode',offset=72,layer=layer,tokens=width,stage='router',kernel='route_simd')]
        for expert,n in sorted(counts.items()):
            ops.append(dict(request_phase='decode',offset=72,layer=layer,tokens=n,stage='routed_expert',kernel='q4_mm',
                matrix=dict(K=640,N=2560,rows=n,bits=4,group=64,quantized=True)))
            if width!=4 or n!=1:
                ops.append(dict(request_phase='decode',offset=72,layer=layer,tokens=n,stage='routed_expert',kernel='scatter_experts'))
        g['operations']=ops
        d.update(tokens=width,routes=routes,new_misses=len(counts),
                 records=[dict(d['records'][0],expert=e) for e in sorted(counts)])
    kernels=Counter(o['kernel'] for g in profile['command_groups'] for o in g['operations'])
    block['target_before']=dict(kernel_dispatches={},dispatches=0,submissions=0,live_command_groups=0,
                                kernels=dict(profile=True,counter_profile=False))
    block['target_after']=dict(kernel_dispatches=dict(kernels),dispatches=sum(kernels.values()),
        submissions=len(profile['command_groups']),live_command_groups=0,kernels=dict(profile=True,counter_profile=False))
    return block,profile


class WidthProfileTests(unittest.TestCase):
    def request(self):
        reference=json.loads((BASE/'pair-0-width-1.json').read_text())
        work=json.loads((BASE/'case-0.json').read_text());raw=copy.deepcopy(reference)
        raw.update(kind='native_mtp_width_profile_v1',performance_measurement=False,
                   profile_workspace_bytes=WORKSPACE,producer_binary_sha256='0'*64)
        raw['admission'].update(profile_workspace_bytes=WORKSPACE,
                               combined_bytes=reference['admission']['combined_bytes']+WORKSPACE)
        raw['profile_window']=window(raw['cycles'],raw['prompt_tokens'],raw['generated_tokens'])[1]
        return raw,reference,work

    def test_full_request_reference_and_explicit_profile_admission(self):
        raw,reference,work=self.request();before=copy.deepcopy(raw)
        result,selected,coverage=validate_request(raw,reference,work,sha(BASE/'case-0.json'),'0'*64)
        self.assertTrue(result['exact_logits_tokens_draft_and_target_state'])
        self.assertEqual(selected,list(range(32,48)))
        self.assertEqual((coverage['omitted_before'],coverage['omitted_after']),(32,80))
        self.assertNotIn('tokens_per_second',result)
        self.assertEqual(raw,before)

    def test_changed_outputs_coverage_admission_and_instrumentation_fail(self):
        edits=[lambda r:r.update(performance_measurement=True),
               lambda r:r.update(producer_binary_sha256='bad'),
               lambda r:r['admission'].update(draft_slots=31),
               lambda r:r['admission'].update(combined_bytes=12*1024**3+1),
               lambda r:r['profile_window'].update(omitted_after=79),
               lambda r:r['row_logits_sha256'].__setitem__(31,'bad'),
               lambda r:r['cycles'][32]['proposals'].__setitem__(0,0),
               lambda r:r['after']['metal']['kernels'].update(counter_profile=True)]
        for edit in edits:
            raw,reference,work=self.request();edit(raw)
            with self.subTest(edit=edit),self.assertRaises((ValueError,KeyError)):
                validate_request(raw,reference,work,sha(BASE/'case-0.json'),'0'*64)

    def test_window_keeps_whole_cycles_with_partial_acceptance(self):
        cycles=[dict(offset=72+3*i,width=4,committed_tokens=3) for i in range(42)]
        cycles.append(dict(offset=198,width=2,committed_tokens=2))
        selected,coverage=window(cycles,72,128)
        self.assertEqual(selected,list(range(11,17)))
        self.assertEqual((coverage['first_output'],coverage['end_output']),(33,51))
        self.assertEqual(coverage['captured_verified'],24)
        with self.assertRaises(ValueError):window(cycles[:12],72,36)
        bad=copy.deepcopy(cycles);bad[12]['offset']+=1
        with self.assertRaises(ValueError):window(bad,72,128)

    def test_single_rows_and_mixed_direct_outputs_reconcile(self):
        for width in (1,4):
            b,p=profile_fixture(width);result=analyze_block(b,p,'b','r')
            self.assertAlmostEqual(sum(result['buckets_ms_per_token'].values()),result['forward_ms_per_token'])
            self.assertEqual(sum(int(k)*v for k,v in result['expert_rows'].items()),48*10*width)
            if width==4:self.assertEqual(result['expert_rows'],{'1':960,'4':240})

    def test_missing_outputs_early_releases_and_other_work_are_rejected(self):
        for kind in ('down','scatter','counter','early','scope','limit'):
            b,p=profile_fixture(4)
            if kind=='down':p['command_groups'][0]['operations'][1]['matrix']['N']=100
            elif kind=='scatter':p['command_groups'][0]['operations'].pop(2)
            elif kind=='counter':b['target_after']['dispatches']+=1
            elif kind=='early':p['expert_dependencies'][0]['records'][0]['released_ns']=0
            elif kind=='scope':p['command_groups'][0]['operations'][0]['request_phase']='draft'
            else:p['entry_limit']=100000
            with self.subTest(kind=kind),self.assertRaises(ValueError):analyze_block(b,p,'b','r')

    def test_resource_blocked_profile_does_not_select_an_optimization(self):
        source=ROOT/'docs/benchmarks/2026-09-17-mtp-width-profile/lru-01/summary.json'
        with tempfile.TemporaryDirectory() as d:
            index=Index(Path(d)/'index.sqlite')
            try:
                index.import_paths([source]);answer=next_experiment(index,str(source))
                self.assertEqual(answer['status'],'partial_diagnostic')
                self.assertEqual(answer['recorded_status'],'resource_blocked')
                self.assertIsNone(answer['possible_request_benefit'])
                self.assertFalse(answer['performance_measurement'])
                self.assertFalse(answer['production_promoted'])
                self.assertNotIn('buckets_ms_per_committed_token',answer)
            finally:index.close()


if __name__=='__main__':unittest.main()
