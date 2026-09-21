import copy
from pathlib import Path
import unittest

import build_q8_expanded as builder
import screen_q8_expanded as screen
from q8_expanded_contract import CASES, KERNEL
from test_q8_block_packed import packed_fixture


def fixture():
    raw=packed_fixture();template=raw['cases'][0];raw['kind']='q8_expanded_operator_v1';raw['cases']=[]
    for spec in CASES:
        case=copy.deepcopy(template);case.update(name=spec['name'],K=spec['K'],N=spec['N'],frequency=spec['frequency'])
        for arm in [*case['warmup'],*[a for p in case['pairs'] for a in p['arms']]]:
            arm['kernel_dispatches']={KERNEL if arm['candidate'] else 'q8_mm_t4':arm['repeats']}
        raw['cases'].append(case)
    return raw


class ExpandedTest(unittest.TestCase):
    def check(self,r):return screen.analyze(r,False,dict(build='build',artifact_revision='revision'),{})

    def test_frequency_weighting(self):
        result=self.check(fixture());self.assertTrue(result['advance_to_verifier_screen'])
        self.assertAlmostEqual(result['median_projection_ms_per_token'],sum(c['frequency'] for c in CASES)*.5/4)

    def test_cannot_advance_with_changed_coverage_or_selection(self):
        def missing(r):r['cases'].pop()
        def frequency(r):r['cases'][0]['frequency']=36
        def kernel(r):r['cases'][0]['pairs'][0]['arms'][1]['kernel_dispatches']={'q8_mm_t4':32}
        def exact(r):r['cases'][0]['pairs'][0]['arms'][1]['output_sha256']='f'*64
        def order(r):r['cases'][0]['pairs'][0]['arms'].reverse()
        def zero(r):r['cases'][0]['pairs'][0]['arms'][0]['gpu_ns']=0
        for mutate in (missing,frequency,kernel,exact,order,zero):
            with self.subTest(mutation=mutate.__name__):
                r=fixture();mutate(r)
                with self.assertRaises(ValueError):self.check(r)

    def test_resource_and_materiality_gates(self):
        r=fixture();r['cases'][0]['warmup'][0]['memory_after']['compressed_peak_bytes']=1
        self.assertFalse(self.check(r)['advance_to_verifier_screen'])
        r=fixture()
        for c in r['cases']:
            for p in c['pairs']:
                for a in p['arms']:
                    if a['candidate']:a['gpu_ns']=950000*32
        self.assertFalse(self.check(r)['advance_to_verifier_screen'])

    def test_source_copy_scope_and_link_ownership(self):
        cfg=builder.settings(Path('/tmp/q8-expanded-test'));sources=builder.sources(cfg)
        for link in cfg['linkers']:self.assertLess(link.index(str(cfg['objects'][0])),link.index('libzerocool_lib.a'))
        text=sources[cfg['generated'][0]]
        self.assertIn('impl_->expanded_scope && impl_->request_phase=="decode" && tokens==4',text)
        self.assertIn('tile==4 && policy.rows==1',text)
        self.assertIn('stage=="gdn" || stage=="attention" || stage=="logits"',text)
        self.assertIn('else if(impl_->config.q4_decode',text)
        self.assertIn('input_bytes>MiB-p.captured_bytes',text)
        self.assertIn('weight_payloads_copied',text)
        self.assertNotIn('@CASES@',sources[cfg['generated'][1]])
        self.assertNotIn('phase_=="decode";\n    verifier_kernels.counter_profile',sources[cfg['verifier']['generated']])


if __name__=='__main__':unittest.main()
