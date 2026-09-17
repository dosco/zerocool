import copy
import unittest
from pathlib import Path

import build_block_gdn as builder
from screen_block_gdn import analyze, SHAPES


def fixture():
    memory = dict(physical_footprint_bytes=1024, physical_footprint_peak_bytes=2048,
        compressed_bytes=0, compressed_peak_bytes=0, decompressions=0, system_swap_used_bytes=0)
    host = dict(thermal_state=0, low_power_mode=False, power_source='AC Power', monotonic_ns=1)
    raw = dict(kind='block_gdn_operator_v1', complete=True, validation=False, build='build', artifact_revision='revision',
        source={}, device='Apple M1 Pro', peak_gpu_bytes=1000, production_promoted=False, normal_request_latency_qualified=False,
        host_before=host, host_after=host, memory_before=memory, memory_after_destroy=memory, cases=[])
    for K, N in SHAPES:
        def arm(candidate, repeats):
            return dict(candidate=candidate, repeats=repeats, exact=True, output_sha256='sha',
                gpu_ns=(500000 if candidate else 1000000)*repeats, wall_ns=2000000*repeats,
                memory_before=copy.deepcopy(memory), memory_after=copy.deepcopy(memory))
        raw['cases'].append(dict(K=K, N=N, tokens=4, reference_sha256='sha', warmup=[arm(False,1),arm(True,1)],
            pairs=[dict(pair=p,arms=[arm(bool(p%2),32),arm(not bool(p%2),32)]) for p in range(5)]))
    return raw


class BlockGDNTest(unittest.TestCase):
    def check(self, raw): return analyze(raw, False, dict(build='build',artifact_revision='revision'), {})

    def test_paired_projection(self):
        result = self.check(fixture())
        self.assertTrue(result['advance_to_verifier_screen'])
        self.assertAlmostEqual(result['median_projection_ms_per_token'], 13.5)
        self.assertEqual(result['weighted_ratios'], [.5]*5)

    def test_disturbance_and_small_effect_do_not_advance(self):
        raw = fixture(); raw['cases'][0]['pairs'][0]['arms'][0]['memory_after']['compressed_peak_bytes'] = 1
        self.assertFalse(self.check(raw)['advance_to_verifier_screen'])
        raw = fixture()
        for c in raw['cases']:
            for p in c['pairs']:
                for arm in p['arms']:
                    if arm['candidate']: arm['gpu_ns'] = 900000*32
        self.assertFalse(self.check(raw)['advance_to_verifier_screen'])

    def test_missing_or_changed_evidence_rejected(self):
        def short(r): r['cases'].pop()
        def order(r): r['cases'][0]['pairs'][0]['arms'].reverse()
        def exact(r): r['cases'][0]['pairs'][0]['arms'][0]['exact'] = False
        def source(r): r['source'] = {'changed': True}
        def pairs(r): r['cases'][0]['pairs'].pop()
        for mutate in (short, order, exact, source, pairs):
            with self.subTest(mutation=mutate.__name__):
                raw=fixture();mutate(raw)
                with self.assertRaises(ValueError): self.check(raw)

    def test_capture_and_operator_link_scope(self):
        cfg = builder.settings(Path('/tmp/block-gdn-test'))
        copied = builder.sources(cfg)[cfg['binary'].parent/'probe.block-profile.cpp']
        self.assertIn('capture_phase="decode"', copied)
        self.assertIn('capture_operator="gdn"', copied)
        self.assertIn('capture_layer=0', copied)
        self.assertIn('if(phase_!="decode") {'+builder.CLEAR_CAPTURE+'}', builder.sources(cfg)[cfg['generated']])
        self.assertIn('if(!decode) {'+builder.CLEAR_CAPTURE+'}', copied)
        self.assertFalse(any(str(p) in cfg['operator_linker'] for p in cfg['objects']))
        self.assertIn(str(cfg['operator_object']), cfg['operator_linker'])


if __name__ == '__main__': unittest.main()
