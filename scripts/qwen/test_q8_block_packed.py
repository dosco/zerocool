import copy
from pathlib import Path
import unittest

import build_q8_block_packed as builder
import screen_q8_block_packed as screen
from test_block_gdn import fixture


def packed_fixture():
    raw = fixture(); raw.update(kind='q8_block_packed_operator_v1', variant=copy.deepcopy(builder.VARIANT))
    for case in raw['cases']:
        case['reference_sha256'] = 'a'*64
        for arm in [*case['warmup'], *[a for p in case['pairs'] for a in p['arms']]]:
            arm['output_sha256'] = 'a'*64
            arm['kernel_dispatches'] = {builder.VARIANT['candidate' if arm['candidate'] else 'control']: arm['repeats']}
    return raw


class PackedBlockTest(unittest.TestCase):
    def check(self, raw): return screen.analyze(raw, False, dict(build='build',artifact_revision='revision'), {})

    def test_materiality_and_exactness(self):
        raw = packed_fixture(); self.assertTrue(self.check(raw)['advance_to_verifier_screen'])
        for case in raw['cases']:
            for pair in case['pairs']:
                for arm in pair['arms']:
                    if arm['candidate']: arm['gpu_ns'] = 900000*32
        self.assertFalse(self.check(raw)['advance_to_verifier_screen'])
        raw['cases'][0]['pairs'][0]['arms'][1]['output_sha256'] = 'b'*64
        with self.assertRaises(ValueError): self.check(raw)

    def test_selection_and_incomplete_evidence_rejected(self):
        def wrong_kernel(r): r['cases'][0]['pairs'][0]['arms'][1]['kernel_dispatches'] = {'q8_mm_r2_t4': 32}
        def missing_dispatch(r): r['cases'][0]['pairs'][0]['arms'][0]['kernel_dispatches']['q8_mm_t4'] = 31
        def wrong_variant(r): r['variant']['output_rows'] = 2
        def missing_shape(r): r['cases'].pop()
        def invalid_time(r): r['cases'][0]['pairs'][0]['arms'][0]['gpu_ns'] = 0
        def missing_hash(r): r['cases'][0]['reference_sha256'] = ''
        for mutate in (wrong_kernel, missing_dispatch, wrong_variant, missing_shape, invalid_time, missing_hash):
            with self.subTest(mutation=mutate.__name__):
                raw = packed_fixture(); mutate(raw)
                with self.assertRaises(ValueError): self.check(raw)

    def test_disturbed_memory_does_not_advance(self):
        raw = packed_fixture(); raw['cases'][0]['warmup'][0]['memory_after']['compressed_peak_bytes'] = 1
        self.assertFalse(self.check(raw)['advance_to_verifier_screen'])

    def test_build_links_isolated_metal_before_archive(self):
        cfg = builder.settings(Path('/tmp/q8-block-packed-test')); sources = builder.sources(cfg)
        self.assertIn('-fobjc-arc', cfg['compiler'][0])
        self.assertLess(cfg['linker'].index(str(cfg['objects'][0])), cfg['linker'].index('libzerocool_lib.a'))
        self.assertIn('std::string(MetalSource)+', sources[cfg['generated'][0]])
        self.assertIn('cfg.affine_rows=1;', sources[cfg['generated'][1]])
        self.assertIn('else gpu.linear_into(l,x,4,{out});', sources[cfg['generated'][1]])
        self.assertNotIn('candidate?2:1', sources[cfg['generated'][1]])


if __name__ == '__main__': unittest.main()
