import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from shared_expert_reference import (LAYERS, REVISION, TOLERANCE, WIDTHS, SelectedWeights,
                                     affine_q8, compare_rows, payload, shared_outputs, validate_manifest)
from verify_checkpoint import fingerprint


def reference_fixture():
    """Small metadata-only complete oracle contract for runner/auditor tests."""
    digest = 'a'*64
    manifest = dict(kind='independent_shared_expert_reference_v1', complete=True, artifact_revision=REVISION,
        artifact_lock_sha256=digest, fixture_manifest_sha256=digest, tolerance=dict(TOLERANCE),
        generator_sha256=digest, bf16_reference_sha256=digest, numpy_version=np.__version__,
        layers={}, files={}, native_bit_exact_qualified=False, full_model_verified=False, quality_qualified=False,
        source_proof=dict(receipt_sha256=digest, files={}, tensors={}))
    for layer in LAYERS:
        row = dict(inputs=dict(file=f'inputs-{layer}.bin', bytes=81920, sha256=digest),
                   offsets=list(range(72, 80)), tensors={})
        filename = f'model-{layer:05}.safetensors'; offset = 64
        manifest['source_proof']['files'][filename] = dict(size=10_000_000, device=1, inode=layer,
                                                         mtime_ns=1, ctime_ns=1, sha256=digest)
        for projection, output, width in (('gate_proj', 640, 2560), ('up_proj', 640, 2560), ('down_proj', 2560, 640)):
            for suffix in ('weight', 'scales', 'biases'):
                shape = [output, width//(4 if suffix == 'weight' else 64)]
                size = int(np.prod(shape))*(4 if suffix == 'weight' else 2)
                key = f'model.layers.{layer}.mlp.shared_expert.{projection}.{suffix}'
                row['tensors'][key] = dict(file=filename, offset=offset, bytes=size, shape=shape,
                    dtype='U32' if suffix == 'weight' else 'BF16', sha256=digest)
                offset += size
        key = f'model.layers.{layer}.mlp.shared_expert_gate.weight'
        row['tensors'][key] = dict(file=filename, offset=offset, bytes=5120, shape=[1, 2560], dtype='BF16', sha256=digest)
        manifest['source_proof']['tensors'].update(copy.deepcopy(row['tensors']))
        for name, width in WIDTHS.items():
            entry = dict(file=f'shared-{layer}.{name}.f32', bytes=8*width*4, shape=[8, width], sha256=digest)
            row[name] = entry; manifest['files'][entry['file']] = copy.deepcopy(entry)
        manifest['layers'][str(layer)] = row
    return manifest


class SharedExpertReferenceTest(unittest.TestCase):
    def test_reference_contract_covers_every_tensor_stage_and_row(self):
        manifest = reference_fixture()
        self.assertEqual(validate_manifest(manifest), manifest)
        for change in ('missing_layer', 'tensor_hash', 'tensor_shape', 'input_rows', 'output_hash', 'tolerance', 'source'):
            altered = copy.deepcopy(manifest); row = altered['layers']['0']
            if change == 'missing_layer': altered['layers'].pop('47')
            elif change == 'tensor_hash': next(iter(row['tensors'].values()))['sha256'] = 'b'*64
            elif change == 'tensor_shape': next(iter(row['tensors'].values()))['shape'][0] -= 1
            elif change == 'input_rows': row['offsets'][-1] = row['offsets'][0]
            elif change == 'output_hash': row['shared']['sha256'] = 'b'*64
            elif change == 'tolerance': altered['tolerance']['scalar_relative_max'] = 1
            else: next(iter(altered['source_proof']['files'].values()))['sha256'] = None
            with self.subTest(change=change):
                with self.assertRaises(ValueError): validate_manifest(altered)

    def test_q8_unsigned_little_endian_codes_and_group_boundaries(self):
        codes = np.array(([0, 1, 127, 255]*16)+([255, 128, 2, 0]*16), dtype=np.uint8).reshape(1, 128)
        packed = codes.copy().view('<u4')
        result = affine_q8(packed, np.array([[0.5, -2.0]]), np.array([[-4.0, 1.0]]))
        expected = np.concatenate((codes[:, :64]*0.5-4, codes[:, 64:].astype(float)*-2+1), axis=1)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype, np.float64)

    def test_q8_rejects_other_bit_layouts_and_nonfinite_metadata(self):
        packed = np.zeros((1, 16), dtype='<u4')
        for weights, scales, biases, group in ((packed.astype('<u2'), [[1]], [[0]], 64),
                (packed, [[1, 1]], [[0]], 64), (packed, [[1]], [[0]], 32),
                (packed, [[float('nan')]], [[0]], 64)):
            with self.assertRaises(ValueError): affine_q8(weights, scales, biases, group)

    def test_shared_chain_preserves_activation_and_original_gate_input(self):
        x = np.array([[3, -2], [1, 4]], np.float32); seen = {}
        def linear(name, value):
            seen[name] = value.copy()
            if name.endswith('gate_proj'): return np.zeros((2, 3), np.float32)
            if name.endswith('up_proj'): return np.full((2, 3), 9, np.float32)
            if name.endswith('down_proj'): return np.full((2, 2), 7, np.float32)
            return np.array([[2], [-1]], np.float32)
        result = shared_outputs(linear, x)
        np.testing.assert_array_equal(result['activation'], np.zeros((2, 3)))
        np.testing.assert_array_equal(seen['shared_expert.down_proj'], result['activation'])
        np.testing.assert_array_equal(seen['shared_expert_gate'], x)
        np.testing.assert_array_equal(result['gate'], [[2], [-1]])  # Raw gate projection, before sigmoid.

    def test_per_row_tolerance_cannot_hide_one_failed_row(self):
        expected = np.ones((8, 640), np.float32); actual = expected.copy()
        actual[3] = 1.02
        checks = compare_rows(actual, expected)
        self.assertEqual([c['row'] for c in checks if not c['passed']], [3])
        actual = expected.copy(); actual[4, :320] *= -1
        self.assertFalse(compare_rows(actual, expected)[4]['passed'])

    def test_scalar_gate_has_explicit_absolute_and_relative_bound(self):
        expected = np.array([[0], [1], [-2]], np.float64)
        within = np.array([[1e-6], [1+1/128], [-2-2/128]], np.float64)
        self.assertTrue(all(r['passed'] for r in compare_rows(within, expected, scalar=True)))
        outside = np.array([[1.001e-6], [1.009], [-2.018]], np.float64)
        self.assertTrue(all(not r['passed'] for r in compare_rows(outside, expected, scalar=True)))

    def test_zero_and_nonfinite_outputs_fail_closed(self):
        self.assertTrue(compare_rows(np.zeros((1, 3)), np.zeros((1, 3)))[0]['passed'])
        self.assertFalse(compare_rows(np.ones((1, 3)), np.zeros((1, 3)))[0]['passed'])
        for value in (float('nan'), float('inf')):
            with self.assertRaises(ValueError): compare_rows([[value, 0]], [[0, 0]])
        with self.assertRaises(ValueError): compare_rows([[1, 2]], [[1]], scalar=True)

    def test_payload_hash_length_and_path_are_enforced(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name); raw = b'fixture'
            (path/'input.bin').write_bytes(raw)
            entry = dict(file='input.bin', bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            self.assertEqual(payload(path, entry), raw)
            for key, value in (('file', '../input.bin'), ('bytes', 1), ('sha256', '0'*64)):
                changed = dict(entry, **{key: value})
                with self.assertRaises(ValueError): payload(path, changed)

    def test_stale_receipts_never_trigger_implicit_shard_rehash(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)
            for filename, data in (('config.json', {'quantization': {}}),
                                   ('model.safetensors.index.json', {'weight_map': {}})):
                (path/filename).write_text(json.dumps(data))
            entries = [dict(path=p.name, size=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                       for p in sorted(path.iterdir())]
            lock = dict(revision=REVISION, files=entries)
            receipt = dict(revision=REVISION, files={e['path']: dict(fingerprint(path/e['path']), sha256=e['sha256'])
                                                   for e in entries})
            (path/'lock.json').write_text(json.dumps(lock))
            (path/'freellm-verification.json').write_text(json.dumps(receipt))
            weights = SelectedWeights(path, path/'lock.json'); weights.proof(); weights.close()
            changed = copy.deepcopy(receipt); changed['files']['config.json']['mtime_ns'] -= 1
            (path/'freellm-verification.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'Stale checkpoint receipt'):
                SelectedWeights(path, path/'lock.json')


if __name__ == '__main__': unittest.main()
