"""Reject misleading layout evidence before any payload reuse is possible."""
import copy
import hashlib
import json
import struct
import unittest
from unittest.mock import patch

from inspect_mixed import check_small, compare, download, parse, read_header


def encoded(tensors):
    raw = json.dumps(tensors).encode()
    return struct.pack('<Q', len(raw))+raw


def matrix(bits):
    base = 'language_model.model.layers.0.mlp.shared_expert.gate_proj'
    rows, width = 2, 64
    values = {}
    for suffix, dtype, cols, unit in [('weight', 'U32', width*bits//32, 4),
                                       ('scales', 'BF16', 1, 2), ('biases', 'BF16', 1, 2)]:
        values[base+'.'+suffix] = dict(dtype=dtype, shape=[rows, cols], bytes=rows*cols*unit)
    config = dict(quantization=dict(bits=4, group_size=64))
    config['quantization'][base.removeprefix('language_model.')] = dict(bits=bits, group_size=64)
    return values, config


class MixedInspection(unittest.TestCase):
    def test_header_rejects_incomplete_or_ambiguous_ranges(self):
        header = {'x': dict(dtype='BF16', shape=[4, 8], data_offsets=[0, 64])}
        raw = encoded(header)
        self.assertEqual(read_header(raw, len(raw)+64)['x']['bytes'], 64)
        for change in [dict(shape=[4.0, 8]), dict(shape=[True, 32]), dict(shape=[0, 8]),
                       dict(data_offsets=[1, 65]), dict(data_offsets=[0, 32]), dict(dtype='UNKNOWN')]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                altered = encoded({'x':dict(header['x'], **change)})
                read_header(altered, len(altered)+64)
        with self.assertRaises(ValueError):
            read_header(encoded(dict(header, y=header['x'])), len(encoded(dict(header, y=header['x'])))+128)
        with self.assertRaises(ValueError):
            read_header(raw[:-1], len(raw)+64)
        with self.assertRaises(ValueError):
            read_header(raw, len(raw)+65)
        with self.assertRaises(ValueError):
            parse('{"x": 1, "x": 2}')

    def test_mixed_geometry_allows_only_declared_resident_precision(self):
        q4, c4 = matrix(4)
        q8, c8 = matrix(8)
        result = compare(q4, q8, c4, c8)
        self.assertEqual(result['quantized_matrices'], {'resident_q8_g64':1})
        self.assertEqual(result['payloads']['resident'], dict(q4_bytes=72, mixed_bytes=136, tensors=3))
        for change in ('shape', 'dtype', 'group', 'missing', 'expert_precision'):
            b, config = copy.deepcopy(q8), copy.deepcopy(c8)
            a, original = copy.deepcopy(q4), copy.deepcopy(c4)
            key = next(iter(b))
            if change == 'shape':
                b[key]['shape'][-1] *= 2
            elif change == 'dtype':
                b[key.replace('weight', 'scales')]['dtype'] = 'F32'
            elif change == 'group':
                next(v for v in config['quantization'].values() if isinstance(v, dict))['bits'] = 3
            elif change == 'missing':
                del b[key]
            else:
                a = {k.replace('shared_expert', 'switch_mlp'):v for k,v in a.items()}
                b = {k.replace('shared_expert', 'switch_mlp'):v for k,v in b.items()}
                for c in (original, config):
                    c['quantization'] = {k.replace('shared_expert', 'switch_mlp'):v for k,v in c['quantization'].items()}
            with self.subTest(change=change), self.assertRaises(ValueError):
                compare(a, b, original, config)

    def test_metadata_uses_payload_hash_for_lfs(self):
        raw = b'payload'
        blob = hashlib.sha1(b'blob 7\0payload').hexdigest()
        check_small(raw, dict(size=7, blobId=blob))
        check_small(raw, dict(size=7, blobId='pointer is not a payload hash', lfs=dict(sha256=hashlib.sha256(raw).hexdigest())))
        with self.assertRaises(ValueError):
            check_small(raw, dict(size=7, blobId=blob, lfs=dict(sha256='wrong')))
        with self.assertRaises(ValueError):
            check_small(raw+b'x', dict(size=7, blobId=blob))

    def test_range_download_cannot_silently_read_a_whole_shard(self):
        class Response:
            def __init__(self, status, content_range, body):
                self.status, self.headers, self.body = status, {'Content-Range':content_range}, body
                self.read_called = False
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, size): self.read_called=True; return self.body[:size]
        for status, extent, body, valid in [(206, 'bytes 0-7/10000000', b'12345678', True),
                (200, None, b'12345678', False), (206, 'bytes 8-15/10000000', b'12345678', False),
                (206, 'bytes 0-7/10000000', b'1234', False),
                (206, 'bytes 0-7/10000000', b'123456789', False)]:
            response = Response(status, extent, body)
            with patch('urllib.request.urlopen', return_value=response):
                if valid:
                    self.assertEqual(download('https://example.test/file', 8, (0, 7), 10000000), body)
                else:
                    with self.assertRaises(ValueError):
                        download('https://example.test/file', 8, (0, 7), 10000000)
                    if status == 200 or extent != 'bytes 0-7/10000000':
                        self.assertFalse(response.read_called)


if __name__ == '__main__':
    unittest.main()
