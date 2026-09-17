#!/usr/bin/env python3
"""Bounded independent CPU reference for four real mixed-precision shared experts.

Only selected tensor bytes are read. A valid existing checkpoint receipt is
required; this tool never hashes entire model shards or runs MLX/Metal.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np

from cache_residency import require
from qualification_evidence import save, sha
from reference_numpy import bf, silu
from native_q4_replay import valid_sha256
from screen_q4_packed import FIXTURES
from verify_checkpoint import fingerprint

ROOT = Path(__file__).resolve().parents[2]
LAYERS = (0, 16, 32, 47)
REVISION = 'b2c422f3c643e36f04227a64d61796b44a4b1029'
TOLERANCE = dict(vector_relative_l2_max=0.01, vector_cosine_min=0.99995,
                 scalar_absolute_max=1e-6, scalar_relative_max=1/128)
WIDTHS = dict(activation=640, shared=2560, gate=1)


def affine_q8(weight, scales, biases, group=64):
    """Decode unsigned little-endian affine Q8 into a mathematical FP64 matrix."""
    weight = np.asarray(weight)
    require(weight.dtype == np.dtype('<u4') and weight.ndim == 2 and group == 64,
            'Expected packed affine Q8/64 matrix')
    rows, packed = weight.shape
    require(packed*4 % group == 0 and np.shape(scales) == np.shape(biases) == (rows, packed*4//group),
            'Q8 scale/bias shape mismatch')
    codes = ((weight[..., None] >> np.arange(0, 32, 8, dtype=np.uint32)) & 255).reshape(rows, packed*4)
    decoded = codes.astype(np.float64)*np.repeat(scales, group, axis=-1)+np.repeat(biases, group, axis=-1)
    require(np.isfinite(decoded).all(), 'Non-finite decoded shared weights')
    return decoded


def shared_outputs(linear, x):
    """Keep model BF16 boundaries; CPU dot/reduction order is independent."""
    gate = linear('shared_expert.gate_proj', x)
    up = linear('shared_expert.up_proj', x)
    activation = bf(silu(gate)*up)
    return dict(activation=activation, shared=linear('shared_expert.down_proj', activation),
                gate=linear('shared_expert_gate', x))


class SelectedWeights:
    def __init__(self, model, lock_path):
        self.model = Path(model)
        lock = json.loads(Path(lock_path).read_text())
        receipt_path = self.model/'freellm-verification.json'
        receipt = json.loads(receipt_path.read_text())
        require(lock.get('revision') == receipt.get('revision') == REVISION, 'Wrong mixed checkpoint revision')
        self.lock = {e['path']: e for e in lock['files'] if not e.get('optional')}
        self.receipt = receipt; self.opened = {}; self.headers = {}; self.states = {}; self.tensors = {}
        self.receipt_sha256 = sha(receipt_path)
        for name in ('config.json', 'model.safetensors.index.json'):
            self.verify_file(name)
            require(sha(self.model/name) == self.lock[name]['sha256'], 'Changed checkpoint metadata')
        self.config = json.loads((self.model/'config.json').read_text())
        self.index = json.loads((self.model/'model.safetensors.index.json').read_text())['weight_map']

    def verify_file(self, name):
        require(Path(name).name == name and name in self.lock, 'Unpinned checkpoint filename')
        current = fingerprint(self.model/name)
        require(current['size'] == self.lock[name]['size'] and
                self.receipt['files'].get(name) == dict(current, sha256=self.lock[name]['sha256']),
                'Stale checkpoint receipt; verify separately before generating bounded fixtures')
        self.states[name] = current

    def get(self, name, shape, dtype):
        full = 'language_model.'+name
        filename = self.index[full]
        if filename not in self.opened:
            self.verify_file(filename)
            stream = (self.model/filename).open('rb', buffering=0)
            self.opened[filename] = stream
            if os.uname().sysname == 'Darwin':
                import fcntl
                fcntl.fcntl(stream.fileno(), 48, 1)
            prefix = stream.read(8)
            require(len(prefix) == 8, 'Truncated safetensors prefix')
            size, = struct.unpack('<Q', prefix)
            require(0 < size <= 16*1024**2, 'Unbounded safetensors metadata')
            data = stream.read(size)
            require(len(data) == size, 'Truncated safetensors metadata')
            self.headers[filename] = (8+size, json.loads(data))
        start, header = self.headers[filename]; tensor = header[full]
        require(tensor.get('dtype') == dtype and tensor.get('shape') == list(shape), 'Changed shared tensor layout')
        lo, hi = tensor['data_offsets']; size = int(np.prod(shape))*(4 if dtype == 'U32' else 2)
        require(type(lo) is int and type(hi) is int and 0 <= lo < hi and hi-lo == size and
                start+hi <= self.states[filename]['size'], 'Shared tensor range outside pinned shard')
        raw = os.pread(self.opened[filename].fileno(), size, start+lo)
        require(len(raw) == size, 'Truncated shared tensor read')
        self.tensors[name] = dict(file=filename, offset=start+lo, bytes=size, shape=list(shape), dtype=dtype,
                                  sha256=hashlib.sha256(raw).hexdigest())
        value = np.frombuffer(raw, dtype='<u4' if dtype == 'U32' else '<u2').reshape(shape)
        return value if dtype == 'U32' else (value.astype(np.uint32) << 16).view(np.float32)

    def linear(self, name, x, output):
        x = np.asarray(x, np.float32); width = x.shape[-1]
        if name.endswith('.shared_expert_gate'):
            require(name not in self.config['quantization'] and output == 1, 'Shared gate unexpectedly quantized')
            matrix = self.get(name+'.weight', (1, width), 'BF16').astype(np.float64)
        else:
            require(self.config['quantization'].get(name) == dict(bits=8, group_size=64), 'Shared projection is not Q8/64')
            matrix = affine_q8(self.get(name+'.weight', (output, width//4), 'U32'),
                               self.get(name+'.scales', (output, width//64), 'BF16'),
                               self.get(name+'.biases', (output, width//64), 'BF16'))
        # Q8 uses ordinary FP32 input sums; the Q4 bias-rounding correction
        # must not be applied. FP64 dot supplies an independent numeric oracle.
        return bf(x.astype(np.float64) @ matrix.T)

    def proof(self):
        require(sha(self.model/'freellm-verification.json') == self.receipt_sha256 and
                self.states == {name: fingerprint(self.model/name) for name in self.states},
                'Checkpoint changed during shared reference computation')
        return dict(receipt_sha256=self.receipt_sha256,
                    files={name: dict(state, sha256=self.lock[name]['sha256']) for name, state in self.states.items()},
                    tensors=self.tensors)

    def close(self):
        for stream in self.opened.values(): stream.close()


def payload(directory, entry):
    name = entry['file']
    require(Path(name).name == name, 'Unsafe shared fixture filename')
    raw = (Path(directory)/name).read_bytes()
    require(len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256'], 'Changed shared fixture bytes')
    return raw


def generate(model, fixtures, output, lock_path=ROOT/'mixed-models.lock.json'):
    fixtures, output = Path(fixtures), Path(output)
    captured = json.loads((fixtures/'manifest.json').read_text()); capture_sha = sha(fixtures/'manifest.json')
    require(captured.get('complete') is True and captured.get('artifact_revision') == REVISION and
            captured.get('origin') == 'existing-Q4-experts' and set(captured['layers']) == {str(l) for l in LAYERS},
            'Requires the complete mixed-artifact four-layer activation capture')
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(kind='independent_shared_expert_reference_v1', complete=False, artifact_revision=REVISION,
        artifact_lock_sha256=sha(lock_path), fixture_manifest_sha256=capture_sha, tolerance=TOLERANCE,
        arithmetic='Independent FP64 decoded-weight CPU dot; BF16 projection and activation boundaries; no Q4 bias correction',
        generator_sha256=sha(Path(__file__)), bf16_reference_sha256=sha(ROOT/'scripts/qwen/reference_numpy.py'),
        numpy_version=np.__version__, layers={}, files={},
        native_bit_exact_qualified=False, full_model_verified=False, quality_qualified=False)
    reader = SelectedWeights(model, lock_path)
    try:
        for layer in LAYERS:
            item = captured['layers'][str(layer)]
            raw = payload(fixtures, item['inputs']); x = np.frombuffer(raw, '<f4')
            require(x.size == 8*2560 and np.isfinite(x).all() and len(item['offsets']) == 8 and
                    len(set(item['offsets'])) == 8, 'Invalid saved shared-expert input rows')
            x = x.reshape(8, 2560); base = f'model.layers.{layer}.mlp.'
            values = shared_outputs(lambda suffix, value: reader.linear(base+suffix, value,
                1 if suffix == 'shared_expert_gate' else 2560 if suffix.endswith('down_proj') else 640), x)
            row = dict(inputs=item['inputs'], offsets=item['offsets'],
                       tensors={k: v for k, v in reader.tensors.items() if k.startswith(base)})
            for name, array in values.items():
                require(array.shape == (8, WIDTHS[name]) and np.isfinite(array).all(), 'Non-finite shared reference result')
                data = np.asarray(array, '<f4').tobytes(); filename = f'shared-{layer}.{name}.f32'
                (output/filename).write_bytes(data)
                entry = dict(file=filename, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), shape=list(array.shape))
                row[name] = entry; manifest['files'][filename] = entry
            manifest['layers'][str(layer)] = row
        manifest['source_proof'] = reader.proof()
        require(sha(fixtures/'manifest.json') == capture_sha, 'Activation manifest changed during reference computation')
        for item in captured['layers'].values(): payload(fixtures, item['inputs'])
        manifest['complete'] = True; validate_manifest(manifest); save(output/'manifest.json', manifest)
        return manifest
    finally:
        reader.close()


def compare_rows(actual, expected, scalar=False):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape and actual.ndim == 2 and actual.shape[0] > 0 and
            (actual.shape[1] == 1 if scalar else actual.shape[1] > 1), 'Shared output shape mismatch')
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), 'Non-finite shared output')
    checks = []
    for index, (a, e) in enumerate(zip(actual.astype(np.float64), expected.astype(np.float64))):
        norm = float(np.linalg.norm(e)); actual_norm = float(np.linalg.norm(a)); error = float(np.linalg.norm(a-e))
        relative = error/max(norm, 1e-30)
        cosine = float(np.dot(a, e)/(norm*actual_norm)) if norm and actual_norm else 1.0 if error == 0 else 0.0
        passed = error <= TOLERANCE['scalar_absolute_max']+TOLERANCE['scalar_relative_max']*norm if scalar else (
            relative <= TOLERANCE['vector_relative_l2_max'] and cosine >= TOLERANCE['vector_cosine_min'])
        checks.append(dict(row=index, passed=bool(passed), relative_l2=relative, cosine=cosine,
                           max_abs=float(np.max(np.abs(a-e))), bit_identical=a.tobytes() == e.tobytes()))
    return checks


def validate_manifest(manifest):
    require(manifest.get('kind') == 'independent_shared_expert_reference_v1' and manifest.get('complete') is True and
            manifest.get('artifact_revision') == REVISION and manifest.get('tolerance') == TOLERANCE and
            set(manifest['layers']) == {str(l) for l in LAYERS}, 'Incomplete shared CPU reference')
    require(all(valid_sha256(manifest.get(key)) for key in
                ('artifact_lock_sha256', 'fixture_manifest_sha256', 'generator_sha256', 'bf16_reference_sha256')) and
            isinstance(manifest.get('numpy_version'), str) and manifest['numpy_version'] and
            manifest.get('native_bit_exact_qualified') is False and manifest.get('full_model_verified') is False and
            manifest.get('quality_qualified') is False, 'Missing reference provenance or invalid qualification claim')
    proof = manifest['source_proof']
    require(valid_sha256(proof.get('receipt_sha256')) and isinstance(proof.get('files'), dict) and
            isinstance(proof.get('tensors'), dict), 'Missing selected checkpoint proof')
    require(all(Path(name).name == name and valid_sha256(entry.get('sha256')) and
                all(type(entry.get(k)) is int and entry[k] >= 0 for k in
                    ('size', 'device', 'inode', 'mtime_ns', 'ctime_ns'))
                for name, entry in proof['files'].items()), 'Invalid verified source file fingerprints')
    expected_files = set(); expected_tensors = set()
    for layer in LAYERS:
        row = manifest['layers'][str(layer)]; base = f'model.layers.{layer}.mlp.'
        source = row['inputs']
        require(Path(source['file']).name == source['file'] and source.get('bytes') == 8*2560*4 and
                valid_sha256(source.get('sha256')) and isinstance(row.get('offsets'), list) and len(row['offsets']) == 8 and
                all(type(p) is int and 0 <= p < 8192 for p in row['offsets']) and
                row['offsets'] == sorted(set(row['offsets'])), 'Invalid reference input coverage')
        tensors = {}
        for projection, shape in (('gate_proj', (640, 2560)), ('up_proj', (640, 2560)), ('down_proj', (2560, 640))):
            output, width = shape
            for suffix in ('weight', 'scales', 'biases'):
                tensors[base+'shared_expert.'+projection+'.'+suffix] = (
                    [output, width//(4 if suffix == 'weight' else 64)], 'U32' if suffix == 'weight' else 'BF16')
        tensors[base+'shared_expert_gate.weight'] = ([1, 2560], 'BF16')
        require(set(row.get('tensors', {})) == set(tensors), 'Incomplete shared tensor coverage')
        for name, (shape, dtype) in tensors.items():
            tensor = row['tensors'][name]; filename = tensor.get('file')
            require(filename in proof['files'] and Path(filename).name == filename and
                    tensor.get('shape') == shape and tensor.get('dtype') == dtype and
                    tensor.get('bytes') == int(np.prod(shape))*(4 if dtype == 'U32' else 2) and
                    type(tensor.get('offset')) is int and tensor['offset'] >= 8 and
                    tensor['offset']+tensor['bytes'] <= proof['files'][filename]['size'] and
                    valid_sha256(tensor.get('sha256')) and tensor == proof['tensors'].get(name),
                    'Changed selected tensor provenance or layout')
            expected_tensors.add(name)
        for name, width in WIDTHS.items():
            entry = row[name]
            require(entry.get('shape') == [8, width] and entry['bytes'] == 8*width*4 and
                    entry.get('file') == f'shared-{layer}.{name}.f32' and valid_sha256(entry.get('sha256')) and
                    manifest['files'].get(entry['file']) == entry, 'Changed shared output layout')
            expected_files.add(entry['file'])
    require(set(manifest['files']) == expected_files and set(proof['tensors']) == expected_tensors,
            'Unexpected shared reference output or tensor coverage')
    return manifest


def verify_reference(reference, fixtures=FIXTURES):
    reference, fixtures = Path(reference), Path(fixtures)
    manifest = validate_manifest(json.loads((reference/'manifest.json').read_text()))
    require(manifest['fixture_manifest_sha256'] == sha(fixtures/'manifest.json'), 'Changed activation fixture manifest')
    captured = json.loads((fixtures/'manifest.json').read_text())
    require(captured.get('complete') is True and captured.get('artifact_revision') == REVISION,
            'Changed activation source artifact')
    for layer in LAYERS:
        row = manifest['layers'][str(layer)]; original = captured['layers'][str(layer)]
        require(row['inputs'] == original['inputs'] and row['offsets'] == original['offsets'],
                'Changed shared reference input identity')
        payload(fixtures, row['inputs'])
        for name in WIDTHS:
            raw = payload(reference, row[name])
            require(np.isfinite(np.frombuffer(raw, '<f4')).all(), 'Non-finite shared reference fixture')
    return manifest


def check_outputs(reference, actual, fixtures=FIXTURES):
    reference, actual = Path(reference), Path(actual)
    manifest = verify_reference(reference, fixtures); checks = []
    for layer in LAYERS:
        for name, width in WIDTHS.items():
            entry = manifest['layers'][str(layer)][name]
            expected = np.frombuffer(payload(reference, entry), '<f4').reshape(8, width)
            raw = (actual/entry['file']).read_bytes()
            require(len(raw) == entry['bytes'], 'Truncated native shared output')
            got = np.frombuffer(raw, '<f4').reshape(8, width)
            checks.append(dict(layer=layer, output=name, native_sha256=hashlib.sha256(raw).hexdigest(),
                               rows=compare_rows(got, expected, scalar=name == 'gate')))
    return dict(kind='independent_shared_expert_output_check_v1', complete=True,
        passed=all(r['passed'] for c in checks for r in c['rows']), reference_manifest_sha256=sha(reference/'manifest.json'),
        tolerance=TOLERANCE, checks=checks, native_bit_exact_qualified=False, full_model_verified=False,
        limitations=['Independent numeric comparison only; native control/candidate byte parity and sealed run identity remain separate requirements.',
                     'CPU dots use FP64 accumulation, not the native SIMD reduction; small BF16 rounding differences are expected.',
                     'The scalar gate is checked with absolute plus one-BF16-spacing relative tolerance; vector rows retain the existing CPU operator tolerance.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='mode', required=True)
    create = sub.add_parser('generate'); create.add_argument('--model', type=Path, default=ROOT/'.cache/qwen-mixed-reference')
    create.add_argument('--fixtures', type=Path, default=FIXTURES); create.add_argument('--output', type=Path, required=True)
    compare = sub.add_parser('check'); compare.add_argument('--reference', type=Path, required=True)
    compare.add_argument('--actual', type=Path, required=True); compare.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'generate': generate(args.model, args.fixtures, args.output)
    else:
        result = check_outputs(args.reference, args.actual); save(args.output, result)
        raise SystemExit(0 if result['passed'] else 1)
