#!/usr/bin/env python3
"""Losslessly prepare pinned Q4 experts/ngrams with bounded memory.

Each file is atomically published with a SHA256 receipt. Interrupted runs reuse
unchanged completed files; --verify rehashes payloads without converting them.
The input checkpoint is never modified except for refreshing its verification
receipt. Production readers require no Python or NumPy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time

from verify_checkpoint import ROOT, fingerprint, verify

ALIGN = 16384
EXPERT_BYTES = 2764800
STRIDE = 2768896


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def nocache(fd):
    if sys.platform == 'darwin':
        import fcntl
        fcntl.fcntl(fd, 48, 1)


def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb', buffering=0) as f:
        nocache(f.fileno())
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


def prepare(model, output, force_verify=False):
    lock = json.loads((ROOT / 'models.lock.json').read_text())
    source = verify(model, lock, cached=True)
    revision = lock['revision']
    output.mkdir(parents=True, exist_ok=True)
    # Fail on a different preparation's output rather than mixing recipes.
    identity = dict(schema=1, source_revision=revision, format='zc-affine-records-v1')
    identity_path = output / 'preparation.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Output belongs to a different preparation')
    atomic_json(identity_path, identity)
    receipt_path = output / 'verification.json'
    old = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    receipt = dict(identity, files={})
    refs, fds = {}, {}
    try:
        weight_map = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
        for name in sorted(set(weight_map.values())):
            if Path(name).name != name or name in ('.', '..'):
                raise ValueError('Unsafe source filename')
            fd = os.open(model / name, os.O_RDONLY)
            fds[name] = fd
            nocache(fd)
            header_size, = struct.unpack('<Q', os.pread(fd, 8, 0))
            header = json.loads(os.pread(fd, header_size, 8))
            for key, value in header.items():
                if key == '__metadata__':
                    continue
                refs[key.removeprefix('language_model.')] = dict(value, file=name, base=8 + header_size)

        def part(key, row, count=1):
            r = refs[key]
            begin, end = r['data_offsets']
            width = (end - begin) // r['shape'][0]
            if row < 0 or row + count > r['shape'][0]:
                raise ValueError('Source row out of bounds')
            b = os.pread(fds[r['file']], count * width, r['base'] + begin + row * width)
            if len(b) != count * width:
                raise IOError('Short source read')
            return b

        files, expert_layers, ngram_shards = [], [], []
        expected_bytes = 48 * 512 * STRIDE
        base = 'model.layers.1.ple.ple_embedding.ngram_embedding.shard_'
        expected_bytes += sum(refs[f'{base}{i}.weight']['shape'][0] * 100 for i in range(128))
        existing = sum(p.stat().st_size for p in output.glob('*.bin'))
        if not force_verify and shutil.disk_usage(output).free < max(0, expected_bytes - existing) + (2 << 30):
            raise RuntimeError('Insufficient space for prepared records plus 2GiB headroom')

        def publish(name, blocks, expected):
            path = output / name
            saved = old.get('files', {}).get(name, {})
            if path.exists() and all(saved.get(k) == v for k, v in fingerprint(path).items()):
                if path.stat().st_size != expected:
                    raise ValueError('Completed file size mismatch')
                if force_verify and digest_file(path) != saved.get('sha256'):
                    raise ValueError(f'Corrupt prepared file: {name}')
                entry = saved
            else:
                if force_verify:
                    raise ValueError(f'Missing or changed prepared file: {name}')
                tmp = output / (name + '.partial')
                h, written = hashlib.sha256(), 0
                with tmp.open('wb', buffering=0) as f:
                    nocache(f.fileno())
                    for block in blocks():
                        h.update(block)
                        view = memoryview(block).cast('B')
                        while view:
                            n = f.write(view)
                            if not n:
                                raise IOError('Short prepared write')
                            written += n
                            view = view[n:]
                    os.fsync(f.fileno())
                if written != expected:
                    raise ValueError(f'Unexpected prepared size: {name}')
                tmp.replace(path)
                entry = dict(fingerprint(path), sha256=h.hexdigest())
            receipt['files'][name] = entry
            # Preserve previous completed entries while an interrupted run resumes.
            old.setdefault('files', {})[name] = entry
            atomic_json(receipt_path, dict(identity, files=old['files']))
            files.append(dict(path=name, size=expected, sha256=entry['sha256']))
            print(f'prepared {name}: {expected} bytes', flush=True)

        projections = [p + '.' + suffix for p in ('gate_proj', 'up_proj', 'down_proj')
                       for suffix in ('weight', 'scales', 'biases')]
        layout, cursor = [], 0
        for p in projections:
            r = refs['model.layers.0.mlp.switch_mlp.' + p]
            size = (r['data_offsets'][1] - r['data_offsets'][0]) // 512
            layout.append(dict(name=p, offset=cursor, length=size, shape=r['shape'][1:],
                               dtype=r['dtype'], bits=4, group_size=64))
            cursor += size
        if cursor != EXPERT_BYTES:
            raise ValueError('Unsupported expert format')
        for layer in range(48):
            def blocks(layer=layer):
                for expert in range(512):
                    for p in projections:
                        yield part(f'model.layers.{layer}.mlp.switch_mlp.{p}', expert)
                    yield bytes(STRIDE - EXPERT_BYTES)
            name = f'experts-{layer:02d}.bin'
            publish(name, blocks, STRIDE * 512)
            expert_layers.append(dict(layer=layer, file=name, count=512, offset=0,
                                      length=EXPERT_BYTES, stride=STRIDE, alignment=ALIGN))

        import numpy as np
        for shard in range(128):
            count = refs[f'{base}{shard}.weight']['shape'][0]

            def blocks(shard=shard, count=count):
                for at in range(0, count, 32768):
                    n = min(32768, count - at)
                    out = np.empty((n, 100), dtype=np.uint8)
                    for suffix, begin, width in [('weight', 0, 80), ('scales', 80, 10), ('biases', 90, 10)]:
                        data = part(f'{base}{shard}.{suffix}', at, n)
                        out[:, begin:begin+width] = np.frombuffer(data, dtype=np.uint8).reshape(n, width)
                    yield out
            name = f'ngram-{shard:03d}.bin'
            publish(name, blocks, count * 100)
            ngram_shards.append(dict(shard=shard, file=name, count=count, offset=0, stride=100,
                                     fields=[dict(offset=0, length=80, dtype='U32', shape=[20]),
                                             dict(offset=80, length=10, dtype='BF16', shape=[5]),
                                             dict(offset=90, length=10, dtype='BF16', shape=[5])]))
        for name, saved in source['files'].items():
            if any(saved[k] != v for k, v in fingerprint(model / name).items()):
                raise RuntimeError(f'Source changed during preparation: {name}')
        manifest = dict(identity, recipe='q4-control-lossless', expert_layout=layout,
                        experts=expert_layers, ngrams=ngram_shards, files=files,
                        prepared_bytes=expected_bytes, ngram_cache_dtype='BF16',
                        source_files={n: x['sha256'] for n, x in source['files'].items()})
        serialized = (json.dumps(manifest, indent=2) + '\n').encode()
        expected = lock.get('prepared_control', {}).get('manifest_sha256')
        if expected and hashlib.sha256(serialized).hexdigest() != expected:
            raise ValueError('Prepared manifest differs from the pinned lossless control')
        atomic_json(output / 'manifest.json', manifest)
        receipt['manifest_sha256'] = digest_file(output / 'manifest.json')
        receipt['verified_at'] = int(time.time())
        atomic_json(receipt_path, receipt)
    finally:
        for fd in fds.values():
            os.close(fd)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, default=ROOT / '.cache/models/qwen38-flash-next')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--verify', action='store_true')
    args = ap.parse_args()
    prepare(args.model, args.output, args.verify)
