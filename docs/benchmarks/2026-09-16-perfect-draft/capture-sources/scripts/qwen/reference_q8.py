#!/usr/bin/env python3
"""Bounded, hash-verified mixed-checkpoint Q8 operator fixtures from MLX.

This is an offline numerical oracle, not a model runner or quality evaluation.
Read only selected matrix rows after verifying the pinned source shard.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from verify_checkpoint import fingerprint


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        fcntl.fcntl(f.fileno(), 48, 1)  # Darwin F_NOCACHE
        for data in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(data)
    return h.hexdigest()


def run(args):
    if mx.__version__ != '0.31.1':
        raise ValueError('The reference requires MLX 0.31.1')
    lock = json.loads(args.lock.read_text())
    if args.download:
        from huggingface_hub import hf_hub_download
        args.source.mkdir(parents=True, exist_ok=True)
        missing = [item for item in lock['files'] if not (args.source/item['path']).exists()]
        if shutil.disk_usage(args.source).free < sum(item['size'] for item in missing) + 5*1024**3:
            raise ValueError('Insufficient disk space for the reference shard and 5GiB reserve')
        # Never replace an existing artifact, including a mistakenly selected Q4 directory.
        for item in lock['files']:
            path = args.source/item['path']
            if path.exists() and (path.stat().st_size != item['size'] or digest(path) != item['sha256']):
                raise ValueError(f'Refusing to replace a different existing artifact: {path}')
        for item in missing:
            hf_hub_download(lock['repo'], item['path'], revision=lock['revision'], local_dir=args.source)
    source_state = {item['path']: fingerprint(args.source/item['path']) for item in lock['files']}
    for item in lock['files']:
        path = args.source / item['path']
        if path.stat().st_size != item['size'] or digest(path) != item['sha256']:
            raise ValueError(f'Pinned source mismatch: {path}')
    config = json.loads((args.source / 'config.json').read_text())
    shard = args.source / 'model-00011.safetensors'
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema=1, kind='mixed_q8_operator_reference', source=lock,
                    lock_sha256=digest(args.lock), generator_sha256=digest(Path(__file__)),
                    mlx=mx.__version__, files={}, matrices={}, cases=[],
                    arithmetic='BF16 inputs, fixed per-token MLX QMV; batch QMM is a separate tolerance comparison',
                    full_model_verified=False, quality_qualified=False)
    mx.set_cache_limit(32 * 1024**2)

    def save(name, data):
        raw = data if isinstance(data, bytes) else np.asarray(data).tobytes()
        (args.output / name).write_bytes(raw)
        manifest['files'][name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        return name

    with shard.open('rb') as f:
        fcntl.fcntl(f.fileno(), 48, 1)
        header_bytes = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_bytes))

        def matrix(name, base, count, start=0):
            fmt = config['quantization'][base]
            if fmt != dict(group_size=64, bits=8):
                raise ValueError(f'Not affine Q8/64: {base}')
            out = dict(source_name=base, start_row=start, rows=count, bits=8, group=64, tensors={})
            arrays = []
            for suffix, dtype in [('weight', '<u4'), ('scales', '<u2'), ('biases', '<u2')]:
                tensor = header['language_model.' + base + '.' + suffix]
                rows, cols = tensor['shape']
                if count <= 0 or start < 0 or start + count > rows:
                    raise ValueError('Fixture row range')
                if tensor['dtype'] != ('U32' if suffix == 'weight' else 'BF16'):
                    raise ValueError('Fixture source dtype')
                stride = cols * np.dtype(dtype).itemsize
                offset = 8 + header_bytes + tensor['data_offsets'][0] + start * stride
                raw = os.pread(f.fileno(), count * stride, offset)
                if len(raw) != count * stride:
                    raise ValueError('Truncated fixture source')
                filename = save(name + '.' + suffix, raw)
                out['tensors'][suffix] = dict(file=filename, source_offset=offset, shape=[count, cols])
                a = np.frombuffer(raw, dtype=dtype).reshape(count, cols)
                if suffix != 'weight':
                    a = (a.astype(np.uint32) << 16).view(np.float32)
                arrays.append(mx.array(a) if suffix == 'weight' else mx.array(a).astype(mx.bfloat16))
            out['input'] = out['tensors']['weight']['shape'][1] * 4
            manifest['matrices'][name] = out
            return arrays

        specifications = [
            ('shared_gate', 'model.layers.7.mlp.shared_expert.gate_proj', 640, 0),
            ('shared_up', 'model.layers.7.mlp.shared_expert.up_proj', 640, 0),
            ('shared_down', 'model.layers.7.mlp.shared_expert.down_proj', 32, 173),
            ('hyper_down', 'model.layers.7.mlp_hyper_connection.input_mix_weight_down', 32, 11),
            ('hyper_up', 'model.layers.7.mlp_hyper_connection.input_mix_weight_up', 7, 319),
            ('attention_q', 'model.layers.7.self_attn.q_proj', 32, 219),
            ('attention_o', 'model.layers.7.self_attn.o_proj', 7, 189),
            ('gdn_qkv', 'model.layers.8.linear_attn.in_proj_qkv', 32, 13),
            ('embedding', 'model.embed_tokens', 32, 760),
            ('lm_head', 'lm_head', 32, 9338),
        ]
        arrays = {name: matrix(name, base, count, start) for name, base, count, start in specifications}
        rng = np.random.default_rng(20260907)

        def qmv(x, weights):
            return mx.quantized_matmul(x, *weights, transpose=True, group_size=64, bits=8)

        def fp32(value):
            mx.eval(value)
            return np.asarray(value.astype(mx.float32))

        for name, weights in arrays.items():
            K = manifest['matrices'][name]['input']
            for tokens in (1, 3, 17):
                x = mx.array(rng.normal(0, 0.3, (tokens, K)).astype(np.float32)).astype(mx.bfloat16)
                expected = mx.concatenate([qmv(x[t:t+1], weights) for t in range(tokens)])
                prefix = f'{name}-{tokens}'
                case = dict(name=prefix, op='linear', matrix=name, tokens=tokens,
                            input=save(prefix + '.input', fp32(x)),
                            expected=save(prefix + '.expected', fp32(expected)),
                            batch_expected=save(prefix + '.batch', fp32(qmv(x, weights))))
                manifest['cases'].append(case)
                if name == 'shared_gate':
                    # Gather repeats and reverses input rows; preserve their output positions.
                    rows = list(reversed(range(tokens))) + [0]
                    gathered = x[mx.array(rows)]
                    gate = mx.concatenate([qmv(gathered[t:t+1], weights) for t in range(len(rows))])
                    up = mx.concatenate([qmv(gathered[t:t+1], arrays['shared_up']) for t in range(len(rows))])
                    manifest['cases'].append(dict(name=prefix+'-fused', op='gated', matrix=name,
                        up='shared_up', tokens=len(rows), rows=rows, input=case['input'],
                        expected=save(prefix+'.fused', fp32(nn.silu(gate)*up))))
            if name == 'embedding':
                ids = [31, 0, 13, 13, 7]
                decoded = mx.dequantize(*weights, group_size=64, bits=8)[mx.array(ids)]
                for copies in (1, 4):
                    manifest['cases'].append(dict(name=f'embedding-copies{copies}', op='embedding',
                        matrix=name, ids=ids, copies=copies,
                        expected=save(f'embedding-copies{copies}.expected', fp32(mx.tile(decoded, (1, copies))))))
            mx.clear_cache()
    if source_state != {item['path']: fingerprint(args.source/item['path']) for item in lock['files']}:
        raise ValueError('Source changed while generating the reference')
    manifest['reference_peak_bytes'] = mx.get_peak_memory()
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Wrote {len(manifest["cases"])} real Q8 cases to {args.output}', flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=root/'.cache/qwen-mixed-reference')
    parser.add_argument('--lock', type=Path, default=root/'q8-reference.lock.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--download', action='store_true', help='Fetch only missing files from the pinned reference lock')
    run(parser.parse_args())
