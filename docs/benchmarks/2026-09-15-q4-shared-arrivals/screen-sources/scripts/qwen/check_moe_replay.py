#!/usr/bin/env python3
"""Compare isolated native router/reduction replay with pinned original MLX ops.

Requires a saved five-token, 48-layer native trace and the qwen_moe_replay
output. Reuses recorded expert outputs; never claims full-forward agreement.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import numpy as np
import mlx.core as mx
from reference_numpy import Weights


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'trace', 'replay', 'replay-report', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    args = ap.parse_args()
    if mx.__version__ != '0.31.1' or version('mlx-lm') != '0.31.1':
        raise RuntimeError('Requires mlx==0.31.1 and mlx-lm==0.31.1')
    mx.set_cache_limit(64 * 1024**2)
    reader = Weights(args.model)
    machine = json.loads(args.replay_report.read_text())
    if machine.get('device') != 'Apple M1 Pro' or not machine.get('build_fingerprint'):
        raise ValueError('Expected a fingerprinted native M1 Pro replay report')
    checks, hashes = [], {}

    def read(root, name, shape, dtype=np.float32):
        path = root / (name + '.bin')
        if path.stat().st_size != int(np.prod(shape)) * np.dtype(dtype).itemsize:
            raise ValueError('Incorrect fixture size: ' + str(path))
        data = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
        a = np.frombuffer(data, dtype).reshape(shape)
        if not np.all(np.isfinite(a)):
            raise ValueError('Non-finite fixture: ' + str(path))
        return a

    for layer in range(48):
        b = f'model.layers.{layer}.mlp'
        x = mx.array(read(args.trace, f'x2_{layer}', (5, 2560)))
        router = x @ mx.array(reader.get(b + '.gate.weight')).T
        ids = mx.argpartition(-router, 9, axis=-1)[..., :10]
        weights = mx.softmax(mx.take_along_axis(router, ids, axis=-1), axis=-1, precise=True)
        original_ids = read(args.trace, f'route_{layer}', (5, 10), np.int32)
        native_ids = read(args.replay, f'route_{layer}', (5, 10), np.int32)
        row = {'layer': layer, 'selection_order_equal': bool(
            np.array_equal(np.array(ids), original_ids) and np.array_equal(native_ids, original_ids))}
        experts = mx.array(read(args.trace, b + '.expert_out', (5, 10, 2560))).astype(mx.bfloat16)
        shared = mx.array(read(args.trace, b + '.shared', (5, 2560))).astype(mx.bfloat16)
        gate = mx.array(read(args.trace, b + '.gate', (5, 1))).astype(mx.bfloat16)
        moe = (experts * weights[..., None]).sum(-2).astype(mx.bfloat16) + mx.sigmoid(gate) * shared
        for key, value in ((f'router_{layer}', router), (b + '.weights', weights), (f'moe_{layer}', moe)):
            reference = np.array(value.astype(mx.float32))
            native = read(args.replay, key, reference.shape)
            row[key.split('.')[-1]] = {'different_values': int(np.count_nonzero(native != reference)),
                                      'max_abs': float(np.max(np.abs(native - reference)))}
        row['passed'] = row['selection_order_equal'] and all(
            v['different_values'] == 0 for v in row.values() if isinstance(v, dict))
        checks.append(row)
    report = {'kind': 'same_input_router_and_expert_reduction', 'full_model_verified': False,
              'mlx': mx.__version__, 'native': machine, 'checks': checks,
              'fixture_sha256': hashes, 'passed': all(row['passed'] for row in checks)}
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(f"{sum(row['passed'] for row in checks)}/48 layers exactly match; full-model agreement is not checked")
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
