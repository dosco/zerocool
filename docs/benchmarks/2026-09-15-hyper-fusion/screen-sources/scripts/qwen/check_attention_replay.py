#!/usr/bin/env python3
"""Compare isolated production attention kernels against pinned MLX on saved Q/K/V.

The query/key/value projections are recorded inputs, not independently checked
here. The first complete attention-layer comparison uses reference_stages.py.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import numpy as np
import mlx.core as mx


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('replay', 'replay-report', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    args = ap.parse_args()
    if mx.__version__ != '0.31.1' or version('mlx-lm') != '0.31.1':
        raise RuntimeError('Requires mlx==0.31.1 and mlx-lm==0.31.1')
    mx.set_cache_limit(64 * 1024**2)
    machine = json.loads(args.replay_report.read_text())
    if machine.get('device') != 'Apple M1 Pro' or not machine.get('build_fingerprint'):
        raise ValueError('Expected a fingerprinted native M1 Pro replay report')
    checks, hashes = [], {}

    def read(name, shape):
        path = args.replay / (name + '.bin')
        if path.stat().st_size != int(np.prod(shape)) * 4:
            raise ValueError('Incorrect fixture size: ' + str(path))
        raw = path.read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        value = np.frombuffer(raw, np.float32).reshape(shape)
        if not np.all(np.isfinite(value)):
            raise ValueError('Non-finite fixture: ' + str(path))
        return value

    for layer in range(3, 48, 4):
        b = f'model.layers.{layer}.self_attn'
        def array(name, shape):
            return mx.array(read(b + '.' + name, shape)).astype(mx.bfloat16)
        q = array('q', (1, 5, 24, 256)).transpose(0, 2, 1, 3)
        k = array('k', (1, 5, 2, 256)).transpose(0, 2, 1, 3)
        v = array('v', (1, 5, 2, 256)).transpose(0, 2, 1, 3)
        qg = array('qg', (1, 5, 24, 512))
        result = mx.fast.scaled_dot_product_attention(q, k, v, scale=1/16, mask='causal')
        result = result.transpose(0, 2, 1, 3) * mx.sigmoid(qg[..., 256:])
        reference = np.array(result.astype(mx.float32))
        native = read(b + '.gated', reference.shape)
        different = int(np.count_nonzero(native != reference))
        checks.append({'layer': layer, 'different_values': different,
                       'max_abs': float(np.max(np.abs(native - reference))), 'passed': different == 0})
    report = {'kind': 'same_input_attention', 'full_model_verified': False, 'mlx': mx.__version__,
              'native': machine, 'fixture_sha256': hashes, 'checks': checks,
              'passed': all(c['passed'] for c in checks)}
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(f"{sum(c['passed'] for c in checks)}/12 attention layers exactly match on identical Q/K/V inputs")
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
