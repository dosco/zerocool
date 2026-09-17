#!/usr/bin/env python3
"""Check the mixed-aware full-model oracle against saved independent Q8 fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from reference_mlx import Reader, array, prepare
from verify_checkpoint import ROOT


def run(args):
    manifest = json.loads((args.fixture/'manifest.json').read_text())
    if mx.__version__ != '0.31.1' or manifest['mlx'] != mx.__version__:
        raise ValueError('Requires the pinned MLX 0.31.1 fixture')
    lock = (ROOT/'q8-reference.lock.json').read_bytes()
    if manifest['lock_sha256'] != hashlib.sha256(lock).hexdigest():
        raise ValueError('Fixture source lock differs')
    for name, entry in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('Unsafe fixture filename')
        raw = (args.fixture/name).read_bytes()
        if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError(f'Changed fixture: {name}')
    class Weights:
        def __init__(self): self.tensors = {}
        def get(self, name, row=None):
            value = self.tensors[name]
            return value if row is None else value[row]
    weights = Weights()
    config = dict(quantization=dict(bits=4, group_size=64))
    for matrix in manifest['matrices'].values():
        base = matrix['source_name']
        config['quantization'][base] = dict(bits=matrix['bits'], group_size=matrix['group'])
        for suffix, item in matrix['tensors'].items():
            raw = np.fromfile(args.fixture/item['file'], dtype='<u4' if suffix=='weight' else '<u2').reshape(item['shape'])
            weights.tensors[base+'.'+suffix] = raw if suffix=='weight' else (raw.astype(np.uint32)<<16).view(np.float32)
    reader = Reader(weights, config)
    checks = []
    for case in manifest['cases']:
        matrix = manifest['matrices'][case['matrix']]
        base = matrix['source_name']
        expected_file = case.get('batch_expected', case['expected'])
        if case['op'] == 'embedding':
            result = mx.tile(reader.embedding(base, case['ids']), (1, case['copies']))
        else:
            values = np.fromfile(args.fixture/case['input'], '<f4').reshape(-1, matrix['input'])
            x = array(values)
            if case['op'] == 'linear':
                result = reader.linear(base, x)
            elif case['op'] == 'gated':
                x = x[mx.array(case['rows'])]
                up = manifest['matrices'][case['up']]['source_name']
                result = mx.concatenate([nn.silu(reader.linear(base,x[i:i+1]))*reader.linear(up,x[i:i+1]) for i in range(len(case['rows']))])
            else:
                raise ValueError('Unknown fixture operation')
        mx.eval(result)
        raw = np.asarray(result.astype(mx.float32)).tobytes()
        expected = (args.fixture/expected_file).read_bytes()
        checks.append(dict(name=case['name'], passed=raw==expected, values=len(raw)//4))
    # Exercise prepare(): it must construct a Q8 module before loading Q8 codes.
    m = manifest['matrices']['shared_gate']
    prefix, leaf = m['source_name'].rsplit('.',1)
    module = nn.Module()
    module[leaf] = nn.Linear(m['input'], m['rows'], bias=False)
    prepare(module, prefix, reader)
    x = array(np.full((1,m['input']), 0.25, np.float32))
    a, b = module[leaf](x), reader.linear(m['source_name'],x)
    mx.eval(a,b)
    checks.append(dict(name='prepared_module_precision', passed=module[leaf].bits==8 and module[leaf].group_size==64 and bool(mx.array_equal(a,b))))
    report = dict(kind='mixed_full_oracle_q8_fixture_check', passed=all(c['passed'] for c in checks), checks=checks,
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        oracle_sha256=hashlib.sha256((ROOT/'scripts/qwen/reference_mlx.py').read_bytes()).hexdigest(),
        fixture_manifest_sha256=hashlib.sha256((args.fixture/'manifest.json').read_bytes()).hexdigest(),
        mlx=mx.__version__, full_mixed_model_verified=False)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(checks)} mixed reference checks: {report["passed"]}')
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
