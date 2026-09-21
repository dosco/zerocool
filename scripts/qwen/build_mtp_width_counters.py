#!/usr/bin/env python3
"""Isolated per-dispatch timing for the current-width target window."""
import argparse
import json
from pathlib import Path
import subprocess

import build_mtp_width_profile as base
from build_block_cache_trace import replace
from cache_residency import require
from qualification_evidence import save, sha

ROOT = base.ROOT


def generated(output):
    sources = base.generated(output)
    p = output/'include/engine/model.hpp'
    sources[p] = replace(sources[p],
        'options_.kernels.profile=active;options_.kernels.counter_profile=false;',
        'options_.kernels.profile=active;options_.kernels.counter_profile=active;')
    p = output/'probe.cpp'
    sources[p] = replace(sources[p], '"native_mtp_width_profile_v1"',
        '"native_mtp_width_counter_profile_v1"')
    return sources


def inputs(cfg): return [*base.inputs(cfg), Path(__file__).resolve()]


def proof(cfg):
    result = base.proof(cfg)
    return dict(result, kind='mtp_width_counter_profile_producer_v1', mode='dispatch',
        inputs={str(p):sha(p) for p in inputs(cfg)},
        generated={str(p):sha(p) for p in generated(cfg['output'])})


def build(output):
    cfg = base.base.settings(output); cfg['output'].mkdir(parents=True, exist_ok=False)
    for p,value in generated(cfg['output']).items():
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(value)
    frozen = proof(cfg); save(cfg['output']/'producer.json', dict(frozen, complete=False))
    with (cfg['output']/'build.log').open('w') as log:
        for cmd in [*cfg['compiler'],cfg['linker']]:
            subprocess.run(cmd,cwd=cfg['native'],stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
    require(frozen==proof(cfg), 'Counter build inputs changed')
    result = dict(frozen, complete=True, binary=str(cfg['binary']), binary_sha256=sha(cfg['binary']),
        objects={str(p):sha(p) for p in cfg['objects']})
    save(cfg['output']/'producer.json', result); return result


def verify(directory):
    cfg = base.base.settings(directory); saved=json.loads((cfg['output']/'producer.json').read_text())
    require(saved==dict(proof(cfg),complete=True,binary=str(cfg['binary']),binary_sha256=sha(cfg['binary']),
        objects={str(p):sha(p) for p in cfg['objects']}), 'Changed counter producer')
    for p,value in generated(cfg['output']).items():
        require(p.read_text()==value, 'Changed counter source copy')
    return cfg,saved


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    result=build(parser.parse_args().output)
    print(json.dumps({k:result[k] for k in ('complete','binary','binary_sha256')}))
