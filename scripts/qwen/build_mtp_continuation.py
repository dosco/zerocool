#!/usr/bin/env python3
"""Build the bounded continuation harness without changing the short-screen producer."""
import argparse
import json
from pathlib import Path
import subprocess

import build_mtp_forward as base
from build_block_cache_trace import replace
from build_identity import build_fingerprint
from cache_residency import require
from qualification_evidence import save, sha

ROOT=base.ROOT
HARNESS=base.SCRIPTS/'probe_mtp_continuation.cpp'


def generated(output):
    sources=base.generated(output);path=output/'probe.cpp';source=sources[path]
    start=source.index('void joint(const std::filesystem::path& model_path,')
    end=source.index('\n}\n}\nint main(',start)+2
    source=source[:start]+HARNESS.read_text()+source[end:]
    source=replace(source,'"native_mtp_forward_v1"','"native_mtp_continuation_v1"')
    source=replace(source,'        check(argc==6,"usage:',
        '        if(argc==2 && std::string_view(argv[1])=="--continuation-self-test") {std::cout<<continuation_self_test().dump()<<\'\\n\';return 0;}\n'
        '        check(argc==6,"usage:')
    sources[path]=source
    return sources


def inputs(cfg):return [*base.inputs(cfg),Path(__file__).resolve(),HARNESS]


def proof(cfg):
    return dict(kind='native_mtp_continuation_producer_v1',base_native_fingerprint=build_fingerprint(ROOT),
        inputs={str(p):sha(p) for p in inputs(cfg)},generated={str(p):sha(p) for p in generated(cfg['output'])},
        compiler=cfg['compiler'],linker=cfg['linker'],production_promoted=False)


def build(output):
    cfg=base.settings(output);cfg['output'].mkdir(parents=True,exist_ok=False)
    for p,value in generated(cfg['output']).items():p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
    frozen=proof(cfg);save(cfg['output']/'producer.json',dict(frozen,complete=False))
    with (cfg['output']/'build.log').open('w') as log:
        for cmd in [*cfg['compiler'],cfg['linker']]:
            subprocess.run(cmd,cwd=cfg['native'],stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
    require(frozen==proof(cfg),'Continuation build inputs changed')
    result=dict(frozen,complete=True,binary=str(cfg['binary']),binary_sha256=sha(cfg['binary']),
        objects={str(p):sha(p) for p in cfg['objects']})
    save(cfg['output']/'producer.json',result);return result


def verify(directory):
    cfg=base.settings(directory);saved=json.loads((cfg['output']/'producer.json').read_text())
    require(saved==dict(proof(cfg),complete=True,binary=str(cfg['binary']),binary_sha256=sha(cfg['binary']),
        objects={str(p):sha(p) for p in cfg['objects']}),'Changed continuation producer')
    for p,value in generated(cfg['output']).items():require(p.read_text()==value,'Changed continuation source copy')
    return cfg,saved


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    r=build(p.parse_args().output);print(json.dumps({k:r[k] for k in ('complete','binary','binary_sha256')}))
