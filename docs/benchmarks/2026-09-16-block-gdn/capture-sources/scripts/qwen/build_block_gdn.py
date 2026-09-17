#!/usr/bin/env python3
"""Build an actual-input capture and bounded Q8 row-pair probe without production edits."""
import argparse
import json
from pathlib import Path
import subprocess

import build_block_profile as profile
from build_block_cache_trace import replace
from build_identity import build_fingerprint
from qualification_evidence import save, sha

ROOT = profile.ROOT
PROBE = ROOT/'scripts/qwen/probe_block_gdn.cpp'


def settings(output):
    cfg = profile.settings(output); obj = cfg['binary'].parent/'operator.o'
    command = list(cfg['compiler'][-1]); command[-1] = str(PROBE); command[command.index('-o')+1] = str(obj)
    link = [v for v in cfg['linker'] if v not in map(str, cfg['objects'])]
    binary = cfg['binary'].parent/'probe-block-gdn'; link[link.index('-o')+1] = str(binary)
    link.insert(link.index('libfreellm_lib.a'), str(obj))
    cfg.update(operator_binary=binary, operator_object=obj, operator_compiler=command, operator_linker=link)
    return cfg


def sources(cfg):
    result = profile.sources(cfg, 'commands'); harness = cfg['binary'].parent/'probe.block-profile.cpp'
    result[cfg['generated']] = replace(result[cfg['generated']], '    gpu_.configure(verifier_kernels);',
        '    if(phase_!="decode") verifier_kernels.operator_capture.clear();\n'
        '    gpu_.configure(verifier_kernels);')
    result[harness] = replace(result[harness], '        options.kernels.profile=true;',
        '        options.kernels.profile=true;\n'
        '        options.kernels.operator_capture=output.parent_path()/"gdn-fixtures";\n'
        '        options.kernels.capture_phase="decode";options.kernels.capture_operator="gdn";options.kernels.capture_layer=0;')
    return result


def inputs(cfg): return [*profile.inputs(cfg), PROBE, Path(__file__).resolve()]


def proof(cfg):
    return dict(kind='block_gdn_producer_v1', base_native_fingerprint=build_fingerprint(ROOT),
        original_inputs={str(p): sha(p) for p in inputs(cfg)}, generated={str(p): sha(p) for p in sources(cfg)},
        compiler=cfg['compiler']+[cfg['operator_compiler']], linker=[cfg['linker'], cfg['operator_linker']],
        capture_binary=str(cfg['binary']), operator_binary=str(cfg['operator_binary']), production_promoted=False)


def build(output):
    cfg = settings(output); cfg['binary'].parent.mkdir(parents=True, exist_ok=False)
    for p, value in sources(cfg).items(): p.write_text(value)
    frozen = proof(cfg); save(cfg['binary'].parent/'producer.json', dict(frozen, complete=False))
    with (cfg['binary'].parent/'build.log').open('w') as log:
        for cmd in [*frozen['compiler'], *frozen['linker']]:
            subprocess.run(cmd, cwd=cfg['directory'], stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True)
    if frozen != proof(cfg): raise ValueError('Build inputs changed')
    result = dict(frozen, complete=True, binaries={str(p): sha(p) for p in (cfg['binary'], cfg['operator_binary'])},
        objects={str(p): sha(p) for p in [*cfg['objects'], cfg['operator_object']]})
    save(cfg['binary'].parent/'producer.json', result); return result


def verify(directory, fingerprint):
    cfg = settings(directory); saved = json.loads((cfg['binary'].parent/'producer.json').read_text())
    expected = dict(proof(cfg), complete=True, binaries={str(p): sha(p) for p in (cfg['binary'], cfg['operator_binary'])},
        objects={str(p): sha(p) for p in [*cfg['objects'], cfg['operator_object']]})
    if saved != expected or saved['base_native_fingerprint'] != fingerprint: raise ValueError('Changed GDN producer')
    for p, value in sources(cfg).items():
        if p.read_text() != value: raise ValueError('Changed source instrumentation')
    return dict(producer=saved, files=[*inputs(cfg), *sources(cfg), *cfg['objects'], cfg['operator_object'],
        cfg['binary'], cfg['operator_binary'], cfg['binary'].parent/'producer.json'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    print(json.dumps(build(p.parse_args().output), indent=2))
