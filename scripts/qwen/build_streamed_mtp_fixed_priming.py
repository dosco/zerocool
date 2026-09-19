#!/usr/bin/env python3
"""Streamed-MTP producer with identical bounded draft-cache priming in both arms."""
import argparse
import json
from pathlib import Path
import subprocess

import build_streamed_mtp as base
from build_block_cache_trace import replace
from cache_residency import require
from qualification_evidence import save, sha

ROOT = base.ROOT
HEADER = ROOT/'scripts/qwen/mtp_fixed_priming.hpp'
KIND = 'streamed_mtp_fixed_priming_producer_v1'
POLICY = 'fixed-lease-batches-32'


def generated(output):
    output = Path(output).resolve();sources = base.generated(output)
    draft = '#include "mtp_fixed_priming.hpp"\n'+(ROOT/'scripts/qwen/mtp_draft.cpp').read_text()
    draft = replace(draft, '    execute_experts(selected,*cache_,reads_,gpu_,4,',
                           '    mtp_fixed_priming::execute(selected,*cache_,reads_,gpu_,4,')
    sources[output/'mtp_draft.cpp'] = draft
    p = output/'probe.cpp';s = '#include "mtp_fixed_priming.hpp"\n'+sources[p]
    s = replace(s, '    // Feed each captured target panel through the draft before reusing it.',
        '    { mtp_fixed_priming::Scope fixed_draft_priming;\n'
        '    // Feed each captured target panel through the draft before reusing it.')
    s = replace(s, '    check(seed && draft_state.position+1==state.tokens',
        '    } // Fixed lease batches apply only to prompt priming.\n'
        '    report["draft_priming_policy"]="fixed-lease-batches-32";\n'
        '    report["draft_priming_before"]=mtp_fixed_priming::stats();\n'
        '    check(seed && draft_state.position+1==state.tokens')
    s = replace(s, '    report["embedding_rows_after"]=',
        '    report["draft_priming_after"]=mtp_fixed_priming::stats();\n'
        '    report["embedding_rows_after"]=')
    sources[p] = s
    return sources


def settings(output):
    cfg = base.settings(output);old = str(ROOT/'scripts/qwen/mtp_draft.cpp')
    commands = [cmd for cmd in cfg['compiler'] if cmd[-1] == old]
    require(len(commands) == 1, 'Missing unique draft compiler command')
    commands[0][-1] = str(cfg['output']/'mtp_draft.cpp')
    return cfg


def inputs(cfg): return [*base.inputs(cfg), Path(__file__).resolve(), HEADER]


def proof(cfg):
    return dict(kind=KIND, base_native_fingerprint=base.build_fingerprint(ROOT),
        inputs={str(p): sha(p) for p in inputs(cfg)}, generated={str(p): sha(p) for p in generated(cfg['output'])},
        compiler=cfg['compiler'], linker=cfg['linker'], production_promoted=False,
        embedding_storage=['resident', 'rows'], widths=[1, 4], real_mtp=True, draft_priming_policy=POLICY)


def build(output):
    cfg = settings(output);cfg['output'].mkdir(parents=True, exist_ok=False)
    for p, s in generated(cfg['output']).items():
        p.parent.mkdir(parents=True, exist_ok=True);p.write_text(s)
    frozen = proof(cfg);save(cfg['output']/'producer.json', dict(frozen, complete=False))
    with (cfg['output']/'build.log').open('w') as log:
        for cmd in [*cfg['compiler'], cfg['linker']]:
            subprocess.run(cmd, cwd=cfg['native'], stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True)
    require(frozen == proof(cfg), 'Fixed-priming build inputs changed')
    result = dict(frozen, complete=True, binary=str(cfg['binary']), binary_sha256=sha(cfg['binary']),
        objects={str(p): sha(p) for p in cfg['objects']})
    save(cfg['output']/'producer.json', result);return result


def verify(output):
    cfg = settings(output);saved = json.loads((cfg['output']/'producer.json').read_text())
    require(saved == dict(proof(cfg), complete=True, binary=str(cfg['binary']), binary_sha256=sha(cfg['binary']),
        objects={str(p): sha(p) for p in cfg['objects']}), 'Fixed-priming producer changed')
    for p, s in generated(cfg['output']).items(): require(p.read_text() == s, 'Fixed-priming source copy changed')
    return cfg, saved


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--output', type=Path, required=True)
    result = build(p.parse_args().output)
    print(json.dumps({k: result[k] for k in ('complete', 'binary', 'binary_sha256')}))
