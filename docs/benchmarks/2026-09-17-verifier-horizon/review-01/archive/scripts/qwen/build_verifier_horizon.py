#!/usr/bin/env python3
"""Eight-row verifier ceiling, isolated from production and real MTP runs."""
import argparse
import json
from pathlib import Path
import subprocess

import build_mtp_widths as base
import build_q8_expanded as packed
from build_block_cache_trace import replace
from build_identity import build_fingerprint
from cache_residency import require
from qualification_evidence import save, sha

ROOT = base.ROOT
HARNESS = ROOT/'scripts/qwen/probe_verifier_horizon.cpp'


def shader():
    # Only the independent token dimension grows. Keep every lane's scalar
    # additions, weight partition, SIMD reduction and BF16 rounding unchanged.
    s = packed.shader().replace('q8_expanded_t4_w8', 'q8_horizon_t8_w8')
    s = s.replace('four tokens', 'eight tokens').replace('p[2]!=4', 'p[2]!=8')
    s = s.replace('[4]={0,0,0,0}', '[8]={0,0,0,0,0,0,0,0}').replace('t<4', 't<8')
    return s


def generated(output):
    output = Path(output).resolve()
    sources = base.generated(output)
    p = output/'model.cpp'; s = sources[p]
    s = replace(s, 'ids.size()!=1 && ids.size()!=2 && ids.size()!=4',
                'ids.size()!=1 && ids.size()!=2 && ids.size()!=4 && ids.size()!=8')
    s = replace(s, 'T!=1 && T!=2 && T!=4', 'T!=1 && T!=2 && T!=4 && T!=8')
    s = s.replace('must contain 1, 2, or 4 tokens', 'must contain 1, 2, 4, or 8 tokens')
    s = replace(s, 'mtp_direct::Scope direct_scope(ids.size()==4 && phase_=="decode");',
                'mtp_direct::Scope direct_scope((ids.size()==4 || ids.size()==8) && phase_=="decode");')
    sources[p] = s
    sources[output/'include/mtp_direct_output.hpp'] = replace(
        (ROOT/'scripts/qwen/mtp_direct_output.hpp').read_text(),
        '!in_verifier || tokens!=4 || rows!=1', '!in_verifier || (tokens!=4 && tokens!=8) || rows!=1')
    p = output/'metal.mm'; s = sources[p]
    s = replace(s, ')PACKED_Q8";', shader()+'\n)PACKED_Q8";')
    s = replace(s, 'impl_->request_phase=="decode" && tokens==4 &&\n       tile==4',
                'impl_->request_phase=="decode" && (tokens==4 || tokens==8) &&\n       tile==tokens')
    s = replace(s, 'dispatch("q8_expanded_t4_w8",',
                'dispatch(tokens==8?"q8_horizon_t8_w8":"q8_expanded_t4_w8",')
    sources[p] = s
    p = output/'probe.cpp'; s = sources[p]
    # Equal eight-row host checkpoint capacity in every new arm. The existing
    # four-row recovery journal is allocated but never used for this ceiling.
    s = replace(s, 'capacity=f>=2 && f<=4?4ull*', 'capacity=f>=2 && f<=4?8ull*')
    s = replace(s, 'state.valid && width && width<=4', 'state.valid && width && width<=8')
    s = s.replace('same four-token capacity before\n            // priming',
                  'same eight-token capacity before\n            // timing')
    start = s.index('void joint(const std::filesystem::path& model_path,')
    end = s.index('\n}\n\n}\nint main(', start)+2
    s = s[:start]+HARNESS.read_text()+s[end:]
    s = replace(s, 'void capture_recovery_fixture(', '[[maybe_unused]] void capture_recovery_fixture(')
    s = replace(s, '"native_mtp_width_v1"', '"native_verifier_horizon_v1"')
    s = replace(s, '        check(argc==6,"usage:',
        '        if(argc==3 && std::string_view(argv[1])=="--horizon-kernel-test") {\n'
        '            check(!std::filesystem::exists(argv[2]),"horizon kernel report exists");\n'
        '            save_report(argv[2],horizon_kernel_test());return 0;\n        }\n'
        '        check(argc==6,"usage:')
    sources[p] = s
    return sources


def settings(output): return base.settings(Path(output).resolve())
def inputs(cfg): return [*base.inputs(cfg), Path(__file__).resolve(), HARNESS]


def proof(cfg):
    return dict(kind='verifier_horizon_producer_v1', base_native_fingerprint=build_fingerprint(ROOT),
        inputs={str(p):sha(p) for p in inputs(cfg)}, generated={str(p):sha(p) for p in generated(cfg['output'])},
        compiler=cfg['compiler'], linker=cfg['linker'], production_promoted=False,
        widths=[1,4,8], perfect_proposals_only=True, normal_request_latency_qualified=False)


def build(output):
    cfg = settings(output); cfg['output'].mkdir(parents=True, exist_ok=False)
    for p,s in generated(cfg['output']).items(): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
    frozen=proof(cfg);save(cfg['output']/'producer.json',dict(frozen,complete=False))
    with (cfg['output']/'build.log').open('w') as log:
        for cmd in [*cfg['compiler'],cfg['linker']]:
            subprocess.run(cmd,cwd=cfg['native'],stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
    require(frozen==proof(cfg),'Horizon inputs changed during build')
    result=dict(frozen,complete=True,binary=str(cfg['binary']),binary_sha256=sha(cfg['binary']),
                objects={str(p):sha(p) for p in cfg['objects']})
    save(cfg['output']/'producer.json',result);return result


def verify(output):
    cfg=settings(output);saved=json.loads((cfg['output']/'producer.json').read_text())
    require(saved==dict(proof(cfg),complete=True,binary=str(cfg['binary']),binary_sha256=sha(cfg['binary']),
                       objects={str(p):sha(p) for p in cfg['objects']}),'Changed horizon producer')
    for p,s in generated(cfg['output']).items(): require(p.read_text()==s,'Changed horizon source copy')
    return cfg,saved


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    r=build(p.parse_args().output);print(json.dumps({k:r[k] for k in ('complete','binary','binary_sha256')}))
