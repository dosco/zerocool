#!/usr/bin/env python3
"""Build direct single-row expert outputs without changing kernels or prior producers."""
import argparse
import json
from pathlib import Path
import subprocess

import build_mtp_ngram_init as base
from build_block_cache_trace import replace
from build_identity import build_fingerprint
from cache_residency import require
from qualification_evidence import save,sha

ROOT=base.ROOT
HEADER=ROOT/'scripts/qwen/mtp_direct_output.hpp'
TEST=ROOT/'scripts/qwen/probe_mtp_direct_output.cpp'
INCLUDE='#include "mtp_direct_output.hpp"\n'


def generated(output):
    sources=base.generated(output)
    p=output/'model.cpp';s=INCLUDE+sources[p]
    s=replace(s,'    mtp_scratch::ForwardScope group_scope(gpu_,group_reuse);',
        '    mtp_scratch::ForwardScope group_scope(gpu_,group_reuse);\n'
        '    mtp_direct::Scope direct_scope(ids.size()==4 && phase_=="decode");')
    sources[p]=s;p=output/'pipeline.cpp';s=INCLUDE+sources[p]
    s=replace(s,'        if(tokens==1 && direct) gpu.linear_into(',
        '        const bool direct_single=mtp_direct::select(tokens,n);\n'
        '        if((tokens==1 && direct) || direct_single) gpu.linear_into(')
    sources[p]=s;p=output/'probe.cpp';s=INCLUDE+sources[p]
    s=replace(s,'struct ContinuationInput {',TEST.read_text()+'\nstruct ContinuationInput {')
    s=replace(s,'    mtp_scratch::enabled=std::string_view(setting)=="on";',
        '    check(std::string_view(setting)=="off","direct-output trial requires scratch off");\n'
        '    mtp_scratch::enabled=false;\n'
        '    const char* direct_setting=std::getenv("FREELLM_MTP_DIRECT_OUTPUT");\n'
        '    check(direct_setting && (std::string_view(direct_setting)=="off" || std::string_view(direct_setting)=="on"),"explicit direct-output arm required");\n'
        '    mtp_direct::enabled=std::string_view(direct_setting)=="on";')
    s=replace(s,'    report["ngram_before_decode"]=NgramAudit::summary(model);',
        '    report["direct_output_before"]=mtp_direct::counters();\n'
        '    report["ngram_before_decode"]=NgramAudit::summary(model);')
    s=replace(s,'    report["ngram_after_decode"]=NgramAudit::summary(model);',
        '    report["direct_output_after"]=mtp_direct::counters();\n'
        '    report["ngram_after_decode"]=NgramAudit::summary(model);')
    s=replace(s,'        check(argc==6,"usage:',
        '        if(argc==4 && std::string_view(argv[1])=="--direct-output-test") {\n'
        '            check(!std::filesystem::exists(argv[3]),"direct-output fixture exists");\n'
        '            save_report(argv[3],direct_output_test(argv[2]));return 0;\n        }\n'
        '        check(argc==6,"usage:')
    sources[p]=s;return sources


def settings(output):return base.base.settings(output)


def inputs(c):return [*base.inputs(c),Path(__file__).resolve(),HEADER,TEST]


def proof(c):
    return dict(kind='mtp_direct_output_producer_v1',base_native_fingerprint=build_fingerprint(ROOT),
        inputs={str(p):sha(p) for p in inputs(c)},generated={str(p):sha(p) for p in generated(c['output'])},
        compiler=c['compiler'],linker=c['linker'],production_promoted=False)


def build(output):
    c=settings(output);c['output'].mkdir(parents=True,exist_ok=False)
    for p,value in generated(c['output']).items():p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
    frozen=proof(c);save(c['output']/'producer.json',dict(frozen,complete=False))
    with (c['output']/'build.log').open('w') as log:
        for cmd in [*c['compiler'],c['linker']]:subprocess.run(cmd,cwd=c['native'],stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
    require(frozen==proof(c),'Direct-output build inputs changed')
    result=dict(frozen,complete=True,binary=str(c['binary']),binary_sha256=sha(c['binary']),objects={str(p):sha(p) for p in c['objects']})
    save(c['output']/'producer.json',result);return result


def verify(directory):
    c=settings(directory);saved=json.loads((c['output']/'producer.json').read_text())
    require(saved==dict(proof(c),complete=True,binary=str(c['binary']),binary_sha256=sha(c['binary']),
        objects={str(p):sha(p) for p in c['objects']}),'Changed direct-output producer')
    for p,value in generated(c['output']).items():require(p.read_text()==value,'Changed direct-output source copy')
    return c,saved


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    r=build(p.parse_args().output);print(json.dumps({k:r[k] for k in ('complete','binary','binary_sha256')}))
