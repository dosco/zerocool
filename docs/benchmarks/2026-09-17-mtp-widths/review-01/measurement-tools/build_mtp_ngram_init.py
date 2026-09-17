#!/usr/bin/env python3
"""Isolate lazy ngram row construction, retaining exact cache capacity and replacement."""
import argparse
import json
from pathlib import Path
import subprocess

import build_mtp_expert_scratch as base
from build_block_cache_trace import replace
from build_identity import build_fingerprint
from cache_residency import require
from qualification_evidence import save,sha

ROOT=base.ROOT
HEADER=ROOT/'scripts/qwen/mtp_ngram_init.hpp'
TEST=ROOT/'scripts/qwen/probe_mtp_ngram_init.cpp'


def generated(output):
    s=base.generated(output)
    p=output/'include/qwen/storage.hpp'
    s[p]=replace((ROOT/'include/qwen/storage.hpp').read_text(),'class NgramStore {','class NgramStore {\n    friend struct NgramAudit;')
    p=output/'include/qwen/model.hpp'
    s[p]=replace(s[p],'    friend struct DraftAccess;','    friend struct DraftAccess;\n    friend struct NgramAudit;')
    p=output/'storage.cpp';text='#include <cstdlib>\n'+s[p]
    text=replace(text,'    rows_.resize(count); lookup_.reserve(count);',
        '    const char* setting=std::getenv("FREELLM_MTP_NGRAM_INIT");\n'
        '    if(!setting || (std::string_view(setting)!="eager" && std::string_view(setting)!="lazy"))\n'
        '        throw std::invalid_argument("explicit ngram initialization mode required");\n'
        '    if(std::string_view(setting)=="lazy") rows_.reserve(count);else rows_.resize(count);\n'
        '    if(rows_.capacity()!=count) throw std::runtime_error("ngram reservation changed capacity");\n'
        '    lookup_.reserve(count);')
    text=replace(text,'        auto& row=rows_[next_]; if(row.key>=0) lookup_.erase(row.key);',
        '        // Construct only the next admitted row, retaining the original FIFO ring.\n'
        '        if(rows_.size()<rows_.capacity()) rows_.emplace_back();\n'
        '        auto& row=rows_[next_]; if(row.key>=0) lookup_.erase(row.key);')
    text=replace(text,'next_=(next_+1)%rows_.size();','next_=(next_+1)%rows_.capacity();')
    s[p]=text;p=output/'probe.cpp';text='#include "mtp_ngram_init.hpp"\n'+s[p]
    text=replace(text,'struct ContinuationInput {',TEST.read_text()+'\nstruct ContinuationInput {')
    text=replace(text,'    report["priming_wall_ns"]=monotonic_ns()-prime_start;',
        '    report["ngram_before_decode"]=NgramAudit::summary(model);\n'
        '    report["ngram_initialization"]=std::getenv("FREELLM_MTP_NGRAM_INIT");\n'
        '    report["priming_wall_ns"]=monotonic_ns()-prime_start;')
    text=replace(text,'    report["expert_scratch_after"]=mtp_scratch::counters();',
        '    report["ngram_after_decode"]=NgramAudit::summary(model);\n'
        '    report["expert_scratch_after"]=mtp_scratch::counters();')
    text=replace(text,'        check(argc==6,"usage:',
        '        if(argc==5 && std::string_view(argv[1])=="--ngram-init-test") {\n'
        '            check(!std::filesystem::exists(argv[4]),"ngram fixture output exists");\n'
        '            save_report(argv[4],ngram_init_test(argv[2],argv[3]));return 0;\n        }\n'
        '        check(argc==6,"usage:')
    s[p]=text;return s


def inputs(c):return [*base.inputs(c),Path(__file__).resolve(),HEADER,TEST]


def proof(c):
    return dict(kind='mtp_ngram_init_producer_v1',base_native_fingerprint=build_fingerprint(ROOT),
        inputs={str(p):sha(p) for p in inputs(c)},generated={str(p):sha(p) for p in generated(c['output'])},
        compiler=c['compiler'],linker=c['linker'],production_promoted=False)


def build(output):
    c=base.settings(output);c['output'].mkdir(parents=True,exist_ok=False)
    for p,value in generated(c['output']).items():p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
    frozen=proof(c);save(c['output']/'producer.json',dict(frozen,complete=False))
    with (c['output']/'build.log').open('w') as log:
        for cmd in [*c['compiler'],c['linker']]:subprocess.run(cmd,cwd=c['native'],stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
    require(frozen==proof(c),'Ngram build inputs changed')
    result=dict(frozen,complete=True,binary=str(c['binary']),binary_sha256=sha(c['binary']),objects={str(p):sha(p) for p in c['objects']})
    save(c['output']/'producer.json',result);return result


def verify(directory):
    c=base.settings(directory);saved=json.loads((c['output']/'producer.json').read_text())
    require(saved==dict(proof(c),complete=True,binary=str(c['binary']),binary_sha256=sha(c['binary']),
        objects={str(p):sha(p) for p in c['objects']}),'Changed ngram producer')
    for p,value in generated(c['output']).items():require(p.read_text()==value,'Changed ngram source copy')
    return c,saved


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    r=build(p.parse_args().output);print(json.dumps({k:r[k] for k in ('complete','binary','binary_sha256')}))
