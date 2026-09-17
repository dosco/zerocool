#!/usr/bin/env python3
"""Horizon with exact SSD-backed embedding rows and unchanged expert capacity."""
import argparse
import json
from pathlib import Path
import subprocess

import build_verifier_horizon as base
from build_block_cache_trace import replace
from build_identity import build_fingerprint
from cache_residency import require
from qualification_evidence import save, sha

ROOT=base.ROOT
HEADER=ROOT/'scripts/qwen/streamed_embeddings.hpp'
TEST=ROOT/'scripts/qwen/probe_streamed_embeddings.cpp'


def generated(output):
    output=Path(output).resolve();sources=base.generated(output)
    p=output/'metal.mm';s='#include "streamed_embeddings.hpp"\n'+sources[p]
    marker='    std::vector<std::string> keys;\n    for(const auto& [key,r]:cp.tensors)'
    s=replace(s,marker,
        '    if(embedding_rows::enabled) {\n'
        '        embedding_rows::need(!embedding_rows::store,"duplicate embedding store");\n'
        '        embedding_rows::store=std::make_unique<embedding_rows::Store>(cp);\n    }\n'+marker)
    s=replace(s,'if(is_resident(key,layers) && (!streaming || !key.starts_with("model.layers.")))',
        'if(is_resident(key,layers) && (!embedding_rows::enabled || !key.starts_with("model.embed_tokens.")) && (!streaming || !key.starts_with("model.layers.")))')
    marker='Linear Resident::linear(const std::string& name) const {\n'
    s=replace(s,marker,marker+
        '    if(embedding_rows::enabled && name=="model.embed_tokens") {\n'
        '        embedding_rows::need(bool(embedding_rows::store),"missing embedding store");\n'
        '        return embedding_rows::store->descriptor();\n    }\n')
    marker='Buf Metal::embedding(const Linear& l,std::span<const int> ids,uint32_t copies) {\n'
    s=replace(s,marker,marker+
        '    if(!l.weight.buffer && embedding_rows::enabled) {\n'
        '        embedding_rows::need(bool(embedding_rows::store),"missing embedding provider");\n'
        '        return embedding_rows::store->gather(*this,l,ids,copies);\n    }\n')
    sources[p]=s;p=output/'model.cpp';s='#include "streamed_embeddings.hpp"\n'+sources[p]
    marker='    plan_=MemoryPlan::make(admitted,gpu_.physical(),gpu_.recommended(),resident_bytes,'
    s=replace(s,marker,'    plan_=MemoryPlan::make(admitted,gpu_.physical(),gpu_.recommended(),embedding_rows::planned_resident(checkpoint_,resident_bytes),')
    sources[p]=s;p=output/'probe.cpp';s='#include "streamed_embeddings.hpp"\n'+sources[p]
    s=replace(s,'Json horizon_kernel_test() {',TEST.read_text()+'\nJson horizon_kernel_test() {')
    s=replace(s,'    const bool validation=mode=="fast-validate";\n',
        '    embedding_rows::Scope streamed_scope;\n    const bool validation=mode=="fast-validate";\n')
    s=replace(s,'    report["before"]=model.stats();report["draft_before"]=draft.stats();report["initial_draft_state"]=',
        '    report["embedding_rows_before"]=embedding_rows::store->stats();\n'
        '    report["before"]=model.stats();report["draft_before"]=draft.stats();report["initial_draft_state"]=')
    s=replace(s,'    check(report["initial_draft_state"]==report["final_draft_state"],',
        '    report["embedding_rows_after"]=embedding_rows::store->stats();\n'
        '    check(report["initial_draft_state"]==report["final_draft_state"],')
    s=replace(s,'        check(argc==6,"usage:',
        '        if(argc==4 && std::string_view(argv[1])=="--streamed-embedding-test") {\n'
        '            check(!std::filesystem::exists(argv[3]),"embedding report exists");\n'
        '            save_report(argv[3],streamed_embedding_test(argv[2]));return 0;\n        }\n'
        '        check(argc==6,"usage:')
    sources[p]=s;return sources


def settings(output):return base.settings(output)
def inputs(cfg):return [*base.inputs(cfg),Path(__file__).resolve(),HEADER,TEST]
def proof(cfg):
    return dict(kind='streamed_verifier_horizon_producer_v1',base_native_fingerprint=build_fingerprint(ROOT),
        inputs={str(p):sha(p) for p in inputs(cfg)},generated={str(p):sha(p) for p in generated(cfg['output'])},
        compiler=cfg['compiler'],linker=cfg['linker'],production_promoted=False,embedding_storage='exact-packed-rows',
        widths=[1,4,8],perfect_proposals_only=True,normal_request_latency_qualified=False)


def build(output):
    cfg=settings(output);cfg['output'].mkdir(parents=True,exist_ok=False)
    for p,s in generated(cfg['output']).items():p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
    frozen=proof(cfg);save(cfg['output']/'producer.json',dict(frozen,complete=False))
    with (cfg['output']/'build.log').open('w') as log:
        for cmd in [*cfg['compiler'],cfg['linker']]:
            subprocess.run(cmd,cwd=cfg['native'],stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
    require(frozen==proof(cfg),'Streamed horizon build changed')
    result=dict(frozen,complete=True,binary=str(cfg['binary']),binary_sha256=sha(cfg['binary']),objects={str(p):sha(p) for p in cfg['objects']})
    save(cfg['output']/'producer.json',result);return result


def verify(output):
    cfg=settings(output);saved=json.loads((cfg['output']/'producer.json').read_text())
    require(saved==dict(proof(cfg),complete=True,binary=str(cfg['binary']),binary_sha256=sha(cfg['binary']),
        objects={str(p):sha(p) for p in cfg['objects']}),'Streamed horizon producer changed')
    for p,s in generated(cfg['output']).items():require(p.read_text()==s,'Streamed source copy changed')
    return cfg,saved


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    r=build(p.parse_args().output);print(json.dumps({k:r[k] for k in ('complete','binary','binary_sha256')}))
