#!/usr/bin/env python3
"""Instrument developer copies of the verifier and cache, leaving production intact."""
import argparse
import json
from pathlib import Path
import subprocess

import build_perfect_draft as base
from build_identity import build_fingerprint
from qualification_evidence import save,sha

ROOT=Path(__file__).resolve().parents[2]
HEADER=ROOT/'scripts/qwen/block_cache_trace.hpp'
TEMPLATE=ROOT/'scripts/qwen/probe_perfect_draft.cpp'
INCLUDE='#include "'+str(HEADER)+'"\n'


def replace(source,old,new):
    if source.count(old)!=1:raise ValueError('Changed or ambiguous block trace seam: '+old[:100])
    return source.replace(old,new,1)


def model_source(source):
    out=INCLUDE+base.instrument(source)
    out=replace(out,base.FORWARD,base.FORWARD+'    block_trace::forward_begin(state.tokens,ids,phase_);\n')
    out=replace(out,'        return result;\n    } catch(...) {',
        '        (void)cache_->json();block_trace::forward_end(state.tokens);\n        return result;\n    } catch(...) {')
    marker='    // Encode shared work now. The first miss batch is submitted before this\n'
    return replace(out,marker,'    block_trace::layer(layer,trace_offset_,tokens,raw,selected);\n'+marker)


def storage_source(source):
    out=INCLUDE+base.instrument_storage(source)
    out=replace(out,'    result["diagnostic_cache_state"]=std::move(hexadecimal);',
        '    result["diagnostic_cache_state"]=std::move(hexadecimal);\n    block_trace::snapshot(diagnostic,result);')
    out=replace(out,'{ ++entry_->pins; }','{ ++entry_->pins; block_trace::lease(entry_->key.value(),entry_->pins,false,false); }')
    out=replace(out,'ExpertCache::Lease::~Lease() { if (entry_) --entry_->pins; }',
        'ExpertCache::Lease::~Lease() { if (entry_) { --entry_->pins; block_trace::lease(entry_->key.value(),entry_->pins,true,ready()); } }')
    out=replace(out,'if (this != &other) { if (entry_) --entry_->pins;',
        'if (this != &other) { if (entry_) { --entry_->pins; block_trace::lease(entry_->key.value(),entry_->pins,true,ready()); }')
    out=replace(out,'        return Lease(e,ready?1:2);',
        '        block_trace::acquire(key.value(),it->second,-1,ready?1:2);\n        return Lease(e,ready?1:2);')
    out=replace(out,'        auto& old=slots_[selected];',
        '        auto& old=slots_[selected];\n        const int64_t trace_victim=old?int64_t(old->key.value()):-1;')
    return replace(out,'        return Lease(e,0);',
        '        block_trace::acquire(key.value(),selected,trace_victim,0);\n        return Lease(e,0);')


def harness_source(source):
    out=INCLUDE+source
    marker='        phase(output,"load_model");'
    out=replace(out,marker,'        check(width==4 && !validation,"block cache capture requires width4 timing mode");\n'
        '        auto trace_path=output;trace_path.replace_extension(".cache.jsonl");\n'
        '        block_trace::open(trace_path,slots);\n'+marker)
    out=replace(out,'plan.at("planned_bytes").get<uint64_t>()+128*MiB+logits_bound<=12*GiB',
        'plan.at("planned_bytes").get<uint64_t>()+128*MiB+logits_bound+block_trace::workspace_bytes<=12*GiB')
    return replace(out,'        report["process_after_destroy"]=process_memory();',
        '        report["expert_cache_trace"]=block_trace::finish();\n        report["process_after_destroy"]=process_memory();')


def settings(output):
    output=Path(output).resolve()
    return base.commands(ROOT,output,output/'probe.block-cache.cpp')


def sources(cfg):
    return {cfg['generated']:model_source(cfg['source'].read_text()),
            cfg['storage_generated']:storage_source(cfg['storage_source'].read_text()),
            cfg['binary'].parent/'probe.block-cache.cpp':harness_source(TEMPLATE.read_text())}


def inputs(cfg):
    return [*base.frozen_inputs(cfg,TEMPLATE),HEADER,Path(__file__).resolve(),
            ROOT/'scripts/qwen/build_identity.py',ROOT/'scripts/qwen/qualification_evidence.py',
            *sorted((ROOT/'include/qwen').glob('*.hpp'))]


def build(output):
    cfg=settings(output);cfg['binary'].parent.mkdir(parents=True,exist_ok=False)
    generated=sources(cfg)
    for path,value in generated.items():path.write_text(value)
    frozen={str(p):sha(p) for p in inputs(cfg)}
    proof=dict(kind='block_cache_trace_producer_v1',complete=False,base_native_fingerprint=build_fingerprint(ROOT),
        original_inputs=frozen,generated={str(p):sha(p) for p in generated},compiler=cfg['compiler'],linker=cfg['linker'],
        binary=str(cfg['binary']),production_promoted=False,performance_measurement=False)
    save(cfg['binary'].parent/'producer.json',proof)
    with (cfg['binary'].parent/'build.log').open('w') as log:
        for cmd in [*cfg['compiler'],cfg['linker']]:
            subprocess.run(cmd,cwd=cfg['directory'],stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
    if frozen!={str(p):sha(p) for p in inputs(cfg)} or build_fingerprint(ROOT)!=proof['base_native_fingerprint']:
        raise ValueError('Build sources changed')
    proof.update(complete=True,binary_sha256=sha(cfg['binary']),objects={str(p):sha(p) for p in cfg['objects']})
    save(cfg['binary'].parent/'producer.json',proof)
    return proof


def verify(binary,fingerprint):
    binary=Path(binary).resolve();cfg=settings(binary.parent);proof=json.loads((binary.parent/'producer.json').read_text())
    expected=dict(kind='block_cache_trace_producer_v1',complete=True,base_native_fingerprint=fingerprint,
        original_inputs={str(p):sha(p) for p in inputs(cfg)},generated={str(p):sha(p) for p in sources(cfg)},
        compiler=cfg['compiler'],linker=cfg['linker'],binary=str(cfg['binary']),production_promoted=False,
        performance_measurement=False,binary_sha256=sha(binary),objects={str(p):sha(p) for p in cfg['objects']})
    if proof!=expected or build_fingerprint(ROOT)!=fingerprint or binary!=cfg['binary']:
        raise ValueError('Changed block trace producer')
    for path,value in sources(cfg).items():
        if path.read_text()!=value:raise ValueError('Generated block trace instrumentation changed')
    return dict(producer=proof,files=[*inputs(cfg),*sources(cfg),*cfg['objects'],binary,binary.parent/'producer.json'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(build(parser.parse_args().output),indent=2))
