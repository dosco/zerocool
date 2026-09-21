#!/usr/bin/env python3
"""Build a bounded expert-temporary experiment; production and older producers are unchanged."""
import argparse
import json
from pathlib import Path
import subprocess

import build_mtp_continuation as base
from build_block_cache_trace import replace
from build_identity import build_fingerprint
from cache_residency import require
from qualification_evidence import save,sha

ROOT=base.ROOT
SCRIPTS=ROOT/'scripts/qwen'
HEADER=SCRIPTS/'mtp_expert_scratch.hpp'
TEST=SCRIPTS/'probe_mtp_expert_scratch.cpp'
INCLUDE='#include "mtp_expert_scratch.hpp"\n'


def generated(output):
    sources=base.generated(output)
    path=output/'model.cpp';s=INCLUDE+sources[path]
    s=replace(s,'    bool completed=false,scratch_started=false;',
        '    const bool group_reuse=mtp_scratch::enabled && ids.size()==4 && phase_=="decode";\n'
        '    if(group_reuse && (options_.decode_scratch!="reuse" || !options_.completion_pipeline ||\n'
        '        options_.decode_path!="reference" || options_.expert_tail!="wait" || plan_.scratch<mtp_scratch::ReserveBytes))\n'
        '        throw std::logic_error("unsupported expert scratch configuration");\n'
        '    mtp_scratch::ForwardScope group_scope(gpu_,group_reuse);\n'
        '    bool completed=false,scratch_started=false;')
    s=replace(s,'        if(options_.decode_scratch=="reuse" && !reuse) gpu_.release_scratch();',
        '        const bool was_group=mtp_scratch::retained_group_owner==&gpu_;\n'
        '        if((options_.decode_scratch=="reuse" && !reuse && !group_reuse) || was_group!=group_reuse) {\n'
        '            gpu_.release_scratch();\n'
        '            if(was_group!=group_reuse) ++mtp_scratch::transitions;\n'
        '        }\n        mtp_scratch::retained_group_owner=group_reuse?&gpu_:nullptr;')
    s=replace(s,'        if(scratch_started) {','        if(scratch_started || group_reuse) {\n            mtp_scratch::retained_group_owner=nullptr;')
    sources[path]=s
    path=output/'pipeline.cpp';s=INCLUDE+(ROOT/'src/engine/pipeline.cpp').read_text()
    s=replace(s,'    auto state=std::make_unique<ExpertTail::Impl>(gpu,reads,detailed);',
        '    const bool pooled=mtp_scratch::target==&gpu;bool scratch_started=false;\n'
        '    if(pooled && (tail || coalesce_reads || encode_group)) throw std::logic_error("unsupported pooled expert schedule");\n'
        '    auto state=std::make_unique<ExpertTail::Impl>(gpu,reads,detailed);')
    s=replace(s,'                    if(!encode_group) encode(group.work.back().lease.key(),record);',
        '                    if(pooled && !scratch_started) {\n'
        '                        gpu.begin_scratch(group_sequence%2,mtp_scratch::PoolBytes);scratch_started=true;\n'
        '                    }\n'
        '                    if(!encode_group) encode(group.work.back().lease.key(),record);')
    s=replace(s,'                group.completion=gpu.submit();',
        '                group.completion=gpu.submit();\n'
        '                // End only after submit: the coordinator must retain the actual completion.\n'
        '                if(scratch_started) {gpu.end_scratch();scratch_started=false;++mtp_scratch::groups;}')
    s=replace(s,'        try {gpu.finish();} catch(...) {} reads.drain();std::rethrow_exception(failure);\n    }\n    return state->report();',
        '        try {gpu.finish();} catch(...) {}\n'
        '        if(pooled) {try {gpu.release_scratch();} catch(...) {}}\n'
        '        reads.drain();std::rethrow_exception(failure);\n    }\n    return state->report();')
    sources[path]=s
    path=output/'metal.mm';s=INCLUDE+sources[path]
    s=replace(s,'Metal::~Metal() {','Metal::~Metal() { if(mtp_scratch::retained_group_owner==this) mtp_scratch::retained_group_owner=nullptr;')
    s=replace(s,'        completion->done.store(true,std::memory_order_release);events->publish();',
        '        mtp_scratch::completion_gate();\n        completion->done.store(true,std::memory_order_release);events->publish();')
    sources[path]=s
    path=output/'probe.cpp';s=INCLUDE+sources[path]
    s=replace(s,'struct ContinuationInput {',TEST.read_text()+'\nstruct ContinuationInput {')
    s=replace(s,'    const ContinuationInput work(input,mode);const auto& prompt=work.prompt;',
        '    const ContinuationInput work(input,mode);const auto& prompt=work.prompt;\n'
        '    const char* setting=std::getenv("ZEROCOOL_MTP_EXPERT_SCRATCH");\n'
        '    check(setting && (std::string_view(setting)=="off" || std::string_view(setting)=="on"),"explicit expert scratch arm required");\n'
        '    mtp_scratch::enabled=std::string_view(setting)=="on";\n'
        '    report["expert_scratch_before"]=mtp_scratch::counters();')
    s=replace(s,'+MtpDraft::budget_bytes(32,8192);','+MtpDraft::budget_bytes(32,8192)+mtp_scratch::ReserveBytes;')
    s=replace(s,'{"host_checkpoint_logits_bytes",host_bound},{"combined_bytes",combined}',
        '{"host_checkpoint_logits_bytes",host_bound},{"combined_bytes",combined},{"expert_scratch_reserve_bytes",mtp_scratch::ReserveBytes}')
    s=replace(s,'    check(model.memory_plan().json()==plan,"continuation memory admission changed");',
        '    report["expert_scratch_after"]=mtp_scratch::counters();\n'
        '    check(gpu.statistics().at("active_scratch_slot")==-1,"expert scratch scope leaked");\n'
        '    check(model.memory_plan().json()==plan,"continuation memory admission changed");')
    s=replace(s,'        check(argc==6,"usage:',
        '        if(argc==4 && std::string_view(argv[1])=="--expert-scratch-test") {\n'
        '            check(!std::filesystem::exists(argv[3]),"scratch test output exists");\n'
        '            save_report(argv[3],expert_scratch_test(argv[2]));return 0;\n        }\n'
        '        check(argc==6,"usage:')
    sources[path]=s;return sources


def settings(output):
    c=base.base.settings(output);source=c['output']/'pipeline.cpp';obj=c['output']/'pipeline.o'
    command=c['compiler'][0].copy();command[command.index('-o')+1]=str(obj);command[-1]=str(source)
    c['compiler'].append(command);c['objects'].append(obj)
    c['linker'].insert(c['linker'].index('libzerocool_lib.a'),str(obj));return c


def inputs(c):return [*base.inputs(c),Path(__file__).resolve(),HEADER,TEST,ROOT/'src/engine/pipeline.cpp']


def proof(c):
    return dict(kind='mtp_expert_scratch_producer_v1',base_native_fingerprint=build_fingerprint(ROOT),
        inputs={str(p):sha(p) for p in inputs(c)},generated={str(p):sha(p) for p in generated(c['output'])},
        compiler=c['compiler'],linker=c['linker'],production_promoted=False)


def build(output):
    c=settings(output);c['output'].mkdir(parents=True,exist_ok=False)
    for p,value in generated(c['output']).items():p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
    frozen=proof(c);save(c['output']/'producer.json',dict(frozen,complete=False))
    with (c['output']/'build.log').open('w') as log:
        for cmd in [*c['compiler'],c['linker']]:subprocess.run(cmd,cwd=c['native'],stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
    require(frozen==proof(c),'Expert scratch build inputs changed')
    result=dict(frozen,complete=True,binary=str(c['binary']),binary_sha256=sha(c['binary']),objects={str(p):sha(p) for p in c['objects']})
    save(c['output']/'producer.json',result);return result


def verify(directory):
    c=settings(directory);saved=json.loads((c['output']/'producer.json').read_text())
    require(saved==dict(proof(c),complete=True,binary=str(c['binary']),binary_sha256=sha(c['binary']),
        objects={str(p):sha(p) for p in c['objects']}),'Changed expert scratch producer')
    for p,value in generated(c['output']).items():require(p.read_text()==value,'Changed expert scratch source copy')
    return c,saved


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    r=build(p.parse_args().output);print(json.dumps({k:r[k] for k in ('complete','binary','binary_sha256')}))
