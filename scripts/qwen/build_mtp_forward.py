#!/usr/bin/env python3
"""Build an isolated native MTP experiment, retaining the qualified Q8 verifier."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

import build_perfect_draft as verifier
import build_q8_expanded as expanded
from build_block_cache_trace import replace
from build_identity import build_fingerprint
from qualification_evidence import save, sha
from cache_residency import require

ROOT=Path(__file__).resolve().parents[2]
SCRIPTS=ROOT/'scripts/qwen'


def generated(output):
    header=replace((ROOT/'include/engine/model.hpp').read_text(),'class Model {\n','class Model {\n    friend struct DraftAccess;\n')
    model=verifier.instrument((ROOT/'src/engine/model.cpp').read_text())
    model='#include "mtp_draft.hpp"\n'+model
    marker='        if(!logits) {gpu_.finish();finish_expert_tail();update.commit();return {};}\n'
    model=replace(model,marker,'        capture_mtp_hidden(gpu_,h,T,state.tokens);\n'+marker)
    # The harness streams <=128 target rows, so the independent layer-major
    # production panel implementation remains linked but is never selected.
    metal=replace(expanded.metal_source(),')PACKED_Q8";',
                  (SCRIPTS/'mtp_draft.metal').read_text()+'\n)PACKED_Q8";')
    helpers=(SCRIPTS/'probe_perfect_draft.cpp').read_text().split('int main(int argc,char** argv) {')[0]
    harness='#include "mtp_draft.hpp"\n'+helpers+(SCRIPTS/'probe_mtp_forward.cpp').read_text()
    return {output/'include/engine/model.hpp':header,output/'model.cpp':model,
            output/'metal.mm':metal,output/'storage.cpp':verifier.instrument_storage((ROOT/'src/engine/storage.cpp').read_text()),
            output/'probe.cpp':harness}


def settings(output):
    output=Path(output).resolve();native=ROOT/'build/qwen'
    db=native/'compile_commands.json';link_file=native/'CMakeFiles/zerocool.dir/link.txt'
    entries=json.loads(db.read_text());commands=[];objects=[]
    for source in (output/'model.cpp',output/'metal.mm',output/'storage.cpp',SCRIPTS/'mtp_draft.cpp',output/'probe.cpp'):
        original=ROOT/'src/engine'/('metal.mm' if source.suffix=='.mm' else 'model.cpp')
        entry=[e for e in entries if Path(e['file']).resolve()==original];require(len(entry)==1,'Missing compiler')
        cmd=shlex.split(entry[0]['command']);obj=output/(source.stem+'.o')
        cmd[1:1]=['-I'+str(output/'include'),'-I'+str(SCRIPTS)]
        cmd[cmd.index('-o')+1]=str(obj);cmd[-1]=str(source);commands.append(cmd);objects.append(obj)
    link=shlex.split(link_file.read_text());link.remove('CMakeFiles/zerocool.dir/src/engine/main.cpp.o')
    binary=output/'probe-mtp-forward';link[link.index('-o')+1]=str(binary)
    at=link.index('libzerocool_lib.a');link[at:at]=list(map(str,objects))
    return dict(output=output,native=native,db=db,link_file=link_file,compiler=commands,linker=link,objects=objects,binary=binary)


def inputs(c):
    dependencies=['build_mtp_forward.py','build_q8_expanded.py','build_q8_block_packed.py','build_block_gdn.py',
        'build_perfect_draft.py','build_block_cache_trace.py','probe_perfect_draft.cpp',
        'mtp_draft.hpp','mtp_draft.cpp','mtp_draft.metal','probe_mtp_forward.cpp',
        'capture_q8_expanded.inc','probe_q8_block_packed.metal','q8_expanded_contract.py']
    return [*[SCRIPTS/p for p in dependencies],*sorted((ROOT/'include/engine').glob('*.hpp')),
        *[ROOT/'src/engine'/p for p in ('model.cpp','storage.cpp','metal.mm')],ROOT/'kernels/metal/qwen.metal',
        c['db'],c['link_file'],c['native']/'libzerocool_lib.a',c['native']/'bin/zerocool']


def proof(c):
    return dict(kind='native_mtp_forward_producer_v1',base_native_fingerprint=build_fingerprint(ROOT),
        inputs={str(p):sha(p) for p in inputs(c)},generated={str(p):sha(p) for p in generated(c['output'])},
        compiler=c['compiler'],linker=c['linker'],production_promoted=False)


def build(output):
    c=settings(output);c['output'].mkdir(parents=True,exist_ok=False)
    for p,value in generated(c['output']).items():p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
    frozen=proof(c);save(c['output']/'producer.json',dict(frozen,complete=False))
    with (c['output']/'build.log').open('w') as log:
        for cmd in [*c['compiler'],c['linker']]:subprocess.run(cmd,cwd=c['native'],stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
    require(frozen==proof(c),'MTP build inputs changed')
    result=dict(frozen,complete=True,binary=str(c['binary']),binary_sha256=sha(c['binary']),objects={str(p):sha(p) for p in c['objects']})
    save(c['output']/'producer.json',result);return result


def verify(directory):
    c=settings(directory);saved=json.loads((c['output']/'producer.json').read_text())
    expected=dict(proof(c),complete=True,binary=str(c['binary']),binary_sha256=sha(c['binary']),objects={str(p):sha(p) for p in c['objects']})
    require(saved==expected,'Changed MTP producer')
    for p,value in generated(c['output']).items():require(p.read_text()==value,'Changed MTP source copy')
    return c,saved


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    result=build(p.parse_args().output);print(json.dumps({k:result[k] for k in ('complete','binary','binary_sha256')}))
