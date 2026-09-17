#!/usr/bin/env python3
"""Build source-copy capture, six-shape probe and verifier for expanded packed Q8."""
import argparse
import json
from pathlib import Path
import subprocess

import build_block_gdn as capture
import build_q8_block_packed as packed
import build_perfect_draft as verifier
from build_block_cache_trace import replace
from build_identity import build_fingerprint
from cache_residency import require
from qualification_evidence import save, sha
from q8_expanded_contract import ROOT, KERNEL, VARIANT, CASES

HOOK = ROOT/'scripts/qwen/capture_q8_expanded.inc'
PROBE = ROOT/'scripts/qwen/probe_q8_expanded.cpp'


def settings(output):
    cfg = packed.settings(output); c = capture.settings(output)
    cfg['capture'] = c; cfg['capture_binary'] = c['binary']
    cfg['verifier'] = verifier.commands(ROOT, output, Path(output)/'probe.expanded-verifier.cpp')
    # Separate objects for non-profiled verifier model and harness.
    v = cfg['verifier']; v['generated'] = cfg['output']/'model.expanded-verifier.cpp'
    v['storage_generated'] = c['storage_generated']
    v['objects'] = [cfg['output']/'model.expanded-verifier.o', c['objects'][1], cfg['output']/'probe.expanded-verifier.o']
    for i in (0,2):
        v['compiler'][i][v['compiler'][i].index('-o')+1] = str(v['objects'][i])
        v['compiler'][i][-1] = str(v['generated'] if i==0 else cfg['output']/'probe.expanded-verifier.cpp')
    v['binary'] = cfg['output']/'probe-expanded-verifier'
    v['linker'] = list(c['linker']); v['linker'][v['linker'].index('-o')+1] = str(v['binary'])
    v['linker'] = [x for x in v['linker'] if x not in map(str,c['objects'])]
    v['linker'][v['linker'].index('libfreellm_lib.a'):v['linker'].index('libfreellm_lib.a')] = list(map(str,v['objects']))
    for link in (c['linker'],v['linker']): link.insert(link.index('libfreellm_lib.a'),str(cfg['objects'][0]))
    cfg['compiler'] = [*cfg['compiler'], *c['compiler'], v['compiler'][0], v['compiler'][2]]
    cfg['linkers'] = [cfg['linker'], c['linker'], v['linker']]
    cfg['all_objects'] = list(dict.fromkeys([*cfg['objects'],*c['objects'],*v['objects']]))
    return cfg


def shader():
    value = packed.SHADER.read_text().replace(packed.KERNEL,KERNEL)
    value = replace(value, '(N==10240 || N==6144)', '(N==10240 || N==6144 || N==12288 || N==248320)')
    return value.replace('These three GDN shapes','These admitted GDN, attention and output shapes')


def metal_source():
    value = replace(packed.METAL.read_text(), '        const auto source=MetalSource;',
        '        const auto source=std::string(MetalSource)+R"PACKED_Q8(\n'+shader()+')PACKED_Q8";')
    start=value.index('void Metal::capture_linear('); end=value.index('Buf Metal::linear(',start)
    value=value[:start]+HOOK.read_text()+'\n'+value[end:]
    value=replace(value, '    bool retained_references=true;',
        '    bool retained_references=true;\n    bool expanded_q8=false,expanded_scope=false;')
    value=replace(value, 'Metal::Metal() : impl_(std::make_unique<Impl>()) {',
        'Metal::Metal() : impl_(std::make_unique<Impl>()) {\n'
        '    if(const char* mode=std::getenv("FREELLM_Q8_EXPANDED")) {\n'
        '        if(std::string_view(mode)!="packed") throw std::runtime_error("invalid expanded Q8 mode");\n'
        '        impl_->expanded_q8=true;\n    }')
    marker='void Metal::label(std::string stage,int layer,uint32_t tokens,uint32_t offset,std::span<const uint32_t> experts) {'
    value=replace(value,marker,marker+'\n    impl_->expanded_scope=(stage=="gdn" || stage=="attention" || stage=="logits");')
    marker='    if(impl_->config.q4_decode=="packed-r2" && impl_->request_phase=="decode" && tokens==1 &&\n'
    value=replace(value,marker,
        '    if(impl_->expanded_q8 && impl_->expanded_scope && impl_->request_phase=="decode" && tokens==4 &&\n'
        '       tile==4 && policy.rows==1 && l.quantized && l.bits==8 && l.group==64 && l.weight.offset%4==0 &&\n'
        '       ((l.input==2560 && (l.output==10240 || l.output==6144 || l.output==12288 || l.output==248320)) ||\n'
        '        (l.input==6144 && l.output==2560))) {\n'
        '        dispatch("'+KERNEL+'",{l.weight,l.scales,l.biases,{x},out},\n'
        '            {l.input,l.output,tokens,l.group,uint32_t(float_output)},l.output*32);\n'
        '    }\n    else '+marker[4:])
    return value


def sources(cfg):
    c=cfg['capture']; result=capture.sources(c); harness=c['binary'].parent/'probe.block-profile.cpp'
    result[harness]=replace(result[harness], 'output.parent_path()/"gdn-fixtures"', 'output.parent_path()/"inputs"')
    result[harness]=replace(result[harness], 'options.kernels.capture_operator="gdn";options.kernels.capture_layer=0;',
        'options.kernels.capture_operator.clear();options.kernels.capture_layer=-2;')
    result.update({cfg['generated'][0]:metal_source(),cfg['generated'][1]:replace(PROBE.read_text(),'@CASES@',json.dumps(CASES))})
    v=cfg['verifier']; result[v['generated']]=verifier.instrument(v['source'].read_text())
    raw=(ROOT/'scripts/qwen/probe_perfect_draft.cpp').read_text()
    raw=replace(raw,'        phase(output,"load_model");const auto setup_start=monotonic_ns();',
        '        report["expanded_q8"]=std::getenv("FREELLM_Q8_EXPANDED")!=nullptr;\n'
        '        phase(output,"load_model");const auto setup_start=monotonic_ns();')
    result[cfg['output']/'probe.expanded-verifier.cpp']=raw
    return result


def inputs(cfg):
    return list(dict.fromkeys([*packed.inputs(cfg), *capture.inputs(cfg['capture']), HOOK, PROBE,
        Path(__file__).resolve(),ROOT/'scripts/qwen/q8_expanded_contract.py']))


def proof(cfg):
    return dict(kind='q8_expanded_producer_v1',base_native_fingerprint=build_fingerprint(ROOT),variant=VARIANT,
        original_inputs={str(p):sha(p) for p in inputs(cfg)},generated={str(p):sha(p) for p in sources(cfg)},
        compiler=cfg['compiler'],linker=cfg['linkers'],production_promoted=False)


def build(output):
    cfg=settings(output);cfg['output'].mkdir(parents=True,exist_ok=False)
    for p,value in sources(cfg).items():p.write_text(value)
    frozen=proof(cfg);save(cfg['output']/'producer.json',dict(frozen,complete=False))
    with (cfg['output']/'build.log').open('w') as log:
        for cmd in [*cfg['compiler'],*cfg['linkers']]:
            subprocess.run(cmd,cwd=cfg['directory'],stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
    require(frozen==proof(cfg),'Expanded build inputs changed')
    result=dict(frozen,complete=True,objects={str(p):sha(p) for p in cfg['all_objects']},
        binaries={str(p):sha(p) for p in (cfg['binary'],cfg['capture_binary'],cfg['verifier']['binary'])})
    save(cfg['output']/'producer.json',result);return result


def verify(output,fingerprint):
    cfg=settings(output);saved=json.loads((cfg['output']/'producer.json').read_text())
    expected=dict(proof(cfg),complete=True,objects={str(p):sha(p) for p in cfg['all_objects']},
        binaries={str(p):sha(p) for p in (cfg['binary'],cfg['capture_binary'],cfg['verifier']['binary'])})
    require(saved==expected and saved['base_native_fingerprint']==fingerprint,'Changed expanded producer')
    for p,value in sources(cfg).items():require(p.read_text()==value,'Changed expanded source copy')
    return dict(producer=saved,files=[*inputs(cfg),*sources(cfg),*cfg['all_objects'],
        *map(Path,saved['binaries']),cfg['output']/'producer.json'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    print(json.dumps(build(p.parse_args().output),indent=2))
