#!/usr/bin/env python3
"""Build bounded four-token diagnostic copies; production sources stay unchanged."""
import argparse
import json
from pathlib import Path
import subprocess

import build_perfect_draft as base
from build_block_cache_trace import replace
from build_identity import build_fingerprint
from qualification_evidence import save, sha

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT/'scripts/qwen/probe_perfect_draft.cpp'
MODES = ('commands', 'dispatch')
WORKSPACE = 512*1024**2


def model_source(source, mode):
    if mode not in MODES: raise ValueError('Unknown profile mode')
    out = base.instrument(source)
    out = replace(out, '    gpu_.configure(verifier_kernels);',
        '    verifier_kernels.profile=phase_=="decode";\n'
        f'    verifier_kernels.counter_profile=phase_=="decode" && {str(mode=="dispatch").lower()};\n'
        '    gpu_.configure(verifier_kernels);')
    return replace(out, '        auto& records=event["records"];',
        '        event["routes"]=std::vector<int>(routes.begin(),routes.end());\n'
        '        event["build"]=gpu_.statistics()["build_fingerprint"];\n'
        '        event["artifact_revision"]=checkpoint_.revision();\n'
        '        auto& records=event["records"];')


def harness_source(source, mode):
    if mode not in MODES: raise ValueError('Unknown profile mode')
    out = replace(source, '        phase(output,"load_model");',
        '        check(width==4 && slots==1460 && !validation,"profile requires width4/1460 timing mode");\n'
        '        options.kernels.profile=true;\n'
        f'        report["block_profile_mode"]="{mode}";report["profile_workspace_bytes"]={WORKSPACE}ull;\n'
        '        phase(output,"load_model");')
    out = replace(out, 'plan.at("planned_bytes").get<uint64_t>()+128*MiB+logits_bound<=12*GiB',
        f'plan.at("planned_bytes").get<uint64_t>()+128*MiB+logits_bound+{WORKSPACE}ull<=12*GiB')
    out = replace(out, '            model.phase("decode");uint64_t total_wall=0;',
        '            check(model.take_profile().at("command_groups").empty(),"unexpected prime GPU profile");\n'
        '            model.phase("decode");uint64_t total_wall=0;')
    out = replace(out, '                    {"checkpoint_ns",checkpoint_ns}',
        '                    {"forward_begin_ns",forward_start},{"forward_end_ns",forward_end},\n'
        '                    {"checkpoint_ns",checkpoint_ns}')
    out = replace(out, '                if(evidence_boundary) save_report(output,report);',
        '                {\n'
        '                    auto profile=model.take_profile();\n'
        '                    size_t entries=0;for(const auto& group:profile.at("command_groups")) entries+=group.at("operations").size();\n'
        '                    check(entries<=20000 && profile.at("expert_dependencies").size()==48,"profile capture bound exceeded");\n'
        '                    const auto serialized=profile.dump();check(serialized.size()<=32*MiB,"profile exceeds file bound");\n'
        '                    auto file=output.parent_path()/(output.stem().string()+"-block-"+std::to_string(at)+".profile.json");\n'
        '                    check(!std::filesystem::exists(file),"profile output already exists");\n'
        '                    std::ofstream stream(file);stream<<serialized;stream.close();check(bool(stream),"profile write failed");\n'
        '                    report["blocks"].back()["profile_file"]=file.filename().string();\n'
        '                    report["blocks"].back()["profile_sha256"]=hash_file(file);\n'
        '                }\n'
        '                if(evidence_boundary) save_report(output,report);')
    return out


def settings(output):
    output = Path(output).resolve()
    return base.commands(ROOT, output, output/'probe.block-profile.cpp')


def sources(cfg, mode):
    return {cfg['generated']: model_source(cfg['source'].read_text(), mode),
        cfg['storage_generated']: base.instrument_storage(cfg['storage_source'].read_text()),
        cfg['binary'].parent/'probe.block-profile.cpp': harness_source(TEMPLATE.read_text(), mode)}


def inputs(cfg):
    return [*base.frozen_inputs(cfg, TEMPLATE), Path(__file__).resolve(),
        ROOT/'scripts/qwen/build_block_cache_trace.py',
        *sorted((ROOT/'include/qwen').glob('*.hpp'))]


def proof(cfg, mode):
    return dict(kind='block_profile_producer_v1', mode=mode,
        base_native_fingerprint=build_fingerprint(ROOT),
        original_inputs={str(p): sha(p) for p in inputs(cfg)},
        generated={str(p): sha(p) for p in sources(cfg, mode)},
        compiler=cfg['compiler'], linker=cfg['linker'], binary=str(cfg['binary']),
        production_promoted=False, performance_measurement=False, workspace_bytes=WORKSPACE)


def build(output, mode):
    cfg = settings(output); cfg['binary'].parent.mkdir(parents=True, exist_ok=False)
    for path, value in sources(cfg, mode).items(): path.write_text(value)
    frozen = proof(cfg, mode)
    save(cfg['binary'].parent/'producer.json', dict(frozen, complete=False))
    with (cfg['binary'].parent/'build.log').open('w') as log:
        for cmd in [*cfg['compiler'], cfg['linker']]:
            subprocess.run(cmd, cwd=cfg['directory'], stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True)
    if frozen != proof(cfg, mode): raise ValueError('Profile build inputs changed')
    result = dict(frozen, complete=True, binary_sha256=sha(cfg['binary']),
        objects={str(p): sha(p) for p in cfg['objects']})
    save(cfg['binary'].parent/'producer.json', result)
    return result


def verify(binary, fingerprint, mode):
    binary = Path(binary).resolve(); cfg = settings(binary.parent)
    saved = json.loads((binary.parent/'producer.json').read_text())
    expected = dict(proof(cfg, mode), complete=True, binary_sha256=sha(binary),
        objects={str(p): sha(p) for p in cfg['objects']})
    if saved != expected or expected['base_native_fingerprint'] != fingerprint or binary != cfg['binary']:
        raise ValueError('Changed profile producer')
    for path, value in sources(cfg, mode).items():
        if path.read_text() != value: raise ValueError('Changed profile source copy')
    return dict(producer=saved, files=[*inputs(cfg), *sources(cfg, mode), *cfg['objects'], binary, binary.parent/'producer.json'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--mode', choices=MODES, required=True)
    a = p.parse_args(); print(json.dumps(build(a.output, a.mode), indent=2))
