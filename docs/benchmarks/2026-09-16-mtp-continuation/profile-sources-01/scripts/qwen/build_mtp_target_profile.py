#!/usr/bin/env python3
"""Source-copy diagnostic of the current real-draft target verifier; never normal timing."""
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
WORKSPACE=512*1024**2


def generated(output):
    sources=base.generated(output);path=output/'model.cpp';source=sources[path]
    source=replace(source,'    gpu_.configure(verifier_kernels);',
        '    verifier_kernels.profile=phase_=="decode";\n    verifier_kernels.counter_profile=false;\n'
        '    gpu_.configure(verifier_kernels);')
    source=replace(source,'        auto& records=event["records"];',
        '        event["routes"]=std::vector<int>(routes.begin(),routes.end());\n'
        '        event["build"]=gpu_.statistics()["build_fingerprint"];event["artifact_revision"]=checkpoint_.revision();\n'
        '        auto& records=event["records"];')
    sources[path]=source;path=output/'probe.cpp';source=sources[path]
    source=replace(source,'"native_mtp_continuation_v1"','"native_mtp_target_profile_v1"')
    source=replace(source,'    Options options;options.artifact=',
        '    check(work.count==16 && !work.validation && work.fast && prompt.size()==72,"profile requires the short real-draft case");\n'
        '    report["performance_measurement"]=false;report["profile_workspace_bytes"]=512*MiB;\n'
        '    Options options;options.artifact=')
    source=replace(source,'options.audit_routes=work.validation;','options.audit_routes=work.validation;options.kernels.profile=true;')
    source=replace(source,'+MtpDraft::budget_bytes(32,8192);','+MtpDraft::budget_bytes(32,8192)+512*MiB;')
    source=replace(source,'{"host_checkpoint_logits_bytes",host_bound},{"combined_bytes",combined}',
        '{"host_checkpoint_logits_bytes",host_bound},{"combined_bytes",combined},{"profile_workspace_bytes",512*MiB}')
    source=replace(source,'    report["priming_wall_ns"]=monotonic_ns()-prime_start;',
        '    {const auto prime_profile=model.take_profile();check(prime_profile.at("command_groups").empty(),"unexpected priming GPU profile");}\n'
        '    report["priming_wall_ns"]=monotonic_ns()-prime_start;')
    source=replace(source,'model.phase("decode");const auto vstart=monotonic_ns();',
        'model.phase("decode");const auto target_before=gpu.statistics();const auto vstart=monotonic_ns();')
    source=replace(source,'verify_ns=monotonic_ns()-vstart;',
        'verify_ns=monotonic_ns()-vstart;auto profile=model.take_profile();const auto target_after=gpu.statistics();')
    source=replace(source,'        good=keep-1;next=',
        '        check(width==4 && keep==4 && !eos,"profile encountered a different proposal path");\n'
        '        good=keep-1;next=')
    source=replace(source,'        if(consumed%16<keep || consumed==work.count || eos) save_report(output,report);',
        '        size_t entries=0;for(const auto& group:profile.at("command_groups")) entries+=group.at("operations").size();\n'
        '        check(entries<=20000 && profile.at("expert_dependencies").size()==48,"incomplete target profile");\n'
        '        const auto serialized=profile.dump();check(serialized.size()<=32*MiB,"target profile exceeds bound");\n'
        '        const auto file=output.parent_path()/(output.stem().string()+"-block-"+std::to_string(consumed-keep)+".profile.json");\n'
        '        check(!std::filesystem::exists(file),"profile file exists");\n'
        '        std::ofstream stream(file);stream<<serialized;stream.close();check(bool(stream),"profile write failed");\n'
        '        report["cycles"].back().update(Json{{"profile_file",file.filename().string()},{"profile_sha256",hash_file(file)},\n'
        '            {"forward_begin_ns",vstart},{"forward_end_ns",vstart+verify_ns},{"forward_ns",verify_ns},\n'
        '            {"target_before",target_before},{"target_after",target_after}});\n'
        '        save_report(output,report);')
    sources[path]=source;return sources


def inputs(cfg):return [*base.inputs(cfg),Path(__file__).resolve()]


def proof(cfg):
    return dict(kind='mtp_target_profile_producer_v1',base_native_fingerprint=build_fingerprint(ROOT),
        inputs={str(p):sha(p) for p in inputs(cfg)},generated={str(p):sha(p) for p in generated(cfg['output'])},
        compiler=cfg['compiler'],linker=cfg['linker'],workspace_bytes=WORKSPACE,performance_measurement=False,production_promoted=False)


def build(output):
    cfg=base.base.settings(output);cfg['output'].mkdir(parents=True,exist_ok=False)
    for p,value in generated(cfg['output']).items():p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
    frozen=proof(cfg);save(cfg['output']/'producer.json',dict(frozen,complete=False))
    with (cfg['output']/'build.log').open('w') as log:
        for cmd in [*cfg['compiler'],cfg['linker']]:subprocess.run(cmd,cwd=cfg['native'],stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
    require(frozen==proof(cfg),'Profile build inputs changed')
    result=dict(frozen,complete=True,binary=str(cfg['binary']),binary_sha256=sha(cfg['binary']),objects={str(p):sha(p) for p in cfg['objects']})
    save(cfg['output']/'producer.json',result);return result


def verify(directory):
    cfg=base.base.settings(directory);saved=json.loads((cfg['output']/'producer.json').read_text())
    require(saved==dict(proof(cfg),complete=True,binary=str(cfg['binary']),binary_sha256=sha(cfg['binary']),
        objects={str(p):sha(p) for p in cfg['objects']}),'Changed profile producer')
    for p,value in generated(cfg['output']).items():require(p.read_text()==value,'Changed profile source copy')
    return cfg,saved


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    r=build(p.parse_args().output);print(json.dumps({k:r[k] for k in ('complete','binary','binary_sha256')}))
