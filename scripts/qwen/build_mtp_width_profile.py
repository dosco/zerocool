#!/usr/bin/env python3
"""Bounded middle-generation profiling of the isolated fixed-width producer."""
import argparse
import json
from pathlib import Path
import subprocess

import build_mtp_widths as base
from build_block_cache_trace import replace
from build_identity import build_fingerprint
from cache_residency import require
from qualification_evidence import save, sha

ROOT = base.ROOT
WORKSPACE = 512*1024**2
ENTRY_LIMIT = 20000
START = 32
TOKENS = 16


def generated(output):
    sources = base.generated(output)
    p = output/'include/qwen/model.hpp'
    sources[p] = replace(sources[p], '    Json take_profile();',
        '    Json take_profile();\n'
        '    void profile_window(bool active) {\n'
        '        options_.kernels.profile=active;options_.kernels.counter_profile=false;\n'
        '        gpu_.configure(options_.kernels);\n'
        '    }')
    p = output/'model.cpp'
    sources[p] = replace(sources[p], '        auto& records=event["records"];',
        '        event["routes"]=std::vector<int>(routes.begin(),routes.end());\n'
        '        event["build"]=gpu_.statistics()["build_fingerprint"];event["artifact_revision"]=checkpoint_.revision();\n'
        '        auto& records=event["records"];')
    p = output/'metal.mm'
    sources[p] = replace(sources[p],
        'uint64_t profile_limit() const {return config.profile_decode_only?120000:100000;}',
        f'uint64_t profile_limit() const {{return {ENTRY_LIMIT};}}')
    p = output/'probe.cpp'; s = sources[p]
    s = replace(s, '"native_mtp_width_v1"', '"native_mtp_width_profile_v1"')
    s = replace(s, '    Options options;options.artifact=',
        '    check(mode=="fast-timing" && work.count==128 && (requested_width==1 || requested_width==4) &&\n'
        '          !state_only && capture_path.empty(),"profile requires the retained 128-token width-one/four path");\n'
        f'    report["performance_measurement"]=false;report["profile_workspace_bytes"]={WORKSPACE}ull;\n'
        '    Options options;options.artifact=')
    s = replace(s, '+mtp_recovery::ReserveBytes;', f'+mtp_recovery::ReserveBytes+{WORKSPACE}ull;')
    s = replace(s, '{"combined_bytes",combined}',
        f'{{"combined_bytes",combined}},{{"profile_workspace_bytes",{WORKSPACE}ull}}')
    s = replace(s, '    uint64_t total=0;uint32_t consumed=0,proposed=0,accepted=0;bool eos=false;',
        '    uint32_t profile_first=0,profile_committed=0,profile_verified=0,profile_calls=0;\n'
        '    uint64_t total=0;uint32_t consumed=0,proposed=0,accepted=0;bool eos=false;')
    s = replace(s, '        model.phase("decode");const auto vstart=monotonic_ns();std::vector<float> logits;',
        f'        const bool profiling=consumed>={START} && profile_committed<{TOKENS};\n'
        '        if(profiling && !profile_calls) profile_first=consumed;\n'
        '        const auto outside=model.take_profile();\n'
        '        check(outside.at("command_groups").empty() && outside.at("expert_dependencies").empty(),"work escaped profile window");\n'
        '        model.profile_window(profiling);model.phase("decode");\n'
        '        const auto target_before=profiling?gpu.statistics():Json::object();\n'
        '        const auto vstart=monotonic_ns();std::vector<float> logits;')
    s = replace(s, '        verify_ns=monotonic_ns()-vstart;',
        '        verify_ns=monotonic_ns()-vstart;auto profile=model.take_profile();\n'
        '        const auto target_after=profiling?gpu.statistics():Json::object();\n'
        '        model.profile_window(false);\n'
        '        check(profiling || (profile.at("command_groups").empty() && profile.at("expert_dependencies").empty()),"uncaptured target emitted profile work");')
    s = replace(s, '        if(consumed%16<keep || consumed==work.count || eos) save_report(output,report);',
        '        if(profiling) {\n'
        '            profile_committed+=keep;profile_verified+=width;++profile_calls;\n'
        f'            check(profile_calls<={TOKENS} && profile_committed<={TOKENS+3},"profile coverage exceeded bound");\n'
        '            size_t entries=0;for(const auto& g:profile.at("command_groups")) entries+=g.at("operations").size();\n'
        f'            check(entries>0 && entries<={ENTRY_LIMIT} && profile.at("truncated")==false &&\n'
        '                  profile.at("expert_dependencies").size()==48,"incomplete bounded profile");\n'
        '            const auto serialized=profile.dump();check(serialized.size()<=32*MiB,"profile serialization exceeds bound");\n'
        '            const auto file=output.parent_path()/(output.stem().string()+"-cycle-"+std::to_string(report["cycles"].size()-1)+".profile.json");\n'
        '            check(!std::filesystem::exists(file),"profile file already exists");\n'
        '            std::ofstream stream(file);stream<<serialized;stream.close();check(bool(stream),"profile write failed");\n'
        '            report["cycles"].back().update(Json{{"profile_file",file.filename().string()},{"profile_sha256",hash_file(file)},\n'
        '                {"forward_begin_ns",vstart},{"forward_end_ns",vstart+verify_ns},{"forward_ns",verify_ns},\n'
        '                {"target_before",target_before},{"target_after",target_after}});\n'
        '        }\n'
        '        if(consumed%16<keep || consumed==work.count || eos) save_report(output,report);')
    s = replace(s, '    const auto request_end=monotonic_ns();',
        '    const auto outside=model.take_profile();\n'
        '    check(outside.at("command_groups").empty() && outside.at("expert_dependencies").empty(),"work escaped final profile window");\n'
        f'    check(profile_calls>0 && profile_committed>={TOKENS} && consumed==128 && !eos,"incomplete middle-generation profile");\n'
        f'    report["profile_window"]={{{{"requested_start",{START}}},{{"requested_committed",{TOKENS}}},\n'
        '        {"first_output",profile_first},{"end_output",profile_first+profile_committed},{"captured_calls",profile_calls},\n'
        '        {"captured_committed",profile_committed},{"captured_verified",profile_verified},\n'
        '        {"omitted_before",profile_first},{"omitted_after",consumed-profile_first-profile_committed}};\n'
        '    const auto request_end=monotonic_ns();')
    sources[p] = s
    return sources


def inputs(cfg): return [*base.inputs(cfg), Path(__file__).resolve()]


def proof(cfg):
    return dict(kind='mtp_width_profile_producer_v1',base_native_fingerprint=build_fingerprint(ROOT),
        inputs={str(p):sha(p) for p in inputs(cfg)},generated={str(p):sha(p) for p in generated(cfg['output'])},
        compiler=cfg['compiler'],linker=cfg['linker'],workspace_bytes=WORKSPACE,
        entry_limit=ENTRY_LIMIT,start_output=START,minimum_committed=TOKENS,
        performance_measurement=False,production_promoted=False)


def build(output):
    cfg=base.settings(output);cfg['output'].mkdir(parents=True,exist_ok=False)
    for p,value in generated(cfg['output']).items():
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
    frozen=proof(cfg);save(cfg['output']/'producer.json',dict(frozen,complete=False))
    with (cfg['output']/'build.log').open('w') as log:
        for cmd in [*cfg['compiler'],cfg['linker']]:
            subprocess.run(cmd,cwd=cfg['native'],stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
    require(frozen==proof(cfg),'Profile build inputs changed')
    result=dict(frozen,complete=True,binary=str(cfg['binary']),binary_sha256=sha(cfg['binary']),
                objects={str(p):sha(p) for p in cfg['objects']})
    save(cfg['output']/'producer.json',result);return result


def verify(directory):
    cfg=base.settings(directory);saved=json.loads((cfg['output']/'producer.json').read_text())
    require(saved==dict(proof(cfg),complete=True,binary=str(cfg['binary']),binary_sha256=sha(cfg['binary']),
            objects={str(p):sha(p) for p in cfg['objects']}),'Changed profile producer')
    for p,value in generated(cfg['output']).items():require(p.read_text()==value,'Changed profile source copy')
    return cfg,saved


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    result=build(parser.parse_args().output)
    print(json.dumps({k:result[k] for k in ('complete','binary','binary_sha256')}))
