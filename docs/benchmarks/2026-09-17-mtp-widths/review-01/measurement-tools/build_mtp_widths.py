#!/usr/bin/env python3
"""Isolated fixed-width MTP candidate; no adaptive policy or production changes."""
import argparse
import json
from pathlib import Path
import re
import subprocess

import build_target_recovery as base
from build_block_cache_trace import replace
from build_identity import build_fingerprint
from cache_residency import require
from qualification_evidence import save, sha

ROOT = base.ROOT


def journal_source():
    # Keep four-row capacity in every arm. Only the populated prefix changes.
    s = base.HEADER.read_text()
    s = replace(s, 'uint32_t offset=0;', 'uint32_t offset=0,recorded_width=4;')
    s = replace(s, 'tokens.size()==4 && at<=8192-4',
                '(tokens.size()==2 || tokens.size()==4) && at<=8192-tokens.size()')
    s = replace(s, 'offset=at;active=true;', 'offset=at;recorded_width=uint32_t(tokens.size());active=true;')
    s = replace(s, 'dt && convolution->bytes==4*10240*4', 'dt && convolution->bytes==recorded_width*10240*4')
    s = replace(s, 'convolution->bytes && a->bytes==4*48*4',
                'convolution->bytes && a->bytes==recorded_width*48*4')
    s = replace(s, 'x->bytes==ple->bytes', 'x->bytes==uint64_t(recorded_width)*Hyper*4')
    s = replace(s, 'gpu.copy(x,0,ple,0,ple->bytes)', 'gpu.copy(x,0,ple,0,x->bytes)')
    s = replace(s, 'keep>=1 && keep<=4 && state.artifact', 'keep>=1 && keep<=recorded_width && state.artifact')
    s = s.replace('offset+4ull', 'uint64_t(offset)+recorded_width').replace('offset+4', 'offset+recorded_width')
    s = replace(s, 'complete && !active && ple_seen && keep>=1',
                'complete && !active && ple_seen && (recorded_width==2 || recorded_width==4) && keep>=1')
    s = replace(s, 'keep>=1 && keep<4 && state.tokens', 'keep>=1 && keep<recorded_width && state.tokens')
    s = replace(s, '{"captures",captures}', '{"recorded_width",recorded_width},{"captures",captures}')
    return s


def generated(output):
    sources = base.generated(output)
    # The generated include directory precedes the developer source directory.
    sources[output/'include/mtp_target_recovery.hpp'] = journal_source()
    p = output/'probe.cpp'
    s = sources[p]
    s = replace(s, 'void check_prefix(const State& state,uint32_t keep) const',
                'void check_prefix(const State& state,uint32_t keep,uint32_t width=4) const')
    s = replace(s, 'keep>=1 && keep<=4 && state.tokens==tokens_+4',
                '(width==2 || width==4) && keep>=1 && keep<=width && state.tokens==tokens_+width')
    s = replace(s, 'r.bytes==4*stride', 'r.bytes==width*stride')
    s = replace(s, 'void restore_prefix(State& state,uint32_t keep) const',
                'void restore_prefix(State& state,uint32_t keep,uint32_t width=4) const')
    s = replace(s, 'check_prefix(state,keep);state.valid=false;', 'check_prefix(state,keep,width);state.valid=false;')

    # The existing width-four fixture remains valid. Extend its independent
    # row-at-a-time synthetic reference to cover width two and the context edge.
    a = s.index('Json recovery_self_test(')
    b = s.index('\nstruct ContinuationInput {', a)
    test = s[a:b]
    test = replace(test, 'const std::string& producer_sha256={})',
                   'const std::string& producer_sha256={},uint32_t width=4)')
    test = replace(test, '    const auto before=process_memory(),host_before=host_conditions();',
                   '    check((width==2 || width==4) && (fixture.empty() || width==4),"invalid recovery fixture width");\n'
                   '    const auto before=process_memory(),host_before=host_conditions();')
    test = replace(test, 'for(uint32_t offset:{3u,8188u})', 'for(uint32_t offset:{3u,8192-width})')
    test = replace(test, 'checkpoint.save(state,4);journal.begin(ids,offset);',
                   'checkpoint.save(state,width);journal.begin(std::span(ids).first(width),offset);')
    test = replace(test,
        '                auto& e=journal.entries[l];fill(e.convolution,l);fill(e.normalized,l+1);fill(e.a,l+2);fill(e.b,l+3);\n'
        '                e.ad=e.dd=1;e.alog=gpu.zeros(48,AllocationClass::Resident);e.dt=gpu.zeros(48,AllocationClass::Resident);e.seen=true;',
        '                auto& e=journal.entries[l];auto c=gpu.zeros(width*10240),n=gpu.zeros(width*10240);\n'
        '                auto a=gpu.zeros(width*48),b=gpu.zeros(width*48);fill(c,l);fill(n,l+1);fill(a,l+2);fill(b,l+3);\n'
        '                auto alog=gpu.zeros(48,AllocationClass::Resident),dt=gpu.zeros(48,AllocationClass::Resident);\n'
        '                journal.capture(gpu,l,c,n,a,b,alog,dt,1,1);gpu.finish();\n'
        '                check(e.seen,"width journal input was not captured");')
    test = replace(test, '            fill(journal.ple,7);journal.ple_seen=true;journal.finish();Json expected=Json::array();',
        '            auto ple_input=gpu.zeros(width*Hyper);fill(ple_input,7);journal.capture_ple(gpu,ple_input);\n'
        '            gpu.finish();ple_input.reset();journal.finish();Json expected=Json::array();')
    test = test.replace('keep<=4', 'keep<=width').replace('keep<4', 'keep<width')
    test = replace(test, 'checkpoint.check_prefix(state,keep);', 'checkpoint.check_prefix(state,keep,width);')
    test = replace(test, 'checkpoint.restore_prefix(state,keep);', 'checkpoint.restore_prefix(state,keep,width);')
    test = replace(test, '{"offset",offset},{"keep",keep}', '{"width",width},{"offset",offset},{"keep",keep}')
    s = s[:a]+test+s[b:]

    # Short validation lengths exercise the final one-row fallback without
    # changing the independent eight-row reference used by older checks.
    s = replace(s, 'count=validation?8u:uint32_t(requested);',
                'count=validation?std::min(8u,uint32_t(requested)):uint32_t(requested);')

    # Restrict changes to the continuation function; legacy capture/replay and
    # fixture helpers retain their explicit four-row format.
    a = s.index('void joint(const std::filesystem::path& model_path,')
    b = s.index('\n}\n\n}\nint main(', a)+2
    joint = s[a:b]
    joint = replace(joint, '    const ContinuationInput work(input,mode);',
        '    const char* width_setting=std::getenv("FREELLM_MTP_WIDTH");\n'
        '    check(width_setting && (std::string_view(width_setting)=="1" || std::string_view(width_setting)=="2" ||\n'
        '        std::string_view(width_setting)=="4"),"explicit MTP width must be 1, 2 or 4");\n'
        '    const uint32_t requested_width=uint32_t(width_setting[0]-\'0\');\n'
        '    check(mode=="fast-validate" || mode=="fast-timing","width trial requires maintained draft state");\n'
        '    check(!input.contains("capture_recovery") || requested_width==4,"capture format requires width four");\n'
        '    report["requested_width"]=requested_width;\n'
        '    const ContinuationInput work(input,mode);')
    joint = replace(joint, 'forced_keep>=1 && forced_keep<=4', 'forced_keep>=1 && forced_keep<=requested_width')
    joint = replace(joint,
        'const uint32_t width=!work.serial && !work.is_eos(next) && work.count-consumed>=4?4:1;',
        'const uint32_t width=!work.is_eos(next) && work.count-consumed>=requested_width?requested_width:1;')
    joint = re.sub(r'\bwidth==4\b', 'width>1', joint).replace('forced_keep<4', 'forced_keep<width')
    joint = replace(joint, 'checkpoint.save(state,4);draft_checkpoint.save(gpu,draft_state,4);',
                    'checkpoint.save(state,width);draft_checkpoint.save(gpu,draft_state,width);')
    joint = replace(joint, 'i<3;++i', 'i+1<width;++i')
    joint = replace(joint, 'proposed+=3;', 'proposed+=width-1;')
    joint = replace(joint, 'checkpoint.check_prefix(state,keep);', 'checkpoint.check_prefix(state,keep,width);')
    joint = replace(joint, 'checkpoint.restore_prefix(state,keep);', 'checkpoint.restore_prefix(state,keep,width);')
    s = s[:a]+joint+s[b:]
    s = replace(s, '"native_mtp_continuation_v2"', '"native_mtp_width_v1"')
    s = replace(s, '        if((argc==3 || argc==4) && std::string_view(argv[1])=="--recovery-self-test")',
        '        if(argc==3 && std::string_view(argv[1])=="--width-recovery-self-test") {\n'
        '            check(!std::filesystem::exists(argv[2]),"width self-test output exists");\n'
        '            Json tests=Json::array();\n'
        '            for(uint32_t width:{2u,4u}) tests.push_back(recovery_self_test({},hash_file(argv[0]),width));\n'
        '            save_report(argv[2],{{"kind","mtp_width_recovery_synthetic_v1"},{"complete",true},\n'
        '                {"real_model_evidence",false},{"performance_measurement",false},{"tests",tests}});return 0;\n'
        '        }\n'
        '        if((argc==3 || argc==4) && std::string_view(argv[1])=="--recovery-self-test")')
    sources[p] = s
    return sources


def settings(output): return base.settings(output)
def inputs(cfg): return [*base.inputs(cfg), Path(__file__).resolve()]


def proof(cfg):
    return dict(kind='mtp_width_producer_v1', base_native_fingerprint=build_fingerprint(ROOT),
                inputs={str(p):sha(p) for p in inputs(cfg)},
                generated={str(p):sha(p) for p in generated(cfg['output'])},
                compiler=cfg['compiler'], linker=cfg['linker'], production_promoted=False)


def build(output):
    cfg = settings(output); cfg['output'].mkdir(parents=True, exist_ok=False)
    for p, value in generated(cfg['output']).items():
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(value)
    frozen = proof(cfg); save(cfg['output']/'producer.json', dict(frozen, complete=False))
    with (cfg['output']/'build.log').open('w') as log:
        for cmd in [*cfg['compiler'], cfg['linker']]:
            subprocess.run(cmd, cwd=cfg['native'], stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True)
    require(frozen == proof(cfg), 'Width build inputs changed')
    result = dict(frozen, complete=True, binary=str(cfg['binary']), binary_sha256=sha(cfg['binary']),
                  objects={str(p):sha(p) for p in cfg['objects']})
    save(cfg['output']/'producer.json', result); return result


def verify(directory):
    cfg = settings(directory); saved = json.loads((cfg['output']/'producer.json').read_text())
    require(saved == dict(proof(cfg), complete=True, binary=str(cfg['binary']), binary_sha256=sha(cfg['binary']),
                         objects={str(p):sha(p) for p in cfg['objects']}), 'Changed width producer')
    for p, value in generated(cfg['output']).items():
        require(p.read_text() == value, 'Changed width source copy')
    return cfg, saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    result = build(parser.parse_args().output)
    print(json.dumps({k:result[k] for k in ('complete','binary','binary_sha256')}))
