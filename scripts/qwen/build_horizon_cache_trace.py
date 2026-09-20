#!/usr/bin/env python3
"""Trace target leases for four/eight-row tiled verification, with an off control."""
import argparse
import json
from pathlib import Path
import subprocess

import build_tiled_verifier_horizon as base
import build_block_cache_trace as trace
import build_perfect_draft as perfect
from cache_residency import require
from qualification_evidence import save, sha

ROOT = base.ROOT
HEADERS = [ROOT/'scripts/qwen'/p for p in
    ('horizon_cache_trace.hpp', 'block_cache_trace.hpp', 'mtp_fixed_priming.hpp')]
INCLUDE = '#include "horizon_cache_trace.hpp"\n'


def generated(output):
    output = Path(output).resolve()
    sources = base.generated(output)
    p = output/'model.cpp'; s = INCLUDE+sources[p]
    s = trace.replace(s, perfect.FORWARD, perfect.FORWARD+
        '    horizon_trace::ForwardScope target_trace;\n'
        '    horizon_trace::forward_begin(state.tokens,ids,phase_);\n')
    s = trace.replace(s, '        return result;\n    } catch(...) {',
        '        (void)cache_->json();horizon_trace::forward_end(state.tokens,route_identity());\n'
        '        return result;\n    } catch(...) {')
    marker = '    // Encode shared work now. The first miss batch is submitted before this\n'
    sources[p] = trace.replace(s, marker,
        '    horizon_trace::layer(layer,trace_offset_,tokens,raw,selected);\n'+marker)
    # Reuse the established acquire/pin/release instrumentation on source copies.
    # Its scoped wrappers ignore the draft, which shares the cache implementation.
    p = output/'storage.cpp'
    sources[p] = INCLUDE+trace.instrument_cache(sources[p]).replace('block_trace::', 'horizon_trace::')
    sources[output/'mtp_draft.cpp'] = '#include "mtp_fixed_priming.hpp"\n'+trace.replace(
        (ROOT/'scripts/qwen/mtp_draft.cpp').read_text(),
        '    execute_experts(selected,*cache_,reads_,gpu_,4,',
        '    mtp_fixed_priming::execute(selected,*cache_,reads_,gpu_,4,')
    p = output/'probe.cpp'; s = INCLUDE+'#include "mtp_fixed_priming.hpp"\n'+sources[p]
    s = trace.replace(s, '    embedding_rows::Scope streamed_scope;',
        '    const char* trace_setting=std::getenv("FREELLM_HORIZON_CACHE_TRACE");\n'
        '    check(trace_setting && (std::string_view(trace_setting)=="on" || std::string_view(trace_setting)=="off"),"explicit cache trace mode required");\n'
        '    horizon_trace::open(output,std::string_view(trace_setting)=="on");\n'
        '    report["cache_trace_enabled"]=horizon_trace::enabled;report["performance_measurement"]=false;\n'
        '    embedding_rows::Scope streamed_scope;')
    s = trace.replace(s, 'prompt.size()<=512 && count>=1 && count<=256',
        'prompt.size()<=128 && count>=1 && count<=64')
    s = trace.replace(s, 'options.audit_routes=validation;', 'options.audit_routes=true;')
    s = trace.replace(s, '+mtp_scratch::ReserveBytes+mtp_recovery::ReserveBytes;',
        '+mtp_scratch::ReserveBytes+mtp_recovery::ReserveBytes+block_trace::workspace_bytes;')
    s = trace.replace(s, '{"target_recovery_reserve_bytes",mtp_recovery::ReserveBytes}};',
        '{"target_recovery_reserve_bytes",mtp_recovery::ReserveBytes},{"trace_workspace_bytes",block_trace::workspace_bytes}};')
    s = trace.replace(s, '    for(uint32_t at=0;at<prompt.size();) {',
        '    { mtp_fixed_priming::Scope fixed_draft_priming;\n    for(uint32_t at=0;at<prompt.size();) {')
    s = trace.replace(s, '    check(draft_state.position+1==state.tokens',
        '    }\n    report["draft_priming_before"]=mtp_fixed_priming::stats();\n'
        '    check(draft_state.position+1==state.tokens')
    s = trace.replace(s, '    check(model.memory_plan().json()==plan,"horizon memory admission changed");',
        '    check(model.memory_plan().json()==plan,"horizon memory admission changed");\n'
        '    report["draft_priming_after"]=mtp_fixed_priming::stats();\n'
        '    report["trace_forwards"]=horizon_trace::forwards;\n'
        '    report["target_cache_trace"]=horizon_trace::finish();')
    sources[p] = s
    return sources


def settings(output):
    cfg = base.settings(output)
    old = str(ROOT/'scripts/qwen/mtp_draft.cpp')
    commands = [cmd for cmd in cfg['compiler'] if cmd[-1] == old]
    require(len(commands) == 1, 'Missing unique draft compiler command')
    commands[0][-1] = str(cfg['output']/'mtp_draft.cpp')
    return cfg


def inputs(cfg): return [*base.inputs(cfg), Path(__file__).resolve(), *HEADERS]


def proof(cfg):
    return dict(kind='horizon_cache_trace_producer_v1', base_native_fingerprint=base.build_fingerprint(ROOT),
        inputs={str(p): sha(p) for p in inputs(cfg)}, generated={str(p): sha(p) for p in generated(cfg['output'])},
        compiler=cfg['compiler'], linker=cfg['linker'], production_promoted=False,
        trace_modes=['off', 'on'], compute_tile_cap=4, widths=[4, 8],
        draft_priming_policy='fixed-lease-batches-32', performance_measurement=False)


def build(output):
    cfg = settings(output); cfg['output'].mkdir(parents=True, exist_ok=False)
    for p, s in generated(cfg['output']).items():
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(s)
    frozen = proof(cfg); save(cfg['output']/'producer.json', dict(frozen, complete=False))
    with (cfg['output']/'build.log').open('w') as log:
        for cmd in [*cfg['compiler'], cfg['linker']]:
            subprocess.run(cmd, cwd=cfg['native'], stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True)
    require(frozen == proof(cfg), 'Trace build inputs changed')
    result = dict(frozen, complete=True, binary=str(cfg['binary']), binary_sha256=sha(cfg['binary']),
        objects={str(p): sha(p) for p in cfg['objects']})
    save(cfg['output']/'producer.json', result)
    return result


def verify(output):
    cfg = settings(output); saved = json.loads((cfg['output']/'producer.json').read_text())
    require(saved == dict(proof(cfg), complete=True, binary=str(cfg['binary']), binary_sha256=sha(cfg['binary']),
        objects={str(p): sha(p) for p in cfg['objects']}), 'Trace producer changed')
    for p, s in generated(cfg['output']).items(): require(p.read_text() == s, 'Trace source copy changed')
    return cfg, saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    result = build(parser.parse_args().output)
    print(json.dumps({k: result[k] for k in ('complete', 'binary', 'binary_sha256')}))
