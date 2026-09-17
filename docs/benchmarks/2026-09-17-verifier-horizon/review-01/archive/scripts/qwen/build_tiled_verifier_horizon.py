#!/usr/bin/env python3
"""Separate eight-token storage grouping from four-token GPU tiles."""
import argparse
import json
from pathlib import Path
import subprocess

import build_streamed_verifier_horizon as base
import build_q8_expanded as packed
from build_block_cache_trace import replace
from build_identity import build_fingerprint
from cache_residency import require
from qualification_evidence import save, sha

ROOT=base.ROOT


def shader():
    s=packed.shader().replace('q8_expanded_t4_w8','q8_horizon8_t4_w8').replace('p[2]!=4','p[2]!=8')
    s=replace(s,'uint tid [[thread_position_in_grid]]','uint2 gid [[thread_position_in_grid]]')
    s=replace(s,'row=tid/32,lane=tid%32','row=gid.x/32,lane=gid.x%32,t0=gid.y*4')
    s=replace(s,'if(row>=N ||','if(row>=N || t0>=8 ||')
    s=replace(s,'x[t*K+base+i]','x[(t0+t)*K+base+i]')
    s=replace(s,'out[t*N+row]','out[(t0+t)*N+row]')
    return s


def generated(output):
    output=Path(output).resolve();sources=base.generated(output)
    p=output/'model.cpp';sources[p]=replace(sources[p],
        'verifier_kernels.token_tile=uint32_t(ids.size());','verifier_kernels.token_tile=std::min(4u,uint32_t(ids.size()));')
    p=output/'metal.mm';s=replace(sources[p],base.base.shader(),shader())
    s=replace(s,'tile==tokens && policy.rows==1','tile==4 && policy.rows==1')
    s=replace(s,'tokens==8?"q8_horizon_t8_w8":"q8_expanded_t4_w8"','tokens==8?"q8_horizon8_t4_w8":"q8_expanded_t4_w8"')
    s=replace(s,'{l.input,l.output,tokens,l.group,uint32_t(float_output)},l.output*32);',
        '{l.input,l.output,tokens,l.group,uint32_t(float_output)},l.output*32,tokens/4);')
    sources[p]=s;p=output/'probe.cpp';s=sources[p]
    s=replace(s,'gpu.dispatch("q8_horizon_t8_w8",','gpu.dispatch("q8_horizon8_t4_w8",')
    s=replace(s,'{K,N,8,64,fp32},N*32);','{K,N,8,64,fp32},N*32,2);')
    s=replace(s,'report["requested_width"]=requested_width;report["prompt_tokens"]=',
        'report["compute_tile_cap"]=4;report["requested_width"]=requested_width;report["prompt_tokens"]=')
    sources[p]=s;return sources


def settings(output):return base.settings(output)
def inputs(cfg):return [*base.inputs(cfg),Path(__file__).resolve()]
def proof(cfg):
    return dict(kind='tiled_verifier_horizon_producer_v1',base_native_fingerprint=build_fingerprint(ROOT),
        inputs={str(p):sha(p) for p in inputs(cfg)},generated={str(p):sha(p) for p in generated(cfg['output'])},
        compiler=cfg['compiler'],linker=cfg['linker'],production_promoted=False,embedding_storage='exact-packed-rows',
        widths=[1,4,8],compute_tile_cap=4,perfect_proposals_only=True,normal_request_latency_qualified=False)


def build(output):
    cfg=settings(output);cfg['output'].mkdir(parents=True,exist_ok=False)
    for p,s in generated(cfg['output']).items():p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
    frozen=proof(cfg);save(cfg['output']/'producer.json',dict(frozen,complete=False))
    with (cfg['output']/'build.log').open('w') as log:
        for cmd in [*cfg['compiler'],cfg['linker']]:
            subprocess.run(cmd,cwd=cfg['native'],stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
    require(frozen==proof(cfg),'Tiled horizon build changed')
    result=dict(frozen,complete=True,binary=str(cfg['binary']),binary_sha256=sha(cfg['binary']),objects={str(p):sha(p) for p in cfg['objects']})
    save(cfg['output']/'producer.json',result);return result


def verify(output):
    cfg=settings(output);saved=json.loads((cfg['output']/'producer.json').read_text())
    require(saved==dict(proof(cfg),complete=True,binary=str(cfg['binary']),binary_sha256=sha(cfg['binary']),
        objects={str(p):sha(p) for p in cfg['objects']}),'Tiled horizon producer changed')
    for p,s in generated(cfg['output']).items():require(p.read_text()==s,'Tiled source copy changed')
    return cfg,saved


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    r=build(p.parse_args().output);print(json.dumps({k:r[k] for k in ('complete','binary','binary_sha256')}))
