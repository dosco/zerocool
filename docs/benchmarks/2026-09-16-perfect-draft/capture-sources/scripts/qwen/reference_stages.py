#!/usr/bin/env python3
"""Trace one original-checkpoint layer on a recorded native input (no arithmetic replacements)."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from reference_mlx import Reader,Weights,Experts,Ngrams,prepare
from verify_checkpoint import ROOT,verify

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--model',type=Path,required=True);ap.add_argument('--tokens',type=Path,required=True)
ap.add_argument('--lock',type=Path,default=ROOT/'models.lock.json')
ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--layer',type=int,required=True);args=ap.parse_args()
if not 0<=args.layer<48: raise SystemExit('Layer must be 0..47')
tokens=json.loads(args.tokens.read_text());T=len(tokens)
if not 0<T<=32: raise SystemExit('Use 1..32 tokens')
mx.set_cache_limit(64*1024**2)
mx.set_memory_limit(768*1024**2)
verify(args.model,json.loads(args.lock.read_text()),cached=True)
spec=importlib.util.spec_from_file_location('freellm_pinned_stages',args.model/'qwen4_exp.py')
ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)
config=json.loads((args.model/'config.json').read_text())
cfg=ref.ModelArgs.from_dict(config).text
reader=Reader(Weights(args.model),config);l=args.layer;b=f'model.layers.{l}'
layer=ref.DecoderLayer(cfg,l);layer.mlp.switch_mlp=Experts(reader,b+'.mlp.switch_mlp')
if l==1:layer.ple.ple_embedding=Ngrams(reader,tokens)
prepare(layer,b,reader);args.out.mkdir(parents=True,exist_ok=True)

class Capture(nn.Module):
    def __init__(self,fn,name,tuple_names=None,post=None):super().__init__();self.fn=fn;self._name=name;self._tuple_names=tuple_names;self._post=post
    def __call__(self,*a,**kw):
        if self._name.startswith('ple.'):
            np.array(a[0].astype(mx.float32)).tofile(args.out/(self._name+'.input.bin'))
        value=self.fn(*a,**kw)
        if self._tuple_names:
            for name,item in zip(self._tuple_names,value):np.array(item.astype(mx.float32)).tofile(args.out/(name+'.bin'))
        else:np.array((self._post(value) if self._post else value).astype(mx.float32)).tofile(args.out/(self._name+'.bin'))
        return value

for key in ['attn_hyper_connection','mlp_hyper_connection']:
    m=getattr(layer,key)
    for attr,label in [('hc_norm','norm'),('input_mix_weight_down','projected'),('input_mix_weight_up','up'),('block_inject_weight','inject_raw')]:
        setattr(m,attr,Capture(getattr(m,attr),b+'.'+key+'.'+label))
layer.attn_hyper_connection=Capture(layer.attn_hyper_connection,'', [f'x1_{l}',f'hyper_{l}',f'inject1_{l}'])
layer.mlp_hyper_connection=Capture(layer.mlp_hyper_connection,'',[f'x2_{l}',f'after_{l}',f'inject2_{l}'])
if (l+1)%4:
    g=layer.linear_attn;prefix=b+'.linear_attn'
    for attr,label in [('in_proj_qkv',f'qkv_{l}'),('in_proj_z',prefix+'.z'),('in_proj_a',prefix+'.a'),('in_proj_b',prefix+'.b'),('norm',prefix+'.gated')]:
        setattr(g,attr,Capture(getattr(g,attr),label))
    g.conv1d=Capture(g.conv1d,prefix+'.conv',post=nn.silu)
    original_delta=ref.gated_delta_update
    def capture_delta(q,k,v,*args_,**kwargs):
        mixed=mx.concatenate([q.reshape(1,T,-1),k.reshape(1,T,-1),v.reshape(1,T,-1)],axis=-1)
        np.array(mixed.astype(mx.float32)).tofile(args.out/(prefix+'.qk.bin'))
        value=original_delta(q,k,v,*args_,**kwargs)
        np.array(value[0].astype(mx.float32)).tofile(args.out/(prefix+'.y.bin'))
        return value
    ref.gated_delta_update=capture_delta
    layer.linear_attn=Capture(g,f'attn_{l}')
else:layer.self_attn=Capture(layer.self_attn,f'attn_{l}')
if (l+1)%4==0:
    original_sdpa=ref.scaled_dot_product_attention
    def capture_sdpa(q,k,v,**kwargs):
        for name,value in [('q',q),('k',k),('v',v)]:
            np.array(value.transpose(0,2,1,3).astype(mx.float32)).tofile(args.out/(b+'.self_attn.'+name+'.bin'))
        result=original_sdpa(q,k,v,**kwargs)
        np.array(result.transpose(0,2,1,3).astype(mx.float32)).tofile(args.out/(b+'.self_attn.sdpa.bin'))
        return result
    ref.scaled_dot_product_attention=capture_sdpa
layer.mlp.gate=Capture(layer.mlp.gate,f'router_{l}')
for attr,label in [('switch_mlp','expert_out'),('shared_expert','shared'),('shared_expert_gate','gate')]:
    setattr(layer.mlp,attr,Capture(getattr(layer.mlp,attr),b+'.mlp.'+label))
layer.mlp=Capture(layer.mlp,f'moe_{l}')
if l==1:
    for name in ['key_proj','value_proj','norm_key','norm_query','norm_conv']:
        setattr(layer.ple,name,Capture(getattr(layer.ple,name),'ple.'+name))
    layer.ple.conv1d=Capture(layer.ple.conv1d,'ple.conv',post=nn.silu)
    layer.ple=Capture(layer.ple,'ple')
h=mx.array(np.fromfile(args.input,np.float32).reshape(1,T,10240)).astype(mx.bfloat16)
h=layer(h,ref.RotaryEmbedding(64,cfg.rope_theta),'causal',None,None,None,mx.array([tokens]),mx.full((1,2),248044,mx.int32))
np.array(h.astype(mx.float32)).tofile(args.out/f'layer_{l}.bin')
print(f'Captured original layer {l} stages in {args.out}')
