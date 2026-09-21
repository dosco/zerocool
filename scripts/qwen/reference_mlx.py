#!/usr/bin/env python3
"""Run the checkpoint's original MLX layer code with bounded diagnostic reads.

Requires mlx==0.31.1 and mlx-lm==0.31.1. Each original decoder layer executes
unchanged; adapters only fetch selected expert and ngram weights from disk.
This is deliberately slow reference tooling, never a production backend.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from reference_numpy import Weights, ngram
from verify_checkpoint import ROOT, fingerprint, verify


def array(x):
    return mx.array(x).astype(mx.bfloat16) if x.dtype==np.float32 else mx.array(x)


class Reader:
    def __init__(self,w,config): self.w=w;self.config=config
    def format(self,b):
        quant=self.config['quantization'];value=quant.get(b,quant)
        bits,group=value['bits'],value['group_size']
        if bits not in (4,8) or group not in (32,64):
            raise ValueError('The oracle only supports the pinned affine Q4/Q8 formats')
        return dict(bits=bits,group_size=group)
    def linear(self,b,x,row=None):
        return mx.quantized_matmul(x,array(self.w.get(b+'.weight',row)),array(self.w.get(b+'.scales',row)),
            array(self.w.get(b+'.biases',row)),transpose=True,**self.format(b))
    def embedding(self,b,ids,group=64):
        fmt=self.format(b)
        if fmt['group_size']!=group: raise ValueError('Embedding group differs from the pinned format')
        return mx.stack([mx.dequantize(array(self.w.get(b+'.weight',int(i)))[None],
            array(self.w.get(b+'.scales',int(i)))[None],array(self.w.get(b+'.biases',int(i)))[None],
            **fmt)[0] for i in ids])


class Experts(nn.Module):
    def __init__(self,r,prefix): super().__init__();self._reader=r;self._prefix=prefix
    def __call__(self,x,indices):
        ids=np.array(indices);result=np.empty((*ids.shape,2560),np.float32)
        r=self._reader;b=self._prefix
        for e in np.unique(ids):
            batch,token,rank=np.nonzero(ids==e)
            selected=x[mx.array(batch),mx.array(token)]
            gate=nn.silu(r.linear(b+'.gate_proj',selected,int(e)));up=r.linear(b+'.up_proj',selected,int(e))
            down=r.linear(b+'.down_proj',gate*up,int(e))
            result[batch,token,rank]=np.array(down.astype(mx.float32))
        return array(result)


class Ngrams(nn.Module):
    def __init__(self,r,ids):
        super().__init__();self._reader=r
        _,self._rows=ngram(r.w,ids)
    def __call__(self,ids,prev):
        b='model.layers.1.ple.ple_embedding.ngram_embedding.shard_'
        per=self._reader.w.tensors[b+'0.weight'][2]['shape'][0]
        out=[]
        for rows in self._rows:
            out.append(mx.concatenate([self._reader.embedding(b+str(i//per),[i%per],32)[0] for i in rows]))
        return mx.stack(out)[None]


def prepare(module,prefix,r):
    nn.quantize(module,class_predicate=lambda path,m:r.format(prefix+'.'+path)
        if hasattr(m,'to_quantized') and prefix+'.'+path+'.scales' in r.w.tensors else False)
    weights=[]
    for name in r.w.tensors:
        if name.startswith(prefix+'.') and '.switch_mlp.' not in name and '.ple_embedding.' not in name:
            weights.append((name[len(prefix)+1:],array(r.w.get(name))))
    module.load_weights(weights,strict=False);module.eval()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,required=True);ap.add_argument('--tokens',type=Path,required=True)
    ap.add_argument('--lock',type=Path,default=ROOT/'models.lock.json')
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--trace',type=Path,required=True)
    ap.add_argument('--layers',type=int,default=48)
    args=ap.parse_args();start=time.monotonic()
    from importlib.metadata import version
    if mx.__version__!='0.31.1' or version('mlx-lm')!='0.31.1':
        raise RuntimeError('The oracle requires mlx==0.31.1 and mlx-lm==0.31.1')
    if not 1<=args.layers<=48: raise ValueError('layers must be 1..48')
    lock=json.loads(args.lock.read_text())
    verify(args.model,lock,cached=True)
    source_state={f['path']:fingerprint(args.model/f['path']) for f in lock['files'] if not f.get('optional')}
    tokens=json.loads(args.tokens.read_text())
    if not 0<len(tokens)<=32: raise ValueError('Reference requires 1..32 tokens')
    mx.set_cache_limit(64*1024**2)
    mx.set_memory_limit(768*1024**2)
    spec=importlib.util.spec_from_file_location('zerocool_pinned_qwen_reference',args.model/'qwen4_exp.py')
    ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)
    config=json.loads((args.model/'config.json').read_text())
    cfg=ref.ModelArgs.from_dict(config).text
    r=Reader(Weights(args.model),config);ids=mx.array([tokens]);prev=mx.full((1,2),248044,mx.int32)
    rope=ref.RotaryEmbedding(64,cfg.rope_theta)
    h=mx.tile(r.embedding('model.embed_tokens',tokens)[None],(1,1,4));mx.eval(h)
    args.trace.mkdir(parents=True,exist_ok=True)
    for l in range(args.layers):
        layer=ref.DecoderLayer(cfg,l);b=f'model.layers.{l}'
        layer.mlp.switch_mlp=Experts(r,b+'.mlp.switch_mlp')
        if l==1: layer.ple.ple_embedding=Ngrams(r,tokens)
        prepare(layer,b,r)
        h=layer(h,rope,'causal',None,None,None,ids,prev);mx.eval(h)
        np.array(h.astype(mx.float32)).tofile(args.trace/f'layer_{l}.bin')
        print(f'Original MLX layer {l+1}/{args.layers}',flush=True)
        del layer;mx.clear_cache()
    result=dict(kind='checkpoint_original_mlx_layers',layers=args.layers,tokens=tokens,mlx=mx.__version__,
        source_repo=lock['repo'],source_revision=lock['revision'],
        file_lock_sha256=hashlib.sha256(args.lock.read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reader_sha256=hashlib.sha256((ROOT/'scripts/qwen/reference_numpy.py').read_bytes()).hexdigest())
    if args.layers==48:
        mixer=ref.GatedResidual(cfg,use_combine=False);prepare(mixer,'model.hyper_connection_mixer',r)
        last=mixer(h)[:,-1,:];mx.eval(last)
        logits=[]
        for first in range(0,248320,256):
            logits.append(np.array(r.linear('lm_head',last,slice(first,min(first+256,248320))).astype(mx.float32)))
        values=np.concatenate(logits,axis=-1).reshape(-1);values.tofile(args.trace/'logits.f32')
        result.update(greedy_id=int(values.argmax()),logits=str(args.trace/'logits.f32'))
    result['elapsed_seconds']=time.monotonic()-start
    result['reference_peak_metal_bytes']=mx.get_peak_memory()
    if source_state!={f['path']:fingerprint(args.model/f['path']) for f in lock['files'] if not f.get('optional')}:
        raise ValueError('Source changed during reference computation')
    args.out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
