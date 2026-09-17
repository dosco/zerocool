#!/usr/bin/env python3
"""Independent CPU equations for small real-weight layer probes.

This reads only the selected expert/embedding rows and dequantizes one matrix
at a time. It does not import the native runtime or its Metal kernels. Equations
follow the pinned checkpoint's qwen4_exp.py; see THIRD_PARTY_NOTICES.md.
It is an operator oracle, not a production backend or a quality evaluation.
"""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import numpy as np


def bf(x):
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    return ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def sigmoid(x):
    x=np.asarray(x,np.float32)
    y=bf(1/bf(1+bf(np.exp(np.abs(x)))))
    return np.where(x<0,y,bf(1-y))


def softplus(x):
    return bf(np.maximum(x,0)+bf(np.log1p(bf(np.exp(-np.abs(x))))))


def silu(x):
    return bf(x * sigmoid(x))


class Weights:
    def __init__(self, root):
        self.tensors = {}
        self.files = []
        index = json.loads((root / 'model.safetensors.index.json').read_text())
        for name in sorted(set(index['weight_map'].values())):
            f = open(root / name, 'rb', buffering=0)
            self.files.append(f)
            size, = struct.unpack('<Q', f.read(8))
            header = json.loads(f.read(size))
            for k, v in header.items():
                if k != '__metadata__':
                    self.tensors[k.removeprefix('language_model.')] = (f, size+8, v)

    def get(self, name, row=None):
        f, start, v = self.tensors[name]
        shape = tuple(v['shape'])
        lo, hi = v['data_offsets']
        if row is not None:
            stride = (hi-lo)//shape[0]
            if isinstance(row,slice):
                lo+=row.start*stride
                hi=lo+(row.stop-row.start)*stride
                shape=(row.stop-row.start,)+shape[1:]
            else:
                lo += row*stride
                hi = lo+stride
                shape = shape[1:]
        dtype = {'BF16':'<u2','U32':'<u4','I64':'<i8','F32':'<f4'}[v['dtype']]
        a = np.frombuffer(os.pread(f.fileno(), hi-lo, start+lo), dtype=dtype).reshape(shape)
        return (a.astype(np.uint32)<<16).view(np.float32) if v['dtype']=='BF16' else a

    def matrix(self, name, row=None, group=64):
        w = self.get(name+'.weight', row)
        if name+'.scales' not in self.tensors:
            return w
        s, b = self.get(name+'.scales', row), self.get(name+'.biases', row)
        q = ((w[...,None] >> np.arange(0,32,4,dtype=np.uint32)) & 15).reshape(*w.shape[:-1],w.shape[-1]*8)
        return q.astype(np.float32)*np.repeat(s,group,-1)+np.repeat(b,group,-1)

    def linear(self, name, x, row=None, floating=False):
        out = x @ self.matrix(name,row).T
        if name+'.scales' in self.tensors:
            # Independent expression for the bias correction imposed by MLX
            # GEMV's BF16 sum-of-four input arithmetic. Quantized codes/scales
            # are unchanged; this rounding is part of the selected arithmetic.
            groups=x.reshape(*x.shape[:-1],-1,4)
            rounded=bf(bf(bf(groups[...,0]+groups[...,1])+groups[...,2])+groups[...,3])
            error=(rounded-groups.sum(-1)).reshape(*x.shape[:-1],-1,16).sum(-1)
            out+=error@self.get(name+'.biases',row).T
        return out if floating else bf(out)


def norm(w, name, x, grouped=False):
    shape=x.shape
    z=x.reshape(-1,2560) if grouped else x
    y=z/np.sqrt(np.mean(z*z,axis=-1,keepdims=True)+1e-6)
    y=bf(y).reshape(shape)
    return bf(y*w.get(name+'.weight'))


def hyper(w, name, x):
    n=norm(w,name+'.hc_norm',x,True)
    low=silu(bf(w.linear(name+'.input_mix_weight_down',n)/4))
    gate=bf(sigmoid(w.linear(name+'.input_mix_weight_up',low)))
    parts=bf(n*gate).reshape(-1,4,2560)
    mixed=bf(bf(bf(parts[:,0]+parts[:,1])+parts[:,2])+parts[:,3])/4
    injection=bf(2*bf(sigmoid(bf(w.linear(name+'.block_inject_weight',n)/4))))
    return mixed,injection


def conv(w, name, x, dilation=1):
    weights=w.get(name+'.weight').reshape(x.shape[-1],4)
    full=np.pad(x,((3*dilation,0),(0,0)))
    out=np.zeros_like(x)
    for k in range(4): out+=full[k*dilation:k*dilation+len(x)]*weights[:,k]
    return silu(bf(out))


def gdn(w,b,x):
    c=conv(w,b+'.conv1d',w.linear(b+'.in_proj_qkv',x))
    q,k,v=np.split(c,[2048,4096],axis=-1)
    q,k=q.reshape(-1,16,128),k.reshape(-1,16,128)
    q=bf(bf(q/np.sqrt((q*q).sum(-1,keepdims=True)+1e-6))*bf(1/np.sqrt(128)))
    k=bf(k/np.sqrt((k*k).sum(-1,keepdims=True)+1e-6))
    q,k=np.repeat(q,3,axis=1),np.repeat(k,3,axis=1)
    v=v.reshape(-1,48,128)
    a=bf(w.linear(b+'.in_proj_a',x)+w.get(b+'.dt_bias'))
    decay=bf(np.exp(-np.exp(w.get(b+'.A_log'))*softplus(a)))
    beta=bf(sigmoid(w.linear(b+'.in_proj_b',x)))
    state=np.zeros((48,128,128),np.float32)
    out=np.empty_like(v)
    for t in range(len(x)):
        state*=decay[t,:,None,None]
        delta=(v[t]-np.einsum('hvk,hk->hv',state,k[t]))*beta[t,:,None]
        state+=delta[:,:,None]*k[t,:,None,:]
        out[t]=bf(np.einsum('hvk,hk->hv',state,q[t]))
    out=norm(w,b+'.norm',out)
    z=w.linear(b+'.in_proj_z',x).reshape(out.shape)
    return w.linear(b+'.out_proj',bf(out*(1/(1+np.exp(-z)))).reshape(len(x),-1))


def rope(x, positions):
    a=np.asarray(positions,dtype=np.float32)[:,None]*np.power(np.float32(1e7),-np.arange(32,dtype=np.float32)/32)
    a=np.concatenate([a,a],axis=-1)
    c,s=bf(np.cos(a)),bf(np.sin(a))
    while c.ndim<x.ndim: c,s=c[:,None],s[:,None]
    part=x[...,:64]
    rotated=np.concatenate([-part[...,32:],part[...,:32]],axis=-1)
    return np.concatenate([bf(bf(part*c)+bf(rotated*s)),x[...,64:]],axis=-1)


def attention(w,b,x):
    T=len(x)
    qg=w.linear(b+'.q_proj',x).reshape(T,24,512)
    q=rope(norm(w,b+'.q_norm',qg[...,:256]),range(T))
    k=rope(norm(w,b+'.k_norm',w.linear(b+'.k_proj',x).reshape(T,2,256)),range(T))
    v=w.linear(b+'.v_proj',x).reshape(T,2,256)
    out=np.empty_like(q)
    for t in range(T):
        for h in range(24):
            score=bf((k[:t+1,h//12]@q[t,h])/16)
            score=np.exp(score-score.max());score=bf(score/score.sum())
            out[t,h]=bf(score@v[:t+1,h//12])
    return w.linear(b+'.o_proj',bf(out*bf(sigmoid(qg[...,256:]))).reshape(T,-1))


def mlp(w,b,x):
    return w.linear(b+'.down_proj',bf(silu(w.linear(b+'.gate_proj',x))*w.linear(b+'.up_proj',x)))


def moe(w,b,x):
    logits=w.linear(b+'.gate',x,floating=True)
    idx=np.argsort(-logits,axis=-1,kind='stable')[:,:10]
    selected=np.take_along_axis(logits,idx,-1)
    probs=np.exp(selected-selected.max(-1,keepdims=True));probs/=probs.sum(-1,keepdims=True)
    results=np.empty((len(x),10,2560),np.float32)
    for e in np.unique(idx):
        t,k=np.nonzero(idx==e)
        gate=silu(w.linear(b+'.switch_mlp.gate_proj',x[t],int(e)))
        up=w.linear(b+'.switch_mlp.up_proj',x[t],int(e))
        results[t,k]=w.linear(b+'.switch_mlp.down_proj',bf(gate*up),int(e))
    out=bf((results*probs[...,None]).sum(axis=1))
    return bf(out+bf(bf(sigmoid(w.linear(b+'.shared_expert_gate',x)))*mlp(w,b+'.shared_expert',x))),idx


def ngram(w,ids):
    b='model.layers.1.ple.ple_embedding.'
    mult=w.get(b+'layer_multipliers');sizes=w.get(b+'ngram_heads_vocab_sizes');offsets=w.get(b+'ngram_heads_offsets')
    history=[248044,248044]+list(ids)
    rows=[]
    for t in range(2,len(history)):
        vals=[history[t],history[t-1],248044 if history[t-1]==248044 else history[t-2]]
        mixed=vals[0]*int(mult[0]); row=[]
        for n in (1,2):
            mixed^=vals[n]*int(mult[n]); signed=(mixed+(1<<63))%(1<<64)-(1<<63)
            row.extend(signed%int(sizes[h])+int(offsets[h]) for h in range((n-1)*8,n*8))
        rows.append(row)
    per=w.tensors[b+'ngram_embedding.shard_0.weight'][2]['shape'][0]
    emb=[]
    for row in rows:
        emb.append(np.concatenate([bf(w.matrix(b+'ngram_embedding.shard_'+str(g//per),g%per,32)) for g in row]))
    return np.array(emb),rows


def ple(w,x,emb):
    b='model.layers.1.ple'
    key=norm(w,b+'.norm_key',w.linear(b+'.key_proj',emb),True).reshape(-1,4,2560)
    query=norm(w,b+'.norm_query',x,True).reshape(-1,4,2560)
    value=w.linear(b+'.value_proj',emb)
    gate=bf(bf(bf(key*query).sum(-1,keepdims=True))/math.sqrt(2560))
    gate=bf(bf(np.sqrt(np.maximum(abs(gate),1e-6)))*np.sign(gate))
    gated=bf(bf(sigmoid(gate))*value[:,None,:]).reshape(len(x),-1)
    return bf(gated+conv(w,b+'.conv1d',norm(w,b+'.norm_conv',gated,True),3))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--trace',type=Path)
    ap.add_argument('--report',type=Path,help='Native truncated probe JSON')
    ap.add_argument('--full-tokens',type=Path,help='Independently compute all 48 layers for these JSON token IDs')
    ap.add_argument('--logits',type=Path,help='Independent full-model FP32 logit output')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.full_tokens:
        import time
        start=time.monotonic();tokens=json.loads(args.full_tokens.read_text());w=Weights(args.model)
        if not 0<len(tokens)<=32: raise ValueError('The independent CPU full-model oracle supports 1..32 tokens')
        if not args.logits: raise ValueError('--full-tokens requires --logits')
        h=np.tile(np.array([bf(w.matrix('model.embed_tokens',t)) for t in tokens]),(1,4))
        ng,_=ngram(w,tokens)
        for l in range(48):
            b=f'model.layers.{l}'
            if l==1: h=bf(h+ple(w,h,ng))
            x,inj=hyper(w,b+'.attn_hyper_connection',h)
            att=gdn(w,b+'.linear_attn',x) if (l+1)%4 else attention(w,b+'.self_attn',x)
            h=bf(h+bf(att[:,None,:]*inj[:,:,None]).reshape(len(tokens),10240))
            x,inj=hyper(w,b+'.mlp_hyper_connection',h);out,_=moe(w,b+'.mlp',x)
            h=bf(h+bf(out[:,None,:]*inj[:,:,None]).reshape(len(tokens),10240))
            print(f'Independent CPU layer {l+1}/48',flush=True)
            if args.trace:
                args.trace.mkdir(parents=True,exist_ok=True);h.tofile(args.trace/f'layer_{l}.bin')
        b='model.hyper_connection_mixer';n=norm(w,b+'.hc_norm',h[-1:],True)
        low=silu(bf(w.linear(b+'.input_mix_weight_down',n)/4));gate=bf(sigmoid(w.linear(b+'.input_mix_weight_up',low)))
        parts=bf(n*gate).reshape(-1,4,2560)
        mixed=bf(bf(bf(parts[:,0]+parts[:,1])+parts[:,2])+parts[:,3])/4
        logits=[]
        for first in range(0,248320,256): logits.append(w.linear('lm_head',mixed,slice(first,min(first+256,248320))))
        logits=np.concatenate(logits,axis=-1).reshape(-1);logits.tofile(args.logits)
        args.out.write_text(json.dumps(dict(kind='independent_numpy_full_model',tokens=tokens,layers=48,
            greedy_id=int(logits.argmax()),elapsed_seconds=time.monotonic()-start,logits=str(args.logits)),indent=2)+'\n')
        return
    if not args.trace or not args.report: raise ValueError('Operator comparison requires --trace and --report')
    run=json.loads(args.report.read_text());tokens=run['tokens'];T=len(tokens)
    w=Weights(args.model);checks=[]
    def read_trace(name,dtype,width):
        files=sorted(args.trace.glob('step_*'),key=lambda p:int(p.name[5:]))
        if not files: files=[args.trace]
        return np.concatenate([np.fromfile(p/(name+'.bin'),dtype).reshape(-1,width) for p in files])
    def actual(name,width=2560): return read_trace(name,np.float32,width)
    def check(name,expected):
        got=actual(name,expected.size//T).reshape(expected.shape)
        rel=float(np.linalg.norm(got-expected)/max(np.linalg.norm(expected),1e-9))
        cosine=float(np.vdot(got,expected)/max(np.linalg.norm(got)*np.linalg.norm(expected),1e-9))
        row=dict(tensor=name,relative_l2=rel,cosine=cosine,finite=bool(np.isfinite(got).all()),passed=rel<0.01 and cosine>0.99995)
        checks.append(row); print(json.dumps(row),flush=True)
    emb=np.array([bf(w.matrix('model.embed_tokens',t)) for t in tokens])
    check('input_0',np.tile(emb,(1,4)))
    ng=None
    for l in range(run['layers']):
        b=f'model.layers.{l}';h=actual(f'input_{l}',10240)
        if l==1:
            ng,rows=ngram(w,tokens);check('ngram',ng)
            h=bf(h+ple(w,h,ng))
        x,inj=hyper(w,b+'.attn_hyper_connection',h);check(f'x1_{l}',x)
        # Each operator receives the independently recorded native input, so
        # an upstream rounding difference is not mistaken for a second bug.
        att=gdn(w,b+'.linear_attn',actual(f'x1_{l}')) if (l+1)%4 else attention(w,b+'.self_attn',actual(f'x1_{l}'))
        check(f'attn_{l}',att)
        after=bf(h+bf(actual(f'attn_{l}')[:,None,:]*inj[:,:,None]).reshape(T,10240))
        x2,inj2=hyper(w,b+'.mlp_hyper_connection',after);check(f'x2_{l}',x2)
        out,idx=moe(w,b+'.mlp',actual(f'x2_{l}'));check(f'moe_{l}',out)
        native_ids=read_trace(f'route_{l}',np.int32,10)
        checks.append(dict(tensor=f'route_{l}',passed=bool(np.array_equal(native_ids,idx))))
        check(f'layer_{l}',bf(after+bf(actual(f'moe_{l}')[:,None,:]*inj2[:,:,None]).reshape(T,10240)))
    report=dict(kind='independent_numpy_real_weight_operators',tokens=tokens,layers=run['layers'],
                build_fingerprint=run['statistics']['metal']['build_fingerprint'],
                tolerance=dict(relative_l2_max=0.01,cosine_min=0.99995),checks=checks,
                passed=all(c['passed'] for c in checks),full_model_verified=False)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__': main()
