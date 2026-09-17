#!/usr/bin/env python3
"""Bounded independent NumPy equations for the prepared MTP forward fixture.

No native library, Metal kernel or MLX is imported. Matrices are decoded in
128-row blocks. Original zero-centred norms are shifted exactly once. This
checks the draft's chosen BF16/affine arithmetic, not original BF16 quality.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np

from reference_numpy import Weights, bf, sigmoid, silu, rope
from prepare_mtp import decode_reference
from qualification_evidence import save, sha


def dot(x,matrix,bias=None,bits=8,floating=False):
    y=x@matrix.T
    if bits==4:
        groups=x.reshape(len(x),-1,4)
        rounded=bf(bf(bf(groups[...,0]+groups[...,1])+groups[...,2])+groups[...,3])
        error=(rounded-groups.sum(-1)).reshape(len(x),-1,16).sum(-1)
        y+=error@bias.T
    return y if floating else bf(y)


class Oracle:
    def __init__(self,prepared,model):
        self.manifest=json.loads((prepared/'manifest.json').read_text())
        self.dense=(prepared/'dense.bin').open('rb',buffering=0)
        self.experts=(prepared/'experts.bin').open('rb',buffering=0)
        self.shared=Weights(model)
        self.quant=json.loads((model/'config.json').read_text())['quantization']
    def part(self,t,name):
        p=t['parts'][name];raw=os.pread(self.dense.fileno(),p['bytes'],p['offset'])
        if len(raw)!=p['bytes']:raise ValueError('truncated MTP tensor')
        return raw
    def norm_weight(self,name):
        t=self.manifest['tensors'][name+'.weight']
        if t['norm_convention']!='zero-centered-original; apply 1+w once':raise ValueError('wrong norm convention')
        return bf(1+(np.frombuffer(self.part(t,'weight'),'<u2').astype('<u4')<<16).view('<f4'))
    def norm(self,x,name,group=None):
        shape=x.shape;z=x.reshape(-1,group or x.shape[-1])
        # FP64 reduction is independent of the SIMD partition used by Metal.
        inv=(1/np.sqrt(np.mean(z.astype(np.float64)**2,-1,keepdims=True)+1e-6)).astype(np.float32)
        return bf(bf(z*inv).reshape(shape)*self.norm_weight(name))
    def linear(self,name,x,expert=None,projection=None,floating=False):
        if expert is not None:
            cols,rows=(640,2560) if projection==2 else (2560,640);bits=4
            base=expert*2768896+projection*921600
            parts=[os.pread(self.experts.fileno(),size,base+offset) for offset,size in ((0,819200),(819200,51200),(870400,51200))]
        else:
            t=self.manifest['tensors'][name+'.weight'];rows,cols=t['shape'];bits=8
            if t['format']=='BF16':
                m=(np.frombuffer(self.part(t,'weight'),'<u2').astype('<u4')<<16).view('<f4').reshape(rows,cols)
                return dot(x,m,floating=floating)
            parts=[self.part(t,p) for p in ('weight','scales','biases')]
        out=[]
        for at in range(0,rows,128):
            n=min(128,rows-at);sizes=[cols*bits//8,cols//64*2,cols//64*2]
            chunk=[p[at*s:(at+n)*s] for p,s in zip(parts,sizes)]
            matrix=decode_reference(chunk,n,cols,bits)
            bias=(np.frombuffer(chunk[2],'<u2').astype('<u4')<<16).view('<f4').reshape(n,-1)
            out.append(dot(x,matrix,bias,bits,floating))
        return np.concatenate(out,-1)
    def shared_linear(self,name,x):
        fmt=self.quant.get(name,self.quant);bits=fmt['bits'];rows=self.shared.tensors[name+'.weight'][2]['shape'][0]
        result=[]
        for at in range(0,rows,128):
            section=slice(at,min(at+128,rows));w=self.shared.get(name+'.weight',section)
            s=self.shared.get(name+'.scales',section);b=self.shared.get(name+'.biases',section)
            codes=((w[...,None]>>np.arange(0,32,bits,dtype=np.uint32))&((1<<bits)-1)).reshape(len(w),-1).astype(np.float32)
            result.append(dot(x,codes*s.repeat(64,-1)+b.repeat(64,-1),b,bits))
        return np.concatenate(result,-1)
    def embedding(self,ids):
        name='model.embed_tokens';bits=self.quant.get(name,self.quant)['bits'];out=[]
        for i in ids:
            w=self.shared.get(name+'.weight',i);s=self.shared.get(name+'.scales',i);b=self.shared.get(name+'.biases',i)
            codes=((w[...,None]>>np.arange(0,32,bits,dtype=np.uint32))&((1<<bits)-1)).reshape(-1).astype(np.float32)
            out.append(bf(codes*s.repeat(64)+b.repeat(64)))
        return np.array(out)
    def hyper(self,x,base,inject=True):
        n=self.norm(x,base+'.hc_norm',2560)
        low=silu(bf(self.linear(base+'.input_mix_weight_down',n)/4))
        gates=sigmoid(self.linear(base+'.input_mix_weight_up',low))
        parts=bf(n*gates).reshape(len(x),4,2560)
        mixed=bf(bf(bf(parts[:,0]+parts[:,1])+parts[:,2])+parts[:,3])/4
        injection=bf(2*sigmoid(bf(self.linear(base+'.block_inject_weight',n)/4))) if inject else None
        return mixed,injection
    def mlp(self,x,base,expert=None):
        if expert is None:
            return self.linear(base+'.down_proj',bf(silu(self.linear(base+'.gate_proj',x))*self.linear(base+'.up_proj',x)))
        return self.linear('',bf(silu(self.linear('',x,expert,0))*self.linear('',x,expert,1)),expert,2)


def forward(w,ids,h):
    out={};T=len(ids)
    def record(name,value):out[name]=value;return value
    embedding=record('embedding',w.embedding(ids))
    en=record('embedding_norm',w.norm(embedding,'mtp.pre_fc_norm_embedding'))
    hn=record('hidden_norm',w.norm(h,'mtp.pre_fc_norm_hidden'))
    e=w.linear('mtp.fc_embedding',en)
    h=record('fused',bf(w.linear('mtp.fc_hidden',hn.reshape(-1,2560)).reshape(T,4,2560)+e[:,None]).reshape(T,10240))
    x,inj=w.hyper(h,'mtp.layers.0.attn_hyper_connection');record('attention_input',x)
    b='mtp.layers.0.self_attn';qg=w.linear(b+'.q_proj',x).reshape(T,24,512)
    q=record('q',rope(w.norm(qg[...,:256],b+'.q_norm'),range(T)))
    k=record('k',rope(w.norm(w.linear(b+'.k_proj',x).reshape(T,2,256),b+'.k_norm'),range(T)))
    v=record('v',w.linear(b+'.v_proj',x).reshape(T,2,256));a=np.empty_like(q)
    for t in range(T):
        for head in range(24):
            score=bf((k[:t+1,head//12]@q[t,head])/16)
            prob=np.exp(score-score.max());prob=bf(prob/prob.sum())
            a[t,head]=bf(prob@v[:t+1,head//12])
    a=record('attention_gated',bf(a*sigmoid(qg[...,256:])).reshape(T,-1))
    a=record('attention',w.linear(b+'.o_proj',a))
    h=record('after_attention',bf(h.reshape(T,4,2560)+bf(a[:,None]*inj[...,None])).reshape(T,10240))
    x,inj=w.hyper(h,'mtp.layers.0.mlp_hyper_connection');record('moe_input',x);b='mtp.layers.0.mlp'
    router=record('router',w.linear(b+'.gate',x,floating=True))
    routes=record('routes',np.argsort(-router,axis=-1,kind='stable')[:,:10].astype('<i4'))
    logits=np.take_along_axis(router,routes,-1);probs=np.exp(logits-logits.max(-1,keepdims=True));probs/=probs.sum(-1,keepdims=True);record('route_weights',probs)
    experts=np.empty((T,10,2560),np.float32)
    for expert in np.unique(routes):
        rows,ranks=np.nonzero(routes==expert);experts[rows,ranks]=w.mlp(x[rows],b,int(expert))
    record('expert_out',experts);shared=record('shared',w.mlp(x,b+'.shared_expert'));gate=record('shared_gate',w.linear(b+'.shared_expert_gate',x))
    contributions=experts*probs[...,None];s=(contributions[:,0]+contributions[:,8])+(contributions[:,1]+contributions[:,9])
    for i in range(2,8):s+=contributions[:,i]
    m=record('moe',bf(bf(s)+bf(sigmoid(gate)*shared)))
    wide=record('wide',bf(h.reshape(T,4,2560)+bf(m[:,None]*inj[...,None])).reshape(T,10240))
    mixed=record('mixed',w.hyper(wide,'mtp.hyper_connection_mixer',False)[0]);record('logits',w.shared_linear('lm_head',mixed))
    return out


def check_reference(prepared,model,input_path,native_path,output):
    raw=json.loads(native_path.read_text());data=json.loads(input_path.read_text())
    if raw.get('complete') is not True or raw.get('mode')!='fixture':raise ValueError('incomplete native fixture')
    hidden=np.fromfile(data['hidden_file'],'<f4').reshape(4,10240)
    values=forward(Oracle(prepared,model),data['ids'],hidden);cases=[]
    for name,expected in values.items():
        file=native_path.parent/'trace'/(name+'.bin');entry=raw['fixture_tensors'][name]
        if file.stat().st_size!=entry['bytes'] or sha(file)!=entry['sha256']:raise ValueError('changed native fixture')
        actual=np.fromfile(file,'<i4' if name=='routes' else '<f4').reshape(expected.shape)
        exact=np.array_equal(actual,expected)
        relative=float(np.linalg.norm(actual.astype(np.float64)-expected)/max(1e-20,np.linalg.norm(expected)))
        maximum=float(np.max(np.abs(actual.astype(np.float64)-expected)))
        # Frozen end-to-end tolerance accommodates independent FP32 reductions
        # and BF16 boundaries. Routing and final greedy decisions must agree.
        passed=exact if name=='routes' else relative<=.02 and np.isfinite(actual).all()
        if name=='logits':passed=passed and np.array_equal(actual.argmax(-1),expected.argmax(-1))
        cases.append(dict(name=name,exact=exact,relative_l2=relative,max_abs=maximum,passed=bool(passed)))
    report=dict(kind='mtp_numpy_forward_reference_v1',complete=True,passed=all(c['passed'] for c in cases),cases=cases,
        native_sha256=sha(native_path),input_sha256=sha(input_path),oracle_sha256=sha(Path(__file__)),
        limits=dict(relative_l2=.02,exact_routes=True,exact_greedy=True),production_promoted=False)
    save(output,report)
    if not report['passed']:raise ValueError('independent MTP forward mismatch: '+str([c for c in cases if not c['passed']]))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('prepared','model','input','native','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();r=check_reference(a.prepared,a.model,a.input,a.native,a.output);print(json.dumps(dict(passed=r['passed'],cases=len(r['cases']))))
