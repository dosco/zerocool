#!/usr/bin/env python3
"""Prepare a draft-only affine Q4/Q8 sidecar from original BF16 MTP ranges.

CPU-only offline preparation. Main-model payloads remain unchanged. RMS weights
stay in their ORIGINAL zero-centred representation; the manifest requires the
future consumer to apply 1+w once. This is not a runnable draft model.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
from mtp_source import (ROOT, REVISION, EXPERT_GATE, EXPERT_DOWN, load_inventory,
                        read_json_bytes, require, sha, leaf, write_all)
from prepare_storage import ALIGN, EXPERT_BYTES, STRIDE, atomic_json, digest_file, nocache

RECIPE='mtp-affine-q4-experts-q8-dense-64-v1'
GROUP=64
BLOCK_ROWS=128
NORM_SUFFIXES=('hc_norm.weight','q_norm.weight','k_norm.weight','q_layernorm.weight',
               'k_layernorm.weight','pre_fc_norm_embedding.weight','pre_fc_norm_hidden.weight')


def rounded_bf16(x):
    x=np.asarray(x,dtype='<f4');u=x.view('<u4')
    require(np.isfinite(x).all(),'Nonfinite affine input')
    return ((u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)).view('<f4')


def bf16_bytes(x): return (rounded_bf16(x).view('<u4')>>16).astype('<u2').tobytes()


def read_bf16(stream, offset, shape):
    size=math.prod(shape)*2;raw=os.pread(stream.fileno(),size,offset)
    require(len(raw)==size,'Truncated original BF16 tensor')
    x=(np.frombuffer(raw,'<u2').astype('<u4')<<16).view('<f4').reshape(shape)
    require(np.isfinite(x).all(),'Nonfinite original BF16 tensor')
    return x


def quantize(matrix,bits):
    require(bits in (4,8) and matrix.ndim==2 and matrix.shape[1]%GROUP==0,'Unsupported affine geometry')
    w=np.asarray(matrix,dtype='<f4');require(np.isfinite(w).all(),'Nonfinite weights')
    groups=w.reshape(w.shape[0],-1,GROUP);lo=groups.min(-1);hi=groups.max(-1)
    scale=rounded_bf16((hi-lo)/np.float32((1<<bits)-1));bias=rounded_bf16(lo)
    require(np.isfinite(scale).all() and np.isfinite(bias).all(),'Nonfinite affine metadata')
    safe=np.where(scale==0,np.float32(1),scale)
    codes=np.clip(np.rint((groups-bias[...,None])/safe[...,None]),0,(1<<bits)-1).astype('<u4')
    codes=np.where(scale[...,None]==0,0,codes).astype('<u4').reshape(w.shape)
    per=32//bits;packed=np.zeros((w.shape[0],w.shape[1]//per),dtype='<u4')
    for lane in range(per):packed|=codes[:,lane::per]<<np.uint32(lane*bits)
    return packed.tobytes(),bf16_bytes(scale),bf16_bytes(bias)


def decode_reference(parts,rows,cols,bits):
    """Independent byte indexing decoder, also used to validate every row block."""
    codes=np.frombuffer(parts[0],np.uint8).reshape(rows,-1)
    if bits==4:
        values=np.empty((rows,cols),np.float32);values[:,0::2]=codes&15;values[:,1::2]=codes>>4
    else:values=codes.astype(np.float32)
    scales=(np.frombuffer(parts[1],'<u2').astype('<u4')<<16).view('<f4').reshape(rows,-1)
    biases=(np.frombuffer(parts[2],'<u2').astype('<u4')<<16).view('<f4').reshape(rows,-1)
    return values*scales.repeat(GROUP,axis=1)+biases.repeat(GROUP,axis=1)


def check_quantization(weights,parts,bits):
    restored=decode_reference(parts,*weights.shape,bits)
    groups=weights.reshape(weights.shape[0],-1,GROUP)
    scale=rounded_bf16((groups.max(-1)-groups.min(-1))/np.float32((1<<bits)-1))
    bias=rounded_bf16(groups.min(-1));maximum=bias+scale*np.float32((1<<bits)-1)
    # Round-to-nearest error plus rounded affine endpoints. This guards packing,
    # not quality: low reconstruction error cannot qualify draft acceptance.
    endpoints=np.maximum(np.abs(bias-groups.min(-1)),np.abs(maximum-groups.max(-1)))
    bound=(scale*.500001+endpoints+np.float32(1e-7)).repeat(GROUP,axis=1)
    error=np.abs(restored-weights);require(np.isfinite(restored).all() and np.all(error<=bound),'Affine reconstruction exceeds rounding bound')
    return dict(elements=weights.size,squared_error=float(np.sum((restored.astype(np.float64)-weights)**2)),
                squared_source=float(np.sum(weights.astype(np.float64)**2)),max_abs_error=float(error.max()))


def merge_error(total,row):
    for k in ('elements','squared_error','squared_source'):total[k]+=row[k]
    total['max_abs_error']=max(total['max_abs_error'],row['max_abs_error'])


def empty_error():return dict(elements=0,squared_error=0.,squared_source=0.,max_abs_error=0.)


def precision(name,tensor):
    if len(tensor['shape'])==1:return 'BF16'
    if name.endswith(('mlp.gate.weight','shared_expert_gate.weight','block_inject_weight.weight')):return 'BF16'
    require(len(tensor['shape'])==2 and tensor['shape'][1]%64==0,'Unknown MTP matrix geometry')
    return 'affine-Q8'


def allocation_plan(inventory,context=8192):
    require(type(context) is int and 0<context<=8192,'Invalid MTP context')
    dense=0;logical=0;counts={};entries={}
    for name,t in sorted(inventory['tensors'].items()):
        if name in (EXPERT_GATE,EXPERT_DOWN):continue
        fmt=precision(name,t);counts[fmt]=counts.get(fmt,0)+1
        sizes=[t['bytes']] if fmt=='BF16' else [math.prod(t['shape']),math.prod(t['shape'])//64*2,math.prod(t['shape'])//64*2]
        for size in sizes:dense=(dense+ALIGN-1)//ALIGN*ALIGN;dense+=size;logical+=size
        entries[name]=dict(format=fmt,shape=t['shape'],bytes=sum(sizes),
            norm_convention='zero-centered-original; apply 1+w once' if name.endswith(NORM_SUFFIXES) else None)
    dense=(dense+ALIGN-1)//ALIGN*ALIGN
    expert=512*STRIDE;attention=(2*512+128)*context*4
    # Explicit planning allowances, not measured native allocations.
    hidden=4*10240*4;logits=4*248320*4;checkpoint_tail=4*(2*512+128)*4
    scratch=64*1024**2;metadata=16*1024**2
    incremental=expert+dense+attention+hidden+logits+checkpoint_tail+scratch+metadata
    return dict(expert_count=512,expert_payload_bytes=EXPERT_BYTES,expert_stride=STRIDE,
        expert_allocated_bytes=expert,dense_allocated_bytes=dense,dense_logical_bytes=logical,
        prepared_bytes=expert+dense,dense_precision_counts=counts,tensors=entries,
        context=context,attention_state_bytes=attention,hidden_workspace_bytes=hidden,logits_workspace_bytes=logits,
        draft_checkpoint_tail_bytes=checkpoint_tail,scratch_allowance_bytes=scratch,metadata_allowance_bytes=metadata,
        shared_embedding_output_additional_bytes=0,incremental_budget_bytes=incremental,
        equivalent_target_cache_slots=math.ceil(incremental/STRIDE),native_allocation_measured=False)


def open_sources(source,inventory):
    receipt=read_json_bytes((source/'download.json').read_bytes())
    require(receipt.get('complete') is True and receipt.get('revision')==REVISION and
        receipt.get('inventory_sha256')==sha((source/'inventory.json').read_bytes()) and
        set(receipt['tensors'])==set(inventory['tensors']),'Incomplete or changed source extraction')
    paths={}
    for name,entry in receipt['tensors'].items():
        path=source/'tensors'/(name+'.bf16');require(entry['file']==str(path.relative_to(source)) and entry['source']==inventory['tensors'][name], 'Changed source binding')
        require(not path.is_symlink() and path.stat().st_size==entry['bytes']==inventory['tensors'][name]['bytes'] and digest_file(path)==entry['sha256'],'Source payload changed')
        paths[name]=path
    return paths,receipt


def validate_dense_layout(manifest, inventory):
    """Bind each tensor to its canonical location, not just any valid interval."""
    expected={};cursor=0
    for name,tensor in sorted(inventory['tensors'].items()):
        if name in (EXPERT_GATE,EXPERT_DOWN):continue
        fmt=precision(name,tensor);elements=math.prod(tensor['shape'])
        entry=dict(source=name,shape=tensor['shape'],format=fmt,parts={})
        if fmt=='BF16':
            parts=[('weight',elements*2,'BF16')]
            entry['norm_convention']='zero-centered-original; apply 1+w once' if name.endswith(NORM_SUFFIXES) else None
        else:
            parts=[('weight',elements,'U32'),('scales',elements//64*2,'BF16'),('biases',elements//64*2,'BF16')]
            entry.update(bits=8,group_size=64)
        for label,size,dtype in parts:
            cursor=(cursor+ALIGN-1)//ALIGN*ALIGN
            entry['parts'][label]=dict(offset=cursor,bytes=size,dtype=dtype)
            cursor+=size
        expected[name]=entry
    require(manifest['tensors']==expected,'Dense MTP tensor binding differs from canonical layout')
    require(manifest['files']['dense.bin']['bytes']==(cursor+ALIGN-1)//ALIGN*ALIGN,'Changed dense allocation')


def prepare(source,output):
    source,output=Path(source),Path(output);inventory=load_inventory(source);plan=allocation_plan(inventory)
    paths,receipt=open_sources(source,inventory)
    require(shutil.disk_usage(source).free>=plan['prepared_bytes']+2*1024**3,'Insufficient output disk headroom')
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(kind='qwen_mtp_prepared_v1',complete=False,recipe=RECIPE,source_revision=REVISION,
        source_inventory_sha256=sha((source/'inventory.json').read_bytes()),source_download_sha256=sha((source/'download.json').read_bytes()),
        producer_sha256=sha(Path(__file__).read_bytes()),source_lock_sha256=sha((ROOT/'mtp-models.lock.json').read_bytes()),source_tool_sha256=sha((ROOT/'scripts/qwen/mtp_source.py').read_bytes()),
        source_full_shard_hash_verified=False,shared_io=inventory['shared_io'],memory_plan=plan,files={},tensors={},
        experts={},quantization_error={},native_draft_implemented=False,acceptance_qualified=False,production_promoted=False)
    shutil.copyfile(source/'inventory.json',output/'source-inventory.json')
    shutil.copyfile(source/'download.json',output/'source-download.json')
    atomic_json(output/'manifest.json',manifest)
    dense_path=output/'dense.bin';dense_tmp=output/'dense.partial'
    with dense_tmp.open('w+b',buffering=0) as out:
        nocache(out.fileno())
        def reserve(size):
            offset=(out.seek(0,2)+ALIGN-1)//ALIGN*ALIGN;out.truncate(offset+size);return offset
        for name,t in sorted(inventory['tensors'].items()):
            if name in (EXPERT_GATE,EXPERT_DOWN):continue
            fmt=precision(name,t);entry=dict(source=name,shape=t['shape'],format=fmt,parts={})
            with paths[name].open('rb',buffering=0) as inp:
                nocache(inp.fileno())
                if fmt=='BF16':
                    value=inp.read();offset=reserve(len(value));out.seek(offset);write_all(out,value)
                    entry['parts']['weight']=dict(offset=offset,bytes=len(value),dtype='BF16')
                    entry['norm_convention']=plan['tensors'][name]['norm_convention']
                    require(sha(value)==receipt['tensors'][name]['sha256'],'BF16 payload changed')
                else:
                    rows,cols=t['shape'];entry.update(bits=8,group_size=64);stats=empty_error()
                    for label,size,dtype in [('weight',rows*cols,'U32'),('scales',rows*cols//64*2,'BF16'),('biases',rows*cols//64*2,'BF16')]:
                        entry['parts'][label]=dict(offset=reserve(size),bytes=size,dtype=dtype)
                    for row in range(0,rows,BLOCK_ROWS):
                        n=min(BLOCK_ROWS,rows-row);w=read_bf16(inp,row*cols*2,(n,cols));parts=quantize(w,8);merge_error(stats,check_quantization(w,parts,8))
                        for (label,part),raw in zip(entry['parts'].items(),parts):out.seek(part['offset']+row*(part['bytes']//rows));write_all(out,raw)
                    manifest['quantization_error'][name]=stats
                manifest['tensors'][name]=entry
            print('prepared '+name,flush=True)
        out.truncate(plan['dense_allocated_bytes']);out.flush();os.fsync(out.fileno())
    dense_tmp.replace(dense_path);manifest['files']['dense.bin']=dict(bytes=dense_path.stat().st_size,sha256=digest_file(dense_path))
    atomic_json(output/'manifest.json',manifest)
    experts_path=output/'experts.bin';stats=empty_error();records=[]
    with paths[EXPERT_GATE].open('rb',buffering=0) as gate,paths[EXPERT_DOWN].open('rb',buffering=0) as down,(output/'experts.partial').open('wb',buffering=0) as out:
        for stream in (gate,down,out):nocache(stream.fileno())
        for expert in range(512):
            layout=[];cursor=0;record=bytearray()
            for projection,stream,rows,cols,offset in (
                ('gate',gate,640,2560,expert*1280*2560*2),
                ('up',gate,640,2560,(expert*1280+640)*2560*2),
                ('down',down,2560,640,expert*2560*640*2)):
                w=read_bf16(stream,offset,(rows,cols));parts=quantize(w,4);merge_error(stats,check_quantization(w,parts,4))
                for label,raw in zip(('weight','scales','biases'),parts):
                    layout.append(dict(projection=projection,part=label,offset=cursor,bytes=len(raw),bits=4,group_size=64));cursor+=len(raw);record.extend(raw)
            require(cursor==EXPERT_BYTES,'Prepared expert incompatible with native record geometry')
            record.extend(bytes(STRIDE-len(record)));write_all(out,record);records.append(sha(record))
            if expert%32==0 or expert==511:print(f'prepared MTP experts {expert+1}/512',flush=True)
        out.flush();os.fsync(out.fileno())
    (output/'experts.partial').replace(experts_path)
    manifest['experts']=dict(file='experts.bin',count=512,stride=STRIDE,payload_bytes=EXPERT_BYTES,layout=layout,record_sha256=records)
    manifest['files']['experts.bin']=dict(bytes=experts_path.stat().st_size,sha256=digest_file(experts_path))
    manifest['quantization_error']['routed_experts']=stats
    require(sum(v['bytes'] for v in manifest['files'].values())==plan['prepared_bytes'],'Prepared allocation differs from plan')
    require(sha((source/'download.json').read_bytes())==manifest['source_download_sha256'],'Source receipt changed during preparation')
    manifest['complete']=True;atomic_json(output/'manifest.json',manifest);return manifest


def verify(output):
    output=Path(output);m=read_json_bytes((output/'manifest.json').read_bytes())
    require(m.get('kind')=='qwen_mtp_prepared_v1' and m.get('complete') is True and m.get('recipe')==RECIPE and
        m.get('source_revision')==REVISION and m.get('production_promoted') is False,'Invalid prepared MTP identity')
    inventory=read_json_bytes((output/'source-inventory.json').read_bytes())
    from mtp_source import validate_inventory
    validate_inventory(inventory)
    require(sha((output/'source-inventory.json').read_bytes())==m['source_inventory_sha256'] and
        sha((output/'source-download.json').read_bytes())==m['source_download_sha256'] and
        m['source_lock_sha256']==sha((ROOT/'mtp-models.lock.json').read_bytes()) and
        m['memory_plan']==allocation_plan(inventory) and m['shared_io']==inventory['shared_io'], 'Changed preparation source or memory identity')
    receipt=read_json_bytes((output/'source-download.json').read_bytes())
    require(receipt['complete'] is True and receipt['revision']==REVISION and set(receipt['tensors'])==set(inventory['tensors']), 'Incomplete saved source receipt')
    require(set(m['tensors'])==set(inventory['tensors'])-{EXPERT_GATE,EXPERT_DOWN},'Changed MTP tensor names')
    require(set(m['files'])=={'dense.bin','experts.bin'} and len(m['tensors'])==29 and len(m['experts']['record_sha256'])==512,'Incomplete prepared MTP')
    validate_dense_layout(m,inventory)
    for name,entry in m['files'].items():
        path=output/leaf(name);require(not path.is_symlink() and path.stat().st_size==entry['bytes'] and digest_file(path)==entry['sha256'],'Prepared MTP payload changed')
    require(m['files']['experts.bin']['bytes']==512*STRIDE and m['experts']['stride']==STRIDE and m['experts']['payload_bytes']==EXPERT_BYTES,'Changed expert geometry')
    intervals=[]
    for name,tensor in m['tensors'].items():
        source_tensor=inventory['tensors'][name]
        require(tensor['source']==name and tensor['shape']==source_tensor['shape'] and
            tensor['format']==precision(name,source_tensor),'Changed MTP tensor format or shape')
        if tensor['format']=='affine-Q8': require(tensor['bits']==8 and tensor['group_size']==64,'Changed affine recipe')
        else: require(tensor['norm_convention']==m['memory_plan']['tensors'][name]['norm_convention'],'Changed RMS convention')
        n=math.prod(tensor['shape'])
        sizes={'weight':n*2} if tensor['format']=='BF16' else {'weight':n,'scales':n//64*2,'biases':n//64*2}
        require(set(tensor['parts'])==set(sizes),'Invalid tensor parts')
        for name,entry in tensor['parts'].items():
            begin=entry['offset'];end=begin+entry['bytes']
            require(type(begin) is int and begin>=0 and begin%ALIGN==0 and entry['bytes']==sizes[name] and end<=m['files']['dense.bin']['bytes'],'Invalid dense interval')
            intervals.append((begin,end))
    require(m['files']['dense.bin']['bytes']==m['memory_plan']['dense_allocated_bytes'] and
        m['experts']['file']=='experts.bin' and m['experts']['count']==512,'Changed file allocation')
    expected_layout=[]
    for projection_index,projection in enumerate(('gate','up','down')):
        for part,offset,size in [('weight',0,819200),('scales',819200,51200),('biases',870400,51200)]:
            expected_layout.append(dict(projection=projection,part=part,offset=projection_index*921600+offset,bytes=size,bits=4,group_size=64))
    require(m['experts']['layout']==expected_layout,'Expert record layout differs from native parser')
    with (output/'dense.bin').open('rb',buffering=0) as stream:
        nocache(stream.fileno())
        for name,tensor in m['tensors'].items():
            if tensor['format']=='BF16':
                part=tensor['parts']['weight'];raw=os.pread(stream.fileno(),part['bytes'],part['offset'])
                require(sha(raw)==receipt['tensors'][name]['sha256'],'BF16 source weights or norms were altered')
    ordered=sorted(intervals);require(all(a[1]<=b[0] for a,b in zip(ordered,ordered[1:])),'Overlapping dense tensors')
    with (output/'experts.bin').open('rb',buffering=0) as stream:
        nocache(stream.fileno())
        for expected in m['experts']['record_sha256']:
            record=stream.read(STRIDE);require(len(record)==STRIDE and sha(record)==expected and not any(record[EXPERT_BYTES:]),'Changed expert record or padding')
    return dict(kind='qwen_mtp_preparation_audit_v1',passed=True,manifest_sha256=sha((output/'manifest.json').read_bytes()),
        prepared_bytes=sum(v['bytes'] for v in m['files'].values()),expert_records=512,native_draft_implemented=False,
        acceptance_qualified=False,production_promoted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('plan','prepare','verify'));p.add_argument('--source',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.action=='plan':result=allocation_plan(load_inventory(a.source));atomic_json(a.output,result)
    elif a.action=='prepare':result=prepare(a.source,a.output)
    else:result=verify(a.output);print(json.dumps(result))
