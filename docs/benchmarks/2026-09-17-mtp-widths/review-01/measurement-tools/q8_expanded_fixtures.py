"""Verify tiny captured inputs and bind selected packed weights to pinned ranges."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import struct

from cache_residency import require
from capture_routes import load
from qualification_evidence import save, sha, verify_seal
from q8_expanded_contract import ROOT, CASES
from shared_expert_reference import SelectedWeights

PROFILE = ROOT/'docs/benchmarks/2026-09-16-block-compute/capture-03'


def frequency_proof():
    seal=sha(PROFILE/'evidence-files.json');verify_seal(PROFILE,seal);report=load(PROFILE/'summary.json')
    require(report['complete'] is True and all(c['exact']['clean_memory'] and c['exact']['clean_host'] for c in report['captures']),
        'Frequency evidence must be complete and clean')
    capture=next(c for c in report['captures'] if c['mode']=='dispatch');a=capture['analysis']
    require(a['captured_blocks']==4 and a['captured_input_tokens']==16,'Changed frequency coverage')
    selected=[]
    for case in CASES:
        rows=[r for r in a['counter_operations'] if r['stage']==case['stage'] and r['kernel']=='q8_mm_t4' and
            (r['K'],r['N'],r['rows'])==(case['K'],case['N'],4)]
        require(len(rows)==1 and rows[0]['dispatches']==case['frequency']*4,'Unproved shape frequency')
        selected.append(dict(name=case['name'],dispatches=rows[0]['dispatches'],frequency=case['frequency']))
    return dict(source=str(PROFILE),seal_sha256=seal,cases=selected,uses_timing=False)


def weight_ranges(reader,case,captured):
    K,N=case['K'],case['N'];base=case['tensor'];result={}
    require(reader.config['quantization'].get(base)==dict(bits=8,group_size=64),'Unexpected precision')
    for key,suffix,dtype,shape in [('w','weight','U32',[N,K//4]),('s','scales','BF16',[N,K//64]),('b','biases','BF16',[N,K//64])]:
        tensor=base+'.'+suffix;name=reader.index['language_model.'+tensor];reader.verify_file(name)
        with (reader.model/name).open('rb',buffering=0) as f:
            fcntl.fcntl(f.fileno(),48,1)  # Darwin F_NOCACHE; no full-shard mmap.
            size=struct.unpack('<Q',f.read(8))[0];require(0<size<16*1024**2,'Unbounded tensor header')
            header=json.loads(f.read(size));entry=header['language_model.'+tensor]
            low,high=entry['data_offsets'];bytes_=N*K if key=='w' else N*(K//64)*2
            require(entry['dtype']==dtype and entry['shape']==shape and type(low) is int and type(high) is int and
                0<=low<high and high-low==bytes_ and 8+size+high<=reader.states[name]['size'],'Changed tensor geometry')
            f.seek(8+size+low);remaining=bytes_;digest=hashlib.sha256()
            while remaining:
                chunk=f.read(min(1024**2,remaining));require(bool(chunk),'Truncated packed tensor')
                digest.update(chunk);remaining-=len(chunk)
        require(captured[key]==dict(bytes=bytes_,sha256=digest.hexdigest()),'Captured weight differs from pinned checkpoint')
        result[key]=dict(tensor=tensor,file=name,offset=8+size+low,bytes=bytes_,sha256=digest.hexdigest(),dtype=dtype,shape=shape)
    return result


def derive(inputs,model,fingerprint):
    manifest=load(inputs/'manifest.json')
    require(manifest['kind']=='q8_expanded_inputs_v1' and manifest['build_fingerprint']==fingerprint and
        manifest['byte_limit']==1024**2 and manifest['case_limit']==6 and manifest['weight_payloads_copied'] is False and
        len(manifest['cases'])==6,'Changed capture bounds or identity')
    reader=SelectedWeights(model,ROOT/'mixed-models.lock.json');cases=[];total=0
    try:
        require(manifest['artifact_revision']==reader.receipt['revision'],'Wrong artifact revision')
        for spec in CASES:
            rows=[c for c in manifest['cases'] if c['context']['stage']==spec['stage'] and c['context']['layer']==spec['layer'] and
                c['matrix']['K']==spec['K'] and c['matrix']['N']==spec['N']]
            require(len(rows)==1,'Missing or duplicate input case');row=rows[0]
            require(row['phase']=='decode' and row['context']['offset']==72 and row['context']['tokens']==4 and
                row['matrix']==dict(K=spec['K'],N=spec['N'],rows=4,group=64,bits=8,fused=False,gathered=False) and
                set(row['tensors'])=={'w','s','b','x'},'Changed capture operation')
            x=row['tensors']['x'];p=inputs/x['file']
            require(Path(x['file']).name==x['file'] and x['bytes']==4*spec['K']*4 and p.stat().st_size==x['bytes'] and
                sha(p)==x['sha256'],'Changed captured activation')
            weights=weight_ranges(reader,spec,row['tensors']);total+=x['bytes']
            cases.append(dict(name=spec['name'],frequency=spec['frequency'],matrix=row['matrix'],phase=row['phase'],
                context=row['context'],tensors=dict(weights,x=x)))
        require(total==manifest['bytes'] and total<=1024**2,'Incomplete input accounting')
        return dict(kind='q8_expanded_fixture_v1',complete=True,build_fingerprint=fingerprint,artifact_revision=manifest['artifact_revision'],
            input_manifest_sha256=sha(inputs/'manifest.json'),input_bytes=total,weight_payloads_copied=False,
            checkpoint_proof=reader.proof(),frequency_proof=frequency_proof(),cases=cases)
    finally:reader.close()


def prepare(inputs,output,model,fingerprint):
    result=derive(inputs,model,fingerprint);output.mkdir(parents=True,exist_ok=False)
    for c in result['cases']:
        name=c['tensors']['x']['file'];shutil.copyfile(inputs/name,output/name)
    save(output/'manifest.json',result);return result


def verify(inputs,output,model,fingerprint):
    expected=derive(inputs,model,fingerprint);require(load(output/'manifest.json')==expected,'Changed prepared fixtures')
    for c in expected['cases']:
        x=c['tensors']['x'];p=output/x['file'];require(p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],'Changed replay activation')
    return expected
