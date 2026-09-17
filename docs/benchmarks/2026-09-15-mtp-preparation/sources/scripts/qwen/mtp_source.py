#!/usr/bin/env python3
"""Bounded extraction of the original checkpoint's MTP tensors, never its trunk.

Range payloads are bound to a pinned HTTPS revision, shard metadata and local
SHA256 receipts. Partial ranges do not prove the upstream full-shard LFS hash.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import urllib.request

from prepare_storage import atomic_json, digest_file, nocache

ROOT = Path(__file__).resolve().parents[2]
REPO = 'Qwen/Qwen3.8-Flash-Next'
REVISION = 'de4b8e4d43b917e7706784d8bb445c9af86a3540'
BASE = f'https://huggingface.co/{REPO}/resolve/{REVISION}/'
ALIGN = 16384
CHUNK = 32*1024**2
MAX_HEADER = 16*1024**2
MAX_SOURCE = 6*1024**3
META = ('config.json', 'model.safetensors.index.json')
TOKENIZERS = ('tokenizer.json','tokenizer_config.json','vocab.json','merges.txt')
EXPERT_GATE = 'mtp.layers.0.mlp.experts.gate_up_proj'
EXPERT_DOWN = 'mtp.layers.0.mlp.experts.down_proj'
SHARED = ('model.language_model.embed_tokens.weight','lm_head.weight')


def require(value, message):
    if not value: raise ValueError(message)


def sha(data): return hashlib.sha256(data).hexdigest()


def leaf(name):
    require(isinstance(name,str) and Path(name).name == name and name not in ('','.','..'), 'Unsafe source filename')
    return name


def read_json_bytes(raw):
    def pairs(items):
        result = {}
        for k,v in items:
            require(k not in result, 'Duplicate JSON key')
            result[k] = v
        return result
    return json.loads(raw, object_pairs_hook=pairs)


def get(url, limit, span=None, opener=urllib.request.urlopen):
    headers = {'Accept-Encoding':'identity','User-Agent':'FreeLLM-MTP-preparation/1'}
    if span is not None:
        start, count, total = span
        require(type(start) is int and start >= 0 and type(count) is int and 0 < count <= limit and
                start+count <= total, 'Invalid requested byte range')
        headers['Range'] = f'bytes={start}-{start+count-1}'
    request = urllib.request.Request(url, headers=headers)
    with opener(request, timeout=60) as response:
        require(response.headers.get('Content-Encoding','identity') == 'identity', 'Encoded range response')
        if span is not None:
            require(response.status == 206 and response.headers.get('Content-Range') ==
                f'bytes {start}-{start+count-1}/{total}', 'Server ignored or changed requested range')
            length = response.headers.get('Content-Length')
            require(length is None or length == str(count), 'Wrong range content length')
        else: require(response.status == 200, 'Unexpected metadata response')
        data = response.read((count if span is not None else limit)+1)
        require(len(data) == count if span is not None else len(data) <= limit, 'Truncated or oversized response')
        return data


def fetch_json(name, output):
    raw = get(BASE+leaf(name), 4*1024**2)
    value = read_json_bytes(raw); (output/name).write_bytes(raw)
    return value, sha(raw)


def validate_tensor(name, entry, base, file_size):
    require(isinstance(entry,dict) and entry.get('dtype') == 'BF16', 'Unexpected MTP source dtype: '+name)
    shape = entry.get('shape'); bounds = entry.get('data_offsets')
    require(isinstance(shape,list) and shape and all(type(v) is int and v>0 for v in shape), 'Invalid tensor shape')
    require(isinstance(bounds,list) and len(bounds)==2 and all(type(v) is int for v in bounds), 'Invalid tensor range')
    lo,hi = bounds; size = math.prod(shape)*2
    require(0 <= lo < hi and hi-lo == size and base+hi <= file_size, 'Invalid tensor storage geometry')
    return dict(shape=shape,dtype='BF16',offset=base+lo,bytes=size)


def validate_inventory(data):
    lock=ROOT/'mtp-models.lock.json'
    if lock.exists(): require(data==read_json_bytes(lock.read_bytes()),'MTP inventory differs from the pinned source lock')
    require(data.get('kind')=='qwen_mtp_source_inventory_v1' and data.get('complete') is True and
        data.get('repo')==REPO and data.get('revision')==REVISION, 'Wrong or incomplete MTP source identity')
    tensors=data['tensors']
    require(len(tensors)==31 and all(k.startswith('mtp.') for k in tensors), 'Unexpected MTP tensor set')
    require(tensors[EXPERT_GATE]['shape']==[512,1280,2560] and
        tensors[EXPERT_DOWN]['shape']==[512,2560,640], 'Unexpected MTP expert geometry')
    require(sum(t['bytes'] for t in tensors.values())==5214301696 and
        all(t['dtype']=='BF16' and t['bytes']==math.prod(t['shape'])*2 for t in tensors.values()), 'Unexpected MTP source bytes')
    require(data['shared_io']['dedicated_embeddings'] is False and
        set(data['shared_io']['tensors'])==set(SHARED) and
        all(t['shape']==[248320,2560] for t in data['shared_io']['tensors'].values()), 'Invalid shared MTP IO')
    return data


def inspect(output, target):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    config,ch=fetch_json('config.json',output);index,ih=fetch_json('model.safetensors.index.json',output)
    api_raw=get(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true',4*1024**2)
    api=read_json_bytes(api_raw);require(api.get('sha')==REVISION,'API returned another revision')
    (output/'hub-metadata.json').write_bytes(api_raw)
    files={x['rfilename']:x for x in api['siblings']}; text=config['text_config']; mtp=text['mtp']
    require(text['mtp_num_hidden_layers']==mtp['num_hidden_layers']==1 and mtp['layer_types']==['full_attention'] and
        text['mtp_use_dedicated_embeddings'] is False and text['hidden_size']==2560 and text['hc_count']==4 and
        text['vocab_size']==248320 and text['num_experts']==512 and text['num_experts_per_tok']==10,'Unsupported MTP architecture')
    selected={k:v for k,v in index['weight_map'].items() if k.startswith('mtp.') or k in SHARED}
    require(len(selected)==33,'Missing or extra MTP/shared tensor names')
    headers={}; shard_info={}
    for filename in sorted(set(selected.values())):
        leaf(filename); info=files[filename]; size=info['size'];lfs=info['lfs'];require(size==lfs['size'],'Inconsistent shard size')
        first=get(BASE+filename,8,(0,8,size)); n,=struct.unpack('<Q',first)
        require(0<n<=MAX_HEADER and n+8<=size,'Invalid safetensors header length')
        raw=get(BASE+filename,MAX_HEADER,(8,n,size)); header=read_json_bytes(raw)
        (output/(filename+'.header.json')).write_bytes(raw);headers[filename]=(n+8,header)
        shard_info[filename]=dict(bytes=size,lfs_sha256=lfs['sha256'],header_sha256=sha(raw),header_bytes=n)
        print('inspected '+filename,flush=True)
    tensors={}
    for name,filename in selected.items():
        base,header=headers[filename]; tensors[name]=dict(file=filename,**validate_tensor(name,header[name],base,files[filename]['size']))
    token_proof={}
    target=Path(target)
    target_lock=read_json_bytes((ROOT/'mixed-models.lock.json').read_bytes())
    for name in ('config.json','model.safetensors.index.json',*TOKENIZERS):
        pinned=next(f for f in target_lock['files'] if f['path']==name)
        require(sha((target/name).read_bytes())==pinned['sha256'],'Target mixed artifact metadata differs: '+name)
    target_config=read_json_bytes((target/'config.json').read_bytes())['text_config']
    for key in ('hidden_size','vocab_size','hc_count','num_experts','num_experts_per_tok','mtp_num_hidden_layers'):
        require(target_config.get(key)==text[key],'Target architecture differs: '+key)
    for filename in TOKENIZERS:
        raw=(target/filename).read_bytes(); git_hash=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        upstream=files[filename]
        compatible=(sha(raw)==upstream['lfs']['sha256'] if 'lfs' in upstream else git_hash==upstream['blobId'])
        require(len(raw)==upstream['size'] and compatible,'Target tokenizer differs: '+filename)
        token_proof[filename]=dict(bytes=len(raw),git_blob_sha1=git_hash,sha256=sha(raw),upstream=upstream)
    data=dict(kind='qwen_mtp_source_inventory_v1',complete=True,repo=REPO,revision=REVISION,
        metadata_sha256={'config.json':ch,'model.safetensors.index.json':ih,'hub-metadata.json':sha(api_raw)},
        tensors={k:v for k,v in tensors.items() if k.startswith('mtp.')},shards=shard_info,
        shared_io=dict(dedicated_embeddings=False,tensors={k:tensors[k] for k in SHARED},
            tokenizer_files=token_proof,target_revision='b2c422f3c643e36f04227a64d61796b44a4b1029'),
        source_bytes=sum(t['bytes'] for k,t in tensors.items() if k.startswith('mtp.')),
        source_parameters=sum(t['bytes']//2 for k,t in tensors.items() if k.startswith('mtp.')),
        source_headers_only=True,full_shard_hash_verified=False,production_promoted=False)
    validate_inventory(data);atomic_json(output/'inventory.json',data);return data


def load_inventory(directory):
    directory=Path(directory);data=validate_inventory(read_json_bytes((directory/'inventory.json').read_bytes()))
    for name,expected in data['metadata_sha256'].items():require(sha((directory/leaf(name)).read_bytes())==expected,'Source metadata changed')
    for name,entry in data['shards'].items():
        raw=(directory/(leaf(name)+'.header.json')).read_bytes();require(sha(raw)==entry['header_sha256'] and len(raw)==entry['header_bytes'],'Source header changed')
        header=read_json_bytes(raw)
        for key,tensor in {**data['tensors'],**data['shared_io']['tensors']}.items():
            if tensor['file']==name:
                require(tensor==dict(file=name,**validate_tensor(key,header[key],8+len(raw),entry['bytes'])),'Tensor range changed')
    return data


def download(directory):
    directory=Path(directory);inventory=load_inventory(directory)
    payload=directory/'tensors';payload.mkdir(exist_ok=True)
    receipt_path=directory/'download.json';old=read_json_bytes(receipt_path.read_bytes()) if receipt_path.exists() else {}
    identity=dict(kind='qwen_mtp_source_download_v1',revision=REVISION,inventory_sha256=sha((directory/'inventory.json').read_bytes()))
    require(not old or all(old.get(k)==v for k,v in identity.items()),'Changed download identity')
    require(inventory['source_bytes']<=MAX_SOURCE,'Source download exceeds bound')
    receipt=dict(identity,complete=False,tensors=old.get('tensors',{}),full_shard_hash_verified=False,application_read_bytes=old.get('application_read_bytes',0))
    require(set(receipt['tensors'])<=set(inventory['tensors']),'Unexpected saved source tensor')
    remaining=sum(t['bytes'] for k,t in inventory['tensors'].items() if k not in receipt['tensors'])
    require(shutil.disk_usage(directory).free>=remaining+4*1024**3,'Insufficient disk space for MTP source plus4GiB reserve')
    atomic_json(receipt_path,receipt)
    for position,(name,tensor) in enumerate(sorted(inventory['tensors'].items())):
        path=payload/(name+'.bf16');saved=receipt['tensors'].get(name)
        if saved:
            require(not path.is_symlink() and path.stat().st_size==tensor['bytes'] and digest_file(path)==saved['sha256'], 'Changed completed source tensor')
            continue
        require(not path.exists(),'Unreceipted source tensor exists')
        temp=path.with_suffix('.partial');h=hashlib.sha256();done=0
        with temp.open('wb',buffering=0) as stream:
            nocache(stream.fileno())
            while done<tensor['bytes']:
                count=min(CHUNK,tensor['bytes']-done)
                raw=get(BASE+tensor['file'],CHUNK,(tensor['offset']+done,count,inventory['shards'][tensor['file']]['bytes']))
                stream.write(raw);h.update(raw);done+=len(raw);receipt['application_read_bytes']+=len(raw)
                atomic_json(receipt_path,receipt)
                if done==tensor['bytes'] or done%(128*1024**2)==0:print(f'{position+1}/31 {name}: {done}/{tensor["bytes"]}',flush=True)
            stream.flush();import os;os.fsync(stream.fileno())
        require(done==tensor['bytes'],'Short source write');temp.replace(path)
        receipt['tensors'][name]=dict(file=str(path.relative_to(directory)),bytes=done,sha256=h.hexdigest(),source=tensor)
        atomic_json(receipt_path,receipt)
    receipt['complete']=True;atomic_json(receipt_path,receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('inspect','download'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--target',type=Path,default=ROOT/'.cache/qwen-mixed-reference');a=p.parse_args()
    result=inspect(a.output,a.target) if a.action=='inspect' else download(a.output)
    print(json.dumps({k:result[k] for k in ('kind','complete','revision')}))
