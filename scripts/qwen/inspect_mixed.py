#!/usr/bin/env python3
"""Audit the pinned mixed artifact using bounded headers, without loading weights.

This is preparation evidence only. Matching layouts do not authorize payload
reuse; the complete mixed shard hashes still need verification before inference.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import urllib.request

from verify_checkpoint import verify

ROOT = Path(__file__).resolve().parents[2]
MAX_HEADER = 16 * 1024**2
UNITS = dict(BF16=2, F16=2, U32=4, F32=4, I32=4, I64=8, U64=8)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def parse(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate metadata key: {key}')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


def safe_name(name):
    if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Unsafe artifact filename')
    return name


def download(url, limit, byte_range=None, total=None):
    headers = {'Accept-Encoding': 'identity'}
    if byte_range:
        start, end = byte_range
        headers['Range'] = f'bytes={start}-{end}'
        # A distinct URL also avoids intermediary caches returning a prior range.
        url += f'?zerocool_range={start}-{end}'
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
        if byte_range and (response.status != 206 or
                           response.headers.get('Content-Range') != f'bytes {start}-{end}/{total}'):
            raise ValueError('Server did not honor the exact bounded range')
        raw = response.read(limit + 1)
    if len(raw) > limit or (byte_range and len(raw) != end-start+1):
        raise ValueError('Oversized or truncated metadata response')
    return raw


def check_small(raw, entry):
    if len(raw) != entry['size']:
        raise ValueError('Metadata size mismatch')
    if 'lfs' in entry:
        valid = sha(raw) == entry['lfs']['sha256']
    else:
        # A Git blob ID hashes the prefix and payload. LFS blobId instead names
        # a pointer file, so it cannot verify the large tokenizer payload.
        valid = hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest() == entry['blobId']
    if not valid:
        raise ValueError('Metadata payload hash mismatch')


def read_header(raw, total):
    if len(raw) < 8:
        raise ValueError('Truncated safetensors prefix')
    length, = struct.unpack('<Q', raw[:8])
    if not 0 < length <= MAX_HEADER or len(raw) != length+8 or len(raw) > total:
        raise ValueError('Invalid safetensors header length')
    tensors = parse(raw[8:])
    ranges = []
    result = {}
    for name, item in tensors.items():
        if name == '__metadata__':
            continue
        shape, offsets = item['shape'], item['data_offsets']
        if (not isinstance(shape, list) or not shape or
                any(type(x) is not int or x <= 0 for x in shape) or
                not isinstance(offsets, list) or len(offsets) != 2 or
                any(type(x) is not int or x < 0 for x in offsets)):
            raise ValueError(f'Invalid tensor geometry: {name}')
        start, end = offsets
        size = math.prod(shape) * UNITS.get(item['dtype'], 0)
        if not size or end-start != size or end > total-len(raw):
            raise ValueError(f'Invalid tensor extent: {name}')
        ranges.append((start, end))
        result[name] = dict(dtype=item['dtype'], shape=shape, bytes=size, offset=len(raw)+start)
    at = 0
    for start, end in sorted(ranges):
        if start != at:
            raise ValueError('Overlapping or uncovered tensor payload')
        at = end
    if at != total-len(raw):
        raise ValueError('Incomplete tensor payload coverage')
    return result


def fetch(directory, recipe, filenames):
    directory.mkdir(parents=True, exist_ok=True)
    repo, revision = recipe['repo'], recipe['revision']
    info = parse(download(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true', MAX_HEADER))
    if info['sha'] != revision or info['id'] != repo:
        raise ValueError('Hub returned a different artifact revision')
    entries = {f['rfilename']: f for f in info['siblings']}
    receipt_file = directory/'metadata-receipt.json'
    old = parse(receipt_file.read_bytes()) if receipt_file.exists() else {}
    if old and (old['repo'], old['revision']) != (repo, revision):
        raise ValueError('Refusing to replace another artifact metadata directory')
    receipt = dict(repo=repo, revision=revision, headers={})
    for name in filenames:
        safe_name(name)
        entry = entries[name]
        if entry['size'] > MAX_HEADER:
            raise ValueError('Only small metadata files may be downloaded whole')
        path = directory/name
        raw = path.read_bytes() if path.exists() else download(
            f'https://huggingface.co/{repo}/resolve/{revision}/{name}', entry['size'])
        check_small(raw, entry)
        path.write_bytes(raw)
    index = parse((directory/'model.safetensors.index.json').read_bytes())
    shards = sorted(set(index['weight_map'].values()))
    for name in shards:
        safe_name(name)
        entry = entries[name]
        if not re.fullmatch(r'model-\d{5}\.safetensors', name) or 'lfs' not in entry:
            raise ValueError('Unexpected mixed shard')
        path = directory/(name+'.header')
        raw = path.read_bytes() if path.exists() else b''
        if not raw or old.get('headers', {}).get(name) != sha(raw):
            url = f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
            prefix = download(url, 8, (0, 7), entry['size'])
            size, = struct.unpack('<Q', prefix)
            if not 0 < size <= MAX_HEADER:
                raise ValueError('Remote safetensors header exceeds bound')
            raw = prefix + download(url, size, (8, size+7), entry['size'])
        read_header(raw, entry['size'])
        path.write_bytes(raw)
        receipt['headers'][name] = sha(raw)
        receipt_file.write_text(json.dumps(receipt, indent=2)+'\n')
        print(f'Inspected {name}: {len(raw):,} header bytes', flush=True)
    (directory/'hub.json').write_text(json.dumps(info, indent=2)+'\n')


def inventory(directory, index, entries, headers_only=False, receipt=None):
    result = {}
    for name in sorted(set(index['weight_map'].values())):
        safe_name(name)
        if headers_only:
            raw = (directory/(name+'.header')).read_bytes()
            if receipt['headers'].get(name) != sha(raw):
                raise ValueError('Cached header identity changed')
        else:
            with (directory/name).open('rb') as f:
                prefix = f.read(8)
                size, = struct.unpack('<Q', prefix)
                if not 0 < size <= MAX_HEADER:
                    raise ValueError('Local safetensors header exceeds bound')
                raw = prefix + f.read(size)
        for key, item in read_header(raw, entries[name]['size']).items():
            if key in result or index['weight_map'].get(key) != name:
                raise ValueError('Index/header disagreement or duplicate tensor')
            result[key] = dict(item, file=name)
    if set(result) != set(index['weight_map']):
        raise ValueError('Index has missing tensors')
    return result


def normalized(name):
    return name.removeprefix('language_model.')


def category(name):
    name = normalized(name)
    if name.startswith(('mtp.', 'vision_tower.', 'model.visual.')):
        return 'excluded'
    if '.switch_mlp.' in name:
        return 'routed_experts'
    if 'ngram_embedding.shard_' in name:
        return 'ngrams'
    return 'resident'


def precision(config, base):
    quant = config['quantization']
    value = quant.get(normalized(base), quant)
    bits, group = value['bits'], value['group_size']
    if type(bits) is not int or type(group) is not int or bits not in (4, 8) or group not in (32, 64):
        raise ValueError('Unsupported affine quantization metadata')
    return bits, group


def compare(q4, mixed, q4_config, mixed_config):
    if set(q4) != set(mixed):
        raise ValueError('Mixed artifact tensor names differ from Q4')
    totals = {kind: dict(q4_bytes=0, mixed_bytes=0, tensors=0) for kind in
              ('resident', 'routed_experts', 'ngrams', 'excluded')}
    counts = Counter()
    for name, a in q4.items():
        b = mixed[name]
        kind = category(name)
        row = totals[kind]
        row['q4_bytes'] += a['bytes']
        row['mixed_bytes'] += b['bytes']
        row['tensors'] += 1
        if name.endswith('.weight') and a['dtype'] == 'U32':
            base = name[:-7]
            formats = []
            for tensors, config in ((q4, q4_config), (mixed, mixed_config)):
                bits, group = precision(config, base)
                weight, scales, biases = (tensors[base+'.'+suffix] for suffix in ('weight', 'scales', 'biases'))
                shape = weight['shape'][:-1] + [weight['shape'][-1]*(32//bits)]
                expected = shape[:-1]+[shape[-1]//group]
                if (weight['dtype'] != 'U32' or shape[-1] % group or
                        scales['dtype'] != 'BF16' or biases['dtype'] != 'BF16' or
                        scales['shape'] != expected or biases['shape'] != expected):
                    raise ValueError(f'Invalid affine projection: {base}')
                formats.append((bits, group, shape))
            if formats[0][2] != formats[1][2] or (kind != 'resident' and formats[0] != formats[1]):
                raise ValueError(f'Incompatible decoded projection: {base}')
            counts[f'{kind}_q{formats[1][0]}_g{formats[1][1]}'] += 1
        elif (a['dtype'], a['shape']) != (b['dtype'], b['shape']):
            raise ValueError(f'Incompatible tensor: {name}')
    resident = {}
    for label, tensors in (('q4', q4), ('mixed', mixed)):
        resident[label] = sum((t['bytes']+16383)//16384*16384 for n,t in tensors.items() if category(n)=='resident')
    return dict(payloads=totals, quantized_matrices=dict(counts), resident_aligned_bytes=resident,
                resident_increase_bytes=resident['mixed']-resident['q4'],
                expert_slot_equivalent=(resident['mixed']-resident['q4'])/2768896)


def run(args):
    control = parse((ROOT/'models.lock.json').read_bytes())
    recipe = parse((ROOT/'recipes.lock.json').read_bytes())['artifacts']['mixed-4_8bit']
    small = [e['path'] for e in control['files'] if not e.get('optional') and not e['path'].endswith('.safetensors')]
    if args.fetch:
        fetch(args.metadata, recipe, small)
    info = parse((args.metadata/'hub.json').read_bytes())
    receipt = parse((args.metadata/'metadata-receipt.json').read_bytes())
    if (info['sha'], info['id']) != (recipe['revision'], recipe['repo']) or (receipt['revision'], receipt['repo']) != (info['sha'], info['id']):
        raise ValueError('Metadata revision mismatch')
    entries = {f['rfilename']:f for f in info['siblings']}
    # The Q4 receipt verifies that headers and common metadata come from the
    # pinned local control. It does not attest any mixed tensor payload bytes.
    verify(args.model, control, cached=True)
    common = {}
    configs = []
    for name in small:
        raw = (args.metadata/name).read_bytes()
        check_small(raw, entries[name])
        baseline = (args.model/name).read_bytes()
        if name == 'config.json':
            configs = [parse(baseline), parse(raw)]
            stripped = [{k:v for k,v in c.items() if k not in ('quantization', 'quantization_config')} for c in configs]
            if stripped[0] != stripped[1]:
                raise ValueError('Mixed architecture configuration differs')
        elif name not in ('model.safetensors.index.json', 'README.md', 'LICENSE'):
            if raw != baseline:
                raise ValueError(f'Tokenizer or architecture file differs: {name}')
            common[name] = sha(raw)
    q4 = inventory(args.model, parse((args.model/'model.safetensors.index.json').read_bytes()), {e['path']:e for e in control['files']})
    mixed_index = parse((args.metadata/'model.safetensors.index.json').read_bytes())
    mixed = inventory(args.metadata, mixed_index, entries, True, receipt)
    result = compare(q4, mixed, *configs)
    result.update(kind='mixed_artifact_layout_preflight', repo=recipe['repo'], revision=recipe['revision'],
        q4_revision=control['revision'], generator_sha256=sha(Path(__file__).read_bytes()),
        tensor_count=len(mixed), common_files_sha256=common, architecture_equal=True, layout_compatible=True,
        headers=receipt['headers'], metadata_files={name:sha((args.metadata/name).read_bytes()) for name in small},
        shard_files=[dict(path=name, size=entries[name]['size'], sha256=entries[name]['lfs']['sha256'])
                     for name in sorted(set(mixed_index['weight_map'].values()))],
        mixed_payload_verified=False, baseline_payload_reuse_authorized=False,
        native_mixed_inference_verified=False, quality_qualified=False)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(f'{len(mixed)} tensor layouts compatible; payload identity is still unverified.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT/'.cache/models/qwen38-flash-next')
    parser.add_argument('--metadata', type=Path, default=ROOT/'.cache/qwen-mixed-metadata')
    parser.add_argument('--fetch', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
