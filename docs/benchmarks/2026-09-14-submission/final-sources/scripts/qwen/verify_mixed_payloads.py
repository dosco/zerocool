#!/usr/bin/env python3
"""Verify whether the mixed artifact can share the Q4 expert/ngram payloads.

Both complete checkpoints must already exist. Whole-file verification receipts
bind each source to its pin; bounded uncached reads then compare every routed
projection and ngram tensor, not samples or just quantization metadata.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import time

from inspect_mixed import ROOT, category, compare, inventory, parse, sha
from verify_checkpoint import fingerprint, verify

BLOCK = 8 * 1024**2


def compare_tensor(a, b, left, right, block=BLOCK):
    if (a['shape'], a['dtype'], a['bytes']) != (b['shape'], b['dtype'], b['bytes']):
        raise ValueError('Payload geometry differs')
    left.seek(a['offset'])
    right.seek(b['offset'])
    first, second = hashlib.sha256(), hashlib.sha256()
    identical = True
    for offset in range(0, a['bytes'], block):
        size = min(block, a['bytes']-offset)
        x, y = left.read(size), right.read(size)
        if len(x) != size or len(y) != size:
            raise ValueError('Truncated payload comparison')
        first.update(x)
        second.update(y)
        identical &= x == y
    return dict(bytes=a['bytes'], q4_sha256=first.hexdigest(), mixed_sha256=second.hexdigest(), identical=identical)


def run(args):
    locks = [parse((ROOT/name).read_bytes()) for name in ('models.lock.json', 'mixed-models.lock.json')]
    paths = [args.q4, args.mixed]
    receipts = [verify(path, lock, cached=not args.rehash) for path, lock in zip(paths, locks)]
    before = [{e['path']:fingerprint(path/e['path']) for e in lock['files'] if not e.get('optional')}
              for path, lock in zip(paths, locks)]
    sources = [inventory(path, parse((path/'model.safetensors.index.json').read_bytes()),
                         {e['path']:e for e in lock['files']}) for path, lock in zip(paths, locks)]
    layout = compare(*sources, *(parse((path/'config.json').read_bytes()) for path in paths))
    entries = {name:None for name in sources[0] if category(name) in ('routed_experts', 'ngrams')}
    start = time.monotonic()
    with ExitStack() as stack:
        files = []
        for path, source in zip(paths, sources):
            handles = {}
            for name in {source[key]['file'] for key in entries}:
                handle = stack.enter_context((path/name).open('rb', buffering=0))
                if os.uname().sysname == 'Darwin':
                    import fcntl
                    fcntl.fcntl(handle.fileno(), 48, 1)  # F_NOCACHE
                handles[name] = handle
            files.append(handles)
        # Sort by the mixed storage location to keep one side sequential while
        # still comparing exactly the corresponding original tensor bytes.
        order = sorted(entries, key=lambda name:(sources[1][name]['file'], sources[1][name]['offset']))
        read_bytes = 0
        for number, name in enumerate(order, 1):
            a, b = (source[name] for source in sources)
            entries[name] = dict(category=category(name), **compare_tensor(a, b, files[0][a['file']], files[1][b['file']]))
            read_bytes += 2*a['bytes']
            if number % 32 == 0 or number == len(order):
                print(f'Compared {number}/{len(order)} tensors; {read_bytes/1024**3:.2f}GiB application reads', flush=True)
    after = [{e['path']:fingerprint(path/e['path']) for e in lock['files'] if not e.get('optional')}
             for path, lock in zip(paths, locks)]
    if before != after:
        raise ValueError('Source files changed during payload comparison')
    result = dict(kind='mixed_q4_complete_expert_ngram_comparison', complete=True,
        all_payloads_identical=all(v['identical'] for v in entries.values()),
        revisions={key:lock['revision'] for key,lock in zip(('q4','mixed'),locks)},
        file_locks_sha256={name:sha((ROOT/name).read_bytes()) for name in ('models.lock.json','mixed-models.lock.json')},
        verification_receipts=receipts, generator_sha256=sha(Path(__file__).read_bytes()),
        source_files_rehashed=args.rehash, application_read_bytes=read_bytes,
        elapsed_seconds=time.monotonic()-start, layout=layout, tensors=entries,
        native_mixed_inference_verified=False, quality_qualified=False)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(f'All {len(entries)} expert/ngram tensors identical: {result["all_payloads_identical"]}', flush=True)
    if not result['all_payloads_identical']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--q4', type=Path, default=ROOT/'.cache/models/qwen38-flash-next')
    parser.add_argument('--mixed', type=Path, default=ROOT/'.cache/qwen-mixed-reference')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rehash', action='store_true', help='Rehash both complete sources in addition to checking their current verification receipts')
    run(parser.parse_args())
