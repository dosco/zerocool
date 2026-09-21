#!/usr/bin/env python3
"""Download one explicit pinned artifact without replacing another recipe."""
import argparse
import json
from pathlib import Path
import shutil
from verify_checkpoint import ROOT, fingerprint, hash_file, verify


def preflight(model, lock):
    entries = [f for f in lock['files'] if not f.get('optional')]
    receipt_path = model/'zerocool-verification.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    if receipt and receipt.get('revision') != lock['revision']:
        raise ValueError('Refusing to replace a different verified artifact; choose its own directory')
    remaining = 0
    for item in entries:
        name = item['path']
        if Path(name).name != name or name in ('', '.', '..'):
            raise ValueError('Unsafe artifact filename')
        path = model/name
        if not path.exists():
            remaining += item['size']
            continue
        before = fingerprint(path)
        if before['size'] != item['size']:
            raise ValueError(f'Refusing to replace a different existing artifact file: {name}')
        saved = receipt.get('files', {}).get(name)
        if saved != dict(before, sha256=item['sha256']):
            if hash_file(path) != item['sha256'] or fingerprint(path) != before:
                raise ValueError(f'Refusing to replace a changed artifact file: {name}')
    if shutil.disk_usage(model).free < remaining + 5 * 1024**3:
        raise ValueError('Insufficient disk space for the pinned checkpoint plus 5GiB reserve')
    return entries


def run(args):
    from huggingface_hub import snapshot_download
    lock = json.loads(args.lock.read_text())
    args.model.mkdir(parents=True, exist_ok=True)
    entries = preflight(args.model, lock)
    snapshot_download(lock['repo'], revision=lock['revision'], local_dir=args.model,
                      allow_patterns=[f['path'] for f in entries], max_workers=2)
    verify(args.model, lock)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, default=ROOT / '.cache/models/qwen38-flash-next')
    ap.add_argument('--lock', type=Path, default=ROOT / 'models.lock.json')
    run(ap.parse_args())
