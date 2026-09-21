#!/usr/bin/env python3
"""Verify every required pinned file, then atomically record its identity.

A receipt avoids rereading 104GB at every startup. It is invalidated by a file's
size, device, inode, or modification time changing. Release checks always hash
the files again; this receipt is a local integrity cache, not a trust boundary.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]


def fingerprint(path):
    st = path.stat()
    return dict(size=st.st_size, device=st.st_dev, inode=st.st_ino,
                mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)


def hash_file(path):
    h = hashlib.sha256()
    with path.open('rb', buffering=0) as f:
        if os.uname().sysname == 'Darwin':
            import fcntl
            fcntl.fcntl(f.fileno(), 48, 1)  # F_NOCACHE
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def verify(model, lock, cached=False):
    receipt_path = model / 'zerocool-verification.json'
    old = json.loads(receipt_path.read_text()) if cached and receipt_path.exists() else {}
    result = dict(schema=1, revision=lock['revision'], verified_at=int(time.time()), files={})
    for entry in lock['files']:
        if entry.get('optional'):
            continue
        path = model / entry['path']
        before = fingerprint(path)
        if before['size'] != entry['size']:
            raise RuntimeError(f"Wrong size: {path}")
        previous = old.get('files', {}).get(entry['path'], {})
        if (cached and old.get('revision') == lock['revision'] and
                previous == dict(before, sha256=entry['sha256'])):
            digest = previous['sha256']
        else:
            # Avoid leaving a 104GB verification scan in the macOS page cache.
            digest = hash_file(path)
        if digest != entry['sha256'] or fingerprint(path) != before:
            raise RuntimeError(f"Hash mismatch or file changed during verification: {path}")
        result['files'][entry['path']] = dict(before, sha256=digest)
        print(f"verified {entry['path']}", flush=True)
    temporary = receipt_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(receipt_path)
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, default=ROOT / '.cache/models/qwen38-flash-next')
    ap.add_argument('--lock', type=Path, default=ROOT / 'models.lock.json')
    ap.add_argument('--check-receipt', action='store_true', help='Reuse unchanged verified files; unsuitable for a release check')
    args = ap.parse_args()
    verify(args.model, json.loads(args.lock.read_text()), args.check_receipt)
