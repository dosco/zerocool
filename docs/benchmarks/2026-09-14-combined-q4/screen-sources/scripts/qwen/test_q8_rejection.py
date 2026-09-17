#!/usr/bin/env python3
"""Explicit real-fixture integrity checks; absent assets fail this command."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def run(args):
    manifest = json.loads((args.fixture/'manifest.json').read_text())
    if len(manifest['cases']) != 35 or not all((args.fixture/name).is_file() for name in manifest['files']):
        raise ValueError('Requires the complete real Q8 fixture')
    first = manifest['matrices']['shared_gate']['tensors']['weight']['file']
    checks = []
    for name, expected in [('missing_weight', 'No such file or directory'), ('truncated_weight', 'fixture size mismatch'),
                           ('changed_weight', 'fixture hash mismatch'), ('wrong_source', 'unpinned Q8 fixture')]:
        with tempfile.TemporaryDirectory(prefix='freellm-q8-rejection-') as directory:
            root = Path(directory)
            shutil.copyfile(args.fixture/'manifest.json', root/'manifest.json')
            for file in manifest['files']:
                (root/file).symlink_to((args.fixture/file).resolve())
            if name == 'wrong_source':
                altered = json.loads((root/'manifest.json').read_text())
                altered['source']['revision'] = '0'*40
                (root/'manifest.json').write_text(json.dumps(altered))
            else:
                (root/first).unlink()
                if name != 'missing_weight':
                    data = bytearray((args.fixture/first).read_bytes())
                    if name == 'truncated_weight':
                        data = data[:-1]
                    else:
                        data[0] ^= 1
                    (root/first).write_bytes(data)
            result = subprocess.run([str(args.binary), str(root), str(args.lock), str(root/'report.json')],
                                    capture_output=True, text=True, timeout=30)
            checks.append(dict(name=name, passed=result.returncode == 1 and expected in result.stderr,
                               returncode=result.returncode, error=result.stderr.strip()))
    report = dict(passed=all(c['passed'] for c in checks), checks=checks)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    if not report['passed']:
        raise SystemExit(1)
    print(f'{len(checks)} Q8 integrity rejection cases passed')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=root/'build/qwen/qwen_q8_check')
    parser.add_argument('--lock', type=Path, default=root/'q8-reference.lock.json')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
