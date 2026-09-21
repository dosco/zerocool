#!/usr/bin/env python3
"""Real-artifact metadata rejection checks; missing assets fail, never skip."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def check(args):
    manifest = json.loads((args.prepared / 'manifest.json').read_text())
    receipt = json.loads((args.prepared / 'verification.json').read_text())
    cases = [
        ('version', lambda m: m.update(schema=99)),
        ('source', lambda m: m.update(source_revision='0'*40)),
        ('projection_offset', lambda m: m['expert_layout'][1].update(offset=0)),
        ('projection_format', lambda m: m['expert_layout'][0].update(bits=3)),
        ('duplicate_layer', lambda m: m['experts'][1].update(file=m['experts'][0]['file'])),
        ('expert_stride', lambda m: m['experts'][0].update(stride=1)),
        ('ngram_mapping', lambda m: m['ngrams'][0].update(count=1)),
        ('ngram_fields', lambda m: m['ngrams'][0]['fields'][1].update(offset=81)),
        ('unsafe_path', lambda m: m['files'][0].update(path='../expert.bin')),
        ('manifest_hash', lambda m: None),
        ('truncated_payload', lambda m: None),
    ]
    results = []
    for name, mutate in cases:
        with tempfile.TemporaryDirectory(prefix='zerocool-prepared-rejection-') as tmp:
            root = Path(tmp)
            for entry in manifest['files']:
                (root / entry['path']).symlink_to((args.prepared / entry['path']).resolve())
            altered = copy.deepcopy(manifest)
            mutate(altered)
            data = (json.dumps(altered, indent=2) + '\n').encode()
            (root / 'manifest.json').write_bytes(data)
            proof = copy.deepcopy(receipt)
            proof['manifest_sha256'] = hashlib.sha256(data).hexdigest()
            if name == 'manifest_hash':
                proof['manifest_sha256'] = '0'*64
            if name == 'truncated_payload':
                file = root / manifest['files'][0]['path']
                file.unlink()  # remove only the temporary symlink, never its target
                file.write_bytes(b'broken')
            (root / 'verification.json').write_text(json.dumps(proof))
            result = subprocess.run([str(args.binary), 'inspect', '--model', str(args.model),
                                     '--prepared', str(root)], capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                raise AssertionError(f'Invalid prepared artifact accepted: {name}')
            if 'zerocool:' not in result.stderr or 'Metal device unavailable' in result.stderr:
                raise AssertionError(f'{name} did not reach artifact validation: {result.stderr}')
            results.append(dict(name=name, passed=True, error=result.stderr.strip()))
    args.output.write_text(json.dumps(dict(kind='prepared_rejection_checks', passed=True, checks=results), indent=2) + '\n')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary', type=Path, default=root / 'build/qwen/bin/zerocool')
    ap.add_argument('--model', type=Path, default=root / '.cache/models/qwen38-flash-next')
    ap.add_argument('--prepared', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    check(ap.parse_args())
