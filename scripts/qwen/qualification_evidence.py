"""Frozen local experiment identity, bounded subprocesses, and immutable evidence."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from build_identity import build_fingerprint
from verify_checkpoint import fingerprint

GiB = 1024**3


class ResourceBlocked(ValueError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def confined(directory, name):
    directory = Path(directory).resolve()
    path = directory / name
    if not path.resolve().is_relative_to(directory):
        raise ValueError('Evidence path escapes its directory')
    return path


def tree_bytes(directory):
    total = 0
    for path in Path(directory).rglob('*'):
        if path.is_symlink():
            raise ValueError('Evidence directories must not contain symbolic links')
        if path.is_file():
            total += path.stat().st_size
    return total


def identity(root, configs, model, prepared, workload):
    root, model, prepared = Path(root), Path(model), Path(prepared)
    lock = json.loads((root/'mixed-models.lock.json').read_text())
    receipt = json.loads((model/'zerocool-verification.json').read_text())
    manifest = json.loads((prepared/'manifest.json').read_text())
    prepared_receipt = json.loads((prepared/'verification.json').read_text())
    if receipt.get('revision') != lock['revision']:
        raise ValueError('Wrong mixed artifact receipt')
    control=json.loads((root/'models.lock.json').read_text())
    if (prepared_receipt.get('manifest_sha256')!=sha(prepared/'manifest.json') or
        prepared_receipt.get('source_revision')!=control['revision'] or
        control['prepared_control']['manifest_sha256']!=sha(prepared/'manifest.json')):
        raise ValueError('Wrong prepared artifact receipt')
    assets = {}
    for entry in lock['files']:
        if entry.get('optional'):
            continue
        path = confined(model, entry['path']); current = fingerprint(path)
        if receipt['files'].get(entry['path']) != dict(current, sha256=entry['sha256']):
            raise ValueError('Stale artifact receipt: ' + entry['path'])
        assets[str(path.resolve())] = current
    for entry in manifest['files']:
        path = confined(prepared, entry['path'])
        current=fingerprint(path)
        if current['size']!=entry['size'] or prepared_receipt['files'].get(entry['path'])!=dict(current,sha256=entry['sha256']):
            raise ValueError('Prepared file changed since verification')
        assets[str(path.resolve())] = current
    paths = [*sorted((root/'scripts/qwen').glob('*.py')),
             *sorted((root/'scripts/qwen').glob('*.cpp')), *sorted((root/'scripts/qwen').glob('*.hpp')),
             root/'tests/test_qwen.cpp', root/'mixed-models.lock.json', root/'models.lock.json',
             root/'mixed-payload-reuse.lock.json', model/'zerocool-verification.json',
             prepared/'manifest.json', prepared/'verification.json', Path(workload)]
    paths += [root/'build/qwen'/name for name in ('bin/zerocool','test_qwen','qwen_cached_recovery','qwen_sparse_replay','qwen_panel_check')]
    return dict(protocol='selector-qualification-v1', root=str(root.resolve()),
                build=build_fingerprint(root), artifact_revision=lock['revision'],
                prepared_manifest_sha256=sha(prepared/'manifest.json'),
                files={str(p.resolve()): sha(p) for p in paths}, assets=assets,
                configurations=configs, sampling=dict(temperature=0,top_p=.95,top_k=20,seed=0),
                budget_bytes=12*GiB, context=8192, device='Apple M1 Pro', physical_bytes=32*GiB)


class EvidenceGuard:
    def __init__(self, evidence, output):
        self.identity = evidence
        self.output = Path(output)

    def check_identity(self):
        if build_fingerprint(Path(self.identity['root'])) != self.identity['build']:
            raise ValueError('Native sources changed during experiment')
        for name, digest in self.identity['files'].items():
            if sha(name) != digest:
                raise ValueError('Frozen experiment file changed: ' + name)
        for name, saved in self.identity['assets'].items():
            if fingerprint(Path(name)) != saved:
                raise ValueError('Artifact payload changed: ' + name)

    def check_resources(self, initial=False):
        free = shutil.disk_usage(self.output).free
        minimum = (7 if initial else 5)*GiB
        if free < minimum:
            raise ResourceBlocked(f'Disk admission requires {minimum} free bytes; available {free}')
        if tree_bytes(self.output) >= 2*GiB:
            raise ResourceBlocked('Experiment evidence reached its 2GiB limit')

    def run(self, command, *, stdout, timeout, env=None, progress=None):
        self.check_identity(); self.check_resources()
        process = subprocess.Popen([str(x) for x in command], cwd=self.identity['root'], stdout=stdout,
                                   stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                if progress is not None: progress()
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    process.wait(timeout=min(2, max(.01, deadline-time.monotonic())))
                except subprocess.TimeoutExpired:
                    self.check_resources()
            self.check_identity(); self.check_resources()
            if process.returncode:
                if getattr(stdout,'name',None):
                    message=Path(stdout.name).read_text(errors='replace')[-8192:]
                    if any(s in message for s in ('insufficient currently available memory',
                        'cached replay budget is not currently admitted','memory admission requires at least',
                        'Common budget/panel unavailable','Disk admission requires',
                        'qualification_evidence.ResourceBlocked:')):
                        raise ResourceBlocked('Native memory/storage admission failed; see '+str(stdout.name))
                raise subprocess.CalledProcessError(process.returncode, command)
        except BaseException:
            if process.poll() is None:
                for sig, delay in ((signal.SIGINT,30),(signal.SIGTERM,10),(signal.SIGKILL,5)):
                    try: os.killpg(process.pid, sig)
                    except ProcessLookupError: break
                    try: process.wait(timeout=delay); break
                    except subprocess.TimeoutExpired: pass
            raise


def seal(directory):
    directory = Path(directory)
    tree_bytes(directory)
    files = {str(p.relative_to(directory)): sha(p) for p in directory.rglob('*')
             if p.is_file() and p != directory/'evidence-files.json'}
    save(directory/'evidence-files.json', files)
    return sha(directory/'evidence-files.json')


def verify_seal(directory, digest):
    directory = Path(directory)
    tree_bytes(directory)
    path = directory/'evidence-files.json'
    if sha(path) != digest:
        raise ValueError('Changed evidence index')
    files=json.loads(path.read_text())
    actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p!=path}
    if not isinstance(files,dict) or set(files)!=actual:
        raise ValueError('Changed evidence file inventory')
    for name, expected in files.items():
        if sha(confined(directory, name)) != expected:
            raise ValueError('Changed raw evidence: ' + name)


def import_sealed(source, destination, digest):
    verify_seal(source, digest)
    if tree_bytes(source) + tree_bytes(Path(destination).parent) >= 2*GiB:
        raise ResourceBlocked('Imported evidence would exceed 2GiB')
    shutil.copytree(source, destination)
    verify_seal(destination, digest)
