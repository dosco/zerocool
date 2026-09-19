"""Run a sealed, unchanged MTP executable independently of later UI rebuilds.

The normal current-source builder remains strict. This explicit execution mode
pins an already-built executable, embedded shaders, host probe, model assets and
the Python modules actually used by the runner. It never relinks the executable.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

from cache_residency import require
from qualification_evidence import save, sha, seal, verify_seal
from stage200 import Experiment, StageGuard
from verify_checkpoint import fingerprint

ROOT = Path(__file__).resolve().parents[2]
KIND = 'streamed_mtp_execution_capsule_v1'


def read(path):
    return json.loads(Path(path).read_text())


def source_proof(source):
    source = Path(source).resolve()
    digest = sha(source/'evidence-files.json')
    verify_seal(source, digest)
    identity, proof = read(source/'identity.json'), read(source/'producer.json')
    require(read(source/'summary.json')['kind'] == 'streamed_mtp_validation_v1' and
        proof['kind'] == 'streamed_mtp_fixed_priming_producer_v1' and proof['complete'] is True and
        identity['build'] == proof['base_native_fingerprint'] and
        identity['files'].get(proof['binary']) == proof['binary_sha256'],
        'Capsule requires a sealed fixed-priming validation producer')
    hosts = [Path(p) for p in identity['files'] if Path(p).name == 'benchmark-host']
    require(len(hosts) == 1, 'Missing unique frozen native host probe')
    host = hosts[0]
    host_proof = read(host.parent/'producer.json')
    require(host_proof['complete'] is True and host_proof['kind'] == 'benchmark_host_producer_v1' and
        host_proof['base_native_fingerprint'] == identity['build'] and
        host_proof['binary_sha256'] == identity['files'][str(host)] and
        sha(host.parent/'producer.json') == identity['files'].get(str(host.parent/'producer.json')),
        'Host probe provenance differs from sealed validation')
    return source, digest, identity, proof, host, host_proof


def create(source, output):
    source, digest, identity, proof, host, host_proof = source_proof(source)
    # Runtime code and Metal shader source are embedded in the executable.
    # Verify every generated source/object too; changed libraries in the shared
    # checkout are build-time inputs, never substituted into the saved binary.
    for path, expected in {**proof['generated'], **proof['objects'],
            proof['binary']: proof['binary_sha256'], str(host): host_proof['binary_sha256']}.items():
        require(sha(path) == expected, 'Saved native producer changed: '+path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for src, name in ((Path(proof['binary']), 'probe-mtp-forward'), (host, 'benchmark-host')):
        shutil.copy2(src, output/name)
    for name in ('producer.json', 'identity.json', 'summary.json', 'evidence-files.json'):
        shutil.copyfile(source/name, output/('source-'+name))
    save(output/'producer.json', proof)
    save(output/'host-producer.json', host_proof)
    receipt = dict(kind=KIND, complete=True, source=str(source), source_seal_sha256=digest,
        binary_sha256=proof['binary_sha256'], host_binary_sha256=host_proof['binary_sha256'],
        base_native_fingerprint=proof['base_native_fingerprint'],
        runtime_policy='unchanged sealed executables; embedded Metal shaders; verified model assets',
        relinked=False, historical_timing_reused=False, production_promoted=False)
    save(output/'capsule.json', receipt)
    seal(output)
    verify(output)
    return receipt


def verify(directory):
    directory = Path(directory).resolve()
    verify_seal(directory, sha(directory/'evidence-files.json'))
    receipt = read(directory/'capsule.json')
    source, digest, identity, proof, _, host_proof = source_proof(receipt['source'])
    expected = dict(kind=KIND, complete=True, source=str(source), source_seal_sha256=digest,
        binary_sha256=proof['binary_sha256'], host_binary_sha256=host_proof['binary_sha256'],
        base_native_fingerprint=proof['base_native_fingerprint'],
        runtime_policy='unchanged sealed executables; embedded Metal shaders; verified model assets',
        relinked=False, historical_timing_reused=False, production_promoted=False)
    require(receipt == expected and read(directory/'producer.json') == proof and
        read(directory/'host-producer.json') == host_proof and
        sha(directory/'probe-mtp-forward') == proof['binary_sha256'] and
        sha(directory/'benchmark-host') == host_proof['binary_sha256'], 'Execution capsule identity changed')
    for name in ('producer.json', 'identity.json', 'summary.json', 'evidence-files.json'):
        require(sha(directory/('source-'+name)) == sha(source/name), 'Capsule provenance copy changed')
    return dict(binary=directory/'probe-mtp-forward'), proof, dict(binary=directory/'benchmark-host'), receipt


class CapsuleGuard(StageGuard):
    def check_identity(self):
        # This mode explicitly identifies the executable's original build. The
        # live checkout's source fingerprint is recorded separately, not used
        # to falsely label the already-linked executable as a new build.
        verify(self.identity['execution_capsule']['path'])
        for name, digest in self.identity['files'].items():
            require(sha(name) == digest, 'Frozen execution input changed: '+name)
        for name, saved in self.identity['assets'].items():
            require(fingerprint(Path(name)) == saved, 'Artifact payload changed since capsule admission')


def experiment(directory):
    directory = Path(directory).resolve()
    if not (directory/'capsule.json').is_file():
        return Experiment

    class CapsuleExperiment(Experiment):
        def prepare(self, configs, work, kind):
            super().prepare(configs, work, kind)
            _, proof, _, receipt = verify(directory)
            current = self.frozen
            original = read(directory/'source-identity.json')
            require(current['assets'] == original['assets'] and all(current[k] == original[k]
                for k in ('artifact_revision', 'prepared_manifest_sha256', 'budget_bytes', 'context',
                          'device', 'physical_bytes')), 'Capsule model assets or experiment controls changed')
            # Keep model receipts/locks and workload. Unrelated developer
            # scripts and rebuilt executables are not runtime dependencies.
            files = {p: h for p, h in current['files'].items()
                if Path(p).suffix == '.json'}
            modules = {Path(m.__file__).resolve() for m in list(sys.modules.values())
                if getattr(m, '__file__', None) and
                Path(m.__file__).resolve().is_relative_to(ROOT/'scripts/qwen')}
            files.update({str(p): sha(p) for p in modules})
            files.update({str(p): sha(p) for p in directory.iterdir() if p.is_file()})
            # Preserve the original-producer lookup used by the offline query
            # tools; execution uses the additional, separately frozen copy.
            files[proof['binary']] = proof['binary_sha256']
            current.update(files=files, runner_workspace_build=current['build'],
                build=proof['base_native_fingerprint'],
                execution_capsule=dict(path=str(directory), seal_sha256=sha(directory/'evidence-files.json'),
                    producer_source=str(directory/'producer.json'), **receipt))
            save(self.out/'identity.json', current)
            self.report['identity']['build'] = current['build']
            self.report['execution_capsule'] = current['execution_capsule']
            self.guard = CapsuleGuard(current, self.out, 2*1024**3)
            self.persist()
    return CapsuleExperiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create(args.source, args.output), indent=2))
