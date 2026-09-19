from pathlib import Path
import tempfile
import unittest

import streamed_mtp_capsule as capsule
from qualification_evidence import save, seal, sha
from verify_checkpoint import fingerprint


class CapsuleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.native = self.root/'native'; self.native.mkdir()
        self.source = self.root/'source'; self.source.mkdir()
        self.host = self.root/'host'; self.host.mkdir()
        binary = self.native/'probe-mtp-forward'; binary.write_bytes(b'original native executable')
        source = self.native/'model.cpp'; source.write_bytes(b'embedded shader and arithmetic')
        obj = self.native/'model.o'; obj.write_bytes(b'original object')
        host = self.host/'benchmark-host'; host.write_bytes(b'original host executable')
        host_proof = dict(kind='benchmark_host_producer_v1', complete=True,
            base_native_fingerprint='a'*64, binary_sha256=sha(host))
        save(self.host/'producer.json', host_proof)
        self.proof = dict(kind='streamed_mtp_fixed_priming_producer_v1', complete=True,
            base_native_fingerprint='a'*64, binary=str(binary), binary_sha256=sha(binary),
            generated={str(source): sha(source)}, objects={str(obj): sha(obj)})
        save(self.source/'producer.json', self.proof)
        save(self.source/'identity.json', dict(build='a'*64, files={str(binary): sha(binary),
            str(host): sha(host), str(self.host/'producer.json'): sha(self.host/'producer.json')}))
        save(self.source/'summary.json', dict(kind='streamed_mtp_validation_v1', complete=False))
        seal(self.source)
        self.output = self.root/'capsule'

    def create(self):
        capsule.create(self.source, self.output)
        return capsule.verify(self.output)

    def test_preserves_exact_executable_and_original_provenance(self):
        cfg, proof, host, receipt = self.create()
        self.assertEqual(proof, self.proof)
        self.assertEqual(sha(cfg['binary']), proof['binary_sha256'])
        self.assertEqual(sha(host['binary']), receipt['host_binary_sha256'])
        self.assertFalse(receipt['relinked'])
        self.assertFalse(receipt['historical_timing_reused'])

    def test_changed_original_binary_is_not_packaged(self):
        Path(self.proof['binary']).write_bytes(b'new executable')
        with self.assertRaisesRegex(ValueError, 'Saved native producer changed'):
            self.create()

    def test_changed_generated_object_is_not_packaged(self):
        Path(next(iter(self.proof['objects']))).write_bytes(b'new object')
        with self.assertRaisesRegex(ValueError, 'Saved native producer changed'):
            self.create()

    def test_modified_copy_fails_even_if_capsule_resealed(self):
        self.create()
        (self.output/'probe-mtp-forward').write_bytes(b'new binary')
        seal(self.output)
        with self.assertRaisesRegex(ValueError, 'capsule identity changed'):
            capsule.verify(self.output)

    def test_modified_host_fails_even_if_capsule_resealed(self):
        self.create()
        (self.output/'benchmark-host').write_bytes(b'new host')
        seal(self.output)
        with self.assertRaisesRegex(ValueError, 'capsule identity changed'):
            capsule.verify(self.output)

    def test_changed_source_seal_cannot_rebind_capsule(self):
        self.create()
        save(self.source/'summary.json', dict(kind='streamed_mtp_validation_v1', complete=True))
        seal(self.source)
        with self.assertRaisesRegex(ValueError, 'capsule identity changed'):
            capsule.verify(self.output)

    def guard(self):
        self.create()
        runner = self.root/'runner.py'; runner.write_bytes(b'original runner')
        artifact = self.root/'model.bin'; artifact.write_bytes(b'original weights')
        identity = dict(execution_capsule=dict(path=str(self.output)),
            files={str(runner): sha(runner), **{str(p): sha(p) for p in self.output.iterdir()}},
            assets={str(artifact): fingerprint(artifact)})
        return capsule.CapsuleGuard(identity, self.root, 2*1024**3), runner, artifact

    def test_unrelated_checkout_rebuild_does_not_change_executed_binary(self):
        guard, _, _ = self.guard()
        (self.root/'libfreellm_lib.a').write_bytes(b'unrelated rebuilt library')
        guard.check_identity()

    def test_changed_runner_is_rejected(self):
        guard, runner, _ = self.guard()
        runner.write_bytes(b'changed runner')
        with self.assertRaisesRegex(ValueError, 'Frozen execution input changed'):
            guard.check_identity()

    def test_changed_model_asset_is_rejected(self):
        guard, _, artifact = self.guard()
        artifact.write_bytes(b'changed weight payload')
        with self.assertRaisesRegex(ValueError, 'Artifact payload changed'):
            guard.check_identity()

    def test_normal_build_path_keeps_existing_guard(self):
        self.assertIs(capsule.experiment(self.native), capsule.Experiment)


if __name__ == '__main__':
    unittest.main()
