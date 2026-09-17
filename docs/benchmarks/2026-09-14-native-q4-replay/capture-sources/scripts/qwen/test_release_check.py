#!/usr/bin/env python3
"""Asset-free regression checks for release rejection paths; never qualifies a model."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from build_identity import build_fingerprint

ROOT = Path(__file__).resolve().parents[2]


class ReleaseRejections(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='freellm-release-test-')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.fingerprint = build_fingerprint(ROOT)
        self.revision = json.loads((ROOT/'models.lock.json').read_text())['revision']
        # Identical synthetic vectors isolate admission of evidence, not model quality.
        for name in ('native_logits.f32', 'reference_logits.f32'):
            np.ones(248320, np.float32).tofile(self.directory / name)
        self.logits = {'native_tokens': [1], 'reference_tokens': [1], 'layers': 48,
                       'build_fingerprint': self.fingerprint, 'artifact_revision': self.revision,
                       'reference': {'source_revision': self.revision}}
        for role in ('native', 'reference'):
            self.logits[role+'_logits_sha256'] = hashlib.sha256((self.directory/(role+'_logits.f32')).read_bytes()).hexdigest()
        self.write('logits.json', self.logits)

    def write(self, name, value):
        (self.directory / name).write_text(json.dumps(value))

    def run_check(self):
        out = self.directory / 'result.json'
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/qwen/release_check.py'),
                                 '--model', str(self.directory / 'missing-model'),
                                 '--evidence', str(self.directory), '--out', str(out)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        report = json.loads(out.read_text())
        self.assertFalse(report['passed'])
        return {c['name']: c for c in report['checks']}

    def test_missing_assets_fail_even_when_vectors_agree(self):
        checks = self.run_check()
        self.assertFalse(checks['checkpoint']['passed'])
        self.assertTrue(checks['full_model_logits']['passed'])
        self.assertFalse(checks['cache_invariance']['passed'])

    def test_matching_vectors_from_an_old_build_fail(self):
        self.write('logits.json', {'native_tokens': [1], 'reference_tokens': [1],
                                 'layers': 48, 'build_fingerprint': 'old-build'})
        check = self.run_check()['full_model_logits']
        self.assertFalse(check['passed'])
        self.assertIn('current native sources', check['error'])

    def test_identical_logits_from_another_artifact_fail(self):
        self.write('logits.json', dict(self.logits, artifact_revision='another-artifact'))
        check = self.run_check()['full_model_logits']
        self.assertFalse(check['passed'])
        self.assertIn('artifact', check['error'])

    def test_changed_logits_cannot_reuse_comparison_metadata(self):
        np.full(248320, 2, np.float32).tofile(self.directory/'native_logits.f32')
        check = self.run_check()['full_model_logits']
        self.assertFalse(check['passed'])
        self.assertIn('payload changed', check['error'])

    def test_diagnostic_schedule_cannot_qualify_performance(self):
        row = {'name': 'prompt_2k', 'output_tokens': 256, 'prompt_tokens': 2048,
               'after': {'artifact_revision': self.revision, 'metal': {'device': 'Apple M1 Pro', 'physical_bytes': 32 * 1024**3,
                                    'build_fingerprint': self.fingerprint},
                         'diagnostic_stream_trunk': True}}
        self.write('performance.json', {'complete': True, 'model_revision': self.revision, 'runs': [row] * 5})
        check = self.run_check()['performance']
        self.assertFalse(check['passed'])
        self.assertIn('Diagnostic schedules', check['error'])

    def test_fewer_than_five_measurements_fail(self):
        self.write('performance.json', {'complete': True, 'runs': [{'name': 'prompt_2k'}] * 3})
        check = self.run_check()['performance']
        self.assertFalse(check['passed'])
        self.assertIn('five measurements', check['error'])

    def test_profiled_and_warm_timings_cannot_qualify(self):
        row=dict(name='prompt_2k',output_tokens=256,prompt_tokens=2048,profiling_enabled=True,repetition=0,
                 after=dict(artifact_revision=self.revision,diagnostic_stream_trunk=False,
                    metal=dict(device='Apple M1 Pro',physical_bytes=32*1024**3,build_fingerprint=self.fingerprint)))
        self.write('performance.json',dict(complete=True,model_revision=self.revision,runs=[row]*5))
        self.assertIn('Profiled timings',self.run_check()['performance']['error'])
        row.update(profiling_enabled=False,repetition=1)
        self.write('performance.json',dict(complete=True,model_revision=self.revision,runs=[row]*5))
        self.assertIn('fresh process',self.run_check()['performance']['error'])


if __name__ == '__main__':
    unittest.main()
