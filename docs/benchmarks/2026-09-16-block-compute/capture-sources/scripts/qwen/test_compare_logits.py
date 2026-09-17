"""Evidence admission tests; synthetic logits never qualify a checkpoint."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from compare_logits import run


class LogitEvidence(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='freellm-logit-evidence-')
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.args = SimpleNamespace(**{name: root / name for name in
            ('native', 'reference', 'native_report', 'reference_report', 'out')})
        np.ones(248320, np.float32).tofile(self.args.native)
        np.ones(248320, np.float32).tofile(self.args.reference)
        self.native = dict(tokens=[760], layers=48, statistics=dict(
            artifact_revision='mixed-revision', diagnostic_stream_trunk=True,
            metal=dict(build_fingerprint='current-build')))
        self.reference = dict(tokens=[760], layers=48, source_revision='mixed-revision')

    def compare(self):
        self.args.native_report.write_text(json.dumps(self.native))
        self.args.reference_report.write_text(json.dumps(self.reference))
        with patch('compare_logits.build_fingerprint', return_value='current-build'), \
                contextlib.redirect_stdout(io.StringIO()):
            return run(self.args)

    def test_matching_evidence_records_hashes_and_diagnostic_scope(self):
        report = self.compare()
        self.assertTrue(report['passed'])
        self.assertTrue(report['diagnostic_stream_trunk'])
        self.assertEqual(report['artifact_revision'], 'mixed-revision')
        for key in ('native_logits_sha256', 'reference_logits_sha256',
                    'native_report_sha256', 'reference_report_sha256'):
            self.assertEqual(len(report[key]), 64)

    def test_identical_vectors_cannot_cross_artifacts(self):
        for revision in ('q4-revision', None):
            with self.subTest(revision=revision):
                self.native['statistics']['artifact_revision'] = revision
                with self.assertRaisesRegex(ValueError, 'artifacts differ'):
                    self.compare()
                self.assertFalse(self.args.out.exists())

    def test_matching_old_build_cannot_qualify_current_sources(self):
        self.native['statistics']['metal']['build_fingerprint'] = 'old-build'
        with self.assertRaisesRegex(ValueError, 'current build'):
            self.compare()
        self.assertFalse(self.args.out.exists())

    def test_nonfinite_logits_and_different_tokens_are_rejected(self):
        self.reference['tokens'] = [314]
        with self.assertRaisesRegex(SystemExit, 'mismatched'):
            self.compare()
        self.reference['tokens'] = [760]
        np.full(248320, np.nan, np.float32).tofile(self.args.native)
        with self.assertRaisesRegex(SystemExit, 'mismatched'):
            self.compare()
        self.assertFalse(self.args.out.exists())


if __name__ == '__main__':
    unittest.main()
