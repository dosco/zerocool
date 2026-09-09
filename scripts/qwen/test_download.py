import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from download import preflight
from verify_checkpoint import verify


class ArtifactDownload(unittest.TestCase):
    def test_rejects_other_recipe_or_changed_files_before_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            original = b'original packed weights'
            (model/'weights').write_bytes(original)
            lock = dict(revision='q4', files=[dict(path='weights', size=len(original), sha256=hashlib.sha256(original).hexdigest())])
            preflight(model, lock)
            verify(model, lock)
            with self.assertRaisesRegex(ValueError, 'different verified artifact'):
                preflight(model, dict(lock, revision='mixed'))
            (model/'weights').write_bytes(b'x'*len(original))
            with self.assertRaisesRegex(ValueError, 'changed artifact file'):
                preflight(model, lock)
            self.assertEqual((model/'weights').read_bytes(), b'x'*len(original))

    def test_disk_admission_counts_missing_files_and_preserves_reserve(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            lock = dict(revision='mixed', files=[dict(path='weights', size=10, sha256='pending')])
            with patch('shutil.disk_usage') as usage:
                usage.return_value.free = 5*1024**3+9
                with self.assertRaisesRegex(ValueError, 'Insufficient disk'):
                    preflight(model, lock)
                usage.return_value.free += 1
                self.assertEqual(preflight(model, lock), lock['files'])
            lock['files'][0]['path'] = '../outside'
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                preflight(model, lock)


if __name__ == '__main__':
    unittest.main()
