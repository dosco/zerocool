import io
import hashlib
import json
from pathlib import Path
import unittest

from verify_mixed_payloads import compare_tensor


class PayloadComparison(unittest.TestCase):
    def test_compiled_reuse_pin_is_backed_by_complete_matching_payload_evidence(self):
        # This verifies the checked-in compatibility proof. It does not replace
        # either checkpoint's current file receipts or a native runtime check.
        root=Path(__file__).resolve().parents[2]
        reuse=json.loads((root/'mixed-payload-reuse.lock.json').read_text())
        raw=(root/reuse['evidence_path']).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), reuse['evidence_sha256'])
        report=json.loads(raw)
        self.assertTrue(report['complete'])
        self.assertTrue(report['all_payloads_identical'])
        self.assertEqual(report['revisions'], dict(q4=reuse['source_revision'], mixed=reuse['consumer_revision']))
        for name, digest in reuse['file_locks_sha256'].items():
            self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(), digest)
            self.assertEqual(report['file_locks_sha256'][name], digest)
        self.assertEqual(len(report['tensors']), reuse['tensor_count'])
        totals=dict(routed_experts=0, ngrams=0)
        for tensor in report['tensors'].values():
            self.assertTrue(tensor['identical'])
            self.assertEqual(tensor['q4_sha256'], tensor['mixed_sha256'])
            totals[tensor['category']] += tensor['bytes']
        self.assertEqual(totals, dict(routed_experts=reuse['expert_payload_bytes'], ngrams=reuse['ngram_payload_bytes']))
        control=json.loads((root/'models.lock.json').read_text())
        self.assertEqual(control['prepared_control']['manifest_sha256'], reuse['prepared_manifest_sha256'])

    def test_offsets_irregular_blocks_and_late_difference(self):
        source = dict(shape=[19], dtype='BF16', bytes=38, offset=3)
        target = dict(source, offset=7)
        payload = bytes(range(38))
        same = compare_tensor(source, target, io.BytesIO(b'xxx'+payload), io.BytesIO(b'1234567'+payload), block=8)
        self.assertTrue(same['identical'])
        self.assertEqual(same['q4_sha256'], same['mixed_sha256'])
        different = compare_tensor(source, target, io.BytesIO(b'xxx'+payload),
                                   io.BytesIO(b'1234567'+payload[:-1]+b'x'), block=8)
        self.assertFalse(different['identical'])
        self.assertNotEqual(different['q4_sha256'], different['mixed_sha256'])

    def test_truncated_or_mismatched_payload_cannot_pass(self):
        source = dict(shape=[8], dtype='U32', bytes=32, offset=0)
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            compare_tensor(source, source, io.BytesIO(b'x'*32), io.BytesIO(b'x'*31), block=8)
        with self.assertRaisesRegex(ValueError, 'geometry'):
            compare_tensor(source, dict(source, dtype='F32'), io.BytesIO(b'x'*32), io.BytesIO(b'x'*32))


if __name__ == '__main__':
    unittest.main()
