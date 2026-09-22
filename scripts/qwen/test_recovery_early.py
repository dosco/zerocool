from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import trial_recovery as trial
from qualification_evidence import save, seal
from screen_recovery_early import diagnostic_pair
import evidence_fixture


def setUpModule():
    evidence_fixture.require('docs/benchmarks/2026-09-15-q4-request-context/capture-02/evidence-files.json')


class EarlyRecoveryTests(unittest.TestCase):
    def bundle(self, root):
        proof=dict(binary='/unused/native',binary_sha256='b'*64)
        save(root/'producer.json',proof)
        save(root/'identity.json',dict(files={proof['binary']:proof['binary_sha256']}))
        save(root/'summary.json',dict(kind='mtp_target_recovery_validation_v1',
                                     status='resource_blocked',complete=False))
        save(root/'case-0.json',trial.correctness_cases()[0])
        for arm in trial.ARMS:
            save(root/f'case-0-pair-0-{arm}.json',dict(target_recovery=arm,producer_binary_sha256='b'*64))
        seal(root)
        return proof

    def test_blocked_numeric_pair_never_imports_resource_qualification(self):
        with tempfile.TemporaryDirectory() as d, \
             patch.object(trial,'observe',return_value=dict(clean_memory=False,clean_host=True)) as observe, \
             patch.object(trial,'comparison',return_value=dict(exact=True)) as compare:
            root=Path(d);proof=self.bundle(root)
            result,files=diagnostic_pair(trial,root,proof)
            self.assertEqual(result['recorded_status'],'resource_blocked')
            self.assertEqual(len(files),7)
            self.assertTrue(all(not s['clean_memory'] for s in result['observations']))
            self.assertIn('no validation timing or resource eligibility',result['use'])
            self.assertEqual(observe.call_count,2)
            self.assertTrue(all(c.args[-1] is True for c in observe.call_args_list))
            compare.assert_called_once()

    def test_changed_seal_producer_workload_and_missing_arm_are_rejected(self):
        for change in ('unsealed','producer','identity','workload','arm','missing'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d, \
                 patch.object(trial,'observe',return_value={}),patch.object(trial,'comparison'):
                root=Path(d);proof=self.bundle(root)
                if change=='unsealed':save(root/'case-0.json',{})
                elif change=='producer':save(root/'producer.json',dict(proof,binary_sha256='c'*64))
                elif change=='identity':save(root/'identity.json',dict(files={}))
                elif change=='workload':save(root/'case-0.json',dict(trial.correctness_cases()[0],force_prefix=2))
                else:
                    path=root/'case-0-pair-0-state-only.json'
                    if change=='missing':path.unlink()
                    else:save(path,dict(target_recovery='full-replay',producer_binary_sha256='b'*64))
                if change!='unsealed':seal(root)
                with self.assertRaises((ValueError,FileNotFoundError)):
                    diagnostic_pair(trial,root,proof)

    def test_mismatched_numerics_are_not_accepted_as_diagnostics(self):
        with tempfile.TemporaryDirectory() as d,patch.object(trial,'observe',return_value={}), \
             patch.object(trial,'comparison',side_effect=ValueError('Different persistent state')):
            root=Path(d);proof=self.bundle(root)
            with self.assertRaisesRegex(ValueError,'Different persistent state'):
                diagnostic_pair(trial,root,proof)


if __name__=='__main__':unittest.main()
