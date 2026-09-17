from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import trial_recovery as trial
from qualification_evidence import save, seal, sha
from resume_recovery_correctness import validation_inventory


class RecoveryResumeTests(unittest.TestCase):
    def bundle(self, root, clean=(True,True), count=2):
        binary=root/'native';binary.write_bytes(b'fixture')
        proof=dict(binary=str(binary),binary_sha256=sha(binary))
        save(root/'producer.json',proof)
        save(root/'identity.json',dict(files={str(binary):sha(binary)}))
        save(root/'case-0.json',trial.correctness_cases()[0])
        samples=[]
        for arm,ok in list(zip(trial.ARMS,clean))[:count]:
            path=root/(arm+'.json')
            save(path,dict(producer_binary_sha256=proof['binary_sha256'],target_recovery=arm,actual_clean=ok))
            samples.append(dict(case=0,arm=arm,source=path.name,sha256=sha(path),clean_memory=True))
        save(root/'summary.json',dict(kind='mtp_target_recovery_numerical_diagnostic_v1',
            complete=False,status='resource_blocked',samples=samples,qualification_eligible=False))
        seal(root)
        return proof

    def observe(self,raw,work,input_sha,validation):
        self.assertIs(validation,True)
        return dict(clean_memory=raw['actual_clean'],clean_host=True)

    def test_clean_independent_pair_reused_without_relabeling_parent(self):
        with tempfile.TemporaryDirectory() as d,patch.object(trial,'observe',side_effect=self.observe), \
             patch.object(trial,'comparison') as compare:
            root=Path(d).resolve();proof=self.bundle(root);before=sha(root/'summary.json')
            chosen,rejected=validation_inventory(trial,[root],proof)
            self.assertEqual(set(chosen),{(0,a) for a in trial.ARMS});self.assertEqual(rejected,[])
            self.assertEqual(sha(root/'summary.json'),before)
            self.assertFalse(trial.read(root/'summary.json')['qualification_eligible'])
            compare.assert_called_once()

    def test_only_complete_clean_runs_reused_and_missing_arm_remains_missing(self):
        for clean,count in [((False,True),2),((True,False),2),((True,True),1)]:
            with self.subTest(clean=clean,count=count),tempfile.TemporaryDirectory() as d, \
                 patch.object(trial,'observe',side_effect=self.observe),patch.object(trial,'comparison'):
                root=Path(d).resolve();proof=self.bundle(root,clean,count)
                chosen,rejected=validation_inventory(trial,[root],proof)
                expected={(0,a) for a,ok in list(zip(trial.ARMS,clean))[:count] if ok}
                self.assertEqual(set(chosen),expected)
                self.assertLess(len(chosen),2)
                self.assertEqual(len(rejected),sum(not ok for ok in clean[:count]))

    def test_timing_source_changed_native_workload_sample_or_duplicates_rejected(self):
        for change in ('timing','binary','workload','sample','duplicate'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d, \
                 patch.object(trial,'observe',side_effect=self.observe),patch.object(trial,'comparison'):
                root=Path(d).resolve();proof=self.bundle(root)
                summary=trial.read(root/'summary.json')
                if change=='timing':summary['kind']='mtp_target_recovery_screen_v1'
                elif change=='binary':(root/'native').write_bytes(b'changed')
                elif change=='workload':save(root/'case-0.json',trial.correctness_cases()[1])
                elif change=='sample':save(root/'state-only.json',dict(actual_clean=True))
                else:summary['samples'].append(summary['samples'][0])
                save(root/'summary.json',summary);seal(root)
                with self.assertRaises(ValueError):validation_inventory(trial,[root],proof)

    def test_numeric_mismatch_never_reused(self):
        with tempfile.TemporaryDirectory() as d,patch.object(trial,'observe',side_effect=self.observe), \
             patch.object(trial,'comparison',side_effect=ValueError('Mismatch')):
            root=Path(d).resolve();proof=self.bundle(root)
            with self.assertRaisesRegex(ValueError,'Mismatch'):
                validation_inventory(trial,[root],proof)


if __name__=='__main__':unittest.main()
