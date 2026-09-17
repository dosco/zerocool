"""Reject stale or cross-artifact replay evidence; synthetic data is not model QA."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from build_identity import build_fingerprint
from check_full_replay import run


class FullReplayEvidence(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='freellm-replay-evidence-')
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)
        root=Path(__file__).resolve().parents[2]
        revision=json.loads((root/'mixed-models.lock.json').read_text())['revision']
        self.sha=lambda raw:hashlib.sha256(raw).hexdigest()
        raw=b'\0'*248320*4
        (self.path/'reference_logits.f32').write_bytes(raw)
        (self.path/'native.f32').write_bytes(raw)
        self.write('reference.json',dict(source_revision=revision,tokens=[760]))
        self.manifest=dict(report_sha256=self.sha((self.path/'reference.json').read_bytes()),
            outputs={'logits.f32':dict(bytes=len(raw),sha256=self.sha(raw))})
        for layer in range(48):
            name=f'layer_{layer}.bin'; (self.path/name).write_bytes(b'\0'*4)
            self.manifest['outputs'][name]=dict(bytes=4,sha256=self.sha(b'\0'*4))
        self.write('reference-identity.json',self.manifest)
        self.report=dict(tokens=[760],layers=48,logits_file=str(self.path/'native.f32'),
            statistics=dict(artifact_revision=revision,metal=dict(build_fingerprint=build_fingerprint(root))),
            state=dict(valid=True,tokens=1,layers=[dict(position=1)]*48))
        self.write('prepared.json',self.report); self.write('source.json',self.report)
        self.args=SimpleNamespace(artifact='mixed-4_8bit',reference=self.path,trace=self.path,
            prepared_report=self.path/'prepared.json',source_report=self.path/'source.json',output=self.path/'out.json')

    def write(self,name,data): (self.path/name).write_text(json.dumps(data))

    def check(self):
        with contextlib.redirect_stdout(io.StringIO()):run(self.args)

    def test_other_artifact_cannot_reuse_identical_outputs(self):
        changed=copy.deepcopy(self.report);changed['statistics']['artifact_revision']='other'
        self.write('source.json',changed)
        with self.assertRaises(SystemExit):self.check()
        r=json.loads(self.args.output.read_text())
        self.assertFalse(r['passed'])
        self.assertFalse(next(c['passed'] for c in r['checks'] if c['name']=='source_identity'))

    def test_changed_reference_logits_are_rejected(self):
        (self.path/'reference_logits.f32').write_bytes(b'\1'*248320*4)
        with self.assertRaisesRegex(ValueError,'logits changed'):self.check()

    def test_changed_reference_identity_is_rejected(self):
        self.write('reference.json',dict(source_revision='other',tokens=[760]))
        with self.assertRaisesRegex(ValueError,'reference identity'):self.check()

    def test_retained_state_difference_cannot_pass_matching_logits(self):
        changed=copy.deepcopy(self.report);changed['state']['layers'][0]['position']=0
        self.write('source.json',changed)
        with self.assertRaises(SystemExit):self.check()
        r=json.loads(self.args.output.read_text())
        self.assertFalse(next(c['passed'] for c in r['checks'] if c['name']=='persistent_state'))


if __name__=='__main__':unittest.main()
