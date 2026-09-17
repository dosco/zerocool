import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import capture_hyper_inputs as capture_tool
import prepare_hyper_fixtures as prep
from qualification_evidence import save, sha


SOURCE = ('std::pair<Buf,Buf> Model::hyper(int x) {\n'
          '    return {out,injection};\n}\nBuf Model::conv(int x) {}\n')


def capture_fixture(root):
    capture = root/'capture'; capture.mkdir()
    (root/'src/qwen').mkdir(parents=True)
    source = root/'src/qwen/model.cpp'; source.write_text(SOURCE)
    (root/'scripts/qwen').mkdir(parents=True)
    helper = root/'scripts/qwen/capture_hyper_inputs.py'; helper.write_text('helper')
    instrumented = root/'instrumented.cpp'; instrumented.write_text(capture_tool.instrument(SOURCE))
    binary = root/'binary'; binary.write_bytes(b'capture-binary')
    producer = dict(kind='hyper_capture_producer_v1', complete=True,
        actual_binary_is_base_native_build=False, normal_request_latency_qualified=False,
        capture_environment=capture_tool.CAPTURE_ENV, base_native_fingerprint='a'*64,
        original_model_sha256=sha(source), instrumented_model_sha256=sha(instrumented),
        helper_sha256=sha(helper), binary_sha256=sha(binary), binary=str(binary),
        source_file=str(instrumented), frozen_build_inputs={str(source):sha(source)})
    save(root/'producer.json', producer)
    report = dict(complete=True, model_revision=prep.REVISION, runs=[dict(prompt_tokens=72, output_tokens=2,
        output_token_ids=[5,6], token_latency_ms=[300.0], finish_reason='length', reused_tokens=0,
        after=dict(metal=dict(build_fingerprint='a'*64)))])
    save(root/'report.json', report)
    for index, (layer, stage, suffix, _) in enumerate(prep.CASES):
        prefix = f'capture_hyper_{index}'
        save(capture/(prefix+'.json'), dict(id=index, layer=layer, stage=stage,
            base=f'model.layers.{layer}.{suffix}', phase='decode', offset=72, tokens=1,
            input_origin='raw_hyper_input'))
        for key, width in (('input',10240),('output',2560),('injection',4)):
            (capture/f'{prefix}_{key}.bin').write_bytes(np.full(width, index+.5, '<f4').tobytes())
    return capture


class FakeWeights:
    def __init__(self, *args):
        self.tensors = {}; self.config = dict(quantization={})
        for layer, _, suffix, _ in prep.CASES:
            for projection in ('input_mix_weight_down', 'input_mix_weight_up'):
                self.config['quantization'][f'model.layers.{layer}.{suffix}.{projection}'] = dict(bits=8,group_size=64)

    def get(self, name, shape, dtype):
        value = np.zeros(shape, '<u4' if dtype == 'U32' else '<f4')
        raw = value.tobytes() if dtype == 'U32' else (value.view('<u4')>>16).astype('<u2').tobytes()
        self.tensors[name] = dict(file='model.safetensors', offset=4096, bytes=len(raw), shape=list(shape),
            dtype=dtype, sha256=hashlib.sha256(raw).hexdigest())
        return value

    def proof(self): return dict(tensors=copy.deepcopy(self.tensors), receipt_sha256='c'*64,files={})
    def close(self): pass


class HyperFixtureTest(unittest.TestCase):
    def test_source_copy_hook_changes_only_hyper_end(self):
        generated = capture_tool.instrument(SOURCE)
        self.assertEqual(generated.replace('#include <cstdlib>\n','',1).replace(capture_tool.HOOK,''), SOURCE)
        self.assertIn('phase_=="decode"', generated)
        self.assertIn('save_capture("input",x,10240*4)', generated)
        self.assertIn('save_capture("output",out,2560*4)', generated)
        self.assertIn('save_capture("injection",injection,4*4)', generated)
        with self.assertRaises(ValueError): capture_tool.instrument(generated)

    def test_raw_capture_proof_requires_completed_real_input_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); capture = capture_fixture(root)
            with patch.object(prep,'ROOT',root):
                cases, proof = prep.capture_proof(capture,root/'producer.json',root/'report.json')
                self.assertEqual(len(cases),4)
                self.assertEqual(proof['capture_bytes'], 204864)
                self.assertFalse(proof['producer']['actual_binary_is_base_native_build'])
                metadata = capture/'capture_hyper_0.json'
                row = json.loads(metadata.read_text()); row['input_origin'] = 'reconstructed_from_normalized'
                save(metadata,row)
                with self.assertRaises(ValueError): prep.capture_proof(capture,root/'producer.json',root/'report.json')

    def test_changed_source_binary_incomplete_request_and_nonfinite_input_fail(self):
        for change in ('binary','source','incomplete','input'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); capture = capture_fixture(root)
                if change == 'binary': (root/'binary').write_bytes(b'changed')
                elif change == 'source': (root/'instrumented.cpp').write_text('changed')
                elif change == 'incomplete':
                    row = json.loads((root/'report.json').read_text()); row['complete'] = False
                    save(root/'report.json',row)
                else:
                    (capture/'capture_hyper_2_input.bin').write_bytes(np.full(10240,np.nan,'<f4').tobytes())
                with patch.object(prep,'ROOT',root), self.assertRaises(ValueError):
                    prep.capture_proof(capture,root/'producer.json',root/'report.json')

    def test_bounded_original_packed_payloads_and_required_output_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); capture = capture_fixture(root)
            (root/'scripts/qwen/shared_expert_reference.py').write_text('reader')
            lock = root/'lock.json'; lock.write_text('{}')
            output = root/'fixtures'
            frequency = dict(eligible_blocks=96,frequencies=[36,36,12,12],
                blocks=[dict(layer=l,stage=s) for l in range(48) for s in ('attention_input','mlp_input')])
            with patch.object(prep,'ROOT',root), patch.object(prep,'SelectedWeights',FakeWeights), \
                    patch.object(prep,'frequency_shape_proof',return_value=frequency):
                result = prep.prepare(capture,root/'producer.json',root/'report.json',root/'model',output,lock_path=lock)
                self.assertEqual(result['total_bytes'], 28467264)
                self.assertLess(result['total_bytes'],32*1024**2)
                self.assertEqual(len(result['files']),44)
                self.assertEqual(prep.verify(output),result)
                self.assertTrue(prep.verify_prepared(output,root/'model',lock_path=lock)['original_weight_bytes_verified'])
            for change in ('output','normalization','weight_sha','frequency','bytes'):
                changed = copy.deepcopy(result)
                if change == 'output': changed['cases'][0].pop('native_output')
                elif change == 'normalization': changed['raw_input_origin']='normalized_input'
                elif change == 'weight_sha': changed['cases'][0]['tensors']['up_w']['sha256']='d'*64
                elif change == 'frequency': changed['cases'][0]['frequency']=48
                else: changed['total_bytes']+=1
                with self.subTest(change=change), self.assertRaises((ValueError,KeyError)):
                    prep.validate_manifest(changed)
            (output/result['cases'][0]['native_output']['file']).write_bytes(b'changed')
            with self.assertRaises(ValueError): prep.verify(output)


if __name__ == '__main__': unittest.main()
