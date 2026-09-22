import io
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import mtp_source as source
import prepare_mtp as prep
from verify_mtp import clean_memory_observations
import evidence_fixture


class Response(io.BytesIO):
    def __init__(self,body,status=206,headers=None):
        super().__init__(body);self.status=status;self.headers=headers or {}

class MtpSourceTest(unittest.TestCase):
    def test_inventory_ignores_only_live_hub_metadata_hash(self):
        original=json.loads((source.ROOT/'mtp-models.lock.json').read_text())
        changed=copy.deepcopy(original)
        changed['metadata_sha256']['hub-metadata.json']='0'*64
        source.validate_inventory(changed)
        for section,key in [('metadata_sha256','config.json'),('metadata_sha256','model.safetensors.index.json')]:
            bad=copy.deepcopy(changed);bad[section][key]='0'*64
            with self.assertRaises(ValueError):source.validate_inventory(bad)
        bad=copy.deepcopy(changed);bad['tensors'][source.EXPERT_GATE]['offset']+=2
        with self.assertRaises(ValueError):source.validate_inventory(bad)
        bad=copy.deepcopy(changed);bad['shards'][next(iter(bad['shards']))]['lfs_sha256']='0'*64
        with self.assertRaises(ValueError):source.validate_inventory(bad)
        with tempfile.TemporaryDirectory() as directory,patch.object(source,'ROOT',Path(directory)):
            with self.assertRaises(FileNotFoundError):source.validate_inventory(original)

    def test_short_writes_are_completed_and_zero_progress_rejected(self):
        class ShortWriter(io.BytesIO):
            def write(self,data):return super().write(data[:3])
        stream=ShortWriter();source.write_all(stream,b'0123456789')
        self.assertEqual(stream.getvalue(),b'0123456789')
        for invalid in (0,None,-1,11,True):
            with patch.object(stream,'write',return_value=invalid):
                with self.assertRaises(ValueError):source.write_all(stream,b'0123456789')

    def test_failed_download_stays_incomplete_and_can_retry(self):
        name='mtp.test';payload=b'0123456789'
        tensor=dict(file='shard.safetensors',offset=0,bytes=len(payload),shape=[5],dtype='BF16')
        inventory=dict(tensors={name:tensor},source_bytes=len(payload),shards={tensor['file']:dict(bytes=len(payload))})
        with tempfile.TemporaryDirectory() as directory,patch.object(source,'load_inventory',return_value=inventory),\
                patch.object(source,'get',return_value=payload),patch.object(source.shutil,'disk_usage') as usage:
            usage.return_value.free=10*1024**3;root=Path(directory);(root/'inventory.json').write_text('{}')
            def interrupted(stream,data):stream.write(data[:2]);raise OSError('simulated interrupted write')
            with patch.object(source,'write_all',side_effect=interrupted):
                with self.assertRaises(OSError):source.download(root)
            receipt=json.loads((root/'download.json').read_text())
            self.assertFalse(receipt['complete']);self.assertEqual(receipt['tensors'],{})
            self.assertFalse((root/'tensors'/f'{name}.bf16').exists())
            result=source.download(root)
            self.assertTrue(result['complete'])
            self.assertEqual((root/'tensors'/f'{name}.bf16').read_bytes(),payload)
            self.assertEqual(result['tensors'][name]['sha256'],hashlib.sha256(payload).hexdigest())
            self.assertEqual(result['application_read_bytes'],2*len(payload))
            result=copy.deepcopy(result);result['tensors'][name]['source']['offset']=2
            (root/'download.json').write_text(json.dumps(result))
            with self.assertRaises(ValueError):source.download(root)

    def test_range_rejects_ignored_misaligned_truncated_and_encoded_responses(self):
        good={'Content-Range':'bytes 4-7/12','Content-Length':'4'}
        def fetch(response):return source.get('https://example.invalid/pinned',4,(4,4,12),opener=lambda *a,**k:response)
        self.assertEqual(fetch(Response(b'abcd',headers=good)),b'abcd')
        for response in [Response(b'abcd',200,good),Response(b'abcd',headers={'Content-Range':'bytes 0-3/12'}),
            Response(b'abc',headers=good),Response(b'abcde',headers=good),Response(b'abcd',headers=dict(good,**{'Content-Encoding':'gzip'}))]:
            with self.assertRaises(ValueError):fetch(response)

    def test_header_geometry_dtype_and_duplicate_keys(self):
        t={'dtype':'BF16','shape':[2,64],'data_offsets':[0,256]}
        self.assertEqual(source.validate_tensor('x',t,100,356)['offset'],100)
        for changed in [dict(t,dtype='F32'),dict(t,shape=[2,True]),dict(t,data_offsets=[0,255]),dict(t,data_offsets=[1,257])]:
            with self.assertRaises(ValueError):source.validate_tensor('x',changed,100,356)
        with self.assertRaises(ValueError):source.read_json_bytes(b'{"x":1,"x":2}')
        for name in ['../weights','a/b','','..']:
            with self.assertRaises(ValueError):source.leaf(name)

class MtpQuantizationTest(unittest.TestCase):
    def test_exact_codes_scale_and_bias_in_native_word_order(self):
        for bits in (4,8):
            values=(np.arange(128,dtype=np.float32)%(1<<bits)).reshape(2,64)
            values=prep.rounded_bf16(values*.125-3)
            parts=prep.quantize(values,bits);decoded=prep.decode_reference(parts,2,64,bits)
            prep.check_quantization(values,parts,bits)
            self.assertEqual(len(parts[0]),128*bits//8)
            self.assertEqual(len(parts[1]),4);self.assertEqual(len(parts[2]),4)
            # Independent scalar unpacking of every word/lane and BF16 metadata.
            words=np.frombuffer(parts[0],'<u4');scales=np.frombuffer(parts[1],'<u2');biases=np.frombuffer(parts[2],'<u2')
            for i in range(128):
                q=(int(words[i//(32//bits)])>>((i%(32//bits))*bits))&((1<<bits)-1)
                s=np.array([int(scales[i//64])<<16],dtype='<u4').view('<f4')[0]
                b=np.array([int(biases[i//64])<<16],dtype='<u4').view('<f4')[0]
                self.assertEqual(decoded.flat[i],np.float32(q)*s+b)

    def test_constant_negative_tiny_extreme_and_irregular_row_blocks(self):
        rng=np.random.default_rng(41)
        cases=[np.zeros((3,64),np.float32),np.full((5,128),-2.5,np.float32),
            np.full((1,64),1e-38,np.float32),prep.rounded_bf16(rng.normal(size=(129,320)).astype(np.float32)),
            np.linspace(-1000,1000,64,dtype=np.float32)[None,:]]
        for value in cases:
            for bits in (4,8):
                full=prep.quantize(value,bits);prep.check_quantization(value,full,bits)
                blocks=[prep.quantize(value[i:i+128],bits) for i in range(0,len(value),128)]
                self.assertEqual(full,tuple(b''.join(block[j] for block in blocks) for j in range(3)))
        for value in [np.full((1,64),np.nan,np.float32),np.full((1,64),np.inf,np.float32),np.zeros((1,63),np.float32)]:
            with self.assertRaises(ValueError):prep.quantize(value,4)

    def test_error_bound_detects_reordered_columns_or_corrupt_metadata(self):
        value=prep.rounded_bf16(np.linspace(-4,4,128,dtype=np.float32).reshape(1,128))
        parts=prep.quantize(value,4);bad=(parts[0][::-1],parts[1],parts[2])
        with self.assertRaises(ValueError):prep.check_quantization(value,bad,4)

    def test_norms_routers_and_injection_are_not_silently_quantized(self):
        self.assertEqual(prep.precision('mtp.pre_fc_norm_hidden.weight',{'shape':[10240]}),'BF16')
        for name,shape in [('mtp.layers.0.mlp.gate.weight',[512,2560]),('mtp.layers.0.mlp.shared_expert_gate.weight',[1,2560]),
            ('mtp.layers.0.attn_hyper_connection.block_inject_weight.weight',[4,10240])]:
            self.assertEqual(prep.precision(name,{'shape':shape}),'BF16')
        self.assertEqual(prep.precision('mtp.fc_hidden.weight',{'shape':[2560,2560]}),'affine-Q8')
        with self.assertRaises(ValueError):prep.allocation_plan({'tensors':{}},9000)

class MtpPreparedValidationTest(unittest.TestCase):
    def test_same_sized_dense_tensors_cannot_exchange_bindings(self):
        evidence_fixture.require('docs/benchmarks/2026-09-15-mtp-preparation/native-01/manifest.json')
        saved=source.ROOT/'docs/benchmarks/2026-09-15-mtp-preparation/native-01'
        manifest=json.loads((saved/'manifest.json').read_text())
        inventory=json.loads((saved/'source-inventory.json').read_text())
        prep.validate_dense_layout(manifest,inventory)
        a=manifest['tensors']['mtp.fc_embedding.weight'];b=manifest['tensors']['mtp.fc_hidden.weight']
        self.assertEqual(a['shape'],b['shape']);a['parts'],b['parts']=b['parts'],a['parts']
        with self.assertRaisesRegex(ValueError,'canonical layout'):prep.validate_dense_layout(manifest,inventory)
        # Exercise the public audit before any payload opens, using saved metadata only.
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            for name in ('source-inventory.json','source-download.json'):(output/name).write_bytes((saved/name).read_bytes())
            (output/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'canonical layout'):prep.verify(output)

    def test_dense_dtype_is_part_of_the_binding(self):
        evidence_fixture.require('docs/benchmarks/2026-09-15-mtp-preparation/native-01/manifest.json')
        saved=source.ROOT/'docs/benchmarks/2026-09-15-mtp-preparation/native-01'
        manifest=json.loads((saved/'manifest.json').read_text())
        inventory=json.loads((saved/'source-inventory.json').read_text())
        manifest['tensors']['mtp.fc_hidden.weight']['parts']['weight']['dtype']='BF16'
        with self.assertRaisesRegex(ValueError,'canonical layout'):prep.validate_dense_layout(manifest,inventory)

    def test_memory_requires_complete_zero_compression_observations(self):
        good=dict(compressed_bytes=0,compressed_peak_bytes=0,decompressions=0,system_swap_used_bytes=100)
        self.assertTrue(clean_memory_observations([good.copy() for _ in range(3)]))
        self.assertFalse(clean_memory_observations([]))
        for key in good:
            for value in (None,True,-1,good[key]+1):
                samples=[good.copy() for _ in range(3)];samples[1][key]=value
                self.assertFalse(clean_memory_observations(samples),(key,value))
            samples=[good.copy() for _ in range(3)];del samples[1][key]
            self.assertFalse(clean_memory_observations(samples),key)

if __name__=='__main__':unittest.main()
