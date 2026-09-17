import io
import unittest
from unittest.mock import patch
import numpy as np
import mtp_source as source
import prepare_mtp as prep

class Response(io.BytesIO):
    def __init__(self,body,status=206,headers=None):
        super().__init__(body);self.status=status;self.headers=headers or {}

class MtpSourceTest(unittest.TestCase):
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

if __name__=='__main__':unittest.main()
