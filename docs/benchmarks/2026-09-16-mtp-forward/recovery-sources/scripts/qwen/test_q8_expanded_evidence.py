import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from q8_expanded_fixtures import weight_ranges
from verify_q8_expanded import observe, KERNEL


class ExpandedEvidenceTest(unittest.TestCase):
    def test_native_scope_counts_and_prime_are_enforced(self):
        raw=dict(expanded_q8=True,host_before=dict(power_source='AC Power'),host_after=dict(power_source='AC Power'),
            before=dict(metal=dict(kernel_dispatches={})),after=dict(metal=dict(kernel_dispatches={KERNEL:532})))
        with patch('verify_q8_expanded.verifier.observe',return_value={}):
            self.assertEqual(observe(raw,{}, {},4,False,'hash',True)['packed_dispatches'],532)
            for count in (0,531,533):
                bad=copy.deepcopy(raw);bad['after']['metal']['kernel_dispatches'][KERNEL]=count
                with self.assertRaises(ValueError):observe(bad,{}, {},4,False,'hash',True)
            bad=copy.deepcopy(raw);bad['before']['metal']['kernel_dispatches'][KERNEL]=1
            bad['after']['metal']['kernel_dispatches'][KERNEL]=533
            with self.assertRaises(ValueError):observe(bad,{}, {},4,False,'hash',True)
            bad=copy.deepcopy(raw);bad['expanded_q8']=False
            with self.assertRaises(ValueError):observe(bad,{}, {},4,False,'hash',True)

    def tensor_source(self,directory,mutation=None):
        parts=[bytes(range(256))*2,bytes(range(16)),bytes(range(16,32))];header={};captured={};at=0
        for key,suffix,dtype,shape,blob in zip(('w','s','b'),('weight','scales','biases'),('U32','BF16','BF16'),
                ([8,16],[8,1],[8,1]),parts):
            header['language_model.test.'+suffix]=dict(dtype=dtype,shape=shape,data_offsets=[at,at+len(blob)]);at+=len(blob)
            captured[key]=dict(bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest())
        if mutation:mutation(header)
        raw=json.dumps(header).encode();name='model.safetensors';file=directory/name
        file.write_bytes(struct.pack('<Q',len(raw))+raw+b''.join(parts))
        reader=SimpleNamespace(model=directory,config=dict(quantization={'test':dict(bits=8,group_size=64)}),
            index={k:name for k in header},states={name:dict(size=file.stat().st_size)},verify_file=lambda _:None)
        return reader,captured

    def test_selected_ranges_are_rehashed(self):
        with tempfile.TemporaryDirectory() as d,patch('q8_expanded_fixtures.fcntl.fcntl'):
            reader,captured=self.tensor_source(Path(d));spec=dict(K=64,N=8,tensor='test')
            out=weight_ranges(reader,spec,captured);self.assertEqual(out['w']['sha256'],captured['w']['sha256'])
            file=reader.model/'model.safetensors'
            with file.open('r+b') as stream:stream.seek(out['w']['offset']);stream.write(b'\xff')
            with self.assertRaises(ValueError):weight_ranges(reader,spec,captured)

    def test_wrong_dtype_and_tensor_bounds_rejected(self):
        def dtype(h):h['language_model.test.weight']['dtype']='F32'
        def bounds(h):h['language_model.test.weight']['data_offsets']=[-1,511]
        for mutate in (dtype,bounds):
            with self.subTest(mutation=mutate.__name__),tempfile.TemporaryDirectory() as d,patch('q8_expanded_fixtures.fcntl.fcntl'):
                reader,captured=self.tensor_source(Path(d),mutate)
                with self.assertRaises(ValueError):weight_ranges(reader,dict(K=64,N=8,tensor='test'),captured)


if __name__=='__main__':unittest.main()
