import copy
import unittest
import numpy as np
from reference_mtp_forward import dot
from screen_mtp_forward import clean,compare_joint
import build_mtp_forward as builder


class MtpForwardTests(unittest.TestCase):
    def test_source_copy_bounds_and_target_arithmetic(self):
        sources=builder.generated(builder.ROOT/'.cache/test-mtp-generated')
        model=sources[builder.ROOT/'.cache/test-mtp-generated/model.cpp']
        self.assertEqual(model.count('capture_mtp_hidden(gpu_,h,T,state.tokens);'),1)
        self.assertIn('original compute_logits',__import__('build_perfect_draft').contracts()['all_row_logits_contract']['one_token'])
        shader=(builder.SCRIPTS/'mtp_draft.metal').read_text()
        self.assertIn('d<10240',shader)
        self.assertNotIn('rms_square_sum(',shader)
    def test_q4_correction_does_not_apply_to_q8(self):
        x=np.full((1,64),.00390625,np.float32);x[0,0]=1
        m=np.ones((2,64),np.float32);b=np.ones((2,1),np.float32)
        q8=dot(x,m,b,8,True);q4=dot(x,m,b,4,True)
        np.testing.assert_array_equal(q8,x@m.T)
        self.assertFalse(np.array_equal(q8,q4))
    def test_missing_memory_is_not_clean(self):
        memory=dict(physical_footprint_bytes=100,physical_footprint_peak_bytes=101,compressed_bytes=0,
            compressed_peak_bytes=0,decompressions=0,system_swap_used_bytes=0)
        host=dict(thermal_state=0,low_power_mode=False,power_source='AC Power')
        raw=dict(mode='fixture',before_load=memory,after_destroy=memory,after=memory,host_before=host,host_after=host)
        self.assertTrue(clean(raw)['clean_memory'])
        raw=copy.deepcopy(raw);raw['after']['compressed_peak_bytes']=1
        self.assertFalse(clean(raw)['clean_memory'])
        del raw['after']['decompressions']
        with self.assertRaises(ValueError):clean(raw)
    def test_incomplete_or_changed_comparison_rejected(self):
        row=dict(complete=True,mode='serial',input_sha256='a',draft_manifest_sha256='b',admission={'bytes':12},
            generated_tokens=16,final_target_state={'hash':'c'},decode_wall_ns=16,
            cycles=[dict(wall_ns=1,committed_tokens=1,forced_rejection=False) for _ in range(16)])
        candidate=copy.deepcopy(row);candidate['mode']='timing'
        self.assertEqual(compare_joint(row,candidate),1)
        for change in ({'complete':False},{'input_sha256':'different'},{'admission':{'bytes':13}},
                       {'final_target_state':{}},{'decode_wall_ns':15}):
            bad=dict(candidate,**change)
            with self.assertRaises(ValueError):compare_joint(row,bad)
        candidate['cycles'][0]['forced_rejection']=True
        with self.assertRaises(ValueError):compare_joint(row,candidate)


if __name__=='__main__':unittest.main()
