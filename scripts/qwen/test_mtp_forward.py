import copy
import unittest
import numpy as np
from reference_mtp_forward import dot
from screen_mtp_forward import clean,compare_joint
import build_mtp_forward as builder
from screen_mtp_recovery import equivalent
from verify_mtp_forward import recovery_sample


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
    def test_recovery_checks_proposals_and_private_state(self):
        a=dict(complete=True,input_sha256='a',admission={},draft_manifest_sha256='b',prime_logits_sha256='c',
            generated_tokens=1,final_target_state={'hash':'d'},boundaries=[{'draft_state':'e'}],decode_wall_ns=3,
            cycles=[dict(width=1,proposals=[9],accepted_proposals=0,committed_tokens=1,forced_rejection=False,next_id=10,wall_ns=3)])
        equivalent(a,copy.deepcopy(a))
        b=copy.deepcopy(a);b['cycles'][0]['proposals']=[12]
        with self.assertRaises(ValueError):equivalent(a,b)
        b=copy.deepcopy(a);b['boundaries']=[{'draft_state':'different'}]
        with self.assertRaises(ValueError):equivalent(a,b)

    @staticmethod
    def recovery_timing():
        memory=dict(physical_footprint_bytes=100,physical_footprint_peak_bytes=101,
            compressed_bytes=0,compressed_peak_bytes=0,decompressions=0,system_swap_used_bytes=0)
        return copy.deepcopy(dict(complete=True,mode='fast-timing',validation=False,input_sha256='input',
            draft_manifest_sha256='draft',admission={},generated_tokens=16,proposed_tokens=12,
            accepted_proposals=12,decode_wall_ns=4,tokens_per_second=4e9,
            before_load=memory,after_destroy=memory,before={'process':memory},after={'process':memory},
            host_before=dict(thermal_state=0,low_power_mode=False,power_source='AC Power'),
            host_after=dict(thermal_state=0,low_power_mode=False,power_source='AC Power'),
            cycles=[dict(width=4,proposals=[1,2,3,4],accepted_proposals=3,committed_tokens=4,
                forced_rejection=False,wall_ns=1,memory_before=dict(memory),memory_after=dict(memory)) for _ in range(4)]))

    def test_recovery_audit_keeps_peak_compression_after_release(self):
        raw=self.recovery_timing()
        self.assertTrue(recovery_sample(raw,'fast-timing','input','draft',{})['clean_memory'])
        # The final gauge can return to zero while a cycle's peak is nonzero.
        raw['cycles'][0]['memory_after']['compressed_peak_bytes']=64
        result=recovery_sample(raw,'fast-timing','input','draft',{})
        self.assertFalse(result['clean_memory'])
        self.assertEqual(result['max_observed_compressed_peak_bytes'],64)

    def test_recovery_audit_rejects_mode_and_coverage_changes(self):
        for changes in ({'validation':True},{'generated_tokens':15},{'proposed_tokens':11},
                        {'accepted_proposals':11},{'decode_wall_ns':3},{'input_sha256':'changed'}):
            with self.assertRaises(ValueError):
                recovery_sample(dict(self.recovery_timing(),**changes),'fast-timing','input','draft',{})

    def test_recovery_audit_rejects_forced_timing_and_missing_counters(self):
        raw=self.recovery_timing();raw['cycles'][0]['forced_rejection']=True
        with self.assertRaises(ValueError):recovery_sample(raw,'fast-timing','input','draft',{})
        raw=self.recovery_timing();del raw['cycles'][0]['memory_after']['decompressions']
        with self.assertRaises(ValueError):recovery_sample(raw,'fast-timing','input','draft',{})


if __name__=='__main__':unittest.main()
