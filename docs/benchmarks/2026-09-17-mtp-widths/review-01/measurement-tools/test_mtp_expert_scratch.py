import copy
import unittest
from pathlib import Path

import build_mtp_expert_scratch as builder
from screen_mtp_expert_scratch import comparison
from test_mtp_continuation import sample


def pair():
    a,_=sample();b=copy.deepcopy(a)
    for r,enabled in ((a,False),(b,True)):
        r['admission'].update(expert_scratch_reserve_bytes=2*1024**2,combined_bytes=11*1024**3)
        r['expert_scratch_after']=dict(enabled=enabled,forwards=1 if enabled else 0,groups=100 if enabled else 0)
        r['before']['metal']=dict(allocation_count=100,scratch_reuses=0,dispatches=50,submissions=10)
        r['after']['metal']=dict(allocation_count=200,scratch_reuses=100 if enabled else 0,dispatches=100,submissions=20,
            active_scratch_slot=-1,live_command_groups=0)
    return a,b


class ScratchTests(unittest.TestCase):
    def test_exact_comparison_and_real_counter_deltas(self):
        a,b=pair();r=comparison(a,b)
        self.assertTrue(r['exact_all_logits_tokens_and_state'])
        self.assertEqual(r['candidate_counters']['scratch_reuses'],100)
        self.assertEqual(r['control_counters']['dispatches'],50)

    def test_changed_results_admission_or_live_users_reject(self):
        for change in ('logits','draft','cache','budget','mode','unused','active','live','incomplete'):
            a,b=pair()
            if change=='logits':b['row_logits_sha256'][0]='b'*64
            elif change=='draft':b['final_draft_state']['position']+=1
            elif change=='cache':b['before']['expert_cache']['diagnostic_cache_state']='different'
            elif change=='budget':a['admission']['combined_bytes']=b['admission']['combined_bytes']=13*1024**3
            elif change=='mode':b['mode']='serial'
            elif change=='unused':b['expert_scratch_after']['forwards']=0
            elif change=='active':b['after']['metal']['active_scratch_slot']=0
            elif change=='live':b['after']['metal']['live_command_groups']=1
            else:b['complete']=False
            with self.subTest(change=change),self.assertRaises(ValueError):comparison(a,b)

    def test_source_copy_preserves_public_layout_and_submission_ownership(self):
        output=Path('/tmp/freellm-scratch-render-test')
        sources=builder.generated(output);baseline=builder.base.generated(output)
        self.assertEqual(sources[output/'include/qwen/model.hpp'],baseline[output/'include/qwen/model.hpp'])
        pipeline=sources[output/'pipeline.cpp']
        self.assertIn('group.completion=gpu.submit();\n                // End only after submit:',pipeline)
        self.assertIn('if(pooled) {try {gpu.release_scratch();}',pipeline)
        self.assertIn('ids.size()==4 && phase_=="decode"',sources[output/'model.cpp'])


if __name__=='__main__':unittest.main()
