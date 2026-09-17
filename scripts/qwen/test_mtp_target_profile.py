import unittest
from collections import Counter

from capture_mtp_target_profile import analyze_block
from test_block_compute import fixture


class TargetProfileTests(unittest.TestCase):
    def example(self):
        block,profile=fixture();counts=Counter(o['kernel'] for g in profile['command_groups'] for o in g['operations'])
        block['target_before']=dict(kernel_dispatches={},dispatches=0,submissions=0)
        block['target_after']=dict(kernel_dispatches=dict(counts),dispatches=sum(counts.values()),submissions=len(profile['command_groups']))
        return block,profile

    def test_target_population_excludes_other_model_work(self):
        block,profile=self.example();result=analyze_block(block,profile,'b','r')
        self.assertEqual(result['dispatches'],528)
        # Unprofiled draft calls still contribute to cumulative GPU counters.
        # Only the delta around this target call may be used for reconciliation.
        block['target_before']['dispatches']+=17;block['target_after']['dispatches']+=17
        block['target_before']['kernel_dispatches']['draft_only']=17
        block['target_after']['kernel_dispatches']['draft_only']=17
        self.assertEqual(result,analyze_block(block,profile,'b','r'))

    def test_missing_captured_target_work_is_rejected(self):
        block,profile=self.example();block['target_after']['kernel_dispatches']['unrecorded']=1
        with self.assertRaises(ValueError):analyze_block(block,profile,'b','r')

    def test_previous_blocks_hidden_copy_cannot_enter_next_capture(self):
        block,profile=self.example();block['forward_begin_ns']=500;block['forward_ns']=49500
        profile['command_groups'].insert(0,dict(submitted_ns=10,completed_ns=40,
            gpu_start_seconds=20/1e9,gpu_end_seconds=30/1e9,
            operations=[dict(request_phase='decode',offset=68,layer=-1,tokens=4,
                stage='logits',kernel='copy_words')]))
        with self.assertRaisesRegex(ValueError,'out-of-block'):
            analyze_block(block,profile,'b','r')


if __name__=='__main__':unittest.main()
