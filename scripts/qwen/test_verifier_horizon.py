import copy
import json
from pathlib import Path
import tempfile
import unittest

import build_verifier_horizon as builder
from screen_verifier_horizon import compare, prefix, validate

ROOT=builder.ROOT
BASE=ROOT/'docs/benchmarks/2026-09-17-mtp-widths/other-coding-01'


def fixture(width=8):
    raw=json.loads((BASE/'case-0-pair-0-width-4.json').read_text())
    work=json.loads((BASE/'case-0.json').read_text())
    work.update(continuation_ids=raw['committed_token_ids'],expected_next_ids=raw['committed_token_ids'][1:]+[raw['next_id']],
        row_logits_sha256=raw['row_logits_sha256'],prime_logits_sha256=raw['prime_logits_sha256'],final_target_state=raw['final_target_state'])
    memory=raw['before']['process'];count=len(work['continuation_ids'])
    raw.update(kind='native_verifier_horizon_v1',perfect_proposals_only=True,requested_width=width,
        initial_draft_state=raw['final_draft_state'],draft_after=raw['draft_before'],
        excluded_costs=['proposal_generation','rejection_recovery','draft_state_catchup'],
        cycles=[dict(offset=raw['prompt_tokens']+i,width=width,committed_tokens=width,wall_ns=1000,verify_ns=900,
            checkpoint_save_ns=50 if width>1 else 0,memory_before=memory,memory_after=memory) for i in range(0,count,width)])
    raw['admission']['checkpoint_rows']=8
    a=raw['admission'];a['host_checkpoint_logits_bytes']=128*1024**2+4*8*248320*4+1024**2
    a['combined_bytes']=a['target']['planned_bytes']+sum(a[k] for k in ('draft_bytes','host_checkpoint_logits_bytes','expert_scratch_reserve_bytes','target_recovery_reserve_bytes'))
    raw['decode_wall_ns']=sum(c['wall_ns'] for c in raw['cycles'])
    raw['verify_wall_ns']=sum(c['verify_ns'] for c in raw['cycles'])
    raw['verified_tokens_per_second']=count*1e9/raw['decode_wall_ns'];raw.pop('tokens_per_second')
    return raw,work


class HorizonTests(unittest.TestCase):
    def test_numerical_allowance_does_not_change_timing_cleanliness(self):
        from screen_horizon_early import diagnostic_resources
        raw,work=fixture();raw['after_destroy']['compressed_peak_bytes']=40*1024**2
        result=validate(raw,work,raw['input_sha256'],raw['producer_binary_sha256'],8,False)
        self.assertFalse(result['clean_memory'])
        diagnostic_resources(result,raw)
        raw['after_destroy']['compressed_peak_bytes']=129*1024**2
        with self.assertRaises(ValueError):diagnostic_resources(result,raw)
        raw['after_destroy']['compressed_peak_bytes']=0
        raw['after_destroy']['system_swap_used_bytes']+=1
        with self.assertRaises(ValueError):diagnostic_resources(result,raw)

    def test_ceiling_is_distinct_and_every_token_is_checked(self):
        raw,work=fixture()
        self.assertTrue(validate(raw,work,raw['input_sha256'],raw['producer_binary_sha256'],8,False)['perfect_proposals_only'])
        changes=[lambda r:r.update(kind='native_mtp_width_v1'),lambda r:r.update(complete=False),
            lambda r:r.update(tokens_per_second=5),lambda r:r.update(perfect_proposals_only=False),
            lambda r:r['row_logits_sha256'].__setitem__(0,'0'*64),lambda r:r['cycles'][0].update(width=4),
            lambda r:r['cycles'][0].update(offset=0),lambda r:r['admission'].update(draft_slots=10),
            lambda r:r.update(excluded_costs=[]),lambda r:r.update(verified_tokens_per_second=5),
            lambda r:r['after']['memory_plan'].update(expert_slots=1536)]
        for edit in changes:
            bad=copy.deepcopy(raw);edit(bad)
            with self.assertRaises(ValueError):validate(bad,work,raw['input_sha256'],raw['producer_binary_sha256'],8,False)
        with self.assertRaises(ValueError):validate(raw,work,raw['input_sha256'],raw['producer_binary_sha256'],True,False)

    def test_resources_are_observed_without_relabelling_bad_runs(self):
        raw,work=fixture();raw['after_destroy']['compressed_peak_bytes']=16384
        result=validate(raw,work,raw['input_sha256'],raw['producer_binary_sha256'],8,False)
        self.assertFalse(result['clean_memory'])
        self.assertFalse(result['normal_request_latency_qualified'])

    def test_streaming_is_an_explicit_fully_accounted_axis(self):
        raw,work=fixture();entry=dict(storage='exact-packed-rows',capacity=256,host_reserve_bytes=2*1024**2,
            fixed_host_bytes=700000,row_bytes=2720,removed_resident_allocation_bytes=675446784,hits=0,misses=0,evictions=0,application_read_bytes=0)
        raw['embedding_rows_before']=dict(entry);raw['embedding_rows_after']=dict(entry,hits=7,misses=1,application_read_bytes=2720)
        for plan in (raw['admission']['target'],raw['before']['memory_plan'],raw['after']['memory_plan']):
            plan['resident_bytes']=5362515968-675446784+2*1024**2
            plan['planned_bytes']=11809357824-675446784+2*1024**2
        raw['admission']['combined_bytes']-=675446784-2*1024**2
        args=(work,raw['input_sha256'],raw['producer_binary_sha256'],8,False)
        self.assertTrue(validate(raw,*args,streamed=True)['exact_full_logits'])
        with self.assertRaises(ValueError):validate(raw,*args)
        raw['embedding_rows_after']['application_read_bytes']=0
        with self.assertRaises(ValueError):validate(raw,*args,streamed=True)

    def test_comparison_requires_equal_allocations_state_and_starting_cache(self):
        a,_=fixture(4);b,_=fixture(8)
        self.assertTrue(compare(a,b)['exact_logits_and_state'])
        for edit in (lambda r:r['admission'].update(checkpoint_rows=4),lambda r:r['final_target_state'].update(tokens=1),
                     lambda r:r['before']['expert_cache'].update(diagnostic_cache_state='changed'),
                     lambda r:r.update(producer_binary_sha256='changed')):
            bad=copy.deepcopy(b);edit(bad)
            with self.assertRaises(ValueError):compare(a,bad)
        with self.assertRaises(ValueError):compare(a,b,numerical_across_producers=True)

    def test_prefix_drops_uncovered_final_state_and_keeps_reference(self):
        _,work=fixture();work['reference']={'sha256':'source'}
        short=prefix(work,9)
        self.assertNotIn('final_target_state',short)
        self.assertEqual(short['reference'],work['reference'])
        self.assertEqual(len(short['row_logits_sha256']),9)
        self.assertEqual(prefix(work,128)['final_target_state'],work['final_target_state'])
        for n in (0,-1,129,True):
            with self.assertRaises(ValueError):prefix(work,n)

    def test_eight_row_shader_preserves_original_independent_token_arithmetic(self):
        # Review the generated shader diff by shrinking only its token dimension.
        import build_q8_expanded as packed
        generated=builder.shader().replace('q8_horizon_t8_w8','q8_expanded_t4_w8')
        generated=generated.replace('eight tokens','four tokens').replace('p[2]!=8','p[2]!=4')
        generated=generated.replace('[8]={0,0,0,0,0,0,0,0}','[4]={0,0,0,0}').replace('t<8','t<4')
        self.assertEqual(generated,packed.shader())

    def test_source_copy_admits_eight_without_changing_earlier_builder(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory).resolve();sources=builder.generated(out)
            self.assertIn('tokens==8?"q8_horizon_t8_w8":"q8_expanded_t4_w8"',sources[out/'metal.mm'])
            self.assertIn('ids.size()!=4 && ids.size()!=8',sources[out/'model.cpp'])
            self.assertIn('T!=4 && T!=8',sources[out/'model.cpp'])
            self.assertIn('(tokens!=4 && tokens!=8)',sources[out/'include/mtp_direct_output.hpp'])
            self.assertIn('capacity=f>=2 && f<=4?8ull*',sources[out/'probe.cpp'])
            joint=sources[out/'probe.cpp'].split('void joint(',1)[1].split('\n}\n\n}',1)[0]
            measured=joint.split('while(consumed<count)',1)[1]
            self.assertNotIn('draft.forward(',measured)
            self.assertNotIn('journal.apply(',measured)
            self.assertIn('checkpoint.save(state,width)',measured)
            self.assertIn('h==hashes[consumed+i]',measured)

    def test_streamed_builder_binds_target_and_draft_to_same_provider(self):
        import build_streamed_verifier_horizon as streamed
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory).resolve();s=streamed.generated(out)
            self.assertIn('embedding_rows::planned_resident(checkpoint_,resident_bytes)',s[out/'model.cpp'])
            metal=s[out/'metal.mm']
            self.assertIn('!embedding_rows::enabled || !key.starts_with("model.embed_tokens.")',metal)
            self.assertIn('return embedding_rows::store->descriptor()',metal)
            self.assertIn('return embedding_rows::store->gather(*this,l,ids,copies)',metal)
            self.assertEqual(s[out/'probe.cpp'].count('embedding_rows::Scope streamed_scope;'),1)
            cfg=streamed.settings(out)
            self.assertIn(streamed.HEADER,streamed.inputs(cfg));self.assertIn(streamed.TEST,streamed.inputs(cfg))

    def test_tile_four_keeps_eight_row_storage_and_complete_two_slice_dispatch(self):
        import build_tiled_verifier_horizon as tiled
        s=tiled.generated(Path('/tmp/freellm-tiled-check'))
        self.assertIn('t0=gid.y*4',s[Path('/tmp/freellm-tiled-check/metal.mm').resolve()])
        self.assertIn('x[(t0+t)*K+base+i]',tiled.shader())
        self.assertIn('out[(t0+t)*N+row]',tiled.shader())
        self.assertIn('l.output*32,tokens/4);',s[Path('/tmp/freellm-tiled-check/metal.mm').resolve()])
        self.assertIn('std::min(4u,uint32_t(ids.size()))',s[Path('/tmp/freellm-tiled-check/model.cpp').resolve()])


if __name__=='__main__':unittest.main()
