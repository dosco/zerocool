import unittest
from pathlib import Path

import build_mtp_direct_output as builder
from screen_mtp_direct_output import comparison,short_rejected
from test_mtp_ngram_init import lazy_pair


def pair():
    a,b=lazy_pair()
    for r,enabled in ((a,False),(b,True)):
        r['expert_scratch_after']=dict(enabled=False,forwards=0,groups=0)
        n=sum(c['width']==4 for c in r['cycles'])
        r['direct_output_before']=dict(enabled=enabled,active=False,forwards=0,eligible_calls=0,direct_writes=0,avoided_copy_bytes=0)
        r['direct_output_after']=dict(enabled=enabled,active=False,forwards=n,eligible_calls=20,
            direct_writes=20 if enabled else 0,avoided_copy_bytes=20*2560*4 if enabled else 0)
    b['after']['metal']['dispatches']-=20
    return a,b


class DirectOutputTests(unittest.TestCase):
    def test_exact_outputs_and_dispatch_accounting(self):
        a,b=pair();r=comparison(a,b)
        self.assertTrue(r['exact_all_logits_tokens_and_state'])
        self.assertEqual(r['direct_writes'],20)
        self.assertEqual(r['avoided_copy_bytes'],204800)

    def test_changed_results_scope_or_measurement_reject(self):
        for change in ('logits','draft','cache','budget','scratch','active','live','unused','extra_dispatch',
                       'coverage','copy_bytes','initial_switch','ngram','incomplete'):
            a,b=pair()
            if change=='logits':b['row_logits_sha256'][0]='b'*64
            elif change=='draft':b['final_draft_state']['position']+=1
            elif change=='cache':b['before']['expert_cache']['diagnostic_cache_state']='changed'
            elif change=='budget':a['admission']['combined_bytes']=b['admission']['combined_bytes']=13*1024**3
            elif change=='scratch':b['expert_scratch_after']['enabled']=True
            elif change=='active':b['direct_output_after']['active']=True
            elif change=='live':b['after']['metal']['live_command_groups']=1
            elif change=='unused':b['direct_output_after']['direct_writes']=0
            elif change=='extra_dispatch':b['after']['metal']['dispatches']-=1
            elif change=='coverage':b['direct_output_after']['forwards']+=1
            elif change=='copy_bytes':b['direct_output_after']['avoided_copy_bytes']+=1
            elif change=='initial_switch':b['direct_output_before']['enabled']=False
            elif change=='ngram':b['ngram_after_decode']['capacity_rows']+=1
            else:b['complete']=False
            with self.subTest(change=change),self.assertRaises(ValueError):comparison(a,b)

    def test_weak_or_reversed_screen_stops(self):
        for ratios in ([0.99],[1.05],[0.97,1.0],[0.979,0.99]):self.assertTrue(short_rejected(ratios))
        for ratios in ([0.98],[0.975,0.98]):self.assertFalse(short_rejected(ratios))
        for ratios in ([],[float('nan')],[0],[1,1,1]):
            with self.assertRaises(ValueError):short_rejected(ratios)

    def test_kernels_layout_and_previous_producer_unchanged(self):
        output=Path('/tmp/freellm-direct-render-test');s=builder.generated(output);old=builder.base.generated(output)
        changed={p.name for p in s if s[p]!=old[p]}
        self.assertEqual(changed,{'pipeline.cpp','model.cpp','probe.cpp'})
        self.assertIn('mtp_direct::Scope direct_scope(ids.size()==4 && phase_=="decode")',s[output/'model.cpp'])
        self.assertIn('mtp_direct::select(tokens,n)',s[output/'pipeline.cpp'])
        self.assertIn('requires scratch off',s[output/'probe.cpp'])


if __name__=='__main__':unittest.main()
