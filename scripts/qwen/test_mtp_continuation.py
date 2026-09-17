import copy
import unittest
from pathlib import Path

from screen_mtp_continuation import compare,observe,select_cases,workloads,ROOT


def sample(mode='fast-timing',eos=False):
    ids=[10,11,99] if eos else [10,11,12,13]
    memory=dict(physical_footprint_bytes=100,physical_footprint_peak_bytes=100,compressed_bytes=0,
        compressed_peak_bytes=0,decompressions=0,system_swap_used_bytes=0)
    host=dict(thermal_state=0,low_power_mode=False,power_source='AC Power')
    work=dict(prompt_ids=[7,8],eos_ids=[99],max_tokens=4)
    groups=[[i] for i in ids] if mode=='serial' else [ids+([14] if eos else [])]
    cycles=[];at=2
    for group in groups:
        keep=len(group)-(1 if eos and mode!='serial' else 0)
        cycles.append(dict(offset=at,width=len(group),proposals=group,committed_tokens=keep,
            accepted_proposals=keep-1,forced_rejection=False,wall_ns=100*keep,draft_ns=0,verify_ns=50*keep,
            recovery_ns=0,memory_before=copy.deepcopy(memory),memory_after=copy.deepcopy(memory),next_id=15))
        at+=keep
    raw=dict(kind='native_mtp_continuation_v1',complete=True,mode=mode,validation=False,input_sha256='input',
        draft_manifest_sha256='draft',admission={},prime_logits_sha256='prime',cycles=cycles,
        committed_token_ids=ids,generated_tokens=len(ids),requested_tokens=4,prompt_tokens=2,eos_ids=[99],
        proposed_tokens=sum(c['width']-1 for c in cycles),accepted_proposals=sum(c['accepted_proposals'] for c in cycles),
        decode_wall_ns=100*len(ids),tokens_per_second=1e7,decode_including_reporting_ns=100*len(ids)+1,
        row_logits_sha256=['a'*64 for _ in ids],stop_reason='eos' if eos else 'length',next_id=15,
        final_target_state=dict(valid=True,tokens=at,layers=[dict(position=at)]*48),
        final_draft_state=None if mode=='serial' else dict(valid=True,position=at-1),
        before_load=copy.deepcopy(memory),after_destroy=copy.deepcopy(memory),host_before=host,host_after=host,
        before=dict(process=copy.deepcopy(memory),expert_cache={'diagnostic_cache_state':'same'}),after=dict(process=copy.deepcopy(memory)))
    return raw,work


class ContinuationTests(unittest.TestCase):
    def test_case_selection_preserves_requested_work_and_order(self):
        rows=[dict(name='first',prompt_ids=[10,11]),dict(name='second',prompt_ids=[12,13])]
        self.assertIs(select_cases(rows,None),rows)
        selected=select_cases(rows,['second','first'])
        self.assertIs(selected[0],rows[1]);self.assertIs(selected[1],rows[0])
        self.assertEqual(select_cases(rows,['first']),[rows[0]])
        for names in ([],['missing'],['first','first']):
            with self.assertRaises(ValueError):select_cases(rows,names)

    def test_every_logit_and_committed_token_matches_serial(self):
        serial,work=sample('serial');draft,_=sample()
        for raw in (serial,draft):self.assertTrue(observe(raw,work,'input',raw['mode'])['clean_memory'])
        self.assertTrue(compare(serial,draft)['exact_all_logits_tokens_and_state'])
        draft['row_logits_sha256'][2]='b'*64
        with self.assertRaises(ValueError):compare(serial,draft)

    def test_eos_truncates_speculative_block_and_does_not_complete_length(self):
        serial,work=sample('serial',True);draft,_=sample(eos=True)
        for raw in (serial,draft):
            observed=observe(raw,work,'input',raw['mode'])
            self.assertEqual(observed['generated_tokens'],3)
            self.assertFalse(observed['completed_requested_length'])
        self.assertTrue(compare(serial,draft)['exact_all_logits_tokens_and_state'])

    def test_missing_rows_positions_or_timing_rejected(self):
        for mutation in ('rows','position','time','acceptance'):
            raw,work=sample()
            if mutation=='rows':raw['row_logits_sha256'].pop()
            elif mutation=='position':raw['cycles'][0]['offset']+=1
            elif mutation=='time':raw['cycles'][0]['recovery_ns']=1000
            else:raw['accepted_proposals']=0
            with self.assertRaises(ValueError):observe(raw,work,'input','fast-timing')

    def test_eos_cannot_be_ignored_or_invented(self):
        raw,work=sample();raw['eos_ids']=[13]
        with self.assertRaises(ValueError):observe(raw,work,'input','fast-timing')
        raw,work=sample();work['eos_ids']=[11];raw['eos_ids']=[11]
        with self.assertRaises(ValueError):observe(raw,work,'input','fast-timing')

    def test_disturbed_memory_stays_disturbed(self):
        raw,work=sample();raw['after_destroy']['compressed_peak_bytes']=16384
        self.assertFalse(observe(raw,work,'input','fast-timing')['clean_memory'])

    def test_prompt_suite_includes_a_panel_boundary(self):
        rows=workloads(ROOT/'.cache/qwen-mixed-reference',128)
        self.assertEqual(len(rows),3)
        self.assertEqual(len(rows[0]['prompt_ids']),72)
        self.assertTrue(any(len(r['prompt_ids'])>128 for r in rows))
        self.assertTrue(all(2<=len(r['prompt_ids'])<=512 and r['max_tokens']==128 for r in rows))


if __name__=='__main__':unittest.main()
