import copy
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

from evidence_index import Index
from mtp_evidence import account, compare, next_experiment
from screen_mtp_widths import compare_widths, observed, validation_cases, reusable_validation
from target_recovery_checks import observe
from qualification_evidence import sha as file_digest
import evidence_fixture


def setUpModule():
    evidence_fixture.require(
        'docs/benchmarks/2026-09-15-q4-request-context/capture-02/evidence-files.json',
        'docs/benchmarks/2026-09-17-mtp-widths/long-lru-01/pair-0-width-4.json',
        'docs/benchmarks/2026-09-17-mtp-widths/numerical-01/case-0.json',
        'docs/benchmarks/2026-09-17-mtp-widths/numerical-01/producer.json',
        'docs/benchmarks/2026-09-17-mtp-widths/other-coding-01/case-0-pair-0-width-4.json',
        'docs/benchmarks/2026-09-17-mtp-widths/validation-03/case-7.json',
        'docs/benchmarks/2026-09-17-target-recovery/numerical-diagnostic-01/case-0-full-replay.json',
    )


ROOT=Path(__file__).resolve().parents[2]
REFERENCE=ROOT/'docs/benchmarks/2026-09-17-target-recovery/numerical-diagnostic-01'


def cycles(widths):
    raw=dict(kind='native_mtp_width_v1',complete=True,validation=False,mode='fast-timing',
             requested_width=max(widths),requested_tokens=sum(widths),prompt_tokens=72,
             eos_ids=[248046],request_id='fixture',cycles=[])
    offset=72
    for i,width in enumerate(widths):
        raw['cycles'].append(dict(cycle_id=i,request_id='fixture',offset=offset,width=width,
            committed_tokens=width,accepted_proposals=width-1,proposals=list(range(offset,offset+width)),
            wall_ns=100,draft_ns=20,verify_ns=40,recovery_ns=20,checkpoint_save_ns=5,
            target_restore_ns=0,target_repair_ns=0,draft_restore_ns=2,draft_catchup_ns=10,
            target_recovery_forward_calls=0,target_recovery_read_bytes=0))
        offset+=width
    count=offset-72
    raw.update(generated_tokens=count,committed_token_ids=list(range(72,offset)),
               decode_wall_ns=100*len(widths),proposed_tokens=count-len(widths),
               accepted_proposals=count-len(widths),tokens_per_second=count*1e9/(100*len(widths)),stop_reason='length')
    return raw


class WidthTests(unittest.TestCase):
    def test_irregular_tails_and_serial_width_account_all_costs(self):
        for shape in ([1]*7,[2,2,2,1],[4,1,1,1]):
            with self.subTest(shape=shape):
                result=account(cycles(shape))
                self.assertEqual(result['captured_committed_tokens'],7)
                self.assertTrue(result['checkpoint_save_measured'])
                self.assertEqual(result['recovery_part_ns']['draft_catchup'],10*len(shape))
                self.assertEqual(result['phase_ns']['other'],15*len(shape))

    def test_mislabelled_width_missing_phase_and_invalid_tail_fail(self):
        edits=[lambda r:r.update(requested_width=3),lambda r:r.update(requested_width=True),
               lambda r:r.update(requested_width=4),lambda r:r['cycles'][-1].update(width=2),
               lambda r:r['cycles'][0].pop('checkpoint_save_ns'),
               lambda r:r['cycles'][0].update(draft_catchup_ns=21)]
        for edit in edits:
            raw=cycles([2,2,2,1]);edit(raw)
            with self.assertRaises(ValueError):account(raw)

    def test_incomplete_width_is_not_timing_evidence(self):
        raw=cycles([2,2,2,1]);raw['complete']=False;raw['cycles']=raw['cycles'][:1]
        result=account(raw)
        self.assertFalse(result['complete']);self.assertIsNone(result['measured_tps'])

    def test_original_recovery_observer_does_not_accept_width_schema(self):
        raw=json.loads((REFERENCE/'case-0-full-replay.json').read_text())
        work=json.loads((REFERENCE/'case-0.json').read_text())
        raw.update(kind='native_mtp_width_v1',requested_width=4)
        result=observed(raw,work,raw['input_sha256'],True)
        self.assertEqual(result['requested_width'],4)
        self.assertEqual(json.loads(json.dumps(result)),result)
        with self.assertRaises(ValueError):observe(raw,work,raw['input_sha256'],True)

    def test_width_comparison_rejects_changed_identity_and_outputs(self):
        raw=json.loads((REFERENCE/'case-0-full-replay.json').read_text())
        raw.update(kind='native_mtp_width_v1',requested_width=4,validation=False,mode='fast-timing')
        self.assertTrue(compare_widths(raw,raw)['exact_all_logits_tokens_and_state'])
        edits=[lambda r:r.update(producer_binary_sha256='wrong'),
               lambda r:r['row_logits_sha256'].__setitem__(0,'wrong'),
               lambda r:r['final_draft_state'].update(keys='wrong'),
               lambda r:r['admission'].update(combined_bytes=1),
               lambda r:r['target_recovery_journal'].update(fixed_allocation_bytes=1)]
        for edit in edits:
            changed=copy.deepcopy(raw);edit(changed)
            with self.assertRaises(ValueError):compare_widths(raw,changed)

    def test_validation_inventory_has_every_prefix_eos_and_odd_tail(self):
        cases=validation_cases()
        normal=[(w,v['force_prefix']) for w,v in cases if v['name']=='irregular-seven']
        self.assertEqual(normal,[(1,1),(2,1),(2,2),(4,1),(4,2),(4,3),(4,4)])
        self.assertEqual([w for w,v in cases if v['name']=='immediate-eos'],[1,2,4])
        self.assertTrue(all(v['max_tokens']==7 for w,v in cases))

    def test_real_width_two_shape_metadata_is_not_a_policy_change(self):
        source=ROOT/'docs/benchmarks/2026-09-17-mtp-widths/numerical-01'
        a,b=[json.loads((source/f'case-{i}.json').read_text()) for i in range(2)]
        self.assertEqual(b['after']['metal']['kernels']['token_tile'],2)
        self.assertTrue(compare_widths(a,b)['exact_all_logits_tokens_and_state'])
        for key,value in [('token_tile',4),('q8_decode_rows',4),('profile',True)]:
            wrong=copy.deepcopy(b);wrong['after']['metal']['kernels'][key]=value
            with self.assertRaises(ValueError):compare_widths(a,wrong)

    def test_clean_native_validation_can_survive_python_checker_correction(self):
        source=ROOT/'docs/benchmarks/2026-09-17-mtp-widths/numerical-01'
        proof=json.loads((source/'producer.json').read_text())
        # Native compiler outputs are local, disposable files. Mock only their
        # hash reads; sealed reports and their inputs still use real bytes.
        native={p:h for key in ('inputs','generated','objects') for p,h in proof[key].items()}
        native[proof['binary']]=proof['binary_sha256']
        def digest(path):
            return native[str(path)] if str(path) in native else file_digest(path)
        with patch('screen_mtp_widths.sha',side_effect=digest):
            self.assertEqual(set(reusable_validation(source,proof)),{0,1})
            native[proof['binary']]='wrong'
            with self.assertRaisesRegex(ValueError,'native sources changed'):
                reusable_validation(source,proof)
        changed=copy.deepcopy(proof);changed['binary_sha256']='wrong'
        with self.assertRaises(ValueError):reusable_validation(source,changed)

    def test_eos_is_valid_numerically_but_not_a_fixed_length_speed_sample(self):
        source=ROOT/'docs/benchmarks/2026-09-17-mtp-widths/validation-03'
        raw=json.loads((source/'case-7.json').read_text())
        self.assertTrue(compare_widths(raw,raw)['exact_all_logits_tokens_and_state'])
        self.assertLess(raw['generated_tokens'],raw['requested_tokens'])
        raw.update(validation=False,mode='fast-timing')
        self.assertTrue(account(raw)['complete'])
        with self.assertRaisesRegex(ValueError,'requested output length'):
            compare_widths(raw,raw)

    def test_real_width_queries_keep_incomplete_screens_separate(self):
        base=ROOT/'docs/benchmarks/2026-09-17-mtp-widths'
        with tempfile.TemporaryDirectory() as d:
            index=Index(Path(d)/'index.sqlite')
            try:
                paths=[base/p/'summary.json' for p in
                       ('long-lru-01','screen-width-1-01','screen-width-2-02')]
                index.import_paths(paths)
                result=compare(index,str(paths[0]),'4','1',['requested_width'])
                self.assertTrue(result['comparable'])
                self.assertTrue(result['full_correctness_stage_passed'])
                self.assertFalse(result['advancement_allowed'])
                self.assertEqual(len(result['pairs']),2)
                self.assertIsNone(result['cases'][0]['confidence_95'])
                self.assertAlmostEqual(result['cases'][0]['geometric_mean_ratio'],.8222658499903038)
                with self.assertRaisesRegex(ValueError,'Unfinished'):
                    compare(index,str(paths[1]),'4','1',['requested_width'])
                next_width=next_experiment(index,str(paths[0]))
                self.assertNotIn('Clean full-model correctness',next_width['missing_evidence'])
                self.assertIn('remaining 128-token',next_width['smallest_experiment'])
                rejected=next_experiment(index,str(paths[2]))
                self.assertEqual(rejected['smallest_experiment'],'Do not repeat the unchanged rejected width.')
            finally:index.close()

    def test_completed_coding_screen_routes_back_to_target_profiling(self):
        path=ROOT/'docs/benchmarks/2026-09-17-mtp-widths/other-coding-01/summary.json'
        with tempfile.TemporaryDirectory() as d:
            index=Index(Path(d)/'index.sqlite')
            try:
                index.import_paths([path])
                result=next_experiment(index,str(path))
                cases=result['validated_comparison']['cases']
                self.assertEqual([c['case'] for c in cases],['merge_intervals','retry_backoff'])
                self.assertTrue(all(c['pairs']==2 and c['confidence_95'] is None for c in cases))
                self.assertIn('target profile',result['smallest_experiment'])
                self.assertNotIn('Clean full-model correctness',result['missing_evidence'])
                self.assertFalse(result['production_promoted'])
            finally:index.close()


if __name__=='__main__':unittest.main()
