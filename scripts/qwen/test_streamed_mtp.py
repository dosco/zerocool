import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import build_streamed_mtp as builder
from screen_streamed_mtp import SAVING, compare, observe, cases, sample, diagnostic_resources, checked_diagnostic, main, reusable_samples
from qualification_evidence import ResourceBlocked
import evidence_fixture


def setUpModule():
    evidence_fixture.require(
        'docs/benchmarks/2026-09-15-q4-request-context/capture-02/evidence-files.json',
        'docs/benchmarks/2026-09-17-mtp-widths/other-coding-01/case-0-pair-0-width-4.json',
        'docs/benchmarks/2026-09-18-streamed-mtp/numerical-02/producer.json',
    )


SOURCE = builder.ROOT/'docs/benchmarks/2026-09-17-mtp-widths/other-coding-01'


def fixture():
    a = json.loads((SOURCE/'case-0-pair-0-width-4.json').read_text())
    work = json.loads((SOURCE/'case-0.json').read_text())
    a.update(embedding_storage='resident', embedding_rows_before=None, embedding_rows_after=None,
             embedding_owner_released=True)
    for c in a['cycles']:
        c.update(unforced_proposals=list(c['proposals']), draft_row_logits_sha256=[], verified_row_logits_sha256=[])
    b = copy.deepcopy(a)
    b['embedding_storage'] = 'rows'
    b['admission']['combined_bytes'] -= SAVING
    for plan in (b['admission']['target'], b['before']['memory_plan'], b['after']['memory_plan']):
        for key in ('resident_bytes', 'planned_bytes'): plan[key] -= SAVING
    row = dict(storage='exact-packed-rows', capacity=256, host_reserve_bytes=2*1024**2, fixed_host_bytes=700000,
        row_bytes=2720, removed_resident_allocation_bytes=675446784, hits=0, misses=1, evictions=0, application_read_bytes=2720)
    b['embedding_rows_before'] = dict(row)
    b['embedding_rows_after'] = dict(row, hits=10, misses=2, application_read_bytes=5440)
    return a, b, work


class StreamedMtpTests(unittest.TestCase):
    def test_exact_storage_is_the_only_comparison_axis(self):
        a, b, work = fixture()
        for arm, raw in (('resident', a), ('rows', b)):
            r = observe(raw, work, raw['input_sha256'], raw['producer_binary_sha256'], 4, arm, False)
            self.assertTrue(r['clean_memory'])
        self.assertEqual(compare(a, b)['planned_saving_bytes'], SAVING)
        for edit in (lambda r: r.update(producer_binary_sha256='other'),
                     lambda r: r.update(requested_width=1),
                     lambda r: r['admission']['target'].update(expert_slots=1536),
                     lambda r: r['cycles'][0]['unforced_proposals'].__setitem__(1, 42),
                     lambda r: r['final_draft_state'].update(keys='0'*64),
                     lambda r: r['row_logits_sha256'].__setitem__(0, '0'*64)):
            bad = copy.deepcopy(b)
            edit(bad)
            with self.assertRaises(ValueError): compare(a, bad)

    def test_missing_reads_owner_or_memory_accounting_fails(self):
        a, b, work = fixture()
        for edit in (lambda r: r.update(embedding_owner_released=False),
                     lambda r: r['embedding_rows_after'].update(application_read_bytes=2720),
                     lambda r: r['embedding_rows_after'].update(capacity=512),
                     lambda r: r['admission'].update(combined_bytes=1),
                     lambda r: r['cycles'][0].update(unforced_proposals=[])):
            bad = copy.deepcopy(b)
            edit(bad)
            with self.assertRaises(ValueError):
                observe(bad, work, b['input_sha256'], b['producer_binary_sha256'], 4, 'rows', False)
        a['embedding_rows_before'] = b['embedding_rows_before']
        with self.assertRaises(ValueError):
            observe(a, work, a['input_sha256'], a['producer_binary_sha256'], 4, 'resident', False)

    def test_compression_remains_unqualified(self):
        _, b, work = fixture()
        b['after_destroy']['compressed_peak_bytes'] = 16384
        r = observe(b, work, b['input_sha256'], b['producer_binary_sha256'], 4, 'rows', False)
        self.assertFalse(r['clean_memory'])
        diagnostic_resources(r, b)
        b['after_destroy']['compressed_peak_bytes'] = 129*1024**2
        with self.assertRaises(ValueError): diagnostic_resources(r, b)
        with self.assertRaises(ValueError):
            sample(None, None, None, None, 4, 'rows', work, 'never-run', False, diagnostic=True)

    def test_completion_counters_can_change_but_starting_cache_cannot(self):
        a, b, _ = fixture()
        b['after']['expert_cache']['ready_hits'] -= 1
        b['after']['expert_cache']['loading_joins'] += 1
        self.assertTrue(compare(a, b)['exact_proposals_logits_and_state'])
        b['before']['expert_cache']['diagnostic_cache_state'] = 'changed'
        with self.assertRaises(ValueError): compare(a, b)

    def test_thermal_stop_is_a_resource_block_not_a_numerical_failure(self):
        _, b, work = fixture()
        b['host_after']['thermal_state'] = 1
        result = observe(b, work, b['input_sha256'], b['producer_binary_sha256'], 4, 'rows', False)
        with self.assertRaisesRegex(ResourceBlocked, 'thermal=1'):
            checked_diagnostic(result, b)

    def test_incomplete_cli_trial_cannot_report_success(self):
        args = ['fixture', '--build', '/unused', '--output', '/unused-output']
        for complete, expected in ((False, 2), (True, 0)):
            with patch('screen_streamed_mtp.fixture', return_value={'complete': complete}):
                self.assertEqual(main(args), expected)

    def test_resume_rechecks_real_samples_and_never_reuses_a_disturbed_process(self):
        source = builder.ROOT/'docs/benchmarks/2026-09-18-streamed-mtp/numerical-02'
        proof = json.loads((source/'producer.json').read_text())
        self.assertEqual(set(reusable_samples(source, proof, False)), {(0, 'rows'), (1, 'rows')})
        self.assertEqual(set(reusable_samples(source, proof, True)),
                         {(0, 'resident'), (0, 'rows'), (1, 'resident'), (1, 'rows')})
        changed = dict(proof, binary_sha256='0'*64)
        with self.assertRaises(ValueError): reusable_samples(source, changed, True)

    def test_validation_compares_rejected_rows_and_primed_state(self):
        a, b, _ = fixture()
        for r in (a, b):
            r.update(validation=True, mode='fast-validate', initial_target_state={'tokens': 71}, initial_draft_state={'position': 70})
            for c in r['cycles']:
                c['draft_row_logits_sha256'] = ['a'*64]*(c['width']-1)
                c['verified_row_logits_sha256'] = ['b'*64]*c['width']
        self.assertFalse(compare(a, b)['timing_used'])
        b['draft_before']['expert_cache']['diagnostic_cache_state'] = 'different eviction order'
        numerical = compare(a, b)
        self.assertTrue(numerical['exact_proposals_logits_and_state'])
        self.assertEqual(numerical['initial_cache_equal'], {'before': True, 'draft_before': False})
        for edit in (lambda r: r['initial_draft_state'].update(position=71),
                     lambda r: r['cycles'][0]['draft_row_logits_sha256'].__setitem__(-1, 'c'*64),
                     lambda r: r['cycles'][0]['verified_row_logits_sha256'].__setitem__(-1, 'c'*64)):
            bad = copy.deepcopy(b)
            edit(bad)
            with self.assertRaises(ValueError): compare(a, bad)

    def test_same_binary_has_explicit_storage_and_retains_four_row_verifier(self):
        out = Path('/tmp/zerocool-streamed-mtp-test').resolve()
        sources = builder.generated(out)
        probe = sources[out/'probe.cpp']
        self.assertIn('ZEROCOOL_MTP_EMBEDDINGS', probe)
        self.assertIn('requested_width==1 || requested_width==4', probe)
        self.assertLess(probe.index('std::unique_ptr<embedding_rows::Scope> embedding_owner'),
                        probe.index('phase(output,"load_target")'))
        self.assertIn('draft_logits.push_back(row_hash(out.logits->floats()))', probe)
        self.assertIn('embedding_owner_released', probe)
        self.assertNotIn('q8_horizon', sources[out/'metal.mm'])
        self.assertIn('embedding_rows::planned_resident', sources[out/'model.cpp'])
        self.assertIn('return embedding_rows::store->gather(*this,l,ids,copies)', sources[out/'metal.mm'])

    def test_cases_cover_serial_all_rejections_tail_and_eos(self):
        rows = cases()
        self.assertEqual(len(rows), 6)
        self.assertEqual([w['force_prefix'] for width, w in rows[1:5]], [1, 2, 3, 4])
        self.assertEqual(rows[0][0], 1)
        self.assertTrue(all(w['max_tokens'] == 7 for _, w in rows))
        self.assertIn(rows[-1][1]['expected_prompt_id'], rows[-1][1]['eos_ids'])

    def test_fixed_priming_is_explicit_bounded_and_ends_before_generation(self):
        import build_streamed_mtp_fixed_priming as fixed
        a, b, work = fixture()
        for r in (a, b):
            r['draft_priming_policy'] = fixed.POLICY
            r['draft_priming_before'] = dict(policy=fixed.POLICY, active=False, calls=5, batches=12, experts=360, peak_leases=32)
            r['draft_priming_after'] = dict(r['draft_priming_before'])
        observe(b, work, b['input_sha256'], b['producer_binary_sha256'], 4, 'rows', False, fixed.POLICY)
        self.assertTrue(compare(a, b)['exact_proposals_logits_and_state'])
        for change in ('missing', 'active', 'decode', 'capacity'):
            bad = copy.deepcopy(b)
            if change == 'missing': bad.pop('draft_priming_policy')
            elif change == 'active': bad['draft_priming_before']['active'] = True
            elif change == 'decode': bad['draft_priming_after']['calls'] += 1
            else: bad['draft_priming_before']['peak_leases'] = 33
            with self.subTest(change=change), self.assertRaises(ValueError):
                observe(bad, work, bad['input_sha256'], bad['producer_binary_sha256'], 4, 'rows', False, fixed.POLICY)
        out = Path('/tmp/zerocool-fixed-priming-test').resolve()
        sources = fixed.generated(out);probe = sources[out/'probe.cpp']
        self.assertIn('mtp_fixed_priming::execute(selected,*cache_', sources[out/'mtp_draft.cpp'])
        self.assertLess(probe.index('mtp_fixed_priming::Scope fixed_draft_priming'), probe.index('report["draft_priming_before"]'))
        self.assertLess(probe.index('report["draft_priming_before"]'), probe.index('const auto request_start=monotonic_ns()'))


if __name__ == '__main__': unittest.main()
