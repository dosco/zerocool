"""Evidence-query tests use transformed old reports, never performance evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evidence_index import Index
from mtp_evidence import compare, cycles, next_experiment
from qualification_evidence import save, seal, sha
from screen_streamed_mtp import SAVING, compare as compare_pair, observe
from streamed_mtp_evidence import labels

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'docs/benchmarks/2026-09-17-mtp-widths/other-coding-01'


def fixture(directory, edit=None):
    parent = directory/'numerical';parent.mkdir();seal(parent)
    root = directory/'screen';root.mkdir()
    work = json.loads((SOURCE/'case-0.json').read_text())
    summary = dict(kind='streamed_mtp_screen_v1', complete=True, status='storage_cost_measured',
        preliminary=True, advancement_allowed=False, production_promoted=False, full_clean_correctness=False,
        numerical_validation=dict(path=str(parent), seal_sha256=sha(parent/'evidence-files.json')),
        samples=[], pairs=[], by_width={})
    producer = None
    for width in (4, 1):
        original = json.loads((SOURCE/f'case-0-pair-0-width-{width}.json').read_text())
        if producer is None:
            producer = dict(kind='streamed_mtp_producer_v1', complete=True, binary='/unused/unit-test-producer',
                binary_sha256=original['producer_binary_sha256'],
                base_native_fingerprint=original['before']['metal']['build_fingerprint'])
        for pair in (0, 1):
            values = {}
            for arm in (('resident', 'rows') if pair == 0 else ('rows', 'resident')):
                raw = copy.deepcopy(original)
                stem = f'width-{width}-pair-{pair}-{arm}'
                inp = root/(stem+'.input.json');save(inp, work)
                raw.update(input_sha256=sha(inp), request_id=sha(inp)+':'+stem,
                    embedding_storage=arm, embedding_owner_released=True,
                    embedding_rows_before=None, embedding_rows_after=None)
                for cycle in raw['cycles']:
                    cycle.update(request_id=raw['request_id'], unforced_proposals=list(cycle['proposals']),
                        draft_row_logits_sha256=[], verified_row_logits_sha256=[])
                if arm == 'rows':
                    raw['admission']['combined_bytes'] -= SAVING
                    for plan in (raw['admission']['target'], raw['before']['memory_plan'], raw['after']['memory_plan']):
                        for key in ('resident_bytes', 'planned_bytes'): plan[key] -= SAVING
                    row = dict(storage='exact-packed-rows', capacity=256, host_reserve_bytes=2*1024**2,
                        fixed_host_bytes=697656, row_bytes=2720, removed_resident_allocation_bytes=675446784,
                        hits=0, misses=1, evictions=0, application_read_bytes=2720)
                    raw.update(embedding_rows_before=dict(row), embedding_rows_after=dict(row))
                observed = observe(raw, work, sha(inp), producer['binary_sha256'], width, arm, False)
                if edit: edit(raw, width, pair, arm)
                path = root/(stem+'.json');save(path, raw)
                summary['samples'].append(dict(source=path.name, input=inp.name, sha256=sha(path), width=width, **observed))
                values[arm] = raw
            # Expected claims remain the original unmodified fixture's values.
            try: result = compare_pair(values['resident'], values['rows'])
            except (ValueError, KeyError): result = dict(latency_ratio=1, physical_peak_saved_bytes=0)
            summary['pairs'].append(dict(width=width, pair=pair, **result))
        ratios = [p['latency_ratio'] for p in summary['pairs'] if p['width'] == width]
        summary['by_width'][str(width)] = dict(pairs=2, geometric_latency_ratio=(ratios[0]*ratios[1])**.5,
            measured_peak_savings_bytes=[p['physical_peak_saved_bytes'] for p in summary['pairs'] if p['width'] == width])
    save(root/'producer.json', producer)
    save(root/'identity.json', dict(files={producer['binary']: producer['binary_sha256']}))
    save(root/'summary.json', summary);seal(root)
    return root


class StreamedEvidenceTests(unittest.TestCase):
    def test_recomputes_pairs_without_pooling_widths_or_promoting_diagnostics(self):
        with tempfile.TemporaryDirectory() as d, patch('streamed_mtp_evidence.prerequisite', return_value=([], False)):
            root = fixture(Path(d));index = Index(Path(d)/'index.sqlite');index.import_paths([root])
            try:
                result = compare(index, str(root/'summary.json'), 'resident', 'rows', ['embedding_storage'])
                self.assertEqual([c['case'] for c in result['cases']], ['width-4', 'width-1'])
                self.assertTrue(all(c['confidence_95'] is None for c in result['cases']))
                self.assertFalse(result['full_correctness_stage_passed'])
                self.assertFalse(result['production_promoted'])
                selected = compare(index, str(root/'summary.json'), 'resident', 'rows', ['embedding_storage'], 'width-1')
                self.assertEqual(len(selected['pairs']), 2)
                self.assertEqual({r['width'] for r in cycles(index, str(root/'summary.json'))['runs']}, {1, 4})
                self.assertIn('cache leases', next_experiment(index, str(root/'summary.json'))['smallest_experiment'])
                with self.assertRaises(ValueError):
                    compare(index, str(root/'summary.json'), 'resident', 'rows', ['requested_width'])
            finally: index.close()

    def test_rejects_changed_payload_resources_and_unaccounted_memory(self):
        for mode in ('logits', 'compression', 'power', 'memory', 'producer'):
            def edit(raw, width, pair, arm):
                if (width, pair, arm) != (4, 0, 'rows'): return
                if mode == 'logits': raw['row_logits_sha256'][0] = '0'*64
                elif mode == 'compression': raw['after_destroy']['compressed_peak_bytes'] = 16384
                elif mode == 'power': raw['host_after']['low_power_mode'] = True
                elif mode == 'memory': raw['admission']['combined_bytes'] += 16384
                else: raw['producer_binary_sha256'] = '0'*64
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d, \
                    patch('streamed_mtp_evidence.prerequisite', return_value=([], False)):
                root = fixture(Path(d), edit);index = Index(Path(d)/'index.sqlite');index.import_paths([root])
                try:
                    with self.assertRaises(ValueError):
                        compare(index, str(root/'summary.json'), 'resident', 'rows', ['embedding_storage'])
                finally: index.close()

    def test_rejects_forged_summary_and_missing_numerical_evidence(self):
        for mode in ('ratio', 'duplicate', 'order', 'numerical', 'qualification'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d, \
                    patch('streamed_mtp_evidence.prerequisite', return_value=([], False)):
                root = fixture(Path(d));path = root/'summary.json';summary = json.loads(path.read_text())
                if mode == 'ratio': summary['pairs'][0]['latency_ratio'] = .01
                elif mode == 'duplicate': summary['samples'].append(summary['samples'][0])
                elif mode == 'order': summary['samples'][:2] = summary['samples'][:2][::-1]
                elif mode == 'numerical': summary['numerical_validation']['seal_sha256'] = '0'*64
                else: summary['full_clean_correctness'] = True
                save(path, summary);seal(root)
                index = Index(Path(d)/'index.sqlite');index.import_paths([root])
                try:
                    with self.assertRaises(ValueError):
                        compare(index, str(path), 'resident', 'rows', ['embedding_storage'])
                finally: index.close()

    def test_incompatible_sample_labels_are_rejected(self):
        good = dict(source='width-4-pair-0-rows.json', width=4, embedding_storage='rows')
        self.assertEqual(labels(good)['arm'], 'rows')
        for bad in (dict(good, width=1), dict(good, source='../width-4-pair-0-rows.json'),
                    dict(good, embedding_storage='resident')):
            with self.assertRaises(ValueError): labels(bad)


if __name__ == '__main__': unittest.main()
