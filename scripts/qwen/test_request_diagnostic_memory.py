import unittest

from request_diagnostic_memory import analyze_memory, GIB, MIB
from test_decode_startup import fixture as decode_fixture


def fixture():
    def state():
        return dict(sample_ns=0, process=dict(physical_footprint_bytes=8*GIB,
            physical_footprint_peak_bytes=9*GIB, compressed_bytes=0,
            compressed_peak_bytes=0, decompressions=0, system_swap_used_bytes=GIB),
            metal=dict(live_command_groups=0), expert_cache={}, expert_dependencies={})
    raw = dict(complete=True, runs=[])
    for phase, offset in enumerate((72, 217)):
        row = decode_fixture(16)
        row.update(name=('initial', 'append')[phase], prompt_tokens=offset, output_tokens=17,
            before=state(), after=state(), phases={key: dict(before=state(), after=state())
                                                 for key in ('ingest', 'decode')})
        for step, sample in enumerate(row['decode_diagnostics']['samples']):
            sample.update(offset=offset+step, before=state(), after=state())
            sample['begin_ns'] += phase*100000000
            sample['end_ns'] += phase*100000000
        raw['runs'].append(row)
    return raw


def chronological(raw):
    for row in raw['runs']:
        yield row['before']['process']
        yield row['phases']['ingest']['before']['process']
        yield row['phases']['ingest']['after']['process']
        yield row['phases']['decode']['before']['process']
        for sample in row['decode_diagnostics']['samples']:
            yield sample['before']['process']
            yield sample['after']['process']
        yield row['phases']['decode']['after']['process']
        yield row['after']['process']


class DiagnosticMemoryTest(unittest.TestCase):
    def test_complete_clean_capture_stays_diagnostic(self):
        result = analyze_memory(fixture())
        self.assertTrue(result['hard_limits_passed'])
        self.assertTrue(result['strict_memory_clean'])
        self.assertEqual(result['coverage']['decode_forwards'], 32)
        self.assertEqual(result['coverage']['observations'], 76)
        self.assertEqual([len(phase['tokens']) for phase in result['phases']], [16, 16])
        self.assertFalse(result['normal_request_latency_qualified'])
        self.assertFalse(result['production_promoted'])

    def test_compression_and_decompression_gaps_are_retained_without_double_count(self):
        raw = fixture()
        for index, process in enumerate(chronological(raw)):
            process.update(decompressions=100+index, compressed_peak_bytes=128*MIB,
                           compressed_bytes=(2 if index%2 else 1)*MIB)
        result = analyze_memory(raw)
        self.assertTrue(result['diagnostic_usable'])
        self.assertFalse(result['strict_memory_clean'])
        self.assertEqual(result['observed_decompressions'], 75)
        self.assertEqual(result['between_requests_process_change']['decompressions'], 1)
        for phase in result['phases']:
            self.assertEqual(phase['decompressions'], dict(ingest=1, decode=33,
                inside_token_forwards=16, outside_token_forwards=17,
                outside_ingest_and_decode=3, request=37))
            self.assertEqual(len(phase['decode_gaps']), 17)
        changes = [o['change_since_previous']['compressed_bytes']
                   for o in result['chronological_observations'][1:]]
        self.assertIn(-MIB, changes)
        self.assertIn(MIB, changes)

    def test_lifetime_compression_peak_includes_before_first_token(self):
        raw = fixture()
        for process in chronological(raw):
            process['compressed_peak_bytes'] = 512*MIB
        allowed = analyze_memory(raw)
        self.assertTrue(allowed['diagnostic_usable'])
        self.assertFalse(allowed['strict_memory_clean'])
        for process in chronological(raw):
            process['compressed_peak_bytes'] += 1
        blocked = analyze_memory(raw)
        self.assertFalse(blocked['hard_limits_passed'])
        self.assertTrue(any('Compression lifetime peak' in reason for reason in blocked['reasons']))
        self.assertEqual(blocked['coverage']['decode_forwards'], 32)

    def test_peak_footprint_bound_checks_even_if_current_is_low(self):
        raw = fixture()
        for process in chronological(raw):
            process['physical_footprint_peak_bytes'] = 12*GIB+1
        result = analyze_memory(raw)
        self.assertFalse(result['hard_limits_passed'])
        self.assertFalse(result['strict_memory_clean'])
        self.assertTrue(any('Physical footprint peak' in reason for reason in result['reasons']))

    def test_swap_decrease_allowed_but_later_regrowth_fails(self):
        raw = fixture()
        for index, process in enumerate(chronological(raw)):
            if index >= 20:
                process['system_swap_used_bytes'] = GIB//2
        result = analyze_memory(raw)
        self.assertTrue(result['hard_limits_passed'])
        self.assertFalse(result['stable_observed_swap'])
        self.assertFalse(result['strict_memory_clean'])
        # Final swap remains lower than initial: adjacent observations still catch growth.
        raw['runs'][-1]['after']['process']['system_swap_used_bytes'] += 1
        self.assertFalse(analyze_memory(raw)['hard_limits_passed'])

    def test_missing_boolean_negative_and_overflow_gauges_are_invalid(self):
        for key in ('physical_footprint_bytes', 'physical_footprint_peak_bytes', 'compressed_bytes',
                    'compressed_peak_bytes', 'decompressions', 'system_swap_used_bytes'):
            for value in (None, True, -1, 1.0, 2**64):
                with self.subTest(key=key, value=value):
                    raw = fixture()
                    raw['runs'][0]['decode_diagnostics']['samples'][9]['after']['process'][key] = value
                    with self.assertRaises(ValueError): analyze_memory(raw)

    def test_reset_or_impossible_peak_is_not_accepted_as_clean(self):
        for key in ('physical_footprint_peak_bytes', 'compressed_peak_bytes', 'decompressions'):
            raw = fixture()
            values = list(chronological(raw))
            if key == 'physical_footprint_peak_bytes':
                values[30][key] -= 1
            else:
                for process in values[:30]: process[key] = 1
            with self.subTest(key=key), self.assertRaises(ValueError): analyze_memory(raw)
        for key, value in (('physical_footprint_bytes', 10*GIB), ('compressed_bytes', 1)):
            raw = fixture(); raw['runs'][0]['before']['process'][key] = value
            with self.assertRaises(ValueError): analyze_memory(raw)

    def test_missing_incomplete_or_misaligned_token_coverage_is_invalid(self):
        mutations = [lambda raw: raw.update(complete=False),
            lambda raw: raw['runs'].pop(),
            lambda raw: raw['runs'][0].update(output_tokens=16),
            lambda raw: raw['runs'][0]['decode_diagnostics'].update(omitted_steps=1),
            lambda raw: raw['runs'][0]['decode_diagnostics']['samples'].pop(),
            lambda raw: raw['runs'][0]['decode_diagnostics']['samples'][3].update(offset=0),
            lambda raw: raw['runs'][0]['decode_diagnostics']['samples'][3].update(input_token_id=5),
            lambda raw: raw['runs'][1]['decode_diagnostics']['samples'][0].update(begin_ns=0),
            lambda raw: raw['runs'][0]['phases']['ingest'].pop('before'),
            lambda raw: raw['runs'][0]['decode_diagnostics']['samples'][3].pop('before')]
        for mutate in mutations:
            raw = fixture(); mutate(raw)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): analyze_memory(raw)

    def test_bad_memory_limits_are_rejected(self):
        for options in (dict(budget_bytes=0), dict(budget_bytes=True),
                        dict(compression_limit_bytes=-1), dict(compression_limit_bytes=13*GIB)):
            with self.assertRaises(ValueError): analyze_memory(fixture(), **options)


if __name__ == '__main__':
    unittest.main()
