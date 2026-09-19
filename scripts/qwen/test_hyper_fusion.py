import unittest

from screen_hyper_fusion import analyze


class HyperFusionEvidenceTest(unittest.TestCase):
    def fixture(self):
        memory = dict(physical_footprint_bytes=40000000, physical_footprint_peak_bytes=50000000,
                      compressed_bytes=0, compressed_peak_bytes=0, decompressions=0, system_swap_used_bytes=0)
        return dict(kind='hyper_fusion_probe_v1', complete=True, validation=False, device='Apple M1 Pro',
            max_shared_buffer_bytes=32000000, gpu_stage_elapsed_seconds=2, dispatch_repeats=32,
            reference_dispatches_per_block=8, candidate_dispatches_per_block=6, edge_cases=12,
            neighbor_guards_intact=True, inputs_unchanged=True,
            cases=[dict(exact=True, native_reference_exact=True, origin=dict(layer=l, stage=s))
                   for l in (0, 3) for s in ('attention_input', 'mlp_input')],
            **{k:dict(memory) for k in ('memory_before', 'memory_after', 'timing_memory_before', 'timing_memory_after')},
            pairs=[dict(pair=p, case=c, arms=[dict(candidate=bool(v),
                sample=dict(wall_us=(4000 if not v else 2000)*(8 if c==4 else 1),
                            gpu_us=(3000 if not v else 1000)*(8 if c==4 else 1), encode_us=10),
                memory_before=dict(memory), memory_after=dict(memory)) for v in (p%2, 1-p%2)])
                   for p in range(5) for c in range(5)])

    def test_screen_cannot_promote_request_performance(self):
        result = analyze(self.fixture())
        self.assertTrue(result['advance_to_request_screen'])
        self.assertEqual(result['median_frequency_projection_ms_per_token'], 192)
        self.assertFalse(result['production_promoted'])
        self.assertFalse(result['normal_request_latency_qualified'])

    def test_gpu_only_gain_does_not_pass_wall_gate(self):
        raw = self.fixture()
        for pair in raw['pairs']:
            for arm in pair['arms']:
                arm['sample']['wall_us'] = 32000
        result = analyze(raw)
        self.assertEqual(result['status'], 'insufficient_benefit')
        self.assertGreater(result['median_gpu_frequency_projection_ms_per_token'], 0)

    def test_small_clear_wall_gain_does_not_pass_material_gate(self):
        raw = self.fixture()
        for pair in raw['pairs']:
            for arm in pair['arms']:
                arm['sample']['wall_us'] = 100000 - int(arm['candidate'])*10
        self.assertFalse(analyze(raw)['advance_to_request_screen'])

    def test_lifetime_pressure_and_between_arm_changes_fail(self):
        for key, val in (('compressed_peak_bytes', 1), ('decompressions', 1), ('system_swap_used_bytes', 1)):
            raw = self.fixture()
            raw['pairs'][3]['arms'][0]['memory_after'][key] = val
            self.assertEqual(analyze(raw)['status'], 'memory_disturbed')

    def test_missing_and_malformed_data_fail_closed(self):
        mutations = [lambda r:r['pairs'].pop(), lambda r:r.update(validation=True),
            lambda r:r.update(gpu_stage_elapsed_seconds=31), lambda r:r.update(edge_cases=0),
            lambda r:r.update(max_shared_buffer_bytes=257*1024**2),
            lambda r:r['cases'][0].update(native_reference_exact=False),
            lambda r:r['cases'][0]['origin'].update(layer=1),
            lambda r:r['pairs'][0]['arms'][0].update(candidate=True),
            lambda r:r['pairs'][0]['arms'][0]['sample'].update(wall_us=float('nan')),
            lambda r:r['memory_before'].pop('compressed_peak_bytes')]
        for mutation in mutations:
            raw = self.fixture();mutation(raw)
            with self.assertRaises(ValueError):analyze(raw)

    def test_pairs_are_units_and_outlier_is_retained(self):
        raw = self.fixture()
        raw['pairs'][-1]['arms'][1]['sample']['wall_us'] = 300000
        result = analyze(raw)
        self.assertFalse(result['advance_to_request_screen'])
        self.assertEqual(result['cases'][-1]['wall_confidence_95']['pairs'], 5)
        self.assertEqual(len(result['frequency_projection_ms_per_token']), 5)


if __name__ == '__main__':unittest.main()
