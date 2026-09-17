import copy
import unittest

from q4_request_profile import analyze_trace, command_classes, compare_traces, DISPATCHES
import test_decode_timeline


def fixture(variant='reference'):
    template_raw, template_profile, template_deps = test_decode_timeline.DecodeTimelineTest().fixture()
    original = template_raw['runs'][0]
    common = dict(artifact_revision='revision', model_id='mixed', diagnostic_stream_trunk=False,
        memory_plan=dict(limit_bytes=12*1024**3, expert_slots=1072),
        execution=dict(cache_policy='clock', residency='core-cache', decode_scratch='reuse'),
        completion_pipeline=True, ready_group=4, chunk_tokens=128, io_workers=8, short_append_tokens=32,
        metal=dict(build_fingerprint='build', device='M1 Pro', dispatches=0,
            kernels=dict(profile=True, profile_decode_only=True, counter_profile=False, q4_decode=variant)))
    raw = dict(complete=True, runs=[], workloads=[], model_revision='revision', sampling=dict(temperature=0, seed=0))
    profile = dict(truncated=False, coverage='decode-only', entry_limit=120000, command_groups=[])
    deps = []
    for phase, low in enumerate((72, 217)):
        row = copy.deepcopy(original)
        row.update(name=('initial', 'append')[phase], prompt_tokens=low, output_tokens=17,
                   output_token_ids=list(range(17)), token_latency_ms=[960/1e6]*16)
        row['before'] = copy.deepcopy(common); row['after'] = copy.deepcopy(common)
        row['before']['metal']['dispatches'] = phase*DISPATCHES//2
        row['after']['metal']['dispatches'] = (phase+1)*DISPATCHES//2
        row['phases'] = dict(decode=dict(before=copy.deepcopy(row['before']), after=copy.deepcopy(row['after'])))
        diagnostic = row['decode_diagnostics']
        diagnostic.update(total_decode_steps=16, captured_steps=16, samples=[])
        for step in range(16):
            shift = (phase*16+step)*20000
            offset = low+step
            sample = copy.deepcopy(original['decode_diagnostics']['samples'][0])
            sample.update(step=step, offset=offset, input_token_id=step,
                          begin_ns=1000+shift, end_ns=1960+shift)
            diagnostic['samples'].append(sample)
            groups = copy.deepcopy(template_profile['command_groups'])
            for group in groups:
                for key in ('submitted_ns', 'completed_ns'):
                    group[key] += shift
                for key in ('gpu_start_seconds', 'gpu_end_seconds'):
                    group[key] += shift/1e9
                for op in group['operations']:
                    op.update(offset=offset, tokens=1,
                              kernel='q4_mm' if op['stage'] == 'routed_expert' else 'normal')
                    if variant == 'packed-r2' and op['kernel'] == 'q4_mm':
                        op['kernel'] = 'q4_down_packed_r2'
            # Keep full expected dispatch coverage without creating thousands
            # of fake groups; these operations intentionally share one group.
            groups[0]['operations'].extend([dict(stage='gdn', layer=0, tokens=1,
                offset=offset, kernel='normal', request_phase='decode')]*3032)
            profile['command_groups'].extend(groups)
            for dep in copy.deepcopy(template_deps):
                dep['offset'] = offset
                for record in dep['records']:
                    for key in record:
                        if key.endswith('_ns'):
                            record[key] += shift
                deps.append(dep)
        raw['runs'].append(row)
        raw['workloads'].append(dict(name=row['name'], tokens=[1, 2], max_tokens=17, append=phase == 1))
    return raw, profile, deps


class RequestProfileTest(unittest.TestCase):
    def test_complete_coverage_and_union_reconcile(self):
        raw, profile, deps = fixture()
        result = analyze_trace(raw, profile, deps)
        self.assertEqual(result['coverage']['dispatches'], 101600)
        self.assertEqual(result['coverage']['selected_experts'], 15360)
        for phase in result['phases']:
            self.assertAlmostEqual(sum(phase['mean_buckets_ms'].values()), phase['mean_forward_ms'])
            self.assertEqual(sum(c['dispatches'] for c in phase['gpu_command_classes']), 50800)
            self.assertEqual(sum(c['dispatches'] for c in phase['layer_command_classes']), 50800)
            self.assertEqual(phase['gap_to_target_ms'], 0)
        self.assertFalse(result['normal_request_latency_qualified'])
        self.assertFalse(result['production_promoted'])

    def test_missing_or_extra_capture_is_rejected(self):
        changes = [lambda r, p, d: p.update(truncated=True),
            lambda r, p, d: p['command_groups'].pop(),
            lambda r, p, d: d.pop(),
            lambda r, p, d: d.append(d[0]),
            lambda r, p, d: d[0]['records'].pop(),
            lambda r, p, d: r['runs'][0]['decode_diagnostics'].update(omitted_steps=1),
            lambda r, p, d: p['command_groups'][0]['operations'][0].update(offset=999),
            lambda r, p, d: p['command_groups'][0].update(completed_ns=999999)]
        for change in changes:
            with self.subTest(change=change):
                raw, profile, deps = fixture(); change(raw, profile, deps)
                with self.assertRaises(ValueError): analyze_trace(raw, profile, deps)

    def test_changed_identity_and_wrong_dependency_clock_are_rejected(self):
        changes = [lambda r, p, d: r['runs'][1]['before']['metal'].update(build_fingerprint='changed'),
            lambda r, p, d: r['runs'][1]['after']['memory_plan'].update(expert_slots=1460),
            lambda r, p, d: d[0].update(artifact_revision='changed'),
            lambda r, p, d: d[0]['records'][0].update(gpu_start_ns=1)]
        for change in changes:
            with self.subTest(change=change):
                raw, profile, deps = fixture(); change(raw, profile, deps)
                with self.assertRaises(ValueError): analyze_trace(raw, profile, deps)

    def test_mixed_commands_are_never_split_or_repeated_by_layer(self):
        groups = [dict(gpu_start_seconds=1, gpu_end_seconds=1.001, operations=[
            dict(layer=1, stage='expert_reduce'), dict(layer=2, stage='router')]),
            dict(gpu_start_seconds=1.0005, gpu_end_seconds=1.0015, operations=[
                dict(layer=2, stage='routed_expert')])]
        rows = command_classes(groups, 1, True)
        mixed = next(r for r in rows if len(r['layers']) == 2)
        self.assertEqual(mixed['stages'], ['expert_reduce', 'router'])
        self.assertEqual(mixed['layers'], [1, 2])
        self.assertAlmostEqual(mixed['gpu_command_ms_per_token'], 1)
        self.assertAlmostEqual(sum(r['gpu_command_ms_per_token'] for r in rows), 2)

    def test_known_gpu_change_and_wait_buckets_are_compared_without_promotion(self):
        a = analyze_trace(*fixture())
        raw, profile, deps = fixture('packed-r2')
        for group in profile['command_groups']:
            if group['operations'][0]['stage'] == 'routed_expert':
                group['gpu_end_seconds'] -= 1/1e9
        for dep in deps:
            for record in dep['records']:
                record['gpu_end_ns'] -= 1
        b = analyze_trace(raw, profile, deps)
        comparison = compare_traces(a, b)
        for phase in comparison['phases']:
            routed = next(r for r in phase['gpu_command_classes'] if r['stages'] == ['routed_expert'])
            self.assertAlmostEqual(routed['delta_gpu_command_ms_per_token'], -48/1e6)
            self.assertAlmostEqual(sum(phase['delta_mean_buckets_ms'].values()), 0)
            self.assertEqual(phase['delta_mean_forward_ms'], 0)
        self.assertIsNone(comparison['causal_savings_estimate_ms'])
        self.assertIsNone(comparison['confidence_95'])

    def test_comparison_rejects_changed_controls_routes_and_history(self):
        a = analyze_trace(*fixture()); b = analyze_trace(*fixture('packed-r2'))
        for key in ('identity', 'workloads', 'sampling', 'output_tokens', 'routes_sha256',
                    'normalized_dispatch_sha256', 'coverage'):
            with self.subTest(key=key):
                changed = copy.deepcopy(b); changed[key] = 'changed'
                with self.assertRaises(ValueError): compare_traces(a, changed)
        with self.assertRaises(ValueError): compare_traces(b, a)


if __name__ == '__main__':
    unittest.main()
