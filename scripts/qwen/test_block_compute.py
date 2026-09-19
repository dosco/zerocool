import copy
import json
import unittest

import block_compute_profile as profile
import build_block_profile as builder
from capture_block_profile import admission_failure, ANCHOR, SOURCE


def fixture(mode='commands'):
    groups, deps = [], []
    for layer in range(48):
        base = 1000+layer*1000
        ops = [dict(request_phase='decode', offset=72, layer=layer, tokens=4, stage='router', kernel='route_simd')]
        ops += [dict(request_phase='decode', offset=72, layer=layer, tokens=4,
            stage='routed_expert', kernel='scatter_experts') for _ in range(10)]
        if mode == 'dispatch':
            for i, op in enumerate(ops):
                op.update(counter_index=2*i, gpu_pass_ns=10, gpu_begin_ticks=base+i*10, gpu_end_ticks=base+i*10+5)
        groups.append(dict(submitted_ns=base+300, completed_ns=base+700,
            gpu_start_seconds=(base+400)/1e9, gpu_end_seconds=(base+600)/1e9, operations=ops))
        records = [dict(expert=e, admitted_ns=base, encoded_ns=base+250, submitted_ns=base+300,
            released_ns=base+750, read_completed_ns=base+200, read_queued_ns=base,
            read_started_ns=base+100, gpu_start_ns=base+400, gpu_end_ns=base+600, acquisition='new_miss') for e in range(10)]
        deps.append(dict(offset=72, layer=layer, tokens=4, request_phase='decode', build='b', artifact_revision='r',
            routes=list(range(10))*4, records=records, ready_hits=0, new_misses=10, loading_joins=0))
    block = dict(offset=72, forward_begin_ns=1, forward_end_ns=50000, forward_ns=49999)
    data = dict(truncated=False, coverage='all-dispatches', entry_limit=100000,
        dependency_capture_limits=dict(passes_per_phase=48, read_records_per_phase=8192),
        timing_kind=('instrumented per-dispatch compute passes; submission boundaries preserved' if mode == 'dispatch'
            else 'existing command groups; mixed stages are not isolated kernel costs'), command_groups=groups, expert_dependencies=deps)
    return block, data


class BlockComputeTest(unittest.TestCase):
    def test_complete_geometry_and_exclusive_overlap(self):
        b, p = fixture(); r = profile.analyze_block(b, p, 'b', 'r', 'commands')
        self.assertEqual(r['expert_rows'], {'4': 480})
        self.assertEqual(r['expert_read_bytes'], 480*2764800)
        self.assertAlmostEqual(sum(r['buckets_ms_per_token'].values()), r['forward_ms_per_token'])
        # Ten simultaneous misses occupy one elapsed interval, never ten waits.
        self.assertAlmostEqual(r['buckets_ms_per_token']['gpu_idle_pending_read'], 48*200/4e6)
        # Mixed commands form a single class, not a full duration per stage.
        self.assertEqual(len(r['classes']), 1)
        self.assertEqual(r['classes'][0]['stages'], ['routed_expert', 'router'])
        self.assertEqual(r, json.loads(json.dumps(r)))

    def test_missing_corrupt_and_early_release_rejected(self):
        def missing(p): p['expert_dependencies'].pop()
        def duplicate(p): p['expert_dependencies'][1]['layer'] = 0
        def wrong_route(p): p['expert_dependencies'][0]['routes'][0] = 1
        def early(p): p['expert_dependencies'][0]['records'][0]['released_ns'] = 1500
        def reversed_read(p): p['expert_dependencies'][0]['records'][0]['read_started_ns'] = 1300
        def lost_dispatch(p): p['command_groups'][0]['operations'].pop()
        def wrong_width(p): p['command_groups'][0]['operations'][0]['tokens'] = 5
        def changed_build(p): p['expert_dependencies'][0]['build'] = 'wrong'
        def truncated(p): p['truncated'] = True
        def missing_router(p): p['command_groups'][0]['operations'][0]['kernel'] = 'wrong'
        for mutate in (missing, duplicate, wrong_route, early, reversed_read, lost_dispatch, wrong_width,
                       changed_build, truncated, missing_router):
            with self.subTest(mutation=mutate.__name__):
                b, p = fixture(); mutate(p)
                with self.assertRaises(ValueError): profile.analyze_block(b, p, 'b', 'r', 'commands')

    def test_counter_coverage(self):
        b, p = fixture('dispatch'); r = profile.analyze_block(b, p, 'b', 'r', 'dispatch')
        self.assertEqual(sum(x['dispatches'] for x in r['counter_operations']), 48*11)
        p['command_groups'][0]['operations'][1]['counter_index'] = 0
        with self.assertRaises(ValueError): profile.analyze_block(b, p, 'b', 'r', 'dispatch')

    def test_no_profile_mutation_or_counter_mixing(self):
        b, p = fixture(); original = copy.deepcopy(p)
        profile.analyze_block(b, p, 'b', 'r', 'commands'); self.assertEqual(p, original)
        p['command_groups'][0]['operations'][0]['gpu_pass_ns'] = 10
        with self.assertRaises(ValueError): profile.analyze_block(b, p, 'b', 'r', 'commands')

    def test_builder_scope_and_seam_drift(self):
        model = (builder.ROOT/'src/qwen/model.cpp').read_text(); harness = builder.TEMPLATE.read_text()
        for mode in builder.MODES:
            copied = builder.model_source(model, mode); probe = builder.harness_source(harness, mode)
            self.assertIn('verifier_kernels.profile=phase_=="decode"', copied)
            self.assertIn('profile requires width4/1460 timing mode', probe)
            self.assertIn(str(builder.WORKSPACE)+'ull', probe)
            self.assertIn('model.take_profile()', probe)
        with self.assertRaises(ValueError): builder.model_source(model.replace('auto& records=event["records"];', ''), 'commands')
        with self.assertRaises(ValueError): builder.harness_source(harness, 'unknown')

    def test_admission_failure_is_not_a_kernel_result(self):
        raw = dict(complete=False, blocks=[], error='expert slots exceed admitted capacity or are below 32')
        self.assertTrue(admission_failure(raw))
        self.assertFalse(admission_failure(dict(raw, complete=True)))
        self.assertFalse(admission_failure(dict(raw, blocks=[{}])))
        self.assertFalse(admission_failure(dict(raw, error='Metal execution failed')))

    def test_real_reference_projection_requires_explicit_profile_flags(self):
        raw = json.loads(ANCHOR.read_text()); frozen = json.loads((SOURCE/'identity.json').read_text())
        work = json.loads((SOURCE/'workload.json').read_text()); control = copy.deepcopy(raw)
        raw.update(block_profile_mode='commands', profile_workspace_bytes=builder.WORKSPACE)
        raw['after']['metal']['kernels']['profile'] = True
        result = profile.exact_observation(raw, frozen, work, raw['input_sha256'], control, 'commands')
        self.assertTrue(result['exact_logits_state_routes'])
        self.assertTrue(raw['after']['metal']['kernels']['profile'])
        raw['after']['metal']['kernels']['counter_profile'] = True
        with self.assertRaises(ValueError):
            profile.exact_observation(raw, frozen, work, raw['input_sha256'], control, 'commands')


if __name__ == '__main__': unittest.main()
