import copy
import unittest

from q4_shared_arrivals import analyze
from q4_read_arrivals import SHARED_KERNELS, SHARED_LAYERS, analyze as base_analyze
from native_q4_replay import COUNTERS, PACKED
from shared_expert_reference import LAYERS, REVISION, TOLERANCE, WIDTHS
from test_q4_read_arrivals import fixture as arrival_fixture


def reference_manifest(raw):
    from test_shared_expert_reference import reference_fixture
    manifest = reference_fixture()
    for layer in LAYERS:
        row = raw['fixture_manifest']['layers'][str(layer)]
        manifest['layers'][str(layer)].update(inputs=row['inputs'], offsets=row['offsets'])
    return manifest


def fixture(mode='timing', condition='shared'):
    raw = arrival_fixture(mode, hits=2, invalidate=True)
    manifest = reference_manifest(raw)
    if condition == 'control': return raw, manifest
    raw['shared_prelude'] = dict(enabled=True, reference_manifest=manifest,
        reference_manifest_sha256='a'*64, expected_native_sha256='b'*64, cpu_reference_passed=True,
        shared_outputs_exact=True, layer_order=SHARED_LAYERS, gpu_scope='shared-and-routed-command-intervals',
        reference_checks=[dict(layer=layer, row=row, activation=dict(relative_l2=0.001, cosine=0.99999),
            shared=dict(relative_l2=0.001, cosine=0.99999),
            gate=dict(expected=2.0, absolute_error=0, limit=1e-6+2/128)) for layer in LAYERS for row in range(8)])
    for pair in raw['pairs']:
        for arm in pair['arms']:
            sample = arm['sample']; before, after = arm['metal_before'], arm['metal_after']
            batches = raw['cycles']*8
            sample.update(shared_chains=batches, scratch_reuses=sample['scratch_reuses']+batches*3)
            after['scratch_reuses'] = before['scratch_reuses']+sample['scratch_reuses']
            sample['kernel_dispatches'].update(dict.fromkeys(SHARED_KERNELS, batches))
            for name in SHARED_KERNELS: after['kernel_dispatches'][name] = before['kernel_dispatches'].get(name, 0)+batches
            if mode != 'trace': continue
            for state in (before, after): state['kernels']['profile'] = True
            names = PACKED if arm['variant'] == 'packed-r2' else ('q4_gate_up', 'q4_mm')
            for batch in sample['batches']:
                batch['shared_layer'] = SHARED_LAYERS[batch['batch']]
                groups = {}
                for event in batch['timing']['records']: groups.setdefault(event['submitted_ns'], []).append(event)
                commands = []
                for index, (submitted, events) in enumerate(sorted(groups.items())):
                    ops = []
                    if index == 0:
                        begin = min(e['admitted_ns'] for e in events)+200
                        ops = [dict(stage='shared_expert', layer=batch['shared_layer'], tokens=1,
                            offset=72+batch['batch'], kernel=kernel, encoded_at_ns=begin+n*10, encode_ns=1)
                            for n, kernel in enumerate(SHARED_KERNELS)]
                    for event in events:
                        ops += [dict(stage='routed_expert', layer=event['layer'], tokens=1,
                            offset=72+batch['batch'], kernel=kernel, encoded_at_ns=event['encoded_ns']+n*2, encode_ns=1)
                            for n, kernel in enumerate((*names, 'scatter_experts'))]
                    commands.append(dict(submitted_ns=submitted, gpu_start_seconds=events[0]['gpu_start_ns']/1e9,
                        gpu_end_seconds=events[0]['gpu_end_ns']/1e9, operations=ops))
                batch['command_profile'] = dict(command_groups=commands, truncated=False, coverage='all-dispatches',
                    timing_kind='existing command groups; mixed stages are not isolated kernel costs',
                    normal_request_latency_qualified=False)
    return raw, manifest


class Q4SharedArrivalsTest(unittest.TestCase):
    def test_separate_control_shared_modes_and_combined_gpu_scope(self):
        for condition in ('control', 'shared'):
            for mode in ('check', 'timing', 'trace'):
                raw, manifest = fixture(mode, condition)
                result = analyze(raw, condition, reference_manifest=manifest)
                self.assertEqual(result['condition'], condition)
                self.assertEqual(result['status'], dict(check='checked', timing='diagnostic_gain', trace='captured')[mode])
                self.assertFalse(result['normal_request_latency_qualified'])
                if condition == 'shared':
                    self.assertEqual(result['gpu_scope'], 'shared-and-routed-command-intervals')
                    self.assertEqual(result['shared_reference']['verified_cases'], 32)
                    if mode == 'trace':
                        for capture in result['captures']:
                            self.assertEqual(len(capture['shared_command_joins']), 8)
                            self.assertTrue(all(j['pure_routed_gpu_ns'] is None for j in capture['shared_command_joins']))
                else:
                    self.assertNotIn('shared_reference', result)

    def test_shared_is_explicit_not_ignored_by_historical_analyzer(self):
        raw, manifest = fixture()
        with self.assertRaises(ValueError): base_analyze(raw)
        with self.assertRaises(ValueError): analyze(raw, 'control', reference_manifest=manifest)
        with self.assertRaises(ValueError): analyze(raw, 'shared')
        raw.pop('shared_prelude')
        with self.assertRaises(ValueError): analyze(raw, 'shared', reference_manifest=manifest)
        for field in ('shared_chains',):
            control, _ = fixture(condition='control'); control['pairs'][0]['arms'][0]['sample'][field] = 0
            with self.assertRaises(ValueError): analyze(control, 'control')

    def test_every_shared_chain_scratch_and_dispatch_are_required(self):
        for change in ('count', 'scratch', 'extra', 'missing', 'submissions'):
            raw, manifest = fixture(); arm = raw['pairs'][0]['arms'][0]; sample = arm['sample']
            if change == 'count': sample['shared_chains'] -= 1
            elif change == 'scratch':
                sample['scratch_reuses'] -= 1; arm['metal_after']['scratch_reuses'] -= 1
            elif change == 'extra':
                sample['kernel_dispatches']['unexpected'] = 1; arm['metal_after']['kernel_dispatches']['unexpected'] = 1
            elif change == 'missing':
                sample['kernel_dispatches']['plain_mm'] -= 1; arm['metal_after']['kernel_dispatches']['plain_mm'] -= 1
            else:
                sample['submissions'] += 1; arm['metal_after']['submissions'] += 1
            with self.subTest(change=change):
                with self.assertRaises(ValueError): analyze(raw, 'shared', reference_manifest=manifest)

    def test_cpu_error_limits_and_oracle_identity_are_enforced(self):
        for change in ('nll', 'cosine', 'gate', 'limit', 'expected', 'nan', 'duplicate', 'missing', 'tolerance', 'manifest'):
            raw, manifest = fixture(); ref = raw['shared_prelude']; check = ref['reference_checks'][0]
            if change == 'nll': check['activation']['relative_l2'] = 0.010001
            elif change == 'cosine': check['shared']['cosine'] = 0.999
            elif change == 'gate': check['gate']['absolute_error'] = 1
            elif change == 'limit': check['gate']['limit'] = 1
            elif change == 'expected': check['gate']['expected'] = 4
            elif change == 'nan': check['gate']['absolute_error'] = float('nan')
            elif change == 'duplicate': ref['reference_checks'][1] = copy.deepcopy(check)
            elif change == 'missing': ref['reference_checks'].pop()
            elif change == 'tolerance':
                manifest = copy.deepcopy(manifest); manifest['tolerance']['vector_relative_l2_max'] = 1
            else: ref['reference_manifest'] = dict(manifest, fixture_manifest_sha256='d'*64)
            with self.subTest(change=change):
                with self.assertRaises(ValueError): analyze(raw, 'shared', reference_manifest=manifest)
        raw, manifest = fixture()
        with self.assertRaises(ValueError): analyze(raw, 'shared', reference_manifest=manifest,
                                                   gate_values={layer: [3.0]*8 for layer in LAYERS})

    def test_profiling_only_in_separate_shared_trace(self):
        for mode in ('check', 'timing', 'trace'):
            for change in ('profile', 'counter'):
                raw, manifest = fixture(mode)
                kernels = raw['pairs'][0]['arms'][0]['metal_before']['kernels']
                kernels['counter_profile' if change == 'counter' else 'profile'] = True if change == 'counter' else not kernels['profile']
                with self.assertRaises(ValueError): analyze(raw, 'shared', reference_manifest=manifest)

    def test_profile_commands_join_and_shared_precedes_routed_without_extra_submission(self):
        for change in ('extra_command', 'missing', 'swap', 'late_shared', 'gpu', 'counter', 'layer', 'row', 'extra_op', 'encode', 'truncated'):
            raw, manifest = fixture('trace'); batch = raw['pairs'][0]['arms'][0]['sample']['batches'][0]
            profile = batch['command_profile']; command = profile['command_groups'][0]; ops = command['operations']
            if change == 'extra_command': profile['command_groups'].append(copy.deepcopy(command))
            elif change == 'missing': ops.pop(0)
            elif change == 'swap': ops[:2] = list(reversed(ops[:2]))
            elif change == 'late_shared': ops[:] = ops[3:]+ops[:3]
            elif change == 'gpu': command['gpu_start_seconds'] += 0.001
            elif change == 'counter': ops[0]['gpu_pass_ns'] = 1
            elif change == 'layer': ops[0]['layer'] = 16
            elif change == 'row': ops[0]['offset'] = 100
            elif change == 'extra_op': ops.append(copy.deepcopy(ops[-1]))
            elif change == 'encode': ops[0]['encoded_at_ns'] = command['submitted_ns']+1
            else: profile['truncated'] = True
            with self.subTest(change=change):
                with self.assertRaises(ValueError): analyze(raw, 'shared', reference_manifest=manifest)

    def test_reference_producer_and_artifact_provenance_match_frozen_sources(self):
        from q4_shared_arrivals import ROOT, FIXTURES, reference_identity
        raw, manifest = fixture()
        mapping = {'generator_sha256': ROOT/'scripts/qwen/shared_expert_reference.py',
                   'bf16_reference_sha256': ROOT/'scripts/qwen/reference_numpy.py',
                   'artifact_lock_sha256': ROOT/'mixed-models.lock.json',
                   'fixture_manifest_sha256': FIXTURES/'manifest.json'}
        frozen = dict(files={str(path.resolve()): manifest[key] for key, path in mapping.items()})
        reference_identity(manifest, frozen)
        for key, path in mapping.items():
            for digest in (None, 'b'*64):
                changed = copy.deepcopy(frozen); changed['files'][str(path.resolve())] = digest
                with self.subTest(key=key, digest=digest):
                    with self.assertRaises(ValueError): reference_identity(manifest, changed)

    def test_storage_and_host_gates_remain_independent_of_shared_gain(self):
        raw, manifest = fixture(); arm = raw['pairs'][0]['arms'][0]
        arm['disk_after'] = copy.deepcopy(arm['disk_before'])
        self.assertEqual(analyze(raw, 'shared', reference_manifest=manifest)['status'], 'unqualified_storage')
        arm['host_after']['thermal_state'] = 2
        self.assertEqual(analyze(raw, 'shared', reference_manifest=manifest)['status'], 'disturbed')


if __name__ == '__main__': unittest.main()
