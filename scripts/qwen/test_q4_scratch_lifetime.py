import copy
import unittest

from native_q4_replay import COUNTERS
from q4_read_arrivals import analyze as base_analyze
from q4_shared_arrivals import analyze as shared_analyze
from q4_scratch_lifetime import SCRATCH_BYTES_PER_PASS, analyze
from test_q4_shared_arrivals import fixture as shared_fixture


def fixture(mode='timing', scope='forward'):
    raw, manifest = shared_fixture(mode)
    old_cycles = raw['cycles']; factor = 6/old_cycles
    raw.update(cycles=6, scratch_scope='48-pass-arena' if scope == 'forward' else 'eight-expert-batch',
        scratch_lifetime=dict(passes=48, scope=scope, retained_payload='shared-and-routed-temporaries',
                              buffers_per_pass=27, buffer_charge_bytes=16384))
    retained = SCRATCH_BYTES_PER_PASS*(48 if scope == 'forward' else 1)
    buffers = retained//16384
    for pair in raw['pairs']:
        for arm in pair['arms']:
            sample = arm['sample']; before, after = arm['metal_before'], arm['metal_after']
            for key, value in list(sample.items()):
                if type(value) is int: sample[key] = int(value*factor)
            sample.update(scope_begins=48 if scope == 'batch' else 1,
                          scope_ends=48 if scope == 'batch' else 1, retained_scratch_bytes=retained)
            sample['kernel_dispatches'] = {k:int(v*factor) for k,v in sample['kernel_dispatches'].items()}
            for key in COUNTERS: after[key] = before[key]+sample[key]
            for key, value in sample['kernel_dispatches'].items():
                after['kernel_dispatches'][key] = before['kernel_dispatches'].get(key,0)+value
            arm['disk_after']['devices'][0]['read_bytes'] = sample['read_bytes']+sample['preparation_read_bytes']
            pool = dict(capacity_bytes=128*1024**2, allocated_bytes=retained, peak_bytes=retained,
                        unused_retained_bytes=0, reuses=100, allocation_count=buffers, wait_ns=100)
            idle = dict(capacity_bytes=0, allocated_bytes=0, peak_bytes=0, unused_retained_bytes=0,
                        reuses=0, allocation_count=0, wait_ns=0)
            before['scratch_pools'] = [copy.deepcopy(pool), copy.deepcopy(idle)]
            after['scratch_pools'] = [dict(pool, reuses=100+1296, wait_ns=100+(48 if scope == 'batch' else 1)), copy.deepcopy(idle)]
            after['live_buffer_bytes'] = before['live_buffer_bytes']
            if mode != 'trace': continue
            original = sample['batches']; batches = []
            for cycle in range(6):
                delta = cycle*10000000
                for previous in original:
                    batch = copy.deepcopy(previous); batch['cycle'] = cycle
                    for event in batch['timing']['records']:
                        for key in ('admitted_ns', 'read_queued_ns', 'read_started_ns', 'read_completed_ns',
                                    'encoded_ns', 'submitted_ns', 'gpu_start_ns', 'gpu_end_ns', 'released_ns'):
                            event[key] += delta
                    for group in batch['command_profile']['command_groups']:
                        group['submitted_ns'] += delta
                        group['gpu_start_seconds'] += delta/1e9; group['gpu_end_seconds'] += delta/1e9
                        for op in group['operations']: op['encoded_at_ns'] += delta
                    batches.append(batch)
            sample['batches'] = batches
            previous = None
            for index, batch in enumerate(batches):
                if previous is None:
                    entering = dict(active_scratch_slot=-1, live_command_groups=0,
                        live_buffer_bytes=before['live_buffer_bytes'], pool=copy.deepcopy(pool),
                        active_prefix_bytes=retained, inferred_used_buffers=buffers)
                else: entering = copy.deepcopy(previous)
                used = (index+1)*27 if scope == 'forward' else 27
                completed_pool = dict(entering['pool'], reuses=100+(index+1)*27,
                    wait_ns=100+(index+1 if scope == 'batch' else 1),
                    unused_retained_bytes=retained-used*16384)
                done = dict(active_scratch_slot=0 if scope == 'forward' and index<47 else -1,
                    live_command_groups=0, live_buffer_bytes=before['live_buffer_bytes'], pool=completed_pool,
                    active_prefix_bytes=used*16384, inferred_used_buffers=used)
                batch['lifetime'] = dict(before_preparation=entering, after_preparation=copy.deepcopy(entering), after_work=done)
                previous = done
    return raw, manifest


class Q4ScratchLifetimeTest(unittest.TestCase):
    def test_six_complete_cycles_in_every_mode_with_exact_lifetime_geometry(self):
        for scope in ('batch', 'forward'):
            for mode in ('check', 'timing', 'trace'):
                raw, manifest = fixture(mode, scope)
                result = analyze(raw, scope, reference_manifest=manifest)
                self.assertEqual(result['status'], dict(check='checked', timing='diagnostic_gain', trace='captured')[mode])
                self.assertEqual(result['coverage']['expert_executions_per_arm'], 384)
                self.assertEqual(result['coverage']['shared_chains_per_arm'], 48)
                life = result['scratch_lifetime']
                self.assertEqual(life['reused_buffers_per_arm'],1296)
                self.assertEqual(life['retained_scratch_bytes'], SCRATCH_BYTES_PER_PASS*(48 if scope=='forward' else 1))
                self.assertFalse(life['complete_model_forward'])
                self.assertFalse(result['production_promoted'])
                if mode == 'trace':
                    self.assertEqual(result['coverage']['detailed_batches'], 4*48)
                    self.assertEqual(result['coverage']['detailed_experts'], 4*384)
                    for capture in life['captures']:
                        self.assertEqual([p['pass_index'] for p in capture['passes']], list(range(48)))
                    for capture in result['captures']:
                        self.assertEqual(len(capture['shared_command_joins']),48)
                        self.assertEqual({c['cycle'] for c in capture['commands']},set(range(6)))

    def test_historical_analyzers_reject_new_lifetime_contract(self):
        for scope in ('batch','forward'):
            raw, manifest=fixture(scope=scope)
            with self.assertRaises(ValueError): base_analyze(raw,shared=True)
            with self.assertRaises(ValueError): shared_analyze(raw,'shared',reference_manifest=manifest)
            raw.pop('scratch_lifetime')
            with self.assertRaises(ValueError): analyze(raw,scope,reference_manifest=manifest)

    def test_pass_coverage_cannot_be_shortened_or_relabelled(self):
        for change in ('cycles','passes','scope','payload','charge','buffers','trace_missing','trace_reorder','trace_duplicate'):
            raw, manifest=fixture('trace'); life=raw['scratch_lifetime']
            if change=='cycles': raw['cycles']=1
            elif change=='passes': life['passes']=8
            elif change=='scope': life['scope']='batch'
            elif change=='payload': life['retained_payload']='all-model-state'
            elif change=='charge': life['buffer_charge_bytes']=4096
            elif change=='buffers': life['buffers_per_pass']=24
            else:
                batches=raw['pairs'][0]['arms'][0]['sample']['batches']
                if change=='trace_missing': batches.pop()
                elif change=='trace_reorder': batches[8:16]=list(reversed(batches[8:16]))
                else: batches[8]['cycle']=0
            with self.subTest(change=change):
                with self.assertRaises(ValueError): analyze(raw,'forward',reference_manifest=manifest)

    def test_preparation_cannot_reset_or_mutate_a_live_arena(self):
        for field in ('active_scratch_slot','live_command_groups','live_buffer_bytes','active_prefix_bytes','inferred_used_buffers'):
            raw,manifest=fixture('trace'); snapshot=raw['pairs'][0]['arms'][0]['sample']['batches'][12]['lifetime']['after_preparation']
            snapshot[field]+=1
            with self.subTest(field=field):
                with self.assertRaises(ValueError): analyze(raw,'forward',reference_manifest=manifest)
        for field in ('capacity_bytes','allocated_bytes','peak_bytes','unused_retained_bytes','reuses','allocation_count','wait_ns'):
            raw,manifest=fixture('trace'); snapshot=raw['pairs'][0]['arms'][0]['sample']['batches'][12]['lifetime']['after_preparation']
            snapshot['pool'][field]+=1
            with self.subTest(field=field):
                with self.assertRaises(ValueError): analyze(raw,'forward',reference_manifest=manifest)

    def test_every_pass_and_final_boundary_preserve_resource_ownership(self):
        for scope in ('batch','forward'):
            for change in ('reuse','cursor','retained','active','live_gpu','pool_arms','discontinuity','first','final'):
                raw,manifest=fixture('trace',scope); arm=raw['pairs'][0]['arms'][0]
                batches=arm['sample']['batches']; snapshot=batches[10]['lifetime']['after_work']
                if change=='reuse': snapshot['pool']['reuses']+=1
                elif change=='cursor': snapshot['inferred_used_buffers']+=1
                elif change=='retained': snapshot['pool']['allocated_bytes']+=16384
                elif change=='active': snapshot['active_scratch_slot']=-1 if scope=='forward' else 0
                elif change=='live_gpu': snapshot['live_command_groups']=1
                elif change=='pool_arms': arm['metal_after']['scratch_pools'][0]['allocated_bytes']+=16384
                elif change=='first': arm['metal_before']['scratch_pools'][0]['wait_ns']+=1
                elif change=='final': arm['metal_after']['scratch_pools'][0]['wait_ns']+=1
                else:
                    for key in ('before_preparation','after_preparation'): batches[11]['lifetime'][key]['pool']['wait_ns']+=1
                with self.subTest(scope=scope,change=change):
                    with self.assertRaises(ValueError): analyze(raw,scope,reference_manifest=manifest)

    def test_scope_boundaries_allocation_counts_and_total_bytes_are_required(self):
        for key in ('scope_begins','scope_ends','retained_scratch_bytes','shared_chains','scratch_reuses','read_bytes','preparation_read_bytes'):
            raw,manifest=fixture(); raw['pairs'][0]['arms'][0]['sample'][key]+=1
            with self.subTest(key=key):
                with self.assertRaises(ValueError): analyze(raw,'forward',reference_manifest=manifest)
        raw,manifest=fixture();arm=raw['pairs'][0]['arms'][0]
        arm['metal_after']['scratch_pools'][0]['allocation_count']+=1
        with self.assertRaises(ValueError): analyze(raw,'forward',reference_manifest=manifest)

    def test_per_arm_storage_and_memory_gates_survive_extended_capture(self):
        raw,manifest=fixture();arm=raw['pairs'][-1]['arms'][-1]
        arm['disk_after']=copy.deepcopy(arm['disk_before'])
        self.assertEqual(analyze(raw,'forward',reference_manifest=manifest)['status'],'unqualified_storage')
        arm['memory_after']['compressed_bytes']=1
        self.assertEqual(analyze(raw,'forward',reference_manifest=manifest)['status'],'disturbed')


if __name__=='__main__':unittest.main()
