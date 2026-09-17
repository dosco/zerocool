import copy
import unittest

import request_diagnostic as diagnostic


def fixture():
    buckets=dict(gpu_active=100.,gpu_idle_submitted=30.,gpu_idle_ready_expert=10.,
                 gpu_idle_pending_read=90.,gpu_idle_callback=5.,gpu_idle_other=15.)
    shared=dict(stages=['shared_expert','routed_expert'],gpu_command_ms_per_token=60.,command_groups=100,dispatches=500)
    trunk=dict(stages=['attention'],gpu_command_ms_per_token=40.,command_groups=100,dispatches=500)
    return dict(kind='q4_request_profile_v1',complete=True,variant='reference',identity={'build':'fixture'},
        coverage=dict(phases=2,forwards=32,dispatches=101600,dependency_passes=1536,
            selected_experts=15360,explicit_dependency_file=True,omitted_tokens=0),
        phases=[dict(name=name,captured_tokens=16,tokens=[{} for _ in range(16)],
            mean_forward_ms=250.,mean_buckets_ms=copy.deepcopy(buckets),mean_gpu_command_ms=100.,
            gpu_command_classes=copy.deepcopy([shared,trunk]),
            layer_command_classes=[dict(shared,layers=[1,2]),dict(trunk,layers=[0])])
            for name in ('initial','append')])


class DiagnosticRankingTests(unittest.TestCase):
    def test_ranks_exclusive_costs_without_adding_command_sums(self):
        result=diagnostic.rank(fixture())
        self.assertEqual(result['largest_observed_category'],'gpu_active')
        self.assertEqual(sum(r['ms_per_token'] for r in result['equal_phase_mean_buckets']),250)
        self.assertEqual(result['phases'][0]['distance_to_200ms'],50)
        self.assertIsNone(result['possible_latency_saving_ms'])
        self.assertFalse(result['causal_bottleneck_identified'])
        self.assertTrue(all(result[k] is False for k in diagnostic.FLAGS))

    def test_mixed_commands_and_layers_remain_whole(self):
        r=diagnostic.rank(fixture())['phases'][0]
        self.assertEqual(len(r['gpu_command_classes']),2)
        self.assertEqual(r['gpu_command_classes'][0]['stages'],['shared_expert','routed_expert'])
        self.assertEqual(r['layer_command_classes'][0]['layers'],[1,2])
        self.assertEqual(r['gpu_command_classes'][0]['gpu_command_ms_per_token'],60)

    def test_bad_coverage_or_nonfinite_values_never_rank(self):
        for value in (-1,float('nan'),float('inf'),None):
            data=fixture();data['phases'][0]['mean_buckets_ms']['gpu_idle_other']=value
            with self.assertRaises(ValueError):diagnostic.rank(data)
        for change in ('omitted','variant','missing_phase','missing_token','missing_bucket','mismatch'):
            data=fixture()
            if change=='omitted':data['coverage']['omitted_tokens']=1
            elif change=='variant':data['variant']='packed-r2'
            elif change=='missing_phase':data['phases'].pop()
            elif change=='missing_token':data['phases'][0]['tokens'].pop()
            elif change=='missing_bucket':data['phases'][0]['mean_buckets_ms'].pop('gpu_idle_other')
            else:data['phases'][0]['mean_forward_ms']=251
            with self.assertRaises(ValueError):diagnostic.rank(data)

    def test_bad_command_class_totals_rejected(self):
        data=fixture();data['phases'][0]['gpu_command_classes'][0]['gpu_command_ms_per_token']=61
        with self.assertRaises(ValueError):diagnostic.rank(data)

    def test_each_phase_keeps_its_own_order_and_target(self):
        data=fixture();data['phases'][1]['mean_buckets_ms']['gpu_idle_pending_read']=110
        data['phases'][1]['mean_forward_ms']=270
        result=diagnostic.rank(data)
        self.assertEqual(result['phases'][0]['exclusive_buckets'][0]['category'],'gpu_active')
        self.assertEqual(result['phases'][1]['exclusive_buckets'][0]['category'],'gpu_idle_pending_read')
        self.assertEqual(result['phases'][1]['distance_to_200ms'],70)


if __name__ == '__main__':unittest.main()
