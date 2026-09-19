import unittest
from benchmark_exact import config_args, is_original_control
from combined_q4 import ORDER, configs, decide
from qualify_exact_sessions import check_configuration


class CombinedQ4Test(unittest.TestCase):
    def rows(self):
        savings = dict(A=0, B=400, C=400, D=900)
        return [dict(pair=p, configuration=arm, memory_screen_passed=True,
                     requests=[dict(request_ms=20000-savings[arm], time_to_first_token_ms=10000,
                                    decode_wall_ms=10000-savings[arm]) for _ in range(2)])
                for p, arm in ORDER]

    def test_only_the_two_planned_axes_vary(self):
        cs = configs(); baseline = {k:v for k,v in cs[0].items() if k not in ('name','expert_slots','q4_decode')}
        self.assertEqual([(c['expert_slots'],c['q4_decode']) for c in cs],
                         [(1072,'reference'),(1072,'packed-r2'),(1460,'reference'),(1460,'packed-r2')])
        for c in cs:
            self.assertEqual({k:v for k,v in c.items() if k not in ('name','expert_slots','q4_decode')}, baseline)
            self.assertIn('--q4-decode', config_args(c))
        self.assertFalse(is_original_control(dict(q4_decode='packed-r2')))

    def test_full_factorial_and_both_decode_phases_are_required(self):
        rows = self.rows(); result = decide(rows)
        self.assertTrue(result['advance_to_confirmation'])
        self.assertEqual(set(result['contrasts']), {'B/A','D/C','C/A','D/B','D/A'})
        for bad in (rows[:-1], rows[::-1], rows+[rows[0]]):
            with self.assertRaises(ValueError): decide(bad)
        for r in rows:
            if r['configuration']=='D': r['requests'][1]['decode_wall_ms']=9500
        self.assertFalse(decide(rows)['advance_to_confirmation'])

    def test_cache_only_gain_cannot_promote_kernel(self):
        rows=self.rows()
        for r in rows:
            if r['configuration']=='B': r['requests'][0]['decode_wall_ms']=10100
        result=decide(rows)
        self.assertFalse(result['gates']['kernel_gain_at_both_capacities'])
        self.assertFalse(result['advance_to_confirmation'])

    def test_memory_regression_and_unavailable_metrics_do_not_pass(self):
        rows=self.rows();rows[3]['memory_screen_passed']=False
        self.assertEqual(decide(rows)['status'],'memory_disturbed')
        rows=self.rows();rows[0]['requests'][0]['decode_wall_ms']=float('nan')
        with self.assertRaises(ValueError): decide(rows)
        rows=self.rows();rows[4]['requests'][1]['time_to_first_token_ms']=11000
        self.assertFalse(decide(rows)['advance_to_confirmation'])

    def test_config_checker_rejects_ignored_q4_flag(self):
        state=dict(metal=dict(kernels=dict(q4_decode='reference')),ready_group=4,chunk_tokens=128,io_workers=8)
        with self.assertRaises(ValueError):check_configuration(state,dict(q4_decode='packed-r2'))
        state['metal']['kernels']['q4_decode']='packed-r2'
        check_configuration(state,dict(q4_decode='packed-r2'))


if __name__ == '__main__': unittest.main()
