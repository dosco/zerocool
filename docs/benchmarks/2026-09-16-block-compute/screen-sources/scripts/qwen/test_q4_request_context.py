import copy
import unittest
from unittest.mock import patch

import q4_request_context as context


def rows(ratios=(1.1, 1.0)):
    reference = [dict(name=name, decode_wall_ms=4800., request_ms=8000.,
                      time_to_first_token_ms=3200.) for name in ('initial','append')]
    packed = [dict(r,decode_wall_ms=r['decode_wall_ms']*ratio) for r,ratio in zip(reference,ratios)]
    return [dict(stem=stem,requests=requests,memory_screen_passed=True)
            for stem,requests in zip(context.ORDER[:2],(reference,packed))]


def memory_raw():
    p = dict(physical_footprint_bytes=1024, physical_footprint_peak_bytes=2048,
             compressed_bytes=0, compressed_peak_bytes=0, decompressions=0,
             system_swap_used_bytes=1024)
    state = dict(process=p)
    return dict(runs=[dict(before=copy.deepcopy(state),after=copy.deepcopy(state),
        phases={name:dict(before=copy.deepcopy(state),after=copy.deepcopy(state))
                for name in ('ingest','decode')})])


class RequestContextTests(unittest.TestCase):
    def test_trigger_is_directional_and_target_uses_decode_forwards(self):
        answer = context.normal_decision(rows((1.,1.01)))
        self.assertTrue(answer['trace_triggered'])
        self.assertIsNone(answer['confidence_95'])
        self.assertEqual(answer['paired_repetitions'],1)
        self.assertEqual(answer['phases'][0]['reference_ms_per_token'],300)
        self.assertEqual(answer['phases'][0]['reference_gap_to_5tps_ms'],100)

    def test_equal_or_faster_ends_without_promotion(self):
        answer = context.finish(rows((1.,.99)),{})
        self.assertEqual(answer['status'],'slowdown_not_reproduced')
        self.assertFalse(answer['production_promoted'])
        self.assertFalse(answer['normal_request_latency_qualified'])
        with self.assertRaises(ValueError): context.finish(rows(),{})

    def test_conditional_order_and_instrumentation_ratios(self):
        data = rows()
        traced_b = dict(data[1],stem='traced-B',requests=[dict(r,decode_wall_ms=r['decode_wall_ms']*1.2) for r in data[1]['requests']])
        traced_a = dict(data[0],stem='traced-A',requests=[dict(r,decode_wall_ms=r['decode_wall_ms']*.9) for r in data[0]['requests']])
        data += [traced_b,traced_a]
        with patch.object(context,'compare_traces',return_value={'complete':True}) as compare:
            answer=context.finish(data,{'traced-A':'a','traced-B':'b'})
            compare.assert_called_once_with('a','b')
            self.assertEqual(answer['trace_to_normal_decode_ratios'],{'A':[.9,.9],'B':[1.2,1.2]})
            with self.assertRaises(ValueError): context.finish(data[:2]+[traced_a,traced_b],{})

    def test_invalid_normal_measurements_do_not_trigger(self):
        for value in (0,-1,float('nan'),float('inf'),None):
            data=rows();data[1]['requests'][0]['decode_wall_ms']=value
            with self.assertRaises(ValueError):context.normal_decision(data)
        with self.assertRaises(ValueError):context.normal_decision(rows()[::-1])

    def test_lifetime_peak_and_missing_gauges_fail_memory(self):
        self.assertTrue(context.memory(memory_raw())['memory_screen_passed'])
        mutations = [('compressed_peak_bytes',1),('compressed_bytes',1),
                     ('physical_footprint_peak_bytes',context.CRITERIA['memory_budget_bytes']+1),
                     ('physical_footprint_bytes',None),('decompressions',1),
                     ('system_swap_used_bytes',2048)]
        for key,value in mutations:
            raw=memory_raw();raw['runs'][0]['after']['process'][key]=value
            self.assertFalse(context.memory(raw)['memory_screen_passed'],key)

    def test_trace_samples_are_included_in_memory_gate(self):
        raw=memory_raw();sample=copy.deepcopy(raw['runs'][0]['phases']['decode'])
        sample['before']['process']['compressed_peak_bytes']=1
        raw['runs'][0]['decode_diagnostics']={'samples':[sample]}
        self.assertFalse(context.memory(raw)['memory_screen_passed'])

    def test_dirty_pair_cannot_finish(self):
        data=rows((1.,1.));data[0]['memory_screen_passed']=False
        with self.assertRaises(ValueError):context.finish(data,{})


if __name__ == '__main__': unittest.main()
