import copy
import unittest

import perfect_draft as verifier
import perfect_draft_capacity as capacity
from test_perfect_draft import fixture, digest


def observation(slots, width, validation=False):
    raw, frozen, work = fixture(width, validation)
    raw['configuration'] = verifier.configuration(width, expert_slots=slots)
    for plan in (raw['memory_plan'], raw['before']['memory_plan'], raw['after']['memory_plan']):
        plan.update(expert_slots=slots, expert_bytes=slots*2768896)
    for state in (raw['before'], raw['after']):
        state['metal']['residency']['bytes_by_class']['expert'] = slots*2768896
    raw['prime']['cache_state'] = digest(slots)
    return verifier.observe(raw, frozen, work, width, validation, 'input', expert_slots=slots)


def rows():
    return [dict(pair=p, slots=s, source=f'{p}-{s}-{w}.json', observation=observation(s,w))
            for p,s,w in capacity.ORDER]


class VerifierCapacityTest(unittest.TestCase):
    def test_actual_capacity_is_required_and_workspace_remains_bounded(self):
        self.assertTrue(observation(1460,4)['clean_memory'])
        raw, frozen, work = fixture(4)
        with self.assertRaises(ValueError):
            verifier.observe(raw,frozen,work,4,False,'input',expert_slots=1460)
        raw['configuration'] = verifier.configuration(4,expert_slots=1460)
        with self.assertRaises(ValueError):
            verifier.observe(raw,frozen,work,4,False,'input',expert_slots=1460)
        for slots in (True, 1072.0, 1461):
            with self.assertRaises(ValueError):verifier.configuration(4,expert_slots=slots)
        raw,frozen,work = fixture(4)
        raw['memory_plan']['planned_bytes'] = verifier.BUDGET
        with self.assertRaises(ValueError):verifier.observe(raw,frozen,work,4,False,'input')

    def test_cross_capacity_relaxes_only_cache_hash_and_keeps_recovery(self):
        a,b = observation(1072,1,True),observation(1460,4,True)
        self.assertTrue(capacity.compare_math(a,b)['exact_row_logits'])
        with self.assertRaises(ValueError):verifier.compare(a,b)
        for change in [lambda r:r['row_logits_sha256'].__setitem__(0,digest('bad')),
                       lambda r:r['prime']['state'].update(tokens=70),
                       lambda r:r['rollback_checks'][-1].update(logits_sha256=digest('bad')),
                       lambda r:r.update(host_logits_bound_bytes=123)]:
            bad=copy.deepcopy(b);change(bad)
            with self.assertRaises(ValueError):capacity.compare_math(a,bad)

    def test_same_capacity_prime_and_timing_prefix_are_checked(self):
        serial=observation(1072,1,True); primes={}; validated=[]
        capacity.check_progress(serial,1072,validated,[],primes,serial['prime'])
        validated.append(dict(observation=serial))
        for slots in (1072,1460):
            o=observation(slots,4,True)
            capacity.check_progress(o,slots,validated,[],primes,serial['prime'])
            validated.append(dict(observation=o))
        timing=observation(1460,4)
        capacity.check_progress(timing,1460,validated,[],primes,serial['prime'])
        for change in [lambda r:r['prime'].update(cache_state=digest('new cache')),
                       lambda r:r['endpoints'][0]['state'].update(tokens=99),
                       lambda r:r['row_logits_sha256'].__setitem__(0,digest('new output'))]:
            bad=copy.deepcopy(timing);change(bad)
            with self.assertRaises(ValueError):
                capacity.check_progress(bad,1460,validated,[],primes,serial['prime'])

    def test_early_stop_needs_clean_complete_candidate_and_never_a_paired_claim(self):
        values=rows(); weak=values[2]['observation']
        weak['decode_wall_ns']=4_000_000_000;weak['verified_tokens_per_second']=4
        self.assertEqual(capacity.decide(values[:2])['status'],'running')
        decision=capacity.decide(values[:3])
        self.assertTrue(decision['early_stop'])
        self.assertFalse(decision['paired_comparison_complete'])
        self.assertFalse(decision['capacity_promising'])
        self.assertEqual(decision['remaining_timing_processes'],3)
        with self.assertRaises(ValueError):capacity.decide(values[:4])
        for field in ('clean_memory','clean_host'):
            changed=copy.deepcopy(values[:3]);changed[2]['observation'][field]=False
            with self.assertRaises(ValueError):capacity.decide(changed)
        changed=copy.deepcopy(values[:3]);changed[2]['observation']['verified_tokens']=4
        with self.assertRaises(ValueError):capacity.decide(changed)

    def test_complete_screen_requires_both_cache_and_serial_advantage(self):
        values=rows()
        for row in values:
            o=row['observation']; wall=2_500_000_000 if row['slots']==1460 and o['width']==4 else 4_000_000_000
            o['decode_wall_ns']=wall;o['verified_tokens_per_second']=16e9/wall
        decision=capacity.decide(values)
        self.assertTrue(decision['capacity_promising'])
        self.assertTrue(decision['paired_comparison_complete'])
        self.assertTrue(all(decision[k] is False for k in verifier.FLAGS))
        values[1]['observation'].update(decode_wall_ns=2_000_000_000,verified_tokens_per_second=8)
        self.assertFalse(capacity.decide(values)['capacity_promising'])
        with self.assertRaises(ValueError):capacity.decide(values[::-1])

    def test_work_counter_resets_or_missing_values_do_not_report_hits(self):
        raw,_,_=fixture(4)
        raw['before']['expert_cache']=dict(hits=1,misses=2,application_read_bytes=20)
        raw['after']['expert_cache']=dict(hits=3,misses=4,application_read_bytes=40)
        self.assertEqual(capacity.counters(raw)['cache_hit_fraction'],.5)
        raw['after']['expert_cache']['misses']=1
        with self.assertRaises(ValueError):capacity.counters(raw)
        raw['after']['expert_cache'].pop('misses')
        with self.assertRaises(ValueError):capacity.counters(raw)


if __name__ == '__main__':unittest.main()
