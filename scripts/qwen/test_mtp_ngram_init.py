import unittest

from test_mtp_expert_scratch import pair
from screen_mtp_ngram_init import comparison


def lazy_pair():
    a,b=pair()
    for r in (a,b):
        r['ngram_initialization']='lazy'
        for phase in ('before','after'):
            r[phase].update(ngram_hits=2,ngram_misses=3)
            r['ngram_'+phase+'_decode']=dict(constructed_rows=3,cached_rows=3,capacity_rows=100,
                initialized_row_bytes=984,reserved_row_bytes=32800,hits=2,misses=3)
    return a,b


class LazyNgramEvidenceTests(unittest.TestCase):
    def test_exact_output_and_logical_cache_comparison(self):
        a,b=lazy_pair();self.assertTrue(comparison(a,b)['lazy_cache_exact'])

    def test_missing_capacity_accounting_and_changed_behavior_reject(self):
        for change in ('mode','capacity','uninitialized','payload','hits','cache_difference'):
            a,b=lazy_pair();row=b['ngram_after_decode']
            if change=='mode':b['ngram_initialization']='eager'
            elif change=='capacity':row['capacity_rows']=2
            elif change=='uninitialized':row['constructed_rows']=2
            elif change=='payload':row['initialized_row_bytes']=row['reserved_row_bytes']+1
            elif change=='hits':row['hits']=100
            else:row['capacity_rows']=101
            with self.subTest(change=change),self.assertRaises(ValueError):comparison(a,b)


if __name__=='__main__':unittest.main()
