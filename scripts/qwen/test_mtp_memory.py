import unittest

from trace_mtp_memory import allocation_categories


SAMPLE='''VM page size:  16384 bytes
==== Summary for process 42
REGION TYPE PAGES PAGES PAGES PAGES PAGES PAGES PAGES COUNT
IOAccelerator (graphics)        600000 600000 600000 0 0 600000 0 3700
MALLOC_LARGE                    3169 23 23 0 0 0 0 1 see MALLOC ZONE table below
owned unmapped memory           1 0 0 1 0 0 0 1
__DATA                         200.25 100.25 30.50 0 0 0 0 100
TOTAL                          603370.25 600123.25 600053.50 1 0 600000 0 3802

MALLOC ZONE PAGES PAGES PAGES PAGES COUNT
DefaultMallocZone              40000 800 800 0 60000 60.4M 0K 0% 26
'''


class MemoryMapTests(unittest.TestCase):
    def test_page_counts_and_categories(self):
        r=allocation_categories(SAMPLE)
        self.assertEqual(r['categories']['IOAccelerator (graphics)']['resident_bytes'],600000*16384)
        self.assertEqual(r['categories']['MALLOC_LARGE']['resident_bytes'],23*16384)
        self.assertEqual(r['categories']['__DATA']['virtual_bytes'],200.25*16384)
        self.assertEqual(set(r['swapped_categories']),{'owned unmapped memory'})
        self.assertEqual(r['total']['swapped_bytes'],16384)
        self.assertFalse(r['allocation_ownership_identified'])
        self.assertFalse(r['performance_measurement'])

    def test_alternate_total_is_not_counted_as_allocation(self):
        text=SAMPLE.replace('\nMALLOC ZONE',
            '\nTOTAL, minus reserved VM space 603370.25 600123.25 600053.50 1 0 600000 0 3802\n\nMALLOC ZONE')
        r=allocation_categories(text)
        self.assertEqual(set(r['swapped_categories']),{'owned unmapped memory'})
        self.assertEqual(r['total_without_reserved'],r['total'])

    def test_incomplete_ambiguous_and_invalid_maps_reject(self):
        for bad in (SAMPLE.replace('VM page size:','Page size:'),SAMPLE.replace('TOTAL','OTHER'),
            SAMPLE.replace('==== Summary for process 42',''),SAMPLE+SAMPLE,
            SAMPLE.replace('1 0 0 1 0 0 0 1','1 0 0 2 0 0 0 1')):
            with self.subTest(text=bad),self.assertRaises(ValueError):allocation_categories(bad)


if __name__=='__main__':unittest.main()
