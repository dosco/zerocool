import copy
import unittest
import json
from pathlib import Path
import tempfile
from evidence_index import Index
from evidence_queries import memory
from measure_ingestion_memory import analyze_trace,expected_boundaries,CLASSES,FIELDS


class IngestionMemoryTest(unittest.TestCase):
    def fixture(self):
        requests=[dict(prompt_tokens=72,reused_tokens=0),dict(prompt_tokens=233,reused_tokens=104)]
        rows=[]
        for i,(life,where) in enumerate(expected_boundaries(requests),1):
            classes={n:{k:0 for k in FIELDS} for n in CLASSES};classes['resident']['allocated_bytes']=16384
            metal=dict(buffer_costs=dict(kind='buffer_costs_v1',classes=classes),live_buffer_bytes=16384,
                peak_buffer_bytes=32768,device_allocated_bytes=32768,encoded_buffer_references=0,live_command_groups=0)
            rows.append(dict(kind='memory_boundary_v1',sequence=i,monotonic_ns=i*100,lifecycle=life,where=where,sample_ns=1,
                process=dict(physical_footprint_bytes=32768,physical_footprint_peak_bytes=65536,compressed_bytes=0,decompressions=0),
                metal=None if where['event'] in ('before_model_load','model_destroyed') else metal))
        coverage=dict(kind='memory_trace_coverage_v1',limit=512,observed=388,captured=388,omitted=0,lifecycle_records=8,records=396)
        return rows,coverage,requests

    def test_expected_layer_microchunk_and_lifecycle_coverage(self):
        rows,cov,req=self.fixture();result=analyze_trace(rows,cov,req)
        self.assertEqual(len(result['lifecycle']),8)
        self.assertTrue(result['memory_values_complete'])
        self.assertFalse(result['normal_request_latency_qualified'])
        self.assertIsNone(result['lifecycle'][-1]['metal_live_bytes'])
        self.assertEqual(sum(r['where']['event']=='microchunk_end' for r in rows),96)

    def test_partial_reordered_or_missing_cleanup_never_passes(self):
        rows,cov,req=self.fixture()
        for mutate in (lambda r:r.pop(),lambda r:r.reverse(),lambda r:r[5]['where'].update(layer=47),
                       lambda r:r[-2]['metal'].update(encoded_buffer_references=1),
                       lambda r:r[5]['metal']['buffer_costs']['classes'].pop('resident')):
            bad=copy.deepcopy(rows);mutate(bad)
            with self.assertRaises(ValueError):analyze_trace(bad,cov,req)
        with self.assertRaises(ValueError):analyze_trace(rows,dict(cov,omitted=1),req)

    def test_missing_memory_stays_unknown_and_onset_is_only_a_boundary(self):
        rows,cov,req=self.fixture();rows[8]['process']['compressed_bytes']=16384
        result=analyze_trace(rows,cov,req)
        self.assertEqual(result['first_observed_compression']['prefill']['previous_boundary'],rows[7]['where'])
        rows[8]['process']['physical_footprint_peak_bytes']=None
        result=analyze_trace(rows,cov,req)
        self.assertFalse(result['memory_values_complete']);self.assertIsNone(result['peaks']['physical_footprint_peak_bytes'])

    def test_delayed_cleanup_requires_actual_elapsed_time(self):
        rows,cov,req=self.fixture();destroyed=rows[-1]['monotonic_ns']
        for delay in (250,1000):
            row=copy.deepcopy(rows[-1]);row.update(sequence=len(rows)+1,monotonic_ns=destroyed+delay*1000000,
                where=dict(event=f'after_destroy_{delay}ms'));rows.append(row)
        cov.update(lifecycle_records=10,records=398)
        result=analyze_trace(rows,cov,req,delayed=True);self.assertEqual(len(result['lifecycle']),10)
        rows[-1]['monotonic_ns']-=1
        with self.assertRaisesRegex(ValueError,'too early'):analyze_trace(rows,cov,req,delayed=True)

    def test_query_exposes_memory_lifecycle_without_inventing_dead_metal_gauges(self):
        rows,_,_=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'memory.jsonl';path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            index=Index(Path(directory)/'index.sqlite')
            try:
                index.import_paths([path]);answer=memory(index,str(path),limit=1,offset=len(rows)-1)
                self.assertEqual(answer['status'],'recorded_snapshots')
                self.assertEqual(answer['snapshots'][0]['pointer'],'line:396')
                self.assertEqual(answer['snapshots'][0]['where']['event'],'model_destroyed')
                self.assertIsNone(answer['snapshots'][0]['live_buffer_bytes'])
                self.assertFalse(answer['coverage']['query_truncated'])
                path.write_text('\n'+'\n'.join(json.dumps(r) for r in rows)+'\n');index.import_paths([path])
                self.assertEqual(memory(index,str(path),limit=1,offset=len(rows)-1)['snapshots'][0]['pointer'],'line:397')
                bad=copy.deepcopy(rows[1]);bad['metal']['buffer_costs']['classes']['resident']=None
                path.write_text(json.dumps(bad)+'\n');index.import_paths([path])
                with self.assertRaisesRegex(ValueError,'class counters'):memory(index,str(path))
                path.write_text('{}\n')
                with self.assertRaises(ValueError):memory(index,str(path))
            finally:index.close()


if __name__=='__main__':unittest.main()
