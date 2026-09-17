import copy
import contextlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from benchmark_exact import configurations, validate
from evidence_index import Index, file_hash, parse, read
from evidence_queries import compare, memory, next_experiment, timeline
from query_evidence import main, record
import test_exact_benchmark as exact_fixture
import test_decode_profile as cached_fixture


class EvidenceQueryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.index = Index(self.root/'index.sqlite')

    def tearDown(self):
        self.index.close(); self.temp.cleanup()

    def put(self, name, value):
        path = self.root/name
        path.write_text(json.dumps(value)+'\n')
        self.index.import_paths([path])
        return str(path)

    def experiment(self, count=5, change=None, summary_change=None):
        a = configurations()[0]; a.update(name='control',q8_decode_rows=0)
        b = dict(a,name='candidate',q8_decode_rows=2)
        rows = []
        for p in range(count):
            for c in ([a,b] if p % 2 == 0 else [b,a]):
                raw = exact_fixture.ExactBenchmarkTest().fixture(); native = raw['runs'][0]
                raw['workloads'] = [dict(name='prompt_2k',tokens=[1]*2048,max_tokens=256)]
                state = native['after']; state['metal']['kernels'].update(profile=False,counter_profile=False,q8_decode_rows=c['q8_decode_rows'])
                state['expert_cache'] = {}; native['before'] = copy.deepcopy(state)
                raw['fixture_pair'] = p
                factor = .8 if c['name'] == 'candidate' else 1
                for key in ('decode_wall_ms','request_ms','time_to_first_token_ms'): native[key] *= factor
                obs = validate(raw,c,dict(build='native',revision='revision'),8*1024**3,256)
                if change is not None and p == 0 and c['name'] == 'candidate': change(raw)
                path = self.put(f'{p}-{c["name"]}.json', raw)
                obs.update(pair=p,configuration=c['name'],report=Path(path).name,report_sha256=file_hash(path))
                rows.append(obs)
        summary = dict(kind='exact_kernel_normal_requests',complete=True,configurations=[a,b],pairs=count,
                       build='native',artifact_revision='revision',budget_bytes=8*1024**3,measurements=rows)
        if summary_change: summary_change(summary)
        return self.put('summary.json',summary)

    def ask(self, source):
        return compare(self.index,source,'control','candidate',['q8_decode_rows'])

    def test_five_actual_pairs_confidence_and_two_pairs_no_confidence(self):
        five = self.ask(self.experiment())
        self.assertTrue(five['comparable'],five)
        self.assertEqual(five['metrics'][0]['confidence_95']['median'],.8)
        self.assertFalse(five['normal_request_latency_qualified'])
        two = self.ask(self.experiment(2))
        self.assertTrue(two['comparable'],two)
        self.assertTrue(all(m['confidence_95'] is None for m in two['metrics']))
        self.assertFalse(two['production_promoted'])

    def test_rejects_artifact_workload_budget_sampling_build_instrumentation_and_outputs(self):
        mutations = [lambda d:d.update(model_revision='wrong'),
            lambda d:d['runs'][0]['before']['prepared'].update(manifest_sha256='wrong'),
            lambda d:d['workloads'][0]['tokens'].__setitem__(0,2),
            lambda d:d['runs'][0]['after']['memory_plan'].update(limit_bytes=7*1024**3),
            lambda d:d['sampling'].update(seed=2),
            lambda d:d['runs'][0]['after']['metal'].update(build_fingerprint='wrong'),
            lambda d:d['runs'][0].update(profiling_enabled=True),
            lambda d:d['runs'][0]['after']['metal']['kernels'].update(counter_profile=True),
            lambda d:d['runs'][0]['output_token_ids'].__setitem__(0,888),
            lambda d:d['workloads'][0].update(max_tokens=64)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                answer=self.ask(self.experiment(1,change=mutation))
                self.assertFalse(answer['comparable'],answer)

    def test_rejects_unfinished_missing_duplicate_reordered_and_tampered_summary(self):
        mutations = [lambda d:d.update(complete=False),
            lambda d:d['measurements'].pop(),
            lambda d:d['measurements'].append(d['measurements'][0]),
            lambda d:d['measurements'].reverse(),
            lambda d:d['measurements'][0].update(request_ms=1)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                answer=self.ask(self.experiment(2,summary_change=mutation))
                self.assertFalse(answer['comparable'],answer)

    def test_controlled_change_must_be_declared_exactly(self):
        source=self.experiment(1)
        answer=compare(self.index,source,'control','candidate',['ready_group'])
        self.assertFalse(answer['comparable'])

    def test_workload_array_is_not_comparison_evidence(self):
        source=self.put('workload.json',[dict(tokens=[1,2])])
        answer=self.ask(source)
        self.assertFalse(answer['comparable'])
        self.assertIn('report object',answer['reason'])

    def test_one_pair_cannot_be_reused_to_fabricate_five_pairs(self):
        def duplicate(summary):
            originals=summary['measurements'][:2]
            summary['measurements']=[dict(row,pair=p) for p in range(5)
                for row in (originals if p%2==0 else originals[::-1])]
        answer=self.ask(self.experiment(5,summary_change=duplicate))
        self.assertFalse(answer['comparable'],answer)

    def test_workloads_cannot_be_relabelled_or_omitted(self):
        def relabel(summary):
            for row in summary['measurements']: row['name']='append_128'
        for mutation in (relabel,lambda s:s.update(cases=['prompt_2k','prompt_4k'])):
            with self.subTest(mutation=mutation):
                self.assertFalse(self.ask(self.experiment(2,summary_change=mutation))['comparable'])

    def test_raw_report_reference_requires_full_hash(self):
        def shorten(summary):
            for row in summary['measurements']: row['report_sha256']=row['report_sha256'][:16]
        self.assertFalse(self.ask(self.experiment(2,summary_change=shorten))['comparable'])

    def test_explicit_failed_status_overrides_complete_claim(self):
        self.assertFalse(self.ask(self.experiment(2,summary_change=lambda d:d.update(status='failed')))['comparable'])

    def test_stale_raw_dependency_rejects_comparison(self):
        source=self.experiment(1)
        (self.root/'0-candidate.json').write_text('{}')
        answer=self.ask(source)
        self.assertFalse(answer['comparable'])
        self.assertIn('Stale',answer['reason'])

    def test_dedup_reimport_and_rebuild_do_not_invent_repetitions(self):
        source=self.put('one.json',dict(complete=False,status='resource_blocked'))
        other=self.root/'copy.json'; other.write_bytes(Path(source).read_bytes())
        self.index.import_paths([other])
        history=self.index.history(); self.assertEqual(history['total'],1)
        self.assertEqual(len(history['results'][0]['sources'][0]['aliases']),2)
        self.assertFalse(history['results'][0]['complete'])
        self.assertEqual(history['results'][0]['status'],'resource_blocked')
        other.write_text('invalid')
        report=self.index.import_paths([other]); self.assertEqual(len(report['issues']),1)
        self.index.import_paths([Path(source)],rebuild=True)
        self.assertEqual(self.index.history()['total'],1)

    def test_incremental_import_refreshes_an_old_metadata_projection(self):
        source=self.put('original.json',dict(status='resource_blocked',complete=False))
        with self.index.db:
            self.index.db.execute('UPDATE documents SET info=?',(json.dumps(dict(status='complete',complete=True)),))
        self.index.import_paths([Path(source)])
        info=json.loads(self.index.db.execute('SELECT info FROM documents').fetchone()[0])
        self.assertEqual(info['status'],'resource_blocked'); self.assertFalse(info['complete'])

    def test_requested_stale_alias_is_reported_even_if_another_copy_verifies(self):
        first=self.put('a.json',dict(complete=True))
        requested=self.put('z.json',dict(complete=True))
        Path(requested).write_text('{}')
        _,source=self.index.source(requested)
        self.assertEqual(source['path'],str(Path(first).resolve()))
        self.assertTrue(any(str(Path(requested).resolve()) in e for e in source['unavailable_aliases']))

    def test_corrupt_and_oversized_files_are_visible_import_issues(self):
        bad=self.root/'bad.json'; bad.write_text('{"value":NaN}')
        huge=self.root/'huge.json'
        with huge.open('wb') as f:f.truncate(64*1024**2+1)
        result=self.index.import_paths([bad,huge])
        self.assertEqual(len(result['issues']),2)
        self.assertEqual(result['documents'],0)

    def test_json_rejects_duplicate_keys_and_numeric_overflow(self):
        for raw in ('{"complete":false,"complete":true}','{"value":1e999}'):
            with self.subTest(raw=raw),self.assertRaises(ValueError): parse(raw)

    def test_file_growth_cannot_exceed_the_read_limit(self):
        path=self.root/'growing.json';path.write_text('{}')
        stream=io.BytesIO(b'x'*17)
        with patch('evidence_index.MAX_BYTES',16),patch.object(Path,'open',return_value=stream):
            with self.assertRaises(ValueError): read(path)

    def test_unrelated_database_is_never_rebuilt(self):
        path=self.root/'unrelated.sqlite'
        with sqlite3.connect(path) as db:
            db.executescript('CREATE TABLE notes(value TEXT); PRAGMA user_version=1;')
        with self.assertRaises(ValueError): Index(path)
        with sqlite3.connect(path) as db:
            self.assertEqual(db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchone()[0],'notes')

    def test_memory_distinguishes_capacity_footprint_and_missing_boundaries(self):
        source=self.put('memory.json',dict(after=dict(memory_plan=dict(resident_bytes=500),metal=dict(live_buffer_bytes=100))))
        answer=memory(self.index,source)
        self.assertEqual(answer['snapshots'][0]['memory_plan']['resident_bytes'],500)
        self.assertEqual(answer['snapshots'][0]['live_buffer_bytes'],100)
        self.assertIsNone(answer['snapshots'][0]['physical_footprint_bytes'])
        self.assertIsNone(answer['snapshots'][0]['live_command_groups'])

    def test_explicit_null_memory_and_omitted_trace_records_remain_missing(self):
        source=self.put('null-memory.json',dict(after=dict(memory_plan={},process=None,metal=dict(residency=None))))
        answer=memory(self.index,source)
        self.assertIsNone(answer['snapshots'][0]['physical_footprint_bytes'])
        self.assertIsNone(answer['snapshots'][0]['pending_retirements'])
        source=self.put('aggregate-only.json',dict(expert_dependencies=[dict(layer=0,offset=0,tokens=1,duration_ns=100)]))
        answer=timeline(self.index,source)
        self.assertIsNone(answer['passes'][0]['recorded_records'])
        self.assertIsNone(answer['coverage']['matching_read_records'])

    def test_partial_timeline_keeps_coverage_and_does_not_invent_blocking_reasons(self):
        source=self.put('profile.json',dict(truncated=False,dependency_capture_limits=dict(passes_per_phase=48),
            expert_dependencies=[dict(layer=2,offset=128,tokens=128,request_phase='append',
                                     coordinator_wait_ns=10,records=[dict(expert=1,gpu_end_ns=50,released_ns=60)])]))
        answer=timeline(self.index,source,phase='append',layer=2,token=129)
        self.assertEqual(answer['coverage']['matching_passes'],1)
        self.assertIsNone(answer['coverage']['whole_request_covered'])
        self.assertIsNone(answer['passes'][0]['read_service_sum_ns'])
        self.assertEqual(timeline(self.index,source,token=300)['status'],'no_matching_captured_events')

    def test_jsonl_is_indexed_and_queries_reference_original_line(self):
        path=self.root/'events.jsonl';path.write_text(json.dumps(dict(layer=3,offset=12,tokens=1,records=[]))+'\n')
        self.index.import_paths([path])
        answer=timeline(self.index,str(path),token=12)
        self.assertEqual(answer['passes'][0]['pointer'],'line:1')
        self.assertEqual(self.index.db.execute('SELECT count(*) FROM events').fetchone()[0],1)

    def test_show_jsonl_returns_compact_metadata(self):
        path=self.root/'events.jsonl';path.write_text('{"layer":1}\n{"layer":2}\n')
        self.index.import_paths([path])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code=main(['--db',str(self.root/'index.sqlite'),'show',str(path)])
        answer=json.loads(output.getvalue())
        self.assertEqual(code,0,answer)
        self.assertEqual(answer['report']['indexed_lines'],2)

    def test_cached_remains_cached_and_requires_same_prepared_artifact(self):
        raw=cached_fixture.CachedComparisonTest().fixture();raw['input_sha256']='original-token-hash'
        for row in raw['runs']:
            for key in ('before','after'):
                state=row[key];state.update(artifact_revision='artifact',prepared=dict(manifest_sha256='prepared'))
                state['metal'].update(device='Apple M1 Pro',physical_bytes=32*1024**3)
        source=self.put('cached.json',raw);answer=self.ask(source)
        self.assertTrue(answer['comparable'],answer);self.assertEqual(answer['scope'],'cached')
        self.assertFalse(compare(self.index,source,'control','candidate',['q8_decode_rows'],case='prompt_7k')['comparable'])
        for flags in (dict(complete=False),dict(status='failed')):
            self.assertFalse(self.ask(self.put('incomplete-cached.json',dict(raw,**flags)))['comparable'])
        raw['runs'][0]['before']['prepared']['manifest_sha256']='wrong'
        self.assertFalse(self.ask(self.put('cached.json',raw))['comparable'])

    def test_ledger_survives_rebuild_and_unfinished_cannot_be_rejected_for_speed(self):
        source=self.put('blocked.json',dict(status='resource_blocked',complete=False,error='memory'))
        entry=dict(hypothesis='GPU selection reduces exposed work',expected_effect='lower request latency',
            controlled_change='sparse_selection',correctness='not completed',outcome='memory admission failed',
            decision='blocked',smallest_experiment='retry short screen after admission',limitations=['No timings'],evidence=[source])
        input_path=self.root/'input.json';input_path.write_text(json.dumps(entry))
        saved=record(self.index,input_path,self.root/'ledger')
        self.index.import_paths([self.root/'ledger',Path(source)],rebuild=True)
        self.assertEqual(self.index.history('GPU selection')['total'],1)
        entry['decision']='rejected';input_path.write_text(json.dumps(entry))
        with self.assertRaises(ValueError):record(self.index,input_path,self.root/'ledger')
        self.assertTrue(Path(saved['path']).exists())

    def test_metadata_cannot_turn_an_unfinished_attempt_into_a_ledger_result(self):
        blocked=self.put('blocked.json',dict(status='resource_blocked'))
        unknown=self.put('unknown.json',dict(note='metadata only'))
        for decision in ('promising','adopted','rejected'):
            entry=dict(hypothesis='h',expected_effect='e',controlled_change='c',correctness='unknown',
                       outcome='no timings',decision=decision,smallest_experiment='retry',
                       limitations=[],evidence=[blocked,unknown])
            input_path=self.root/'input.json';input_path.write_text(json.dumps(entry))
            with self.subTest(decision=decision),self.assertRaises(ValueError):
                record(self.index,input_path,self.root/'ledger')

    def test_optional_ledger_tags_and_revisit_rationale_survive_rebuild(self):
        source=self.put('result.json',dict(complete=True,status='rejected'))
        entry=dict(hypothesis='h',expected_effect='e',controlled_change='c',correctness='passed',
            outcome='below threshold',decision='rejected',smallest_experiment='changed mechanism only',
            limitations=[],evidence=[source],tags=dict(optimization=['target-state-recovery'],rejection_reason=['slow']),
            rerun_rationale='The new design avoids target forwards rather than just their copies')
        p=self.root/'entry.json';p.write_text(json.dumps(entry));saved=record(self.index,p,self.root/'ledger')
        self.index.import_paths([self.root/'ledger'],rebuild=True)
        result=self.index.history('target-state-recovery')['results'];self.assertEqual(len(result),1)
        self.assertEqual(result[0]['tags'],entry['tags']);self.assertEqual(result[0]['rerun_rationale'],entry['rerun_rationale'])
        bad=dict(entry,tags={'unknown':['value']});p.write_text(json.dumps(bad))
        with self.assertRaises(ValueError):record(self.index,p,self.root/'ledger')
        self.assertTrue(Path(saved['path']).exists())

    def test_next_prefers_missing_dependency_evidence_over_a_cached_speedup_claim(self):
        source=self.put('attribution.json',dict(kind='instrumented_cached_decode_attribution',kernels=[dict(name='big',mean_ms_per_token=500)]))
        answer=next_experiment(self.index,source)
        self.assertIsNone(answer['possible_request_benefit'])
        self.assertIn('normal-request',answer['missing_evidence'][0])


if __name__ == '__main__': unittest.main()
