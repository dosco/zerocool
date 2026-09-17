import contextlib
from functools import lru_cache
import io
import itertools
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from cache_simulation import (Belady, Clock, SegmentedLRU, LAYERS, PAYLOAD_BYTES,
                              SLOT_BYTES, cache_curve)
from evidence_index import Index, file_hash
from query_evidence import main


REVISION = 'aa7c790e804bbf9d491ddb109c3d61bc4a555f7c'


def window(offset=0, tokens=1, **extra):
    return [dict(layer=layer, offset=offset, tokens=tokens, routes=list(range(10)) * tokens,
                 build='b' * 64, artifact_revision=REVISION, **extra) for layer in range(LAYERS)]


def optimum(sequence, capacity):
    """Independent exhaustive victim enumeration with mandatory admission."""
    @lru_cache(None)
    def solve(position, resident):
        if position == len(sequence):
            return 0
        key = sequence[position]
        if key in resident:
            return solve(position + 1, resident)
        if capacity == 0:
            return 1 + solve(position + 1, ())
        if len(resident) < capacity:
            return 1 + solve(position + 1, tuple(sorted((*resident, key))))
        return 1 + min(solve(position + 1, tuple(sorted(set(resident) - {victim} | {key})))
                       for victim in resident)
    return solve(0, ())


class CachePolicyTest(unittest.TestCase):
    def test_belady_matches_exhaustive_optimum_and_bounds_other_policies(self):
        for length in range(7):
            for sequence in itertools.product(range(3), repeat=length):
                for capacity in range(4):
                    oracle = Belady(capacity, sequence)
                    misses = sum(not oracle.access(k, i) for i, k in enumerate(sequence))
                    self.assertEqual(misses, optimum(sequence, capacity), (sequence, capacity))
                    for cache in (Clock(capacity), SegmentedLRU(capacity)):
                        other = sum(not cache.access(k, i) for i, k in enumerate(sequence))
                        self.assertLessEqual(misses, other, (sequence, capacity, type(cache)))

    def test_clock_matches_independent_second_chance_queue(self):
        rng = random.Random(710)
        for capacity in range(1, 9):
            sequence = [rng.randrange(20) for _ in range(1000)]
            cache = Clock(capacity)
            queue = []
            for i, key in enumerate(sequence):
                hits = [r for r in queue if r[0] == key]
                if hits:
                    hits[0][1] = True
                else:
                    if len(queue) == capacity:
                        while queue[0][1]:
                            old = queue.pop(0)
                            old[1] = False
                            queue.append(old)
                        queue.pop(0)
                    queue.append([key, True])
                self.assertEqual(cache.access(key, i), bool(hits))
                self.assertEqual(set(cache.lookup), {r[0] for r in queue})

    def test_protected_entries_demote_and_probation_stays_evictable(self):
        cache = SegmentedLRU(4)
        for i, key in enumerate([0, 1, 2, 3, 0, 1, 2, 3]):
            cache.access(key, i)
        self.assertEqual(list(cache.protected), [1, 2, 3])
        self.assertEqual(list(cache.probation), [0])
        self.assertFalse(cache.access(4, 8))
        self.assertNotIn(0, cache.probation)
        self.assertTrue(cache.access(4, 9))
        self.assertEqual(list(cache.probation), [1])
        self.assertFalse(cache.access(5, 10))
        self.assertNotIn(1, cache.probation)
        one = SegmentedLRU(1)
        self.assertEqual([one.access(k, i) for i, k in enumerate([0, 0, 1, 0])],
                         [False, True, False, False])

    def test_oracle_heap_bounded_on_long_hit_trace(self):
        sequence = [0, 1] * 5000
        cache = Belady(2, sequence)
        misses = 0
        for i, key in enumerate(sequence):
            misses += not cache.access(key, i)
            self.assertLessEqual(len(cache.heap), 4)
        self.assertEqual(misses, 2)


class CacheQueryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / 'index.sqlite'
        self.index = Index(self.db)

    def tearDown(self):
        self.index.close()
        self.temp.cleanup()

    def put(self, rows, name='routes.jsonl'):
        path = self.root / name
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        self.index.import_paths([path])
        return str(path)

    def query(self, rows, budgets=None, **kw):
        return cache_curve(self.index, self.put(rows), [0, 1280] if budgets is None else budgets, **kw)

    def test_exact_bytes_layer_keys_and_cold_zero_full_cache(self):
        data = self.query(window() + window(1), per_layer=True)
        zero, full = data['curves']
        self.assertEqual(zero['slots'], 0)
        self.assertEqual(zero['policies'][0]['misses'], 960)
        self.assertEqual(zero['policies'][0]['application_miss_bytes'], 960 * PAYLOAD_BYTES)
        self.assertEqual(full['slots'], 1280 * 1024**2 // SLOT_BYTES)
        self.assertEqual(full['slot_allocation_bytes'] + full['unused_budget_bytes'], full['budget_bytes'])
        for result in full['policies']:
            self.assertEqual((result['hits'], result['misses']), (480, 480))
            self.assertEqual(result['byte_hit_fraction'], .5)
            self.assertTrue(all(r['hits'] == r['misses'] == 10 for r in result['layers']))
        self.assertIsNone(data['predicted_tokens_per_second'])
        self.assertFalse(data['normal_request_latency_qualified'])
        self.assertFalse(data['production_promoted'])
        self.assertIsNone(data['coverage']['whole_request_covered'])
        self.assertEqual(data['source_result'], dict(status='unknown', complete=None))
        self.assertEqual(len(data['sources'][0]['sha256']), 64)

    def test_prefill_dedup_is_not_counted_as_cache_hits_or_decode(self):
        data = self.query(window(tokens=5))
        self.assertEqual(data['coverage']['logical_router_selections'], 2400)
        self.assertEqual(data['coverage']['within_pass_grouped_selections'], 1920)
        self.assertEqual(data['coverage']['compulsory_misses'], 480)
        self.assertEqual(data['coverage']['segments'][0]['single_token_windows'], 0)
        for curve in data['curves']:
            self.assertTrue(all(r['hits'] == 0 and r['demands'] == 480 for r in curve['policies']))

    def test_restarts_gaps_phases_and_session_changes_reset(self):
        tails = [window(), window(2), window(1, request_phase='append'), window(1, session_id='new')]
        for tail in tails:
            data = self.query(window() + tail)
            self.assertEqual(data['coverage']['cold_starts'], 2)
            self.assertEqual(data['curves'][-1]['policies'][0]['hits'], 0)

    def test_incomplete_windows_are_excluded_and_break_continuation(self):
        data = self.query(window() + window(1)[:20] + window(1))
        self.assertEqual(data['coverage']['excluded_incomplete_passes'], 20)
        self.assertEqual(data['coverage']['included_passes'], 96)
        self.assertEqual(data['coverage']['cold_starts'], 2)
        self.assertEqual(data['curves'][-1]['policies'][0]['hits'], 0)
        only_partial = self.query(window()[1:])
        self.assertEqual(only_partial['status'], 'insufficient_evidence')
        self.assertEqual(only_partial['curves'], [])
        backwards = self.query(list(reversed(window())))
        self.assertEqual(backwards['coverage']['included_passes'], 0)

    def test_missing_routes_never_reconstructed_from_read_records(self):
        rows = window()
        rows[15].pop('routes')
        rows[15]['records'] = [dict(expert=i) for i in range(10)]
        data = self.query(rows)
        self.assertEqual(data['status'], 'insufficient_evidence')
        self.assertEqual(data['coverage']['missing_route_passes'], 1)

    def test_phase_filter_does_not_stitch_over_omitted_work(self):
        rows = window(request_phase='decode') + window(1, request_phase='cached_replay') + window(1, request_phase='decode')
        data = self.query(rows, phase='decode')
        self.assertEqual(data['coverage']['filtered_passes'], 48)
        self.assertEqual(data['coverage']['cold_starts'], 2)
        self.assertEqual(data['curves'][-1]['policies'][0]['hits'], 0)

    def test_read_completion_order_and_measured_hit_counts_do_not_change_curve(self):
        rows = window() + window(1)
        original = self.query(rows)['curves']
        for row in rows:
            row.update(ready_hits=10, new_misses=0, records=[dict(expert=i, admitted_ns=100-i)
                                                         for i in range(9, -1, -1)])
            row['routes'].reverse()
        self.assertEqual(self.query(rows)['curves'], original)

    def test_rejects_unknown_or_mixed_artifact_build_and_invalid_routes(self):
        mutations = [lambda r:r.update(artifact_revision='c' * 40),
                     lambda r:r.update(build='c' * 64), lambda r:r.update(build='short'),
                     lambda r:r.update(layer=True), lambda r:r.update(offset=8192),
                     lambda r:r.update(tokens=2), lambda r:r.update(routes=[0] * 10),
                     lambda r:r.update(expert_bits=3), lambda r:r.update(expert_payload_bytes=1),
                     lambda r:r['routes'].__setitem__(0, True),
                     lambda r:r['routes'].__setitem__(0, -1),
                     lambda r:r['routes'].__setitem__(0, 512)]
        for mutate in mutations:
            rows = window()
            mutate(rows[1])
            with self.assertRaises(ValueError):
                self.query(rows)

    def test_profile_capture_limits_and_unfinished_status_preserved(self):
        path = self.put([dict(expert_dependencies=window(), complete=False, status='interrupted',
                              dependency_capture_limits=dict(passes_per_phase=48))], 'profile.json')
        data = cache_curve(self.index, path, [0])
        self.assertEqual(data['source_result'], dict(status='interrupted', complete=False))
        self.assertEqual(data['coverage']['capture_limits'], dict(passes_per_phase=48))
        self.assertIsNone(data['coverage']['whole_request_covered'])

    def test_budget_and_work_bounds_fail_instead_of_silent_partial_curve(self):
        path = self.put(window())
        for budgets in ([], [1] * 13, [1, 1], [-1], [22529], [True], [1.5]):
            with self.assertRaises(ValueError):
                cache_curve(self.index, path, budgets)
        for name, limit in [('MAX_LINES', 47), ('MAX_ACCESSES', 479), ('MAX_POLICY_STEPS', 1439)]:
            with patch('cache_simulation.' + name, limit), self.assertRaises(ValueError):
                cache_curve(self.index, path, [0])

    def test_profile_identity_and_format_cannot_override_route_identity(self):
        for changed in [dict(build='c' * 64), dict(artifact_revision='c' * 40), dict(expert_bits=3)]:
            path = self.put([dict(expert_dependencies=window(), **changed)], 'profile.json')
            with self.assertRaises(ValueError):
                cache_curve(self.index, path, [0])

    def test_cli_read_only_stale_evidence_and_empty_profile(self):
        source = self.put(window() + window(1))
        before = file_hash(source)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(['--db', str(self.db), 'cache', source, '--budget-mib', '1280'])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())['status'], 'simulated')
        self.assertEqual(file_hash(source), before)
        Path(source).write_text('{}\n')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--db', str(self.db), 'cache', source]), 2)
        empty = self.put([dict(expert_dependencies=[dict(layer=0, tokens=1)])], 'empty.json')
        self.assertEqual(cache_curve(self.index, empty)['status'], 'insufficient_evidence')


if __name__ == '__main__':
    unittest.main()
