import copy
import json
from pathlib import Path
import tempfile
import unittest

import build_horizon_cache_trace as builder
import horizon_cache_replay as replay
from screen_verifier_horizon import validate
from test_block_cache import synthetic_trace
from test_verifier_horizon import fixture
import evidence_fixture


def setUpModule():
    evidence_fixture.require('docs/benchmarks/2026-09-17-mtp-widths/other-coding-01/case-0-pair-0-width-4.json')


def trace_fixture(width=4, count=16):
    work = dict(prompt_ids=list(range(72)), continuation_ids=list(range(count)))
    coverage = replay.patterns(work, width)
    events, old, work = synthetic_trace(tuple(p[1] for p in coverage), 1460)
    forwards = [dict(position=at+rows, routes=routes) for (at, rows, _), routes in zip(coverage,
        [old['prime']['routes']]+[b['routes'] for b in old['blocks']])]
    raw = dict(cache_trace_enabled=True, performance_measurement=False, compute_tile_cap=4,
        requested_width=width, trace_forwards=forwards, before=old['before'], after=old['after'])
    return events, raw, work


def decode(events, raw, work):
    data = b''.join((json.dumps(e, separators=(',', ':'))+'\n').encode() for e in events)
    raw = copy.deepcopy(raw)
    raw['target_cache_trace'] = dict(complete=True, events=len(events), bytes=len(data),
                                   workspace_bound_bytes=replay.WORKSPACE, performance_measurement=False)
    return replay.decode(data, raw, work, 'build')


class HorizonCacheTests(unittest.TestCase):
    def test_complete_four_eight_and_irregular_tail_replay(self):
        for width, count in ((4, 16), (8, 16), (4, 9), (8, 11)):
            events, raw, work = trace_fixture(width, count)
            trace = decode(events, raw, work); result = replay.summarize(trace)
            self.assertTrue(result['native_replay']['native_slot_state_exact'])
            self.assertEqual(sum(f['tokens'] for f in trace['forwards'][1:]), count)
            self.assertEqual(trace['snapshots_verified'], len(replay.patterns(work, width)))
            self.assertEqual(result['row_use'][1]['distinct_row_count_histogram'], {str(width): 480})
            self.assertEqual(result['row_use'][1]['single_row_records'], 0)
            for curve in result['curves']:
                self.assertIsNone(curve['latency_prediction'])
                self.assertEqual(curve['decode_misses'], 0)
                self.assertIsNone(curve['miss_reduction_fraction'])
            self.assertEqual([c['capacity_admitted'] for c in result['curves']], [True, False, False])

    def test_coverage_is_derived_from_workload_not_trace_claims(self):
        events, raw, work = trace_fixture(8)
        for mutation in (lambda r: r['trace_forwards'].pop(),
                         lambda r: r['trace_forwards'][1].update(position=73),
                         lambda r: r['trace_forwards'][1]['routes'][0].update(sha256='0'*64),
                         lambda r: r.update(requested_width=4),
                         lambda r: r.update(performance_measurement=True)):
            bad = copy.deepcopy(raw); mutation(bad)
            with self.assertRaises(ValueError): decode(events, bad, work)
        for width, count in ((True, 16), (1, 16), (8, 65), (4, 0)):
            with self.assertRaises(ValueError):
                replay.patterns(dict(prompt_ids=[1], continuation_ids=[1]*count), width)

    def test_draft_leaks_and_missing_or_unready_events_fail(self):
        events, raw, work = trace_fixture(8)
        changes = [lambda e: e.pop(), lambda e: e[30].update(sequence=30),
            lambda e: next(x for x in e if x['event'] == 'acquire')['detail'].update(key=48*512),
            lambda e: next(x for x in e if x['event'] == 'release')['detail'].update(ready=False)]
        for edit in changes:
            bad = copy.deepcopy(events); edit(bad)
            with self.assertRaises(ValueError): decode(bad, raw, work)

    def test_trace_allowance_is_explicit_and_in_combined_admission(self):
        raw, work = fixture()
        args = (work, raw['input_sha256'], raw['producer_binary_sha256'], 8, False)
        with self.assertRaises(ValueError): validate(raw, *args, trace_workspace_bytes=replay.WORKSPACE)
        raw['admission']['trace_workspace_bytes'] = replay.WORKSPACE
        with self.assertRaises(ValueError): validate(raw, *args, trace_workspace_bytes=replay.WORKSPACE)
        raw['admission']['combined_bytes'] += replay.WORKSPACE
        self.assertTrue(validate(raw, *args, trace_workspace_bytes=replay.WORKSPACE)['exact_full_logits'])
        with self.assertRaises(ValueError): validate(raw, *args)

    def test_omitted_middle_snapshot_cannot_hide_behind_final_counts(self):
        events, raw, work = trace_fixture(4)
        index = [i for i, e in enumerate(events) if e['event'] == 'snapshot'][1]
        events.pop(index)
        for i, event in enumerate(events): event['sequence'] = i+1
        with self.assertRaises(ValueError): decode(events, raw, work)

    def test_comparison_never_uses_trace_timing_and_requires_exact_routes(self):
        a, _ = fixture(4)
        a.update(cache_trace_enabled=False, performance_measurement=False, compute_tile_cap=4,
            trace_forwards=[{'routes': ['route']}], draft_priming_before={'calls': 2},
            draft_priming_after={'calls': 2}, embedding_rows_before={'hits': 1}, embedding_rows_after={'hits': 2})
        b = copy.deepcopy(a); b['cache_trace_enabled'] = True; b['decode_wall_ns'] *= 5
        self.assertEqual(replay.compare_modes(a, b), dict(exact_logits_and_state=True, exact_routes=True,
            exact_initial_caches=True, width=4, timing_used=False, performance_measurement=False))
        for field in ('trace_forwards', 'draft_before', 'draft_priming_before', 'final_target_state'):
            bad = copy.deepcopy(b); bad[field] = None
            with self.assertRaises(ValueError): replay.compare_modes(a, bad)

    def test_source_copy_preserves_lazy_storage_and_scopes_target_only(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory).resolve(); sources = builder.generated(out)
            storage = sources[out/'storage.cpp']
            self.assertIn('rows_.reserve(count)', storage)
            self.assertIn('next_=(next_+1)%rows_.capacity()', storage)
            self.assertEqual(storage.count('horizon_trace::lease'), 3)
            model = sources[out/'model.cpp']
            self.assertIn('horizon_trace::ForwardScope target_trace', model)
            self.assertIn('horizon_trace::forward_end(state.tokens,route_identity())', model)
            draft = sources[out/'mtp_draft.cpp']
            self.assertNotIn('horizon_trace', draft)
            self.assertIn('mtp_fixed_priming::execute', draft)
            probe = sources[out/'probe.cpp']
            self.assertIn('options.audit_routes=true', probe)
            self.assertIn('prompt.size()<=128 && count>=1 && count<=64', probe)
            self.assertIn('report["performance_measurement"]=false', probe)
            self.assertIn('+block_trace::workspace_bytes', probe)
            self.assertIn('mtp_fixed_priming::Scope fixed_draft_priming', probe)
            cfg = builder.settings(out)
            self.assertIn(str(out/'mtp_draft.cpp'), [cmd[-1] for cmd in cfg['compiler']])


if __name__ == '__main__': unittest.main()
