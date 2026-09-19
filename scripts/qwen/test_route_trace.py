import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from cache_simulation import cache_curve, route_segments
from capture_routes import append_tokens, run, validate_capture
from evidence_index import Index
from route_trace import KIND, decode, progress


def fixture(steps=1, append=False):
    events = []
    request_id = forward_id = 0
    def emit(event, **details):
        events.append(dict(kind=KIND, sequence=len(events)+1, monotonic_ns=len(events)+1,
                           event=event, request_id=request_id, forward_id=forward_id, details=details))
    emit('trace_begin', identity=dict(build='b'*64, artifact_revision='b2c422f3c643e36f04227a64d61796b44a4b1029',
         budget_bytes=12*1024**3, prepared_manifest_sha256='c'*64, device='Apple M1 Pro', physical_bytes=32*1024**3))
    request_id = 1
    emit('request_begin', name=f'coding_routes_{steps}', input_token_ids=[760], max_tokens=steps+1, prime=False)
    for i in range(steps+1):
        forward_id += 1
        emit('forward_begin', session_id='1', offset=i, tokens=1, input_token_ids=[760 if i==0 else 10],
             request_phase='prefill' if i==0 else 'decode', expert_slots=1000)
        for layer in range(48): emit('routes', layer=layer, routes=list(range(10)))
        emit('forward_commit', position=i+1)
    emit('request_end', output_token_ids=[10]*(steps+1), finish_reason='length', reused_tokens=0)
    if append:
        request_id = 2
        emit('request_begin', name='append_128', input_token_ids=[760]+[10]*(steps+1)+[20]*128, max_tokens=33, prime=False)
        offset = steps+1
        for i in range(33):
            forward_id += 1
            inputs = [10]+[20]*128 if i==0 else [11]
            emit('forward_begin', session_id='1', offset=offset, tokens=len(inputs), input_token_ids=inputs,
                 request_phase='append' if i==0 else 'decode', expert_slots=1000)
            for layer in range(48): emit('routes', layer=layer, routes=list(range(10))*len(inputs))
            offset += len(inputs)
            emit('forward_commit', position=offset)
        emit('request_end', output_token_ids=[11]*33, finish_reason='length', reused_tokens=steps+1)
    emit('trace_end', status='complete')
    return events


def capture_fixture(steps=32, append=False):
    trace=decode(encoded(fixture(steps,append)))
    work=[dict(name=f'coding_routes_{steps}',tokens=[760],max_tokens=steps+1)]
    if append: work.append(dict(name='append_128',tokens=[20]*128,max_tokens=33,append=True))
    state=dict(diagnostic_stream_trunk=False, memory_plan=dict(panel_tokens=512), phase_memory=dict(pressure_resizes=0))
    raw=dict(complete=True,workloads=work,model_revision=trace['identity']['artifact_revision'],
             sampling=dict(temperature=0,top_k=20,top_p=.95,seed=0),runs=[])
    for i,r in enumerate(trace['requests']):
        raw['runs'].append(dict(name=work[i]['name'],profiling_enabled=True,
            runtime_cache_state='retained' if i else 'empty_at_process_start',repetition=0,
            prompt_tokens=len(r['input_token_ids']),
            pending_tokens_ingested=int(i>0),before=copy.deepcopy(state),after=copy.deepcopy(state),
            output_tokens=len(r['result']['output_token_ids']),**r['result']))
    return raw,trace,trace['identity'],work,dict(panel=512)


def encoded(events):
    return ''.join(json.dumps(r)+'\n' for r in events).encode()


class RouteTraceTest(unittest.TestCase):
    def test_committed_inputs_and_phase_progress(self):
        trace = decode(encoded(fixture()))
        self.assertTrue(trace['complete'])
        self.assertEqual(len(trace['rows']), 96)
        self.assertEqual([f['request_phase'] for f in trace['committed']], ['prefill','decode'])
        self.assertEqual(trace['requests'][0]['result']['output_token_ids'], [10,10])
        self.assertIsNone(trace['active_forward'])

    def test_partial_forward_is_not_committed_even_with_all_routes(self):
        events = fixture()
        for cut in (3, 10, 51):
            data = decode(encoded(events[:cut]))
            self.assertFalse(data['complete'])
            self.assertEqual(data['rows'], [])
            self.assertEqual(data['uncommitted_layer_passes'], cut-3)
        # An interrupted second forward cannot erase the first committed one.
        partial = decode(encoded(events[:60]) + b'{"kind":')
        self.assertTrue(partial['partial_final_line'])
        self.assertEqual(len(partial['rows']), 48)
        self.assertEqual(partial['active_forward']['request_phase'], 'decode')

    def test_abort_and_incomplete_end_preserve_completed_prefix_only(self):
        events = fixture()[:60]
        last = events[-1]
        for event, details in [('forward_abort',dict(captured_layers=7)), ('trace_end',dict(status='incomplete'))]:
            events.append(dict(kind=KIND, sequence=len(events)+1, monotonic_ns=len(events)+1,
                               event=event, details=details, request_id=1, forward_id=last['forward_id']))
        data = decode(encoded(events))
        self.assertFalse(data['complete'])
        self.assertEqual(data['aborted_forwards'], 1)
        self.assertEqual(len([r for _,r in data['rows'] if 'routes' in r]), 48)

    def test_corrupt_lifecycles_routes_positions_and_output_are_rejected(self):
        mutations = [lambda e:e[2].update(sequence=99), lambda e:e[2].update(monotonic_ns=0),
            lambda e:e[3].update(request_id=8), lambda e:e[3].update(forward_id=7),
            lambda e:e[3]['details'].update(layer=1), lambda e:e[3]['details'].update(routes=[0]*10),
            lambda e:e[2]['details'].update(offset=1), lambda e:e[51]['details'].update(position=2),
            lambda e:e[-2]['details'].update(output_token_ids=[11,10]),
            lambda e:e[-2]['details'].update(finish_reason='stop'), lambda e:e[-2]['details'].update(reused_tokens=1)]
        for mutate in mutations:
            events = fixture(); mutate(events)
            with self.assertRaises(ValueError): decode(encoded(events))
        with self.assertRaises(ValueError): decode(encoded(fixture()) + b'{')
        with self.assertRaises(ValueError): decode(encoded(fixture()+fixture()))

    def test_committed_trace_warms_across_prefill_decode_but_phase_filter_is_cold(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'trace.jsonl';path.write_bytes(encoded(fixture()))
            index=Index(Path(temp)/'index.sqlite')
            try:
                index.import_paths([path])
                all_phases=cache_curve(index,str(path),[1280])
                self.assertEqual(all_phases['coverage']['segment_count'],1)
                self.assertTrue(all_phases['coverage']['whole_request_covered'])
                self.assertTrue(all_phases['coverage']['session_boundaries_fully_known'])
                self.assertEqual(all_phases['curves'][0]['policies'][0]['hits'],480)
                decode_only=cache_curve(index,str(path),[1280],phase='decode')
                self.assertEqual(decode_only['curves'][0]['policies'][0]['hits'],0)
                self.assertTrue(decode_only['coverage']['phase_filter_starts_cold'])
                self.assertEqual(progress(path)['last_event']['event'],'trace_end')
            finally: index.close()

    def test_explicit_cache_reset_breaks_reuse(self):
        events=fixture();events.insert(52,dict(kind=KIND,event='cache_reset',request_id=1,forward_id=1,details={}))
        for i,e in enumerate(events): e.update(sequence=i+1,monotonic_ns=i+1)
        data=decode(encoded(events))
        self.assertNotEqual(data['rows'][0][1]['session_id'],data['rows'][-1][1]['session_id'])
        segments,coverage=route_segments(data['rows'],explicit_boundaries=True)
        self.assertEqual(len(segments),2)
        self.assertEqual(coverage['explicit_boundary_events'],1)
        self.assertEqual(coverage['missing_route_passes'],0)

    def test_capture_requires_32_actual_decode_forwards_and_bound_output(self):
        trace=decode(encoded(fixture(32))); evidence=trace['identity']
        work=[dict(name='coding_routes_32',tokens=[760],max_tokens=33)]
        state=dict(diagnostic_stream_trunk=False, memory_plan=dict(panel_tokens=512), phase_memory=dict(pressure_resizes=0))
        raw=dict(complete=True,workloads=work,model_revision=evidence['artifact_revision'],
                 sampling=dict(temperature=0,top_k=20,top_p=.95,seed=0),runs=[dict(name='coding_routes_32',
                 profiling_enabled=True,runtime_cache_state='empty_at_process_start',repetition=0,prompt_tokens=1,
                 reused_tokens=0,before=state,after=copy.deepcopy(state),output_token_ids=[10]*33,output_tokens=33,finish_reason='length')])
        with patch('capture_routes.check_machine'),patch('capture_routes.check_configuration'):
            result=validate_capture(raw,trace,evidence,work,dict(panel=512))
            self.assertTrue(result['target_32_steps_met'])
            raw['runs'][0]['output_tokens']=34
            with self.assertRaises(ValueError): validate_capture(raw,trace,evidence,work,dict(panel=512))

    def test_extended_capture_checks_pending_token_and_actual_reuse(self):
        args=capture_fixture(256,True)
        with patch('capture_routes.check_machine'),patch('capture_routes.check_configuration'):
            result=validate_capture(*args,decode_steps=256)
            self.assertTrue(result['target_steps_met'])
            first,followup=result['requests']
            self.assertEqual(first['committed_decode_steps'],256)
            self.assertEqual((followup['committed_decode_steps'],followup['reused_tokens'],followup['new_input_tokens']),
                             (32,257,128))
            self.assertEqual(followup['pending_tokens_ingested'],1)
            mutations=[lambda a:a[0]['runs'][1].update(pending_tokens_ingested=0),
                       lambda a:a[0]['runs'][1].update(reused_tokens=258),
                       lambda a:a[0]['runs'][1].update(runtime_cache_state='empty_at_process_start'),
                       lambda a:a[0]['runs'][1]['after']['memory_plan'].update(panel_tokens=256),
                       lambda a:a[1]['committed'][-1].update(session_id='2'),
                       lambda a:a[1]['requests'][1]['result'].update(reused_tokens=0),
                       lambda a:a[3][1].update(append=False)]
            for mutate in mutations:
                changed=copy.deepcopy(args);mutate(changed)
                with self.assertRaises(ValueError): validate_capture(*changed,decode_steps=256)

    def test_phase_counts_preserve_warmth_and_separate_requests(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'trace.jsonl';path.write_bytes(encoded(fixture(2,True)))
            index=Index(Path(temp)/'index.sqlite')
            try:
                index.import_paths([path]);curve=cache_curve(index,str(path),[0,1280])
                self.assertEqual(curve['coverage']['segment_count'],1)
                self.assertNotIn('access_spans',curve['coverage']['segments'][0])
                for capacity in curve['curves']:
                    for policy in capacity['policies']:
                        phases=policy['request_phases']
                        self.assertEqual([(p['request_id'],p['phase']) for p in phases],
                            [(1,'prefill'),(1,'decode'),(2,'append'),(2,'decode')])
                        for key in ('hits','misses','demands','application_miss_bytes'):
                            self.assertEqual(sum(p[key] for p in phases),policy[key])
                        self.assertEqual([p['demands'] for p in phases],[480,960,480,15360])
                        self.assertEqual([p['misses'] for p in phases],
                            [480,960,480,15360] if capacity['slots']==0 else [480,0,0,0])
            finally: index.close()

    def test_early_eos_remains_complete_without_meeting_decode_target(self):
        events=fixture(2)
        events[1]['details'].update(name='coding_routes_256',max_tokens=257)
        events[-2]['details'].update(output_token_ids=[10,10,248044],finish_reason='stop')
        args=list(capture_fixture(2));args[1]=decode(encoded(events))
        args[3][0].update(name='coding_routes_256',max_tokens=257)
        args[0]['runs'][0].update(name='coding_routes_256',output_token_ids=[10,10,248044],finish_reason='stop')
        with patch('capture_routes.check_machine'),patch('capture_routes.check_configuration'):
            result=validate_capture(*args,decode_steps=256)
        self.assertTrue(result['trace_complete'])
        self.assertFalse(result['target_steps_met'])
        self.assertEqual(result['committed_decode_steps'],2)

    def test_append_framing_is_preserved_and_only_body_is_truncated(self):
        pieces=dict(prefix=dict(tokens=[1,2]),body=dict(tokens=[3]*150),suffix=dict(tokens=[4,5,6]))
        self.assertEqual(append_tokens(pieces),[1,2]+[3]*123+[4,5,6])
        for part,tokens in [('prefix',[1]*125),('body',[3]),('suffix',[True])]:
            changed=copy.deepcopy(pieces);changed[part]['tokens']=tokens
            with self.assertRaises(ValueError): append_tokens(changed)

    def test_deadline_stops_before_any_inference_and_retains_unfinished_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'run'
            args=argparse.Namespace(output=output,time_limit=1)
            with patch('capture_routes.subprocess.run',side_effect=subprocess.TimeoutExpired('render',1)),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(args),2)
            summary=json.loads((output/'summary.json').read_text())
            self.assertFalse(summary['complete'])
            self.assertEqual(summary['status'],'time_budget_exhausted')
            self.assertFalse((output/'routes.jsonl').exists())
            self.assertTrue((output/'evidence-files.json').exists())


if __name__=='__main__': unittest.main()
