import json
from pathlib import Path
import unittest

from decode_timeline import partition,BUCKETS,summarize
from screen_cache import validate_request
from screen_route_selection import configs
import evidence_fixture


class DecodeTimelineTest(unittest.TestCase):
    def fixture(self):
        groups=[];dependencies=[]
        for layer in range(48):
            start=1000+layer*20
            for stage,submit,a,b,end in (('router',1,2,4,5),('routed_expert',10,11,15,16)):
                operations=[dict(request_phase='decode',offset=72,layer=layer,stage=stage)]
                if stage=='router' and layer:
                    operations.append(dict(request_phase='decode',offset=72,layer=layer-1,stage='expert_reduce'))
                groups.append(dict(submitted_ns=start+submit,completed_ns=start+end,
                    gpu_start_seconds=(start+a)/1e9,gpu_end_seconds=(start+b)/1e9,
                    operations=operations))
            records=[dict(expert=e,acquisition='new_miss',admitted_ns=start+6,read_queued_ns=start+5,
                read_started_ns=start+6,read_completed_ns=start+8,encoded_ns=start+9,submitted_ns=start+10,
                gpu_start_ns=start+11,gpu_end_ns=start+15,released_ns=start+16) for e in range(10)]
            dependencies.append(dict(build='build',artifact_revision='revision',layer=layer,offset=72,
                tokens=1,routes=list(range(10)),records=records))
        counters=dict(sample_ns=0,process={},metal=dict(live_command_groups=0))
        row=dict(profiling_enabled=True,prompt_tokens=72,output_token_ids=[1,2],token_latency_ms=[960/1e6],
            before=dict(metal=dict(dispatches=0)),
            after=dict(metal=dict(build_fingerprint='build',dispatches=143,kernels=dict(profile=True,counter_profile=False)),
                artifact_revision='revision',diagnostic_stream_trunk=False),
            decode_diagnostics=dict(kind='decode_step_diagnostics_v1',max_steps=32,total_decode_steps=1,
                captured_steps=1,omitted_steps=0,samples=[dict(step=0,offset=72,input_token_id=1,
                    begin_ns=1000,end_ns=1960,forward_ms=960/1e6,before=counters,after=counters)]))
        return dict(complete=True,runs=[row]),dict(truncated=False,command_groups=groups),dependencies

    def test_exclusive_overlap_and_duplicate_intervals(self):
        intervals=[('gpu_active',20,50),('gpu_active',30,50),('gpu_idle_pending_read',0,70),
                   ('gpu_idle_ready_expert',40,80),('gpu_idle_submitted',75,90),('gpu_idle_callback',85,95)]
        result=partition(0,100,intervals)
        self.assertEqual(result,dict(zip(BUCKETS,(30,15,25,20,5,5))))
        self.assertEqual(sum(result.values()),100)

    def test_clipping_empty_and_half_open_boundaries(self):
        self.assertEqual(partition(10,20,[('gpu_active',0,15),('gpu_idle_pending_read',15,30)])['gpu_active'],5)
        self.assertEqual(partition(0,10,[])['gpu_idle_other'],10)
        self.assertEqual(partition(0,10,[('gpu_active',5,5)])['gpu_active'],0)
        with self.assertRaises(ValueError):partition(0,10,[('gpu_active',5,4)])
        with self.assertRaises(ValueError):partition(5,5,[])

    def test_complete_join_and_order_independence(self):
        native,profile,deps=self.fixture();result=summarize(native,profile,deps)
        self.assertEqual(result['captured_tokens'],1)
        self.assertAlmostEqual(sum(result['mean_buckets_ms'].values()),960/1e6)
        self.assertEqual(result['tokens'][0]['expert_read_bytes'],480*2764800)
        self.assertFalse(result['normal_request_latency_qualified'])
        profile['command_groups'].reverse();deps.reverse()
        self.assertEqual(result,summarize(native,profile,deps))

    def test_incomplete_mismatched_and_reversed_sources_fail(self):
        changes=[lambda n,p,d:n.update(complete=False),lambda n,p,d:p.update(truncated=True),
            lambda n,p,d:p['command_groups'].pop(),lambda n,p,d:d.pop(),
            lambda n,p,d:d[0]['records'].pop(),lambda n,p,d:d[0].update(build='changed'),
            lambda n,p,d:d[0]['records'][0].update(read_started_ns=99999),
            lambda n,p,d:p['command_groups'][0].update(gpu_end_seconds=100),
            lambda n,p,d:d[0]['records'][0].update(submitted_ns=1011),
            lambda n,p,d:d[0]['records'][0].update(gpu_start_ns=1900)]
        for change in changes:
            n,p,d=self.fixture();change(n,p,d)
            with self.assertRaises(ValueError):summarize(n,p,d)

    def test_intervening_attention_does_not_inflate_boundary_gap(self):
        native,profile,deps=self.fixture()
        original=summarize(native,profile,deps)['mean_layer_boundary_submission_gap_ms']
        group=profile['command_groups'][2]
        router=group['operations'].pop(0)
        profile['command_groups'].append(dict(submitted_ns=1026,completed_ns=1029,
            gpu_start_seconds=1027/1e9,gpu_end_seconds=1028/1e9,operations=[router]))
        self.assertEqual(summarize(native,profile,deps)['mean_layer_boundary_submission_gap_ms'],original)

    def test_explicit_short_output_keeps_default_confirmation_length(self):
        evidence_fixture.require('docs/benchmarks/2026-09-12-route-five-pairs/raw/pair-0-candidate.json')
        root=Path(__file__).resolve().parents[2]/'docs/benchmarks/2026-09-12-route-five-pairs/raw'
        raw=json.loads((root/'pair-0-candidate.json').read_text());evidence=json.loads((root/'summary.json').read_text())['identity']
        raw['runs']=raw['runs'][:1];raw['workloads']=raw['workloads'][:1]
        raw['workloads'][0]['max_tokens']=17
        row=raw['runs'][0];row['output_tokens']=17;row['output_token_ids']=row['output_token_ids'][:17];row['token_latency_ms']=row['token_latency_ms'][:16]
        result=validate_request(raw,evidence,configs()[1],raw['workloads'],{},output_tokens=17)
        self.assertEqual(result[0]['tokens_per_second'],16000/row['decode_wall_ms'])
        with self.assertRaises(ValueError):validate_request(raw,evidence,configs()[1],raw['workloads'],{})
        with self.assertRaises(ValueError):validate_request(raw,evidence,configs()[1],raw['workloads'],{},output_tokens=16)


if __name__=='__main__':unittest.main()
