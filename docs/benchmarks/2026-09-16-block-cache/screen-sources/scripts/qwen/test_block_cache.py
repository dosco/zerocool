import copy
import hashlib
import json
import random
import struct
import unittest

import block_cache_replay as replay
import build_block_cache_trace as builder
import capture_block_cache as capture
from cache_simulation import Clock,SegmentedLRU
from test_perfect_draft import fixture


def synthetic_trace():
    raw,_,work=fixture(4);cache=replay.Cache(1072);events=[];histories=[[] for _ in range(48)]
    def emit(name,detail):events.append(dict(sequence=len(events)+1,event=name,detail=detail))
    emit('begin',dict(kind='qwen_block_cache_events_v1',capacity=1072,policy='clock',build='build',
        payload_bytes=replay.PAYLOAD_BYTES,slot_bytes=replay.SLOT_BYTES,workspace_bound_bytes=32*1024**2))
    all_tokens=work['prompt_ids']+work['continuation_ids']
    for f,(offset,count) in enumerate([(0,72),(72,4),(76,4),(80,4),(84,4)]):
        emit('forward_begin',dict(offset=offset,tokens=count,input=all_tokens[offset:offset+count],phase='prefill' if f==0 else 'decode'))
        for layer in range(48):
            routes=list(range(10))*count
            emit('layer',dict(layer=layer,offset=offset,tokens=count,routes=routes,selected=list(range(10))))
            histories[layer].extend(routes)
            for expert in range(10):
                key=layer*512+expert;kind=1 if key in cache.lookup else 0
                _,slot,victim=cache.acquire(key,kind)
                emit('acquire',dict(key=key,slot=slot,victim=victim,acquisition=kind))
                emit('pin',dict(key=key,pins=cache.lease(key,False),ready=None))
                emit('release',dict(key=key,pins=cache.lease(key,True),ready=True))
        state=cache.state();digest=hashlib.sha256(json.dumps(state,separators=(',',':')).encode()).hexdigest()
        stats=dict(diagnostic_cache_state=digest,hits=cache.hits,misses=cache.misses,evictions=cache.evictions,
            application_read_bytes=cache.misses*replay.PAYLOAD_BYTES,ready_hits=cache.ready_hits,loading_joins=cache.joins,
            layer_hits=cache.layer_hits.copy(),layer_misses=cache.layer_misses.copy())
        emit('snapshot',dict(state=state,stats=stats))
        target=[dict(layer=i,tokens=offset+count,sha256=hashlib.sha256(struct.pack('<'+'i'*len(v),*v)).hexdigest())
                for i,v in enumerate(histories)]
        if f==0:raw['before']['expert_cache']=stats;raw['prime']['routes']=target
        else:raw['blocks'][f-1]['routes']=target
        emit('forward_end',dict(position=offset+count))
    raw['after']['expert_cache']=stats
    emit('end',dict(complete=True))
    return events,raw,work


def decode(events,raw,work):
    data=b''.join((json.dumps(e,separators=(',',':'))+'\n').encode() for e in events)
    raw=copy.deepcopy(raw)
    raw['expert_cache_trace']=dict(complete=True,events=len(events),bytes=len(data),workspace_bound_bytes=32*1024**2,performance_measurement=False)
    return replay.decode(data,raw,work,'build')


class BlockCacheTest(unittest.TestCase):
    def test_unleased_policies_match_existing_independent_simulators(self):
        rng=random.Random(723)
        for policy,legacy in [('clock',Clock),('slru',SegmentedLRU)]:
            for capacity in (1,3,17):
                a=replay.Cache(capacity,policy);b=legacy(capacity)
                for i in range(1500):
                    key=rng.randrange(40);hit,_,_=a.acquire(key)
                    self.assertEqual(hit,b.access(key,i))
                    a.lease(key,False);a.lease(key,True)
                if policy=='clock':
                    self.assertEqual(a.hand,b.hand)
                    self.assertEqual([e['key'] if e else None for e in a.slots],b.slots)
                    self.assertEqual([e['referenced'] if e else False for e in a.slots],b.referenced)

    def test_clock_keeps_pinned_reference_bits_and_rejects_exhaustion(self):
        c=replay.Cache(2)
        for key in (0,1):c.acquire(key);c.lease(key,False)
        with self.assertRaises(ValueError):c.acquire(2)
        c.lease(0,True);hit,_,victim=c.acquire(2)
        self.assertFalse(hit);self.assertEqual(victim,0)
        self.assertTrue(c.slots[c.lookup[1]]['referenced'])
        with self.assertRaises(ValueError):c.lease(0,True)

    def test_slru_can_evict_unpinned_protected_when_probation_is_busy(self):
        c=replay.Cache(3,'slru')
        for _ in range(2):c.acquire(0);c.lease(0,False);c.lease(0,True)
        for key in (1,2):c.acquire(key);c.lease(key,False)
        self.assertEqual(c.acquire(3)[2],0)

    def test_complete_trace_reconciles_routes_snapshots_and_report_counts(self):
        events,raw,work=synthetic_trace();t=decode(events,raw,work)
        self.assertEqual(len(t['forwards']),5);self.assertEqual(t['snapshots_verified'],5)
        self.assertTrue(t['native_counts_exact']);self.assertTrue(t['native_slot_state_exact'])
        self.assertEqual(replay.simulate(t,1072,'clock')['decode_misses'],0)

    def test_missing_reordered_unready_and_changed_events_fail_closed(self):
        events,raw,work=synthetic_trace()
        mutations=[lambda e:e.pop(),lambda e:e[20].update(sequence=20),
            lambda e:next(x for x in e if x['event']=='acquire')['detail'].update(victim=99),
            lambda e:next(x for x in e if x['event']=='release')['detail'].update(ready=False),
            lambda e:next(x for x in e if x['event']=='pin')['detail'].update(pins=2),
            lambda e:next(x for x in e if x['event']=='snapshot')['detail']['state'].update(hand=99),
            lambda e:next(x for x in e if x['event']=='layer')['detail']['routes'].__setitem__(0,1),
            lambda e:next(x for x in e if x['event']=='forward_begin')['detail']['input'].__setitem__(0,99)]
        for change in mutations:
            with self.subTest(change=change):
                bad=copy.deepcopy(events);change(bad)
                with self.assertRaises(ValueError):decode(bad,raw,work)

    def test_builder_rejects_drift_and_preserves_production_inputs(self):
        root=builder.ROOT
        storage=(root/'src/qwen/storage.cpp').read_text();generated=builder.storage_source(storage)
        self.assertIn('block_trace::acquire(key.value(),selected,trace_victim,0)',generated)
        self.assertEqual(generated.count('block_trace::lease'),3)
        with self.assertRaises(ValueError):builder.storage_source(storage.replace('        return Lease(e,0);','return {};'))
        harness=builder.harness_source(builder.TEMPLATE.read_text())
        self.assertIn('logits_bound+block_trace::workspace_bytes<=12*GiB',harness)
        self.assertIn('check(width==4 && !validation',harness)
        self.assertIn('block_trace::forward_end(state.tokens)',builder.model_source((root/'src/qwen/model.cpp').read_text()))

    def test_selection_requires_both_orders_and_joint_memory_headroom(self):
        def captures(a,b):
            return [dict(slots=s,observation=dict(clean_memory=True,clean_host=True,
                native_replay=dict(native_counts_exact=True,native_slot_state_exact=True),curves=[
                dict(policy='clock',capacity=1460,decode_misses=1000),dict(policy='slru',capacity=1460,decode_misses=m),
                dict(policy='clock',capacity=1536,decode_misses=850)])) for s,m in zip((1072,1460),(a,b))]
        budgets=[dict(target_slots=s,fits_with_headroom=True) for s in (1460,1536)]
        self.assertEqual(capture.select(captures(800,810),budgets)['selected_candidate'],dict(policy='slru',slots=1460))
        self.assertEqual(capture.select(captures(800,950),budgets)['selected_candidate'],dict(policy='clock',slots=1536))
        budgets[1]['fits_with_headroom']=False
        self.assertIsNone(capture.select(captures(800,950),budgets)['selected_candidate'])
        disturbed=captures(800,810);disturbed[0]['observation']['clean_memory']=False
        with self.assertRaises(ValueError):capture.select(disturbed,budgets)


if __name__=='__main__':unittest.main()
