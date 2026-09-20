"""Lease-aware fixed-order cache replay. Counts application reads, never latency."""
from collections import OrderedDict
import hashlib
import json
import struct

from cache_simulation import PAYLOAD_BYTES, SLOT_BYTES, LAYERS, EXPERTS, TOP_K, integer


def require(ok,message):
    if not ok:raise ValueError(message)


class Cache:
    def __init__(self,capacity,policy='clock'):
        integer(capacity,1,4096,'capacity');require(policy in ('clock','slru'),'Unknown replay policy')
        self.capacity,self.policy=capacity,policy
        self.slots=[None]*capacity;self.lookup={};self.hand=0
        self.probation=OrderedDict();self.protected=OrderedDict()
        self.hits=self.misses=self.evictions=self.ready_hits=self.joins=0
        self.layer_hits=[0]*LAYERS;self.layer_misses=[0]*LAYERS

    def acquire(self,key,recorded_kind=None):
        integer(key,0,LAYERS*EXPERTS-1,'expert key')
        if key in self.lookup:
            slot=self.lookup[key];e=self.slots[slot];e['referenced']=True;self.hits+=1;self.layer_hits[key//EXPERTS]+=1
            if self.policy=='slru':
                self.probation.pop(key,None);self.protected.pop(key,None);self.protected[key]=None
                while len(self.protected)>3*self.capacity//4:
                    demoted,_=self.protected.popitem(last=False);self.probation[demoted]=None
            if recorded_kind==1:self.ready_hits+=1
            elif recorded_kind==2:self.joins+=1
            return True,slot,-1
        selected=None
        if self.policy=='slru' and len(self.lookup)==self.capacity:
            selected=next((self.lookup[k] for q in (self.probation,self.protected) for k in q
                           if not self.slots[self.lookup[k]]['pins']),None)
        else:
            for _ in range(2*self.capacity+1):
                slot=self.hand;self.hand=(self.hand+1)%self.capacity;e=self.slots[slot]
                if e:
                    if self.policy=='slru' or e['pins']:continue
                    if e['referenced']:e['referenced']=False;continue
                selected=slot;break
        require(selected is not None,'Replay cache exhausted by leases')
        old=self.slots[selected];victim=old['key'] if old else -1
        if old:
            del self.lookup[victim];self.probation.pop(victim,None);self.protected.pop(victim,None);self.evictions+=1
        self.slots[selected]=dict(key=key,referenced=True,pins=0);self.lookup[key]=selected
        if self.policy=='slru':self.probation[key]=None
        self.misses+=1;self.layer_misses[key//EXPERTS]+=1
        return False,selected,victim

    def lease(self,key,release):
        require(key in self.lookup,'Lease refers to evicted or missing entry')
        e=self.slots[self.lookup[key]];e['pins']+=-1 if release else 1
        require(e['pins']>=0,'Release without a live lease')
        return e['pins']

    def pins(self):return sum(e['pins'] for e in self.slots if e)

    def state(self):
        require(self.policy=='clock' and self.pins()==0,'Full snapshot needs drained CLOCK state')
        return dict(kind='expert_cache_eviction_state_v1',policy='clock',capacity=self.capacity,stride=SLOT_BYTES,
            hand=self.hand,protected=0,oldest=[None,None],newest=[None,None],slots=[None if e is None else
                dict(slot=i,key=e['key'],referenced=e['referenced'],pins=0,future_valid=True,ready=True,
                     queue=0,previous=None,next=None) for i,e in enumerate(self.slots)])


def decode(data,raw,work,fingerprint,*,expected_patterns=None):
    require(len(data)<=32*1024**2,'Oversized block trace')
    lines=data.splitlines();require(2<=len(lines)<=100000,'Missing or excessive events')
    events=[json.loads(line) for line in lines]
    require(all(set(e)=={'sequence','event','detail'} and type(e['sequence']) is int and e['sequence']==i+1
                for i,e in enumerate(events)), 'Missing, duplicate or reordered event sequence')
    begin=events[0]
    require(begin['event']=='begin' and events[-1]['event']=='end' and events[-1]['detail']==dict(complete=True),
            'Capture is incomplete')
    capacity=raw['configuration']['expert_slots']
    require(begin['detail']==dict(kind='qwen_block_cache_events_v1',capacity=capacity,policy='clock',build=fingerprint,
        payload_bytes=PAYLOAD_BYTES,slot_bytes=SLOT_BYTES,workspace_bound_bytes=32*1024**2),'Changed trace identity/format')
    meta=raw['expert_cache_trace']
    require(meta==dict(complete=True,events=len(events),bytes=len(data),workspace_bound_bytes=32*1024**2,
                      performance_measurement=False),'Native trace receipt differs')
    cache=Cache(capacity);forwards=[];active=None;pending_pin=None;snapshots=[]
    histories=[[] for _ in range(LAYERS)];requests=[];max_pins=0
    patterns=expected_patterns if expected_patterns is not None else [(0,72,'prefill')]+[(i,4,'decode') for i in (72,76,80,84)]
    require(isinstance(patterns,list) and 2<=len(patterns)<=65,'Missing bounded forward coverage')
    position=0
    for i,(at,count,phase) in enumerate(patterns):
        integer(at,0,8191,'forward offset');integer(count,1,128,'forward tokens')
        require(at==position and phase==('prefill' if i==0 else 'decode'),'Noncontiguous expected forwards')
        position+=count
    require(position==len(work['prompt_ids'])+len(work['continuation_ids']) and
        patterns[0][1]==len(work['prompt_ids']), 'Expected coverage does not match workload')
    tokens=work['prompt_ids']+work['continuation_ids']
    for event in events[1:-1]:
        name,d=event['event'],event['detail']
        require(pending_pin is None or name=='pin','Acquisition was not pinned immediately')
        if name=='forward_begin':
            require(active is None and cache.pins()==0 and len(forwards)<len(patterns),'Overlapping/extra forward or live prior leases')
            at,count,phase=patterns[len(forwards)]
            expected=dict(offset=at,tokens=count,input=tokens[at:at+count],phase=phase)
            require(json.dumps(d,sort_keys=True)==json.dumps(expected,sort_keys=True),'Changed forward coverage or tokens')
            active=dict(offset=at,tokens=count,phase=phase,layers=0,selected=[],index=0,
                        hits_before=cache.hits,misses_before=cache.misses,demands=[],snapshots=0)
        elif name=='layer':
            require(active is not None and cache.pins()==0 and active['index']==len(active['selected']),
                    'Layer boundary omitted demands or live users')
            layer=active['layers'];routes=d['routes'];selected=d['selected']
            for key,low,high in [('layer',0,47),('offset',0,8191),('tokens',1,8192)]:integer(d[key],low,high,key)
            require(layer<LAYERS and d['layer']==layer and d['offset']==active['offset'] and
                    d['tokens']==active['tokens'] and len(routes)==active['tokens']*TOP_K,'Invalid layer coverage')
            for start in range(0,len(routes),TOP_K):
                group=routes[start:start+TOP_K]
                require(len(set(group))==TOP_K,'Duplicate routed expert')
                for key in group:integer(key,0,EXPERTS-1,'route')
            ascending=sorted(set(routes));keys=[layer*EXPERTS+k for k in ascending]
            for expert in selected:integer(expert,0,EXPERTS-1,'selected expert')
            expected=ascending if active['tokens']>32 else ([k%EXPERTS for k in keys if k in cache.lookup]+
                                                         [k%EXPERTS for k in keys if k not in cache.lookup])
            require(selected==expected,'Native hit-first demand ordering differs')
            histories[layer].extend(routes)
            active.update(layers=layer+1,selected=[layer*EXPERTS+k for k in selected],index=0)
        elif name=='acquire':
            require(active is not None and active['index']<len(active['selected']) and
                    d['key']==active['selected'][active['index']], 'Acquisition does not match selected order')
            kind=integer(d['acquisition'],0,2,'acquisition kind')
            integer(d['slot'],0,capacity-1,'slot');integer(d['victim'],-1,LAYERS*EXPERTS-1,'victim')
            hit,slot,victim=cache.acquire(d['key'],kind)
            require(hit==(kind!=0) and (slot,victim)==(d['slot'],d['victim']), 'Native CLOCK decision does not replay')
            active['index']+=1;active['demands'].append(d['key']);pending_pin=d['key']
            requests.append(dict(event='acquire',key=d['key'],forward=len(forwards)))
        elif name in ('pin','release'):
            release=name=='release';key=d['key']
            integer(key,0,LAYERS*EXPERTS-1,'lease key');integer(d['pins'],0,32,'pin count')
            if not release:require(key==pending_pin,'Pin does not own preceding acquisition')
            require(d['ready'] is True if release else d['ready'] is None,
                    'Lease released before its read completed')
            require(d['pins']==cache.lease(key,release),'Native pin count differs')
            max_pins=max(max_pins,cache.pins());require(max_pins<=32,'Lease window exceeded')
            if not release:pending_pin=None
            requests.append(dict(event=name,key=key,forward=len(forwards)))
        elif name=='snapshot':
            if active is not None:
                require(active['layers']==48 and active['index']==len(active['selected']),
                        'Snapshot precedes completed layer work')
                active['snapshots']+=1
            state=cache.state();encoded=json.dumps(state,separators=(',',':'))
            require(json.dumps(d['state'],separators=(',',':'))==encoded,'Native cache state does not replay')
            digest=hashlib.sha256(encoded.encode()).hexdigest();stats=d['stats']
            for key in ('hits','misses','evictions','application_read_bytes','ready_hits','loading_joins'):
                integer(stats[key],0,100000*PAYLOAD_BYTES,key)
            for key in ('layer_hits','layer_misses'):
                for count in stats[key]:integer(count,0,100000,key)
            require(stats['diagnostic_cache_state']==digest and
                    all(stats[k]==v for k,v in dict(hits=cache.hits,misses=cache.misses,evictions=cache.evictions,
                        application_read_bytes=cache.misses*PAYLOAD_BYTES,ready_hits=cache.ready_hits,loading_joins=cache.joins,
                        layer_hits=cache.layer_hits,layer_misses=cache.layer_misses).items()),'Native cache counters do not replay')
            snapshots.append(dict(hash=digest,hits=cache.hits,misses=cache.misses))
        elif name=='forward_end':
            require(active is not None and active['layers']==48 and active['index']==len(active['selected']) and
                    active['snapshots']>=1 and cache.pins()==0 and
                    d==dict(position=active['offset']+active['tokens']), 'Incomplete forward')
            target=raw['prime']['routes'] if not forwards else raw['blocks'][len(forwards)-1]['routes']
            expected=[dict(layer=i,tokens=d['position'],sha256=hashlib.sha256(
                struct.pack('<'+'i'*len(v),*v)).hexdigest()) for i,v in enumerate(histories)]
            require(target==expected,'Captured routes differ from native state identity')
            forwards.append(dict(offset=active['offset'],tokens=active['tokens'],phase=active['phase'],
                hits=cache.hits-active['hits_before'],misses=cache.misses-active['misses_before'],
                distinct_records=len(set(active['demands'])),demands=active['demands']))
            active=None
        else:raise ValueError('Unknown cache trace event '+name)
    require(active is None and pending_pin is None and len(forwards)==len(patterns) and cache.pins()==0,'Incomplete terminal state')
    for prefix,state in [('before',forwards[:1]),('after',forwards)]:
        hits=sum(f['hits'] for f in state);misses=sum(f['misses'] for f in state);actual=raw[prefix]['expert_cache']
        require((actual['hits'],actual['misses'],actual['application_read_bytes'])==(hits,misses,misses*PAYLOAD_BYTES) and
                dict(hash=actual['diagnostic_cache_state'],hits=hits,misses=misses) in snapshots,
                'Trace misses native report boundary')
    return dict(capacity=capacity,events=events,requests=requests,forwards=forwards,
                snapshots_verified=len(snapshots),max_pins=max_pins,native_counts_exact=True,native_slot_state_exact=True)


def simulate(trace,capacity,policy):
    cache=Cache(capacity,policy);blocks=[dict(hits=0,misses=0) for _ in trace['forwards']]
    for event in trace['requests']:
        if event['event']=='acquire':
            hit,_,_=cache.acquire(event['key']);blocks[event['forward']]['hits' if hit else 'misses']+=1
        else:cache.lease(event['key'],event['event']=='release')
    require(cache.pins()==0,'Simulation left live leases')
    hit=sum(b['hits'] for b in blocks[1:]);miss=sum(b['misses'] for b in blocks[1:])
    return dict(capacity=capacity,policy=policy,blocks=blocks,decode_hits=hit,decode_misses=miss,
        decode_application_read_bytes=miss*PAYLOAD_BYTES,decode_hit_fraction=hit/(hit+miss),
        simulated=True,latency_prediction=None,ordering='captured native acquire/pin/release order')


def curves(trace):
    return [simulate(trace,n,p) for n in (1072,1460,1536) for p in ('clock','slru')]
