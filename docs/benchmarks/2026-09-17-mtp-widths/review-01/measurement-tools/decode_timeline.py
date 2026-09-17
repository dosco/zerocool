"""Join existing command, expert-read and token timestamps without adding wait sums.

Buckets describe observed overlap, not a causal speedup prediction. The input
is an instrumented normal request with complete command and dependency capture.
"""
from collections import defaultdict
import math
import statistics

from diagnose_decode_startup import analyze_steps


BUCKETS=('gpu_active','gpu_idle_submitted','gpu_idle_ready_expert',
         'gpu_idle_pending_read','gpu_idle_callback','gpu_idle_other')


def ns(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:
        raise ValueError('Invalid timestamp')
    return int(value)


def partition(begin,end,intervals):
    """Union overlapping intervals; each nanosecond belongs to one bucket."""
    if not 0<=begin<end:raise ValueError('Invalid token interval')
    events=defaultdict(list);events[begin];events[end]
    for kind,a,b in intervals:
        if kind not in BUCKETS[:-1] or a>b:raise ValueError('Invalid interval')
        a,b=max(begin,a),min(end,b)
        if a<b:events[a].append((kind,1));events[b].append((kind,-1))
    active=defaultdict(int);totals=dict.fromkeys(BUCKETS,0);previous=begin
    for at in sorted(events):
        kind=next((k for k in BUCKETS[:-1] if active[k]),BUCKETS[-1])
        totals[kind]+=at-previous
        for key,change in events[at]:active[key]+=change
        previous=at
    assert sum(totals.values())==end-begin
    return totals


def summarize(native,profile,dependencies):
    if (native.get('complete') is not True or len(native.get('runs',[]))!=1 or
        profile.get('truncated') is not False):raise ValueError('Incomplete normal trace')
    row=native['runs'][0];analyze_steps(row)
    state=row['after'];build=state['metal']['build_fingerprint'];revision=state['artifact_revision']
    if (not state['metal']['kernels']['profile'] or state['metal']['kernels']['counter_profile'] or
        state['diagnostic_stream_trunk']):raise ValueError('Requires original command groups without per-kernel passes')
    groups=profile['command_groups'];by_submit={};intervals=[]
    for g in groups:
        submitted,completed=ns(g['submitted_ns']),ns(g['completed_ns'])
        start,end=ns(g['gpu_start_seconds']*1e9),ns(g['gpu_end_seconds']*1e9)
        if not submitted<=start<=end<=completed or submitted in by_submit:
            raise ValueError('Missing, reversed or duplicate command timestamps')
        by_submit[submitted]=g
        intervals.extend((('gpu_active',start,end),('gpu_idle_submitted',submitted,start),
                          ('gpu_idle_callback',end,completed)))
    coverage=profile.get('coverage','all-dispatches')
    if coverage not in ('all-dispatches','decode-only'):raise ValueError('Unknown profile coverage')
    if coverage=='decode-only':
        a,b=(row['phases']['decode'][k]['metal'] for k in ('before','after'))
        expected=b['dispatches']-a['dispatches']
    else:expected=state['metal']['dispatches']-row['before']['metal']['dispatches']
    if sum(len(g['operations']) for g in groups)!=expected:raise ValueError('Missing dispatch coverage')
    for d in dependencies:
        if d.get('build')!=build or d.get('artifact_revision')!=revision:raise ValueError('Dependency identity differs')
    results=[]
    for sample in row['decode_diagnostics']['samples']:
        begin,end=ns(sample['begin_ns']),ns(sample['end_ns']);offset=sample['offset']
        passes=[d for d in dependencies if d['tokens']==1 and d['offset']==offset]
        if sorted(d['layer'] for d in passes)!=list(range(48)):raise ValueError('Incomplete or duplicate layer coverage')
        extra=[];last_experts={};read_bytes=0
        for d in passes:
            records=d['records'];routes=d['routes']
            if (len(records)!=10 or len(set(routes))!=10 or
                sorted(r['expert'] for r in records)!=sorted(routes)):
                raise ValueError('Incomplete selected-expert coverage')
            last_experts[d['layer']]=0
            for r in records:
                admitted,ready,encoded,submitted,released=[ns(r[k]) for k in
                    ('admitted_ns','read_completed_ns','encoded_ns','submitted_ns','released_ns')]
                if not begin<=admitted<=encoded<=submitted<=released<=end:
                    raise ValueError('Expert ownership lies outside its token or reverses')
                ready=max(admitted,ready)
                if ready>encoded or submitted not in by_submit:raise ValueError('Data not ready or missing GPU user')
                g=by_submit[submitted]
                if abs(r['gpu_start_ns']-int(g['gpu_start_seconds']*1e9))>2 or abs(r['gpu_end_ns']-int(g['gpu_end_seconds']*1e9))>2:
                    raise ValueError('Expert and command clocks differ')
                last_experts[d['layer']]=max(last_experts[d['layer']],ns(r['gpu_end_ns']))
                if r['acquisition'] not in ('ready_hit','new_miss','loading_join'):raise ValueError('Unknown acquisition')
                if r['acquisition']!='ready_hit':
                    q,s,c=[ns(r[k]) for k in ('read_queued_ns','read_started_ns','read_completed_ns')]
                    if not q<=s<=c:raise ValueError('Reversed read timing')
                    extra.append(('gpu_idle_pending_read',admitted,ready))
                    if r['acquisition']=='new_miss':read_bytes+=2764800
                extra.append(('gpu_idle_ready_expert',ready,submitted))
        active_groups=[g for g in groups if any(o['request_phase']=='decode' and o['offset']==offset for o in g['operations'])]
        # Gap after the final expert GPU use before its reduction group is
        # submitted. The next router can require intervening attention GPU
        # work, so measuring all the way to that router would overstate this.
        # Other work can occupy the same interval;
        # report it as a separate overlap measure, never add it to buckets.
        boundary_gaps=[]
        for layer in range(47):
            following=[g for g in active_groups if any(o['stage']=='expert_reduce' and o['layer']==layer for o in g['operations'])]
            if len(following)!=1:raise ValueError('Missing expert reduction command')
            boundary_gaps.append(max(0,following[0]['submitted_ns']-last_experts[layer]))
        buckets=partition(begin,end,intervals+extra)
        results.append(dict(step=sample['step'],offset=offset,forward_ms=(end-begin)/1e6,
            buckets_ms={k:v/1e6 for k,v in buckets.items()},expert_read_bytes=read_bytes,
            command_groups=len(active_groups),layer_boundary_submission_gap_ms=sum(boundary_gaps)/1e6))
    if not results:raise ValueError('No committed decode samples')
    aggregate={k:statistics.mean(r['buckets_ms'][k] for r in results) for k in BUCKETS}
    return dict(kind='decode_overlap_timeline_v1',complete=True,build=build,artifact_revision=revision,
        captured_tokens=len(results),target_ms_per_token=200,mean_forward_ms=statistics.mean(r['forward_ms'] for r in results),
        mean_buckets_ms=aggregate,
        mean_layer_boundary_submission_gap_ms=statistics.mean(r['layer_boundary_submission_gap_ms'] for r in results),
        tokens=results,normal_request_latency_qualified=False,production_promoted=False,
        limitations=['Original command groups are preserved, but trace collection and JSON writing add overhead.',
            'Bucket precedence is GPU active, submitted command, ready expert, pending read, callback, other; buckets sum to captured forward time.',
            'Overlap does not establish why a coordinator waited or predict a speedup; ready work can be blocked by buffer or GPU capacity.',
            'Layer boundary gaps overlap the buckets and must not be added to them.',
            'Only the captured short-request token positions are covered; no 2K/4K acceptance claim.'])
