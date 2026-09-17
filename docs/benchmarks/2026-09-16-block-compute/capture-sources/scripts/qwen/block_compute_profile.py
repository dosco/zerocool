"""Join bounded verifier blocks with real expert dependencies and GPU work."""
from collections import Counter, defaultdict
import copy
import math

from cache_residency import require
from decode_timeline import partition, ns
import perfect_draft as verifier
from q4_request_profile import command_classes, digest
from build_block_profile import WORKSPACE

LIMITATIONS = [
    'Four fixed four-token blocks after a 72-token prompt; no long-context, real-draft or sustained-use qualification.',
    'Command profiling preserves submissions but adds metadata collection; per-dispatch profiling additionally splits compute passes. Neither measures normal speed.',
    'Exclusive overlap buckets describe observed concurrency, not causality or removable latency. Mixed commands are never charged in full to each kernel.',
    'Counter durations rank actual four-token work but cannot be subtracted from normal request latency or joined across different processes as one timeline.',
    'Only complete clean captures may support a next experiment; no production promotion or automatic kernel selection.']


def exact_observation(raw, frozen, work, input_sha, reference, mode):
    require(mode in ('commands', 'dispatch') and raw.get('block_profile_mode') == mode and
        raw.get('profile_workspace_bytes') == WORKSPACE, 'Changed profiling mode or workspace')
    # Validate the explicitly different instrumentation before projecting onto
    # the existing verifier's arithmetic/state/memory contract. Raw evidence is
    # never rewritten, and instrumented timing is never used for performance.
    normalized = copy.deepcopy(raw)
    for when in ('before', 'after'):
        kernels = raw[when]['metal']['kernels']
        require(kernels['profile'] is (when == 'after') and
            kernels['counter_profile'] is (when == 'after' and mode == 'dispatch') and
            kernels['profile_decode_only'] is False, 'Unexpected profiling phase')
        normalized[when]['metal']['kernels']['profile'] = False
        normalized[when]['metal']['kernels']['counter_profile'] = False
    actual = verifier.observe(normalized, frozen, work, 4, False, input_sha, expert_slots=1460)
    control = verifier.observe(reference, frozen, work, 4, False, input_sha, expert_slots=1460)
    require(all(actual[k] == control[k] for k in ('prime', 'row_logits_sha256', 'endpoints',
        'snapshot_allocated_bytes', 'host_logits_bound_bytes')), 'Profile changed exact model results or prime cache')
    require(raw['memory_plan']['planned_bytes']+128*1024**2+raw['host_logits_bound_bytes']+WORKSPACE
        <= verifier.BUDGET, 'Profile workspace is not admitted')
    return dict(exact_logits_state_routes=True, exact_initial_cache=True,
        **{k: actual[k] for k in ('clean_memory', 'clean_host', 'peak_physical_bytes', 'peak_compressed_bytes')})


def analyze_block(block, profile, build, revision, mode):
    offset = block['offset']; begin, end = ns(block['forward_begin_ns']), ns(block['forward_end_ns'])
    require(end > begin and end-begin == block['forward_ns'], 'Invalid forward timestamps')
    require(profile.get('truncated') is False and profile.get('coverage') == 'all-dispatches' and
        profile.get('entry_limit') == 100000 and profile.get('dependency_capture_limits') ==
        dict(passes_per_phase=48, read_records_per_phase=8192), 'Incomplete or changed profile bounds')
    expected_kind = ('instrumented per-dispatch compute passes; submission boundaries preserved' if mode == 'dispatch'
        else 'existing command groups; mixed stages are not isolated kernel costs')
    require(profile.get('timing_kind') == expected_kind, 'Changed counter instrumentation')
    groups = profile['command_groups']; deps = profile['expert_dependencies']
    by_submit = {}; intervals = []; kernels = Counter(); signatures = Counter(); routers = Counter()
    operations = defaultdict(lambda: [0, 0]); expert_rows = Counter()
    for g in groups:
        submitted, completed = ns(g['submitted_ns']), ns(g['completed_ns'])
        start, stop = ns(g['gpu_start_seconds']*1e9), ns(g['gpu_end_seconds']*1e9)
        require(begin <= submitted <= start <= stop <= completed <= end and submitted not in by_submit,
            'Missing, reversed, duplicate or out-of-block command timestamps')
        by_submit[submitted] = g
        intervals += [('gpu_active', start, stop), ('gpu_idle_submitted', submitted, start),
            ('gpu_idle_callback', stop, completed)]
        ops = g['operations']; require(bool(ops), 'Empty captured command')
        if mode == 'dispatch':
            require([o.get('counter_index') for o in ops] == list(range(0, 2*len(ops), 2)),
                'Missing or duplicate GPU counters')
        for op in ops:
            require(op['request_phase'] == 'decode' and op['offset'] == offset and
                type(op['layer']) is int and -1 <= op['layer'] < 48 and
                type(op['tokens']) is int and 1 <= op['tokens'] <= 4 and
                isinstance(op['stage'], str) and bool(op['stage']) and isinstance(op['kernel'], str),
                'Changed block operation scope')
            matrix = op.get('matrix', {})
            shape = (matrix.get('K'), matrix.get('N'), matrix.get('rows'))
            kernels[op['kernel']] += 1
            signatures[(offset, op['layer'], op['stage'], op['kernel'], op['tokens'], *shape)] += 1
            if op['kernel'] == 'route_simd': routers[op['layer']] += 1
            if op['stage'] == 'routed_expert' and op['kernel'] == 'scatter_experts': expert_rows[op['tokens']] += 1
            if mode == 'dispatch':
                duration = op.get('gpu_pass_ns')
                require(type(duration) is int and duration > 0 and
                    type(op.get('gpu_begin_ticks')) is int and type(op.get('gpu_end_ticks')) is int and
                    op['gpu_end_ticks'] > op['gpu_begin_ticks'], 'Invalid GPU pass duration')
                key = (op['stage'], op['kernel'], *shape)
                operations[key][0] += duration; operations[key][1] += 1
            else: require('gpu_pass_ns' not in op, 'Unexpected counter duration')
    require(0 < sum(kernels.values()) <= 20000 and routers == Counter({i: 1 for i in range(48)}),
        'Missing block dispatch/layer coverage')
    require(len(deps) == 48 and sorted(d['layer'] for d in deps) == list(range(48)),
        'Missing or duplicate dependency layer')
    reads = Counter(); queues = []; services = []; ready_delays = []; per_layer = []; routes = []
    for d in deps:
        require(d['offset'] == offset and d['tokens'] == 4 and d['request_phase'] == 'decode' and
            d['build'] == build and d['artifact_revision'] == revision, 'Dependency identity differs')
        route = d['routes']; require(len(route) == 40 and all(type(e) is int and 0 <= e < 512 for e in route) and
            all(len(set(route[t:t+10])) == 10 for t in range(0, 40, 10)), 'Invalid top-ten routes')
        records = d['records']; selected = set(route)
        require(len(records) == len(selected) and {r['expert'] for r in records} == selected,
            'Incomplete selected-expert coverage')
        routes.append((offset, d['layer'], route)); layer_intervals = []; last_gpu = 0; layer_reads = Counter()
        for r in records:
            admitted, encoded, submitted, released = (ns(r[k]) for k in
                ('admitted_ns', 'encoded_ns', 'submitted_ns', 'released_ns'))
            require(begin <= admitted <= encoded <= submitted <= released <= end and submitted in by_submit,
                'Invalid expert lifetime or missing GPU user')
            ready = max(admitted, ns(r['read_completed_ns']))
            require(ready <= encoded, 'Expert encoded before data ready')
            g = by_submit[submitted]
            require(abs(r['gpu_start_ns']-int(g['gpu_start_seconds']*1e9)) <= 2 and
                abs(r['gpu_end_ns']-int(g['gpu_end_seconds']*1e9)) <= 2 and
                r['gpu_end_ns'] <= released, 'Expert GPU clocks differ or lease released early')
            kind = r['acquisition']; require(kind in ('ready_hit', 'new_miss', 'loading_join'), 'Unknown read acquisition')
            reads[kind] += 1; layer_reads[kind] += 1
            if kind != 'ready_hit':
                q, s, c = (ns(r[k]) for k in ('read_queued_ns', 'read_started_ns', 'read_completed_ns'))
                require(q <= s <= c, 'Reversed read times')
                layer_intervals.append(('gpu_idle_pending_read', admitted, ready))
                queues.append(s-q); services.append(c-s)
            layer_intervals.append(('gpu_idle_ready_expert', ready, submitted))
            ready_delays.append(submitted-ready); last_gpu = max(last_gpu, r['gpu_end_ns'])
        intervals += layer_intervals
        require(d['ready_hits'] == layer_reads['ready_hit'] and d['new_misses'] == layer_reads['new_miss'] and
            d['loading_joins'] == layer_reads['loading_join'], 'Dependency counters differ')
        per_layer.append(dict(layer=d['layer'], selected=len(records), **dict(layer_reads),
            last_expert_gpu_ns=last_gpu))
    require(sum(expert_rows.values()) == sum(reads.values()) and
        sum(rows*count for rows, count in expert_rows.items()) == 48*40,
        'Routed row geometry does not reconcile')
    buckets = partition(begin, end, intervals)
    operation_rows = [dict(stage=k[0], kernel=k[1], K=k[2], N=k[3], rows=k[4],
        gpu_pass_ms_per_token=v[0]/4e6, dispatches=v[1]) for k, v in
        sorted(operations.items(), key=lambda kv: -kv[1][0])]
    return dict(offset=offset, tokens=4, forward_ms_per_token=(end-begin)/4e6,
        buckets_ms_per_token={k: v/4e6 for k, v in buckets.items()},
        dispatches=sum(kernels.values()), kernels=dict(kernels), command_groups=len(groups),
        classes=command_classes(groups, 4), layers=per_layer, routes=sorted(routes),
        read_counts=dict(reads), expert_read_bytes=reads['new_miss']*2764800,
        expert_rows={str(k): v for k, v in sorted(expert_rows.items())},
        dispatch_signature=digest(sorted((*k, v) for k, v in signatures.items())),
        counter_operations=operation_rows,
        mean_read_queue_ms=sum(queues)/max(1, len(queues))/1e6,
        mean_read_service_ms=sum(services)/max(1, len(services))/1e6,
        mean_ready_to_submit_ms=sum(ready_delays)/len(ready_delays)/1e6)


def analyze(raw, profiles, build, revision, mode):
    require(len(profiles) == len(raw['blocks']) == 4, 'Missing complete four-block capture')
    blocks = [analyze_block(b, p, build, revision, mode) for b, p in zip(raw['blocks'], profiles)]
    require([b['offset'] for b in blocks] == [72, 76, 80, 84], 'Changed block positions')
    a, b = raw['before']['metal'], raw['after']['metal']; counts = Counter()
    for block in blocks: counts.update(block['kernels'])
    expected = Counter({k: b['kernel_dispatches'].get(k, 0)-a['kernel_dispatches'].get(k, 0)
        for k in set(a['kernel_dispatches']) | set(b['kernel_dispatches'])})
    require(counts == expected and sum(counts.values()) == b['dispatches']-a['dispatches'] and
        sum(block['command_groups'] for block in blocks) == b['submissions']-a['submissions'],
        'Missing native command or dispatch population')
    mean_buckets = {k: sum(block['buckets_ms_per_token'][k] for block in blocks)/4
        for k in blocks[0]['buckets_ms_per_token']}
    mean_forward = sum(block['forward_ms_per_token'] for block in blocks)/4
    require(math.isclose(sum(mean_buckets.values()), mean_forward, abs_tol=1e-6), 'Buckets do not partition forward time')
    operations = defaultdict(lambda: [0, 0]); rows = Counter()
    for block in blocks:
        rows.update(block['expert_rows'])
        for op in block['counter_operations']:
            key = (op['stage'], op['kernel'], op['K'], op['N'], op['rows'])
            operations[key][0] += op['gpu_pass_ms_per_token']/4
            operations[key][1] += op['dispatches']
    return dict(mode=mode, captured_blocks=4, captured_input_tokens=16, dependency_passes=192,
        dispatches=sum(counts.values()), mean_forward_ms_per_token=mean_forward,
        mean_buckets_ms_per_token=mean_buckets, expert_rows=dict(sorted(rows.items())), blocks=blocks,
        counter_operations=[dict(stage=k[0], kernel=k[1], K=k[2], N=k[3], rows=k[4],
            gpu_pass_ms_per_token=v[0], dispatches=v[1]) for k, v in sorted(operations.items(), key=lambda kv: -kv[1][0])],
        performance_measurement=False, production_promoted=False, limitations=LIMITATIONS)


def compare(commands, dispatch):
    require(commands['mode'] == 'commands' and dispatch['mode'] == 'dispatch', 'Reversed profile modes')
    require(len(commands['blocks']) == len(dispatch['blocks']) == 4 and
        [b['offset'] for b in commands['blocks']] == [72, 76, 80, 84], 'Incomplete profile comparison')
    for a, b in zip(commands['blocks'], dispatch['blocks']):
        require(a['offset'] == b['offset'] and a['dispatch_signature'] == b['dispatch_signature'] and
            a['routes'] == b['routes'] and a['expert_rows'] == b['expert_rows'], 'Counter instrumentation changed operation/route population')
    return dict(same_dispatches_routes_shapes=True,
        instrumented_forward_ratio=dispatch['mean_forward_ms_per_token']/commands['mean_forward_ms_per_token'],
        normal_speedup=None, selected_kernel=None, production_promoted=False)
