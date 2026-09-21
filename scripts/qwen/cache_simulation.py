"""Bounded, offline expert-cache what-ifs. No runtime or latency prediction."""
from collections import OrderedDict
import heapq
import re

from evidence_index import MAX_LINES, metadata, parse

# Current Q4 and mixed-4/8 checkpoints have identical fixed-size Q4 experts.
# Byte accounting matches include/engine/storage.hpp; this is NOT a Q3 model.
LAYERS, EXPERTS, TOP_K = 48, 512, 10
PAYLOAD_BYTES, SLOT_BYTES = 2_764_800, 2_768_896
REVISIONS = {'aa7c790e804bbf9d491ddb109c3d61bc4a555f7c',
             'b2c422f3c643e36f04227a64d61796b44a4b1029'}
MAX_ACCESSES = 250_000
MAX_POLICY_STEPS = 9_000_000
DEFAULT_BUDGETS = [0, 1024, 2048, 4096, 6144, 8192]


class Clock:
    """Native second chance when all reads are ready and no leases are pinned."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.slots = [None] * capacity
        self.referenced = [False] * capacity
        self.lookup = {}
        self.hand = 0

    def access(self, key, position):
        if key in self.lookup:
            self.referenced[self.lookup[key]] = True
            return True
        if not self.capacity:
            return False
        while self.referenced[self.hand]:
            self.referenced[self.hand] = False
            self.hand = (self.hand + 1) % self.capacity
        slot = self.hand
        if self.slots[slot] is not None:
            del self.lookup[self.slots[slot]]
        self.slots[slot] = key
        self.lookup[key] = slot
        self.referenced[slot] = True
        self.hand = (slot + 1) % self.capacity
        return False


class SegmentedLRU:
    """One probation/protected candidate: protected target floor(3*slots/4)."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.protected_limit = 3 * capacity // 4
        self.probation = OrderedDict()
        self.protected = OrderedDict()

    def access(self, key, position):
        if key in self.protected:
            self.protected.move_to_end(key)
            return True
        if key in self.probation:
            del self.probation[key]
            self.protected[key] = None
            if len(self.protected) > self.protected_limit:
                demoted, _ = self.protected.popitem(last=False)
                self.probation[demoted] = None
            return True
        if not self.capacity:
            return False
        if len(self.protected) + len(self.probation) == self.capacity:
            self.probation.popitem(last=False)
        self.probation[key] = None
        return False


class Belady:
    """MIN for this fixed access order, equal-size records, mandatory admission."""
    def __init__(self, capacity, sequence):
        self.capacity = capacity
        self.next = [len(sequence)] * len(sequence)
        future = {}
        for i in range(len(sequence) - 1, -1, -1):
            key = sequence[i]
            self.next[i] = future.get(key, len(sequence))
            future[key] = i
        self.resident = {}
        self.heap = []

    def access(self, key, position):
        hit = key in self.resident
        if not self.capacity:
            return False
        if not hit and len(self.resident) == self.capacity:
            while True:
                negative_next, victim = heapq.heappop(self.heap)
                if self.resident.get(victim) == -negative_next:
                    del self.resident[victim]
                    break
        self.resident[key] = self.next[position]
        heapq.heappush(self.heap, (-self.next[position], key))
        # Bound stale heap entries even for long all-hit traces.
        if len(self.heap) > 2 * max(1, self.capacity):
            self.heap = [(-n, k) for k, n in self.resident.items()]
            heapq.heapify(self.heap)
        return hit


def integer(value, minimum, maximum, label):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{label} must be an integer in {minimum}..{maximum}')
    return value


def check_format(row):
    for key, expected in (('expert_payload_bytes', PAYLOAD_BYTES), ('expert_stride_bytes', SLOT_BYTES),
                          ('expert_bits', 4), ('expert_group_size', 64)):
        if key in row and (type(row[key]) is not int or row[key] != expected):
            raise ValueError('Variable-size or alternate expert formats are not supported: ' + key)


def route_segments(rows, phase=None, explicit_boundaries=False):
    """Keep complete 48-layer windows in file order; never transpose a panel."""
    segments, pending = [], []
    identity = None
    excluded = missing_routes = filtered = incomplete = 0
    explicit_events = 0
    logical_routes = passes = 0
    barriers = []
    can_continue = False

    def discard(reason, pointer):
        nonlocal pending, excluded, can_continue, incomplete
        if pending:
            excluded += len(pending)
            incomplete += 1
        pending = []
        can_continue = False
        barriers.append(dict(reason=reason, source_pointer=pointer))

    for pointer, row in rows:
        if not isinstance(row, dict):
            raise ValueError('Route trace entries must be objects')
        if explicit_boundaries and 'cache_boundary' in row:
            explicit_events += 1
            discard(row['cache_boundary'], pointer)
            continue
        if phase is not None and row.get('request_phase') != phase:
            discard('phase_filter', pointer)
            filtered += 1
            continue
        if 'routes' not in row:
            discard('missing_routes', pointer)
            missing_routes += 1
            continue
        revision, build = row.get('artifact_revision'), row.get('build')
        if revision not in REVISIONS:
            raise ValueError('Cache simulation requires a pinned Q4 or mixed-4/8 artifact with Q4 experts')
        if not isinstance(build, str) or not re.fullmatch('[0-9a-f]{64}', build):
            raise ValueError('Route entry requires a full native build fingerprint')
        current = (revision, build)
        if identity is not None and identity != current:
            raise ValueError('Do not combine artifact revisions or native builds in a cache curve')
        identity = current
        check_format(row)
        layer = integer(row.get('layer'), 0, LAYERS - 1, 'layer')
        offset = integer(row.get('offset'), 0, 8191, 'offset')
        tokens = integer(row.get('tokens'), 1, 8192 - offset, 'tokens')
        routes = row['routes']
        if not isinstance(routes, list) or len(routes) != tokens * TOP_K:
            raise ValueError('Routes must contain exactly tokens * top-10 expert IDs')
        for route in routes:
            integer(route, 0, EXPERTS - 1, 'expert')
        for start in range(0, len(routes), TOP_K):
            if len(set(routes[start:start + TOP_K])) != TOP_K:
                raise ValueError('A token cannot select the same expert twice')
        # Pass dedup models the existing grouping of token rows by expert.
        # The fixed ascending order is shared by ALL policies/capacities. Native
        # hit-first admission and asynchronous leases are deliberately not replayed.
        event = dict(pointer=pointer, layer=layer, offset=offset, tokens=tokens,
                     phase=row.get('request_phase'), session=row.get('session_id'), request_id=row.get('request_id'),
                     keys=[layer * EXPERTS + e for e in sorted(set(routes))])
        if explicit_boundaries:
            integer(event['request_id'], 1, MAX_ACCESSES, 'request_id')
        if any(event[k] is not None and not isinstance(event[k], str) for k in ('phase', 'session')):
            raise ValueError('Phase/session identity must be a string when recorded')
        if pending and (layer != len(pending) or any(event[k] != pending[0][k]
                           for k in ('offset', 'tokens', 'phase', 'session', 'request_id'))):
            discard('incomplete_or_discontinuous_window', pointer)
        if not pending and layer != 0:
            excluded += 1
            can_continue = False
            continue
        pending.append(event)
        if len(pending) != LAYERS:
            continue
        first = pending[0]
        if (can_continue and segments[-1]['end_offset'] == offset and
                (explicit_boundaries or segments[-1]['phase'] == first['phase']) and segments[-1]['session_id'] == first['session']):
            segment = segments[-1]
        else:
            if segments and can_continue:
                barriers.append(dict(reason='position_phase_or_session_boundary', source_pointer=first['pointer']))
            segment = dict(start_offset=offset, end_offset=offset, phase=first['phase'],
                           session_id=first['session'], first_source_pointer=first['pointer'],
                           last_source_pointer=pointer, windows=0, single_token_windows=0, accesses=[])
            segments.append(segment)
            if explicit_boundaries: segment.update(phases=[], access_spans=[])
        if explicit_boundaries and first['phase'] not in segment['phases']:
            segment['phases'].append(first['phase'])
        start = len(segment['accesses'])
        for entry in pending:
            segment['accesses'].extend(entry['keys'])
        if explicit_boundaries:
            spans = segment['access_spans']
            if spans and (spans[-1]['request_id'], spans[-1]['phase']) == (first['request_id'], first['phase']):
                spans[-1]['end'] = len(segment['accesses'])
            else:
                spans.append(dict(start=start, end=len(segment['accesses']),
                                  request_id=first['request_id'], phase=first['phase']))
        segment['end_offset'] = offset + tokens
        segment['last_source_pointer'] = pointer
        segment['windows'] += 1
        segment['single_token_windows'] += int(tokens == 1)
        logical_routes += tokens * TOP_K * LAYERS
        passes += LAYERS
        pending = []
        can_continue = True
        if sum(len(s['accesses']) for s in segments) > MAX_ACCESSES:
            raise ValueError('Cache trace exceeds bounded demand count; select a smaller source')
    if pending:
        discard('incomplete_final_window', pending[-1]['pointer'])
    demands = sum(len(s['accesses']) for s in segments)
    return segments, dict(
        identity=dict(artifact_revision=identity[0], build=identity[1]) if identity else None,
        input_passes=len(rows)-explicit_events, explicit_boundary_events=explicit_events,
        included_passes=passes, excluded_incomplete_passes=excluded,
        missing_route_passes=missing_routes, filtered_passes=filtered,
        incomplete_windows=incomplete, logical_router_selections=logical_routes,
        demand_record_accesses=demands, within_pass_grouped_selections=logical_routes - demands,
        compulsory_misses=sum(len(set(s['accesses'])) for s in segments),
        segment_count=len(segments), cold_starts=len(segments),
        boundary_count=len(barriers), boundaries=barriers[:20],
        boundary_details_truncated=len(barriers) > 20,
        segments=[{k: v for k, v in s.items() if k not in ('accesses', 'access_spans')} for s in segments[:20]],
        segment_details_truncated=len(segments) > 20,
        whole_request_covered=None, session_boundaries_fully_known=False,
        phase_filter=phase)


def counts(segments, capacity, policy, per_layer=False):
    hits = [0] * LAYERS
    misses = [0] * LAYERS
    request_phases = {}
    for segment in segments:
        sequence = segment['accesses']
        cache = Belady(capacity, sequence) if policy == 'belady' else (
            Clock(capacity) if policy == 'clock' else SegmentedLRU(capacity))
        spans, span_index = segment.get('access_spans', []), 0
        for position, key in enumerate(sequence):
            hit = cache.access(key, position)
            bucket = hits if hit else misses
            bucket[key // EXPERTS] += 1
            if spans:
                while position >= spans[span_index]['end']: span_index += 1
                span = spans[span_index]
                phase = request_phases.setdefault((span['request_id'], span['phase']),
                    dict(request_id=span['request_id'], phase=span['phase'], hits=0, misses=0))
                phase['hits' if hit else 'misses'] += 1
    hit_count, miss_count = sum(hits), sum(misses)
    answer = dict(policy=policy, demands=hit_count + miss_count, hits=hit_count, misses=miss_count,
                  application_miss_bytes=miss_count * PAYLOAD_BYTES,
                  byte_hit_fraction=hit_count / (hit_count + miss_count) if hit_count + miss_count else None)
    if request_phases:
        answer['request_phases'] = [dict(r, demands=r['hits']+r['misses'],
            application_miss_bytes=r['misses']*PAYLOAD_BYTES,
            byte_hit_fraction=r['hits']/(r['hits']+r['misses'])) for r in request_phases.values()]
    if per_layer:
        answer['layers'] = [dict(layer=i, hits=hits[i], misses=misses[i],
                                application_miss_bytes=misses[i] * PAYLOAD_BYTES) for i in range(LAYERS)]
    return answer


def cache_curve(index, selector, budgets=None, phase=None, per_layer=False):
    budgets = DEFAULT_BUDGETS if budgets is None else budgets
    if not isinstance(budgets, list) or not 1 <= len(budgets) <= 12:
        raise ValueError('Choose 1..12 cache budgets')
    for budget in budgets:
        integer(budget, 0, 22 * 1024, 'expert cache MiB')
    if len(set(budgets)) != len(budgets):
        raise ValueError('Cache budgets must be distinct')
    raw, source = index.source(selector)
    data = {}
    committed_trace = None
    if source['path'].endswith('.jsonl'):
        lines = raw.splitlines()
        if len(lines) > MAX_LINES:
            raise ValueError('Cache simulation refuses a line-truncated trace')
        rows = [(f'line:{i}', parse(line)) for i, line in enumerate(lines, 1) if line.strip()]
        recorded_status = dict(status='unknown', complete=None)
        capture = None
        if rows and isinstance(rows[0][1], dict) and rows[0][1].get('kind') == 'qwen_route_trace_v1':
            from route_trace import decode
            committed_trace = decode(raw)
            rows = committed_trace['rows']
            data = committed_trace['identity']
            check_format(data)
            recorded_status = {k:committed_trace[k] for k in ('status', 'complete')}
    else:
        data = parse(raw)
        if not isinstance(data, dict) or not isinstance(data.get('expert_dependencies'), list):
            raise ValueError('Select native route JSONL or a profile with expert_dependencies')
        if len(data['expert_dependencies']) > MAX_LINES:
            raise ValueError('Cache simulation refuses an oversized pass list')
        check_format(data)
        rows = [(f'/expert_dependencies/{i}', r) for i, r in enumerate(data['expert_dependencies'])]
        recorded_status = {k: metadata(data)[k] for k in ('status', 'complete')}
        capture = data.get('dependency_capture_limits')
    segments, coverage = route_segments(rows, phase, explicit_boundaries=committed_trace is not None)
    identity = coverage['identity'] or {}
    for key in ('build', 'artifact_revision'):
        if identity and key in data and data[key] != identity[key]:
            raise ValueError('Profile identity differs from recorded routes: ' + key)
    coverage.update(capture_limits=capture, query_truncated=False)
    if committed_trace is not None:
        coverage.update(route_trace_protocol='qwen_route_trace_v1',
            whole_request_covered=committed_trace['complete'] if phase is None else None,
            session_boundaries_fully_known=True, committed_forwards=len(committed_trace['committed']),
            aborted_forwards=committed_trace['aborted_forwards'],
            uncommitted_layer_passes=committed_trace['uncommitted_layer_passes'],
            completed_requests=len(committed_trace['requests']), phase_filter_starts_cold=phase is not None)
    limits = [
        'Simulation of expert reads only; no measured latency, tokens/s, device traffic or quality gain.',
        'Fixed ascending expert-ID demand order within each recorded layer pass, with within-pass deduplication. '
        'No token-major reconstruction of prefill panels. Native hit-first admission can change the order.',
        'Cold start at each position/phase/session discontinuity or omitted pass. Contiguous positions imply '
        'continuation only as a simulation assumption; unrecorded resets cannot be detected.',
        'Immediate read completion and release; no pinned GPU users, loading joins, prefetch, OS cache or read overlap. '
        'CLOCK here matches the native rule only under those assumptions.',
        'Belady is a future-aware minimum for the SAME fixed order, equal-size records and mandatory admission. '
        'It is not a bound for arbitrary native schedules, variable-size quantization or end-to-end latency.',
        'Budgets cover expert slots only, including alignment. Resident weights, metadata, ngrams, state, scratch and '
        'driver reserve need separate admission; this curve cannot approve an engine memory budget.',
        'Artifact/build IDs are recorded claims, not newly verified model weights. No prompt identity, terminal request '
        'marker or representative steady-generation coverage is inferred from these route records.',
    ]
    if committed_trace is not None:
        limits[2] = ('Only committed forwards are included. Explicit session/cache resets and omitted work start '
                     'cold segments. Unfiltered captures retain simulated prefill warmth into decode; phase filters do not.')
        limits[-1] = ('Request token inputs and terminal markers are checked against committed forwards. '
                      'Artifact identities are recorded claims, and a complete short capture does not establish representative locality.')
        limits.append('Request/phase counters partition the unfiltered simulation without resetting its cache. '
                      'Belady minimizes total misses for each segment, not misses in every individual phase.')
    total_demands = sum(len(s['accesses']) for s in segments)
    if total_demands * len(budgets) * 3 > MAX_POLICY_STEPS:
        raise ValueError('Cache sweep exceeds bounded policy steps; select fewer budgets or a smaller source')
    curves = []
    for mib in sorted(budgets) if segments else []:
        budget_bytes = mib * 1024**2
        slots = budget_bytes // SLOT_BYTES
        results = [counts(segments, slots, policy, per_layer)
                   for policy in ('clock', 'segmented_lru', 'belady')]
        clock_bytes, _, oracle_bytes = [r['application_miss_bytes'] for r in results]
        curves.append(dict(budget_mib=mib, budget_bytes=budget_bytes, slots=slots,
                           slot_allocation_bytes=slots * SLOT_BYTES,
                           unused_budget_bytes=budget_bytes % SLOT_BYTES, policies=results,
                           clock_minus_oracle_read_bytes=clock_bytes - oracle_bytes))
    return dict(kind='expert_cache_simulation_v1', status='simulated' if segments else 'insufficient_evidence',
                scope='fixed_order_serial_demand_simulation', sources=[source], limitations=limits,
                build=identity.get('build'), artifact_revision=identity.get('artifact_revision'),
                source_result=recorded_status, coverage=coverage,
                layout=dict(format='affine_q4_group64', expert_payload_bytes=PAYLOAD_BYTES,
                            aligned_slot_bytes=SLOT_BYTES, layers=LAYERS, experts_per_layer=EXPERTS, top_k=TOP_K),
                policy_parameters=dict(segmented_lru_protected_fraction=.75, mandatory_admission=True),
                curves=curves, normal_request_latency_qualified=False, production_promoted=False,
                predicted_tokens_per_second=None, predicted_request_speedup=None,
                next_experiment='First capture 32 successive tokens of normal generation with explicit '
                    'session/reset/phase markers and complete routes after a prompt, under a fixed deadline. '
                    'If promising, extend to 256 tokens and a retained-history append. Screen one cache policy '
                    'at the same admitted byte budget before full qualification.')
