"""Audit full verifier cache traces before making fixed-order read simulations."""
from collections import Counter

import block_cache_replay as replay
from cache_residency import require
from screen_verifier_horizon import compare, validate

WORKSPACE = 32*1024**2
CAPACITIES = (1460, 1909, 2048)


def patterns(work, width):
    require(type(width) is int and width in (4, 8), 'Unsupported trace width')
    prompt, count = len(work['prompt_ids']), len(work['continuation_ids'])
    require(1 <= prompt <= 128 and 1 <= count <= 64, 'Unbounded trace workload')
    result = [(0, prompt, 'prefill')]
    consumed = 0
    while consumed < count:
        rows = width if count-consumed >= width else 1
        result.append((prompt+consumed, rows, 'decode'))
        consumed += rows
    return result


def observe(raw, work, input_sha, producer_sha, width, capture):
    require(type(capture) is bool and raw.get('cache_trace_enabled') is capture and
        raw.get('performance_measurement') is False and raw.get('compute_tile_cap') == 4,
        'Undeclared trace mode or compute policy')
    result = validate(raw, work, input_sha, producer_sha, width, False, True,
                      trace_workspace_bytes=WORKSPACE)
    result.pop('verified_tps')
    require(raw['draft_priming_before'] == raw['draft_priming_after'], 'Draft ran during trace continuation')
    priming = raw['draft_priming_before']
    require(priming['policy'] == 'fixed-lease-batches-32' and priming['active'] is False and
        priming['calls'] > 0 and 0 < priming['peak_leases'] <= 32, 'Missing bounded draft priming')
    coverage = patterns(work, width)
    forwards = raw['trace_forwards']
    require(len(forwards) == len(coverage) and all(set(f) == {'position', 'routes'} and
        f['position'] == at+rows and len(f['routes']) == 48
        for f, (at, rows, _) in zip(forwards, coverage)), 'Incomplete trace route coverage')
    for f in forwards:
        require([r['layer'] for r in f['routes']] == list(range(48)) and
            all(r['tokens'] == f['position'] and len(r['sha256']) == 64 and
                all(c in '0123456789abcdef' for c in r['sha256']) for r in f['routes']),
            'Invalid route identity')
    require(isinstance(raw['target_cache_trace'], dict) if capture else raw['target_cache_trace'] is None,
        'Missing trace or unexpected trace in control')
    return dict(result, performance_measurement=False, timing_used=False,
                captured_forwards=len(forwards), captured_tokens=sum(p[1] for p in coverage))


def compare_modes(control, captured):
    require(control['cache_trace_enabled'] is False and captured['cache_trace_enabled'] is True and
        control['performance_measurement'] is captured['performance_measurement'] is False and
        control['requested_width'] == captured['requested_width'], 'Trace comparison changed width or modes')
    # Reuse the established full-logit, persistent-state and admission checks;
    # explicitly discard instrumented timing rather than comparing its ratio.
    compare(control, captured)
    for key in ('trace_forwards', 'draft_priming_before', 'draft_priming_after', 'draft_before',
                'embedding_rows_before', 'embedding_rows_after', 'compute_tile_cap'):
        require(control[key] == captured[key], 'Trace changed '+key)
    return dict(exact_logits_and_state=True, exact_routes=True, exact_initial_caches=True,
                width=control['requested_width'], timing_used=False, performance_measurement=False)


def decode(data, raw, work, fingerprint):
    require(raw['cache_trace_enabled'] is True and raw['performance_measurement'] is False and
        raw['compute_tile_cap'] == 4, 'Not a target trace capture')
    coverage = patterns(work, raw['requested_width'])
    forwards = raw['trace_forwards']
    require(len(forwards) == len(coverage) and all(f['position'] == at+rows
        for f, (at, rows, _) in zip(forwards, coverage)), 'Changed forward positions')
    adapter = dict(configuration=dict(expert_slots=1460), expert_cache_trace=raw['target_cache_trace'],
        prime=dict(routes=forwards[0]['routes']), blocks=[dict(routes=f['routes']) for f in forwards[1:]],
        before=raw['before'], after=raw['after'])
    result = replay.decode(data, adapter, work, fingerprint, expected_patterns=coverage)
    require(result['snapshots_verified'] == len(coverage), 'Missing or extra target snapshots')
    return result


def summarize(trace):
    curves = [replay.simulate(trace, capacity, 'clock') for capacity in CAPACITIES]
    actual_hits = sum(f['hits'] for f in trace['forwards'][1:])
    actual_misses = sum(f['misses'] for f in trace['forwards'][1:])
    require((curves[0]['decode_hits'], curves[0]['decode_misses']) == (actual_hits, actual_misses),
            'Control simulation differs from native continuation')
    for curve in curves:
        curve['miss_reduction_fraction'] = 1-curve['decode_misses']/actual_misses if actual_misses else None
        curve['capacity_admitted'] = curve['capacity'] == 1460
    row_use = []
    for event in trace['events']:
        if event['event'] == 'forward_begin':
            d = event['detail']
            row_use.append(dict(offset=d['offset'], tokens=d['tokens'], phase=d['phase'],
                                distinct_row_count_histogram=Counter()))
        elif event['event'] == 'layer':
            # Top-k is unique per token, so occurrences equal distinct token rows.
            row_use[-1]['distinct_row_count_histogram'].update(Counter(event['detail']['routes']).values())
    for window in row_use:
        histogram = dict(sorted(window['distinct_row_count_histogram'].items()))
        window.update(distinct_row_count_histogram={str(k): v for k, v in histogram.items()},
                      selected_records=sum(histogram.values()), single_row_records=histogram.get(1, 0))
    return dict(native_replay={k: trace[k] for k in ('capacity', 'snapshots_verified', 'max_pins',
        'native_counts_exact', 'native_slot_state_exact')},
        forwards=[{k: v for k, v in f.items() if k != 'demands'} for f in trace['forwards']],
        row_use=row_use, curves=curves, latency_prediction=None, performance_measurement=False,
        production_promoted=False, normal_request_latency_qualified=False)
