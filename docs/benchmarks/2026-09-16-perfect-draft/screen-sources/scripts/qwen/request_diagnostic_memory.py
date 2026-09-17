"""Memory boundary evidence for one complete, reference-only request diagnostic."""
from diagnose_decode_startup import analyze_steps


GIB = 1024**3
MIB = 1024**2
UINT64_MAX = 2**64-1
GAUGES = ('physical_footprint_bytes', 'physical_footprint_peak_bytes',
          'compressed_bytes', 'compressed_peak_bytes', 'decompressions',
          'system_swap_used_bytes')
CUMULATIVE = ('physical_footprint_peak_bytes', 'compressed_peak_bytes', 'decompressions')


def uint(value):
    return type(value) is int and 0 <= value <= UINT64_MAX


def require(condition, message):
    if not condition:
        raise ValueError(message)


def difference(a, b):
    return {key: b[key]-a[key] for key in GAUGES}


def analyze_memory(raw, *, budget_bytes=12*GIB, compression_limit_bytes=512*MIB):
    """Reject missing/reset observations; report limits without claiming qualification.

    The compression allowance applies to the process lifetime peak, including
    initialization. There is no cumulative compressed-traffic byte counter.
    """
    require(uint(budget_bytes) and budget_bytes > 0 and uint(compression_limit_bytes) and
            compression_limit_bytes <= budget_bytes, 'Invalid diagnostic memory limits')
    require(isinstance(raw, dict) and raw.get('complete') is True and
            isinstance(raw.get('runs'), list) and len(raw['runs']) == 2,
            'Memory diagnostic requires two complete request phases')
    observations = []
    phases = []
    reasons = []
    previous_end = 0

    def observe(state, name, label, **context):
        require(isinstance(state, dict) and isinstance(state.get('process'), dict),
                f'Missing process observation at {name}:{label}')
        process = {key: state['process'].get(key) for key in GAUGES}
        require(all(uint(value) for value in process.values()),
                f'Missing or invalid unsigned memory gauge at {name}:{label}')
        require(0 < process['physical_footprint_bytes'] <= process['physical_footprint_peak_bytes'] and
                process['compressed_bytes'] <= process['compressed_peak_bytes'],
                f'Current memory exceeds its lifetime peak at {name}:{label}')
        delta = None
        if observations:
            prior = observations[-1]['process']
            require(all(process[key] >= prior[key] for key in CUMULATIVE),
                    f'Cumulative memory counter reset at {name}:{label}')
            delta = difference(prior, process)
            if delta['system_swap_used_bytes'] > 0:
                reasons.append(f'Observed system swap growth at {name}:{label}')
        if process['physical_footprint_peak_bytes'] > budget_bytes:
            reasons.append(f'Physical footprint peak exceeds budget at {name}:{label}')
        if process['compressed_peak_bytes'] > compression_limit_bytes:
            reasons.append(f'Compression lifetime peak exceeds diagnostic allowance at {name}:{label}')
        index = len(observations)
        observations.append(dict(index=index, phase=name, boundary=label, process=process,
                                 change_since_previous=delta, **context))
        return process, index

    def phase_boundary(row, phase, boundary, name):
        require(isinstance(row.get('phases'), dict), 'Missing phase memory boundaries')
        data = row['phases'].get(phase, {})
        require(isinstance(data, dict), f'Missing {phase} memory boundaries')
        return observe(data.get(boundary), name, f'{phase}.{boundary}')

    names = [row.get('name') for row in raw['runs'] if isinstance(row, dict)]
    require(len(names) == 2 and all(isinstance(name, str) and name for name in names) and
            len(set(names)) == 2, 'Missing or duplicate request phase names')
    for row in raw['runs']:
        name = row['name']
        require(type(row.get('output_tokens')) is int and row['output_tokens'] == 17 and
                isinstance(row.get('output_token_ids'), list) and len(row['output_token_ids']) == 17 and
                isinstance(row.get('token_latency_ms'), list) and len(row['token_latency_ms']) == 16,
                'Memory diagnostic requires 17 outputs and 16 forwards per phase')
        diagnostic = row.get('decode_diagnostics', {})
        require(isinstance(diagnostic, dict) and all(uint(diagnostic.get(key)) for key in
                ('max_steps', 'total_decode_steps', 'captured_steps', 'omitted_steps')),
                'Missing or invalid decode coverage counters')
        # Keep the existing independent token/position/timestamp coverage contract.
        try:
            steps = analyze_steps(row)
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError('Incomplete decode observation') from error
        require(steps['captured_steps'] == 16 and steps['omitted_steps'] == 0,
                'Incomplete per-token memory coverage')
        request_before, first = observe(row.get('before'), name, 'request.before')
        ingest_before, _ = phase_boundary(row, 'ingest', 'before', name)
        ingest_after, _ = phase_boundary(row, 'ingest', 'after', name)
        decode_before, _ = phase_boundary(row, 'decode', 'before', name)
        tokens = []
        gaps = []
        prior = decode_before
        for sample in diagnostic['samples']:
            step = sample['step']
            require(uint(step) and uint(sample.get('offset')) and uint(sample.get('begin_ns')) and
                    uint(sample.get('end_ns')) and sample['begin_ns'] >= previous_end,
                    'Nonchronological decode observation across request phases')
            previous_end = sample['end_ns']
            before, start = observe(sample['before'], name, 'token.before', step=step,
                offset=sample['offset'], forward_begin_ns=sample['begin_ns'])
            gaps.append(dict(boundary='decode_to_first_token' if step == 0 else 'between_tokens',
                preceding_step=None if step == 0 else step-1, following_step=step,
                process_change=difference(prior, before)))
            after, end = observe(sample['after'], name, 'token.after', step=step,
                offset=sample['offset'], forward_end_ns=sample['end_ns'])
            tokens.append(dict(step=step, offset=sample['offset'], input_token_id=sample['input_token_id'],
                forward_ms=sample['forward_ms'], before_observation=start, after_observation=end,
                memory_before=before, memory_after=after, process_change=difference(before, after)))
            prior = after
        decode_after, _ = phase_boundary(row, 'decode', 'after', name)
        gaps.append(dict(boundary='last_token_to_decode', preceding_step=15, following_step=None,
                         process_change=difference(prior, decode_after)))
        request_after, last = observe(row.get('after'), name, 'request.after')
        inside = sum(token['process_change']['decompressions'] for token in tokens)
        outside = sum(gap['process_change']['decompressions'] for gap in gaps)
        decode_total = decode_after['decompressions']-decode_before['decompressions']
        require(inside+outside == decode_total, 'Decode decompression accounting does not reconcile')
        outside_phases = (ingest_before['decompressions']-request_before['decompressions'] +
                          decode_before['decompressions']-ingest_after['decompressions'] +
                          request_after['decompressions']-decode_after['decompressions'])
        phases.append(dict(name=name, first_observation=first, last_observation=last,
            tokens=tokens, decode_gaps=gaps,
            ingest_process_change=difference(ingest_before, ingest_after),
            decode_process_change=difference(decode_before, decode_after),
            request_process_change=difference(request_before, request_after),
            decompressions=dict(ingest=ingest_after['decompressions']-ingest_before['decompressions'],
                decode=decode_total, inside_token_forwards=inside, outside_token_forwards=outside,
                outside_ingest_and_decode=outside_phases,
                request=request_after['decompressions']-request_before['decompressions'])))

    processes = [observation['process'] for observation in observations]
    between = difference(processes[phases[0]['last_observation']], processes[phases[1]['first_observation']])
    total_decompressions = processes[-1]['decompressions']-processes[0]['decompressions']
    require(sum(phase['decompressions']['request'] for phase in phases)+between['decompressions'] ==
            total_decompressions, 'Request decompression accounting does not reconcile')
    stable_swap = len({process['system_swap_used_bytes'] for process in processes}) == 1
    strict_clean = (not reasons and all(process['compressed_bytes'] == process['compressed_peak_bytes'] == 0
                        for process in processes) and total_decompressions == 0 and stable_swap)
    return dict(kind='request_diagnostic_memory_v1', complete=True, hard_limits_passed=not reasons,
        reasons=reasons, diagnostic_usable=not reasons, strict_memory_clean=strict_clean,
        budget_bytes=budget_bytes, compression_peak_limit_bytes=compression_limit_bytes,
        chronological_observations=observations, phases=phases,
        between_requests_process_change=between, observed_decompressions=total_decompressions,
        stable_observed_swap=stable_swap,
        maxima={key: max(process[key] for process in processes) for key in GAUGES},
        coverage=dict(request_phases=2, decode_forwards=32, omitted_decode_forwards=0,
                      observations=len(observations), includes_ingestion_and_between_token_gaps=True),
        normal_request_latency_qualified=False, production_promoted=False,
        limitations=[
            'These are lifecycle and token boundary observations plus lifetime peaks, not a continuous physical-memory guard.',
            'The compression peak allowance is an operational diagnostic bound, not an empirical safety or performance threshold.',
            'Compressed-byte changes are signed net changes between samples. Gross compression traffic and unsampled churn are unknown.',
            'Token before/after process gauges bracket the forward; their exact sampling timestamps are not recorded.',
            'Between-token and phase-boundary decompressions are reported separately and are not attributed to a kernel or SSD read.',
            'System swap observations include other processes. An observed increase fails the diagnostic bound; decreases are permitted but do not satisfy strict stable-swap evidence.',
            'Decompressions before the first process observation are outside the measured interval. Lifetime compression and physical peaks include initialization.',
            'Every token is retained. Compression-disturbed diagnostic timing cannot qualify normal-request latency.'])
