"""Strict reconstruction of the bounded memory experiment from original reports."""
import math
import statistics

from benchmark_exact import configurations
from screen_cache import validate_request
from screen_residency import decide as residency_decision, pair_order

PROBE = 'resident-q8-v1'
CAPS = {'control': 1848, 'candidate': 1460}
KINDS = {'screen': 'cache_capacity_screen_v1', 'confirm': 'cache_capacity_confirmation_v1'}


def configs():
    return [dict(configurations()[0], name=name, expert_slots=slots, cache_policy='clock',
                 residency='off', decode_path='reference', prefill_pipeline='serial', phase_memory='fixed')
            for name, slots in CAPS.items()]


def order(pairs):
    return [(i, 'control' if arm == 'off' else 'candidate') for i, arm in pair_order(pairs)]


def finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def validate_probe(probe):
    if (probe.get('mode') != PROBE or probe.get('kernel') != 'q8_mm' or
        (probe.get('input'), probe.get('output'), probe.get('group')) != (2560, 6144, 64) or
        probe.get('temporary_bytes') != 49152 or probe.get('live_bytes_before') != probe.get('live_bytes_after') or
        not finite(probe.get('live_bytes_before')) or probe.get('live_command_groups') != 0 or
        probe.get('cache_unchanged') is not True):
        raise ValueError('GPU reference identity, allocation or cleanup changed')
    samples = probe.get('samples', [])
    if len(samples) != 4: raise ValueError('Missing GPU reference samples')
    hashes = set(); warm = []
    for i, sample in enumerate(samples):
        if sample.get('sample') != i or sample.get('dispatches') != (8 if i else 1):
            raise ValueError('GPU reference sample order changed')
        if (not finite(sample.get('wall_ns'), True) or not finite(sample.get('submitted_ns'), True) or
            not finite(sample.get('completed_ns'), True) or sample['completed_ns'] < sample['submitted_ns']):
            raise ValueError('Invalid CPU reference timestamps')
        checksum = sample.get('output_sha256')
        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
            raise ValueError('Missing GPU reference checksum')
        hashes.add(checksum)
        start, end = sample.get('gpu_start_seconds'), sample.get('gpu_end_seconds')
        valid = finite(start, True) and finite(end, True) and end > start
        duration, delay = sample.get('gpu_ns'), sample.get('submission_delay_ns')
        if valid:
            if not finite(duration, True) or not math.isclose(duration, (end-start)*1e9, rel_tol=1e-10, abs_tol=1):
                raise ValueError('GPU duration differs from timestamps')
            if i: warm.append(duration/8)
        elif duration is not None:
            raise ValueError('Unavailable GPU timing represented as a duration')
        ordered = valid and start >= sample['submitted_ns']/1e9 and end <= sample['completed_ns']/1e9
        if ordered:
            if not finite(delay) or not math.isclose(delay, (start-sample['submitted_ns']/1e9)*1e9, abs_tol=1):
                raise ValueError('Submission delay differs from timestamps')
        elif delay is not None:
            raise ValueError('Invalid GPU timestamps have a derived submission delay')
        for key in ('memory_before', 'memory_after'):
            if not isinstance(sample.get(key), dict): raise ValueError('Missing per-probe memory observation')
    if len(hashes) != 1: raise ValueError('GPU reference changed output')
    median = statistics.median(warm) if len(warm) == 3 else None
    if probe.get('warm_gpu_ns_per_dispatch') != median: raise ValueError('Changed warm reference median')
    for name in ('host_before', 'host_after'):
        host = probe.get(name)
        if not isinstance(host, dict) or not host.get('source') or not host.get('limitation'):
            raise ValueError('Missing host-condition availability information')
    return median


def observations(raw, evidence, config, workload, expected, mode):
    refs = raw.get('gpu_references')
    if raw.get('gpu_reference_mode') != mode: raise ValueError('Missing explicit reference protocol')
    if mode == PROBE:
        if not isinstance(refs, list) or [(r.get('repetition'),r.get('boundary')) for r in refs] != [(0,'before'),(0,'after')]:
            raise ValueError('Missing or misplaced conversation boundary probes')
        for ref in refs:
            validate_probe(ref)
            if ref['memory_plan'] != raw['runs'][0]['before']['memory_plan']:
                raise ValueError('Probe changed admitted memory plan')
        if refs[0]['samples'][0]['output_sha256'] != refs[1]['samples'][0]['output_sha256']:
            raise ValueError('Conversation changed reference output')
    elif mode != 'off' or refs != []:
        raise ValueError('Unexpected GPU reference protocol')
    for row in raw.get('runs', []):
        if 'decode_diagnostics' in row: raise ValueError('Additional decode instrumentation')
        for state in (row['before'], row['after'], *[v[k] for v in row['phases'].values() for k in ('before','after')]):
            if state['metal']['kernels'].get('profile') or state['metal']['kernels'].get('counter_profile'):
                raise ValueError('Additional GPU profiling')
            if state.get('short_append_tokens') != 32: raise ValueError('Changed short-append threshold')
    return validate_request(raw, evidence, config, workload, expected, gpu_reference=mode, capacity_axis=True)


def decide(measurements, pairs):
    # Keep precisely the already-declared residency timing gates and intervals.
    translated = [dict(pair=r['pair'], residency='off' if r['configuration']=='control' else 'core',
                       requests=r['requests']) for r in measurements]
    result = residency_decision(translated, pairs)
    result.pop('advance_to_long_validation', None)
    result['advance_to_confirmation'] = bool(result.pop('advance_to_five_pairs', False))
    result['candidate_for_later_qualification'] = pairs == 5 and result['status'] == 'promising'
    return result


def revalidate(summary, resolve):
    """resolve(sha256) returns the original JSON; never trust summary timings."""
    phase = summary.get('mode'); pairs = 2 if phase == 'screen' else 5
    mode = PROBE if phase == 'screen' else 'off'
    if (phase not in KINDS or summary.get('kind') != KINDS[phase] or summary.get('complete') is not True or
        summary.get('configurations') != configs() or summary.get('gpu_reference_mode') != mode or
        summary.get('metal_validation') is not False):
        raise ValueError('Incomplete or changed capacity experiment protocol')
    rows = summary.get('measurements', [])
    if [(r['pair'],r['configuration']) for r in rows] != order(pairs):
        raise ValueError('Requires the original complete alternating capacity pairs')
    expected = {}; used = set()
    for row in rows:
        digest = row['sha256']
        if digest in used: raise ValueError('Repeated raw report cannot create another pair')
        used.add(digest)
        raw = resolve(digest)
        config = next(c for c in configs() if c['name'] == row['configuration'])
        result = observations(raw, summary['identity'], config, summary['workload'], expected, mode)
        if result != row['requests']: raise ValueError('Summary differs from original requests')
    decision = decide(rows, pairs)
    if any(summary.get(k) != v for k,v in decision.items()): raise ValueError('Changed predeclared decision')
    return decision
