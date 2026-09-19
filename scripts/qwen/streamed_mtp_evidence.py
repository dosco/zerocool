"""Recompute storage comparisons from sealed normal-request evidence."""
import math
import re
from pathlib import Path

from mtp_evidence import account, child, need, original, screening_context
from qualification_evidence import sha, verify_seal
from screen_streamed_mtp import ARMS, compare as compare_pair, observe, prerequisite


def labels(sample):
    match = re.fullmatch(r'width-([14])-pair-([01])-(resident|rows)\.json', sample['source'])
    need(match is not None, 'Unexpected storage sample name')
    width, pair, arm = int(match[1]), int(match[2]), match[3]
    need(sample['width'] == width and sample['embedding_storage'] == arm, 'Storage sample labels changed')
    return dict(case=f'width-{width}', width=width, pair=pair, arm=arm)


def compare(runs, source, data, control, candidate, changes, case=None):
    need((control, candidate) == ARMS and changes == ['embedding_storage'],
         'Declare only resident/rows embedding storage')
    context = screening_context(data)
    root = Path(source['path']).parent
    verify_seal(root, sha(root/'evidence-files.json'))
    producer, pref = original(root/'producer.json')
    frozen, fref = original(root/'identity.json')
    need(producer['complete'] is True and producer['kind'] in
         ('streamed_mtp_producer_v1', 'streamed_mtp_fixed_priming_producer_v1') and
         frozen['files'].get(producer['binary']) == producer['binary_sha256'], 'Missing frozen storage producer')
    validation = data['numerical_validation']
    parent = Path(validation['path'])
    verify_seal(parent, validation['seal_sha256'])
    _, clean = prerequisite(parent, producer, preliminary=True)
    need(data['full_clean_correctness'] is clean, 'Changed numerical prerequisite qualification')
    sources = [source, pref, fref, dict(path=str(parent/'evidence-files.json'), sha256=validation['seal_sha256'])]
    keyed = {}; workload = None
    for raw, ref, label in runs:
        width, pair, arm = (label[k] for k in ('width', 'pair', 'arm'))
        key = (width, pair, arm)
        need(key not in keyed, 'Duplicate storage sample')
        sample = next(s for s in data['samples'] if s['sha256'] == ref['sha256'])
        inp = child(root, sample['input'])
        work, wref = original(inp)
        if workload is None: workload = work
        need(work == workload, 'Storage workloads differ')
        result = observe(raw, work, wref['sha256'], producer['binary_sha256'], width, arm, False,
                         producer.get('draft_priming_policy', 'completion'))
        need(result['clean_memory'] and result['clean_host'], 'Resource-disturbed storage timing')
        need(all(sample.get(k) == v for k, v in result.items()), 'Storage sample summary differs from raw observation')
        need(producer['base_native_fingerprint'] == raw['before']['metal']['build_fingerprint'], 'Native base changed')
        account(raw)
        keyed[key] = raw
        sources.extend((ref, wref))
    need(set(k[0] for k in keyed) == {1, 4}, 'Missing storage width coverage')
    pairs = []; grouped = []; all_pairs = []
    for width in (4, 1):
        numbers = sorted({p for w, p, _ in keyed if w == width})
        need(numbers in ([0], [0, 1]), 'Noncontiguous storage pairs')
        values = []
        for pair in numbers:
            need(all((width, pair, arm) in keyed for arm in ARMS), 'Incomplete storage pair')
            order = [label['arm'] for _, _, label in runs if label['width'] == width and label['pair'] == pair]
            need(order == list(ARMS if pair == 0 else ARMS[::-1]), 'Nonalternating storage pair')
            result = compare_pair(*(keyed[width, pair, arm] for arm in ARMS))
            claim = dict(width=width, pair=pair, **result)
            all_pairs.append(claim); values.append(result)
            if case is None or case == f'width-{width}': pairs.append(claim)
        ratios = [v['latency_ratio'] for v in values]
        summary = dict(pairs=len(values), geometric_latency_ratio=math.prod(ratios)**(1/len(ratios)),
            measured_peak_savings_bytes=[v['physical_peak_saved_bytes'] for v in values])
        need(data['by_width'][str(width)] == summary, 'Storage width summary differs from raw evidence')
        if case is None or case == f'width-{width}':
            grouped.append(dict(case=f'width-{width}', **summary, confidence_95=None))
    need(data['pairs'] == all_pairs and bool(pairs), 'Missing, changed or unselected storage comparisons')
    return dict(comparable=True, status='measured', **context, controlled_change={'embedding_storage': list(ARMS)},
        pairs=pairs, cases=grouped, sources=sources, normal_request_latency_qualified=False, production_promoted=False,
        limitations=['One coding workload and at most two alternating pairs per width; no confidence bounds.',
            'Each width is a separate storage comparison; workloads, widths and historical timings are not pooled.',
            'Memory savings do not establish a speedup from a larger expert cache.',
            'A diagnostic numerical prerequisite does not qualify clean correctness or production adoption.',
            'Long context, retained conversations and sustained coding remain unqualified.'])
