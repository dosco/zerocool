"""Summarize complete short request traces without attributing mixed GPU costs.

File seals, artifact admission, host resources and full-model state proofs remain
the runner's responsibility. This module joins the bounded native evidence.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math

from cache_residency import require
from profile_resident_decode import dependency_details
from stage200 import profile_row

DISPATCHES = 101600
STEPS = 16
LAYERS = 48
TARGET_MS = 200
PACKED = {'q4_gate_up_packed_r2': 'q4_gate_up', 'q4_down_packed_r2': 'q4_mm'}
FLAGS = dict(normal_request_latency_qualified=False, production_promoted=False)
LIMITATIONS = [
    'These are traced normal requests; instrumentation and dependency JSON writing add overhead. They are not normal-speed measurements or a promotion decision.',
    'GPU-active time is an interval union. Command-duration sums may overlap and must not be added to exclusive wait buckets or treated as removable latency.',
    'Mixed-stage and mixed-layer commands retain their full class; no duration is attributed to an isolated kernel or duplicated into each constituent layer.',
    'The explicit dependency JSONL path covers every selected pass. The separate embedded summary remains capped at 48 passes per phase and is not used here.',
    'Two short phases contain sixteen decode forwards each. Initial prompt and append processing, longer contexts, and sustained coding are outside trace coverage.',
    'Waiting buckets describe observed overlap, not the cause of a stall or an attainable speedup. The gap to 200ms is a target distance, not a savings estimate.']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def identity(state):
    metal = state['metal']; kernels = metal['kernels']
    require(kernels.get('profile') is True and kernels.get('profile_decode_only') is True and
            kernels.get('counter_profile') is False and
            kernels.get('q4_decode') in ('reference', 'packed-r2') and
            state.get('diagnostic_stream_trunk') is False,
            'Changed trace instrumentation or diagnostic execution')
    return dict(build=metal['build_fingerprint'], device=metal['device'],
        artifact_revision=state['artifact_revision'], model_id=state['model_id'],
        memory_plan=state['memory_plan'], execution=state['execution'],
        runtime={key: state[key] for key in
            ('completion_pipeline', 'ready_group', 'chunk_tokens', 'io_workers', 'short_append_tokens')},
        kernels={k: v for k, v in kernels.items() if k != 'q4_decode'})


def command_classes(groups, steps, per_layer=False):
    """Assign each command once, retaining all stages/layers on that command."""
    values = defaultdict(lambda: [0.0, 0, 0])
    for group in groups:
        operations = group['operations']
        stages = tuple(sorted({op['stage'] for op in operations}))
        layers = tuple(sorted({op['layer'] for op in operations})) if per_layer else ()
        key = (layers, stages)
        values[key][0] += (group['gpu_end_seconds']-group['gpu_start_seconds'])*1000
        values[key][1] += 1
        values[key][2] += len(operations)
    return [dict(stages=list(stages), **(dict(layers=list(layers)) if per_layer else {}),
                 gpu_command_ms_per_token=duration/steps, command_groups=count,
                 dispatches=dispatches)
            for (layers, stages), (duration, count, dispatches) in sorted(values.items())]


def analyze_trace(raw, profile, deps):
    require(raw.get('complete') is True and len(raw.get('runs', [])) == 2 and
            len(raw.get('workloads', [])) == 2 and
            profile.get('truncated') is False and profile.get('coverage') == 'decode-only' and
            profile.get('entry_limit') == 120000, 'Incomplete or changed decode trace scope')
    rows = raw['runs']; common = identity(rows[0]['after'])
    variant = rows[0]['after']['metal']['kernels']['q4_decode']
    require(raw.get('model_revision') == common['artifact_revision'], 'Changed model identity')
    samples = {}; counts = []; signatures = []; phases = []
    for row, work in zip(rows, raw['workloads']):
        require(row['name'] == work['name'] and work.get('max_tokens') == 17 and
                row.get('output_tokens') == len(row.get('output_token_ids', [])) == 17 and
                len(row.get('token_latency_ms', [])) == STEPS,
                'Changed request names or output window')
        diagnostic = row['decode_diagnostics']
        require(diagnostic.get('captured_steps') == diagnostic.get('total_decode_steps') == STEPS and
                diagnostic.get('omitted_steps') == 0 and len(diagnostic.get('samples', [])) == STEPS,
                'Incomplete token coverage')
        for boundary in (row['before'], row['after']):
            require(identity(boundary) == common and
                    boundary['metal']['kernels']['q4_decode'] == variant,
                    'Changed build, artifact, configuration or memory plan within trace')
        phase = row['phases']['decode']
        count = phase['after']['metal']['dispatches']-phase['before']['metal']['dispatches']
        require(type(count) is int and count == DISPATCHES//2,
                'Changed native decode dispatch population')
        counts.append(count)
        for sample in diagnostic['samples']:
            offset = sample['offset']
            require(offset not in samples, 'Ambiguous token offsets across requests')
            samples[offset] = sample
    require(len(samples) == 2*STEPS and len({r['name'] for r in rows}) == 2,
            'Duplicate request or token coverage')
    groups = profile.get('command_groups', [])
    require(sum(len(g['operations']) for g in groups) == sum(counts) == DISPATCHES,
            'Missing or extra command dispatch coverage')
    for group in groups:
        ops = group['operations']; offsets = {op.get('offset') for op in ops}
        require(bool(ops) and len(offsets) == 1 and next(iter(offsets)) in samples and
                all(op.get('request_phase') == 'decode' and op.get('tokens') == 1 and
                    type(op.get('layer')) is int and -1 <= op['layer'] < LAYERS and
                    isinstance(op.get('stage'), str) and bool(op['stage']) and
                    isinstance(op.get('kernel'), str) and bool(op['kernel']) for op in ops),
                'Command crosses tokens or contains unknown phase/layer/stage/kernel')
        sample = samples[next(iter(offsets))]
        require(sample['begin_ns'] <= group['submitted_ns'] <= group['completed_ns'] <= sample['end_ns'],
                'Command lifetime lies outside its captured token')
    require(len(deps) == STEPS*2*LAYERS and
            {(d['offset'], d['layer']) for d in deps} ==
                {(offset, layer) for offset in samples for layer in range(LAYERS)} and
            all(d['tokens'] == 1 for d in deps), 'Incomplete or extra explicit dependency coverage')
    for index, row in enumerate(rows):
        timeline = profile_row(raw, index, profile, deps)
        require(timeline['captured_tokens'] == STEPS, 'Incomplete joined token coverage')
        low = row['prompt_tokens']; high = low+STEPS
        selected = [g for g in groups if low <= g['operations'][0]['offset'] < high]
        details = dependency_details(row, deps, selected)
        layer_classes = command_classes(selected, STEPS, True)
        classes = command_classes(selected, STEPS)
        # A count signature detects changes hidden by the same total dispatch
        # count, while allowing the intended two Q4 kernel replacements only.
        signature = Counter((op['offset'], op['layer'], op['stage'],
                             PACKED.get(op['kernel'], op['kernel']))
                            for g in selected for op in g['operations'])
        signatures.append(digest(sorted((*key, count) for key, count in signature.items())))
        total = sum(c['gpu_command_ms_per_token'] for c in classes)
        require(math.isclose(total, sum(c['gpu_command_ms_per_token'] for c in layer_classes), abs_tol=1e-7),
                'Command class attribution does not reconcile')
        phases.append(dict(name=row['name'], captured_tokens=STEPS,
            offsets=list(range(low, high)), mean_forward_ms=timeline['mean_forward_ms'],
            gap_to_target_ms=max(0, timeline['mean_forward_ms']-TARGET_MS),
            mean_buckets_ms=timeline['mean_buckets_ms'],
            mean_gpu_command_ms=total, gpu_command_classes=classes,
            layer_command_classes=layer_classes, dependency_details=details,
            mean_layer_boundary_submission_gap_ms=timeline['mean_layer_boundary_submission_gap_ms'],
            tokens=timeline['tokens']))
    routes = sorted((d['offset'], d['layer'], d['routes']) for d in deps)
    return dict(kind='q4_request_profile_v1', complete=True, variant=variant, identity=common,
        workloads=raw['workloads'], sampling=raw['sampling'],
        output_tokens=[r['output_token_ids'] for r in rows],
        routes_sha256=digest(routes), normalized_dispatch_sha256=signatures,
        coverage=dict(phases=2, forwards=32, dispatches=DISPATCHES,
                      dependency_passes=len(deps), selected_experts=len(deps)*10,
                      explicit_dependency_file=True, omitted_tokens=0),
        target_ms_per_token=TARGET_MS, phases=phases, limitations=LIMITATIONS, **FLAGS)


def class_deltas(reference, packed, per_layer=False):
    def key(row):
        return (tuple(row['layers']) if per_layer else (), tuple(row['stages']))
    left = {key(r): r for r in reference}; right = {key(r): r for r in packed}
    results = []
    for layers, stages in sorted(left.keys() | right.keys()):
        a = left.get((layers, stages), {}); b = right.get((layers, stages), {})
        results.append(dict(stages=list(stages), **(dict(layers=list(layers)) if per_layer else {}),
            reference_gpu_command_ms_per_token=a.get('gpu_command_ms_per_token', 0),
            packed_gpu_command_ms_per_token=b.get('gpu_command_ms_per_token', 0),
            delta_gpu_command_ms_per_token=b.get('gpu_command_ms_per_token', 0)-a.get('gpu_command_ms_per_token', 0),
            reference_command_groups=a.get('command_groups', 0), packed_command_groups=b.get('command_groups', 0),
            reference_dispatches=a.get('dispatches', 0), packed_dispatches=b.get('dispatches', 0)))
    return sorted(results, key=lambda r: -r['delta_gpu_command_ms_per_token'])


def compare_traces(reference, packed):
    for record, variant in ((reference, 'reference'), (packed, 'packed-r2')):
        require(record.get('kind') == 'q4_request_profile_v1' and record.get('complete') is True and
                record.get('variant') == variant and all(record.get(k) is False for k in FLAGS),
                'Incomplete or reversed trace comparison')
    for key in ('identity', 'workloads', 'sampling', 'output_tokens', 'routes_sha256',
                'normalized_dispatch_sha256', 'coverage', 'target_ms_per_token'):
        require(reference[key] == packed[key], 'Incompatible trace comparison: '+key)
    require(len(reference['phases']) == len(packed['phases']) == 2, 'Missing comparison phase')
    phases = []
    for a, b in zip(reference['phases'], packed['phases']):
        require(a['name'] == b['name'] and a['offsets'] == b['offsets'] and
                a['captured_tokens'] == b['captured_tokens'] == STEPS,
                'Changed phase history or capture window')
        delta = {k: b['mean_buckets_ms'][k]-v for k, v in a['mean_buckets_ms'].items()}
        forward_delta = b['mean_forward_ms']-a['mean_forward_ms']
        require(math.isclose(sum(delta.values()), forward_delta, abs_tol=1e-6),
                'Exclusive bucket differences do not reconcile')
        phases.append(dict(name=a['name'], reference_mean_forward_ms=a['mean_forward_ms'],
            packed_mean_forward_ms=b['mean_forward_ms'], delta_mean_forward_ms=forward_delta,
            delta_mean_buckets_ms=delta,
            reference_gap_to_target_ms=a['gap_to_target_ms'], packed_gap_to_target_ms=b['gap_to_target_ms'],
            gpu_command_classes=class_deltas(a['gpu_command_classes'], b['gpu_command_classes']),
            layer_command_classes=class_deltas(a['layer_command_classes'], b['layer_command_classes'], True)))
    return dict(kind='q4_request_trace_comparison_v1', complete=True,
        source_analysis_sha256=dict(reference=digest(reference), packed=digest(packed)),
        identity=reference['identity'], phases=phases, confidence_95=None,
        causal_savings_estimate_ms=None, limitations=LIMITATIONS, **FLAGS)
