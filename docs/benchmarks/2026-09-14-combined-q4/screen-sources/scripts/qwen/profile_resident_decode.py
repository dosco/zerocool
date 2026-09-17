#!/usr/bin/env python3
"""Bounded current-residency dependency capture; no performance promotion."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from cache_residency import configs, memory_observation, PRIOR, require
from capture_routes import load
from confirm_expert_residency import validate_requests
from measure_buffer_costs import delta, FIELDS, CLASSES
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
from screen_cache import validate_request
from stage200 import Experiment, profile_row
from submission_timeline import analyze as driver_analysis
from verify_stage200 import verify_sources


def dependency_details(row, deps, groups):
    low = row['prompt_tokens']
    steps = len(row['decode_diagnostics']['samples'])
    selected_passes = [d for d in deps if d['tokens'] == 1 and low <= d['offset'] < low+steps]
    passes = {(d['offset'], d['layer']): d for d in selected_passes}
    require(steps > 0 and len(passes) == len(selected_passes) == steps*48,
            'Missing or duplicate per-layer dependency coverage')
    misses = [r for d in passes.values() for r in d['records'] if r['acquisition'] == 'new_miss']
    def timing(values):
        if not values: return None
        values = sorted(values)
        require(all(v >= 0 for v in values), 'Reversed read timing')
        return dict(mean=statistics.mean(values), median=statistics.median(values),
                    p95=values[int(.95*(len(values)-1))], p99=values[int(.99*(len(values)-1))])
    proposed = selected = missed = 0
    for (offset, layer), d in passes.items():
        if layer == 0 or (offset-1, layer) not in passes: continue
        guesses = passes[offset-1, layer]['routes'][:2]
        proposed += len(guesses)
        selected += sum(e in d['routes'] for e in guesses)
        cold = {r['expert'] for r in d['records'] if r['acquisition'] == 'new_miss'}
        missed += sum(e in cold for e in guesses)
    gpu = defaultdict(float)
    for g in groups:
        if any(low <= o['offset'] < low+steps and o['tokens'] == 1 for o in g['operations']):
            stages = tuple(sorted({o['stage'] for o in g['operations']}))
            gpu[stages] += (g['gpu_end_seconds']-g['gpu_start_seconds'])*1000/steps
    return dict(name=row['name'],
        queue_ms=timing([(r['read_started_ns']-r['read_queued_ns'])/1e6 for r in misses]),
        service_ms=timing([(r['read_completed_ns']-r['read_started_ns'])/1e6 for r in misses]),
        gpu_command_classes=[dict(stages=list(k), gpu_ms_per_token=v)
                             for k, v in sorted(gpu.items(), key=lambda kv: -kv[1])],
        previous_top_two=dict(proposed=proposed, selected_again=selected, observed_new_misses=missed,
            live_prefetch_run=False, predicted_latency_saving_ms=None,
            scope='Retrospective overlap with actual misses, excluding first captured token and layer zero. '
                  'No simulated admission, evictions, bandwidth contention or proof of readiness at prefetch time.'))


def buffer_intervals(row):
    samples = row['decode_diagnostics']['samples']
    require(len(samples) == 16, 'Requires sixteen decode samples')
    a, b = (row['phases']['decode'][k]['metal'] for k in ('before', 'after'))
    total = delta(a, b)
    steps = [delta(s['before']['metal'], s['after']['metal']) for s in samples]
    for name in CLASSES:
        for key in FIELDS:
            require(sum(s['classes'][name][key] for s in steps) == total['classes'][name][key],
                    'Uncovered buffer work')
    for key in ('retired_groups', 'group_retirement_ns'):
        require(sum(s[key] for s in steps) == total[key], 'Uncovered group retirement')
    memory = [s[k]['process'] for s in samples for k in ('before', 'after')]
    require(all(s.get('compressed_bytes') == 0 for s in memory) and
            all(type(s.get('decompressions')) is int for s in memory) and
            len({s['decompressions'] for s in memory}) == 1, 'Decode memory disturbance')
    return dict(allocation_ms_per_token=sum(c['allocation_ns'] for c in total['classes'].values())/16e6,
        group_retirement_ms_per_token=total['group_retirement_ns']/16e6,
        outside_owner_release_ms_per_token=sum(c['outside_release_ns'] for c in total['classes'].values())/16e6,
        allocation_count_per_token=sum(c['allocations'] for c in total['classes'].values())/16,
        costs=total, causal_latency_saving_ms=None)


def reconstruct(out):
    out = Path(out)
    verify_seal(out, sha(out/'evidence-files.json'))
    summary, frozen = load(out/'summary.json'), load(out/'identity.json')
    require(summary.get('complete') is True and summary['configurations'] == [configs()[0]],
            'Incomplete or changed resident profile')
    require(all(w['max_tokens'] == 17 for w in summary['workload']), 'Changed trace window')
    expected = {}
    for name in ('normal', 'traced'):
        record = summary[name]
        require(record['source'] == name+'.json' and sha(out/record['source']) == record['sha256'],
                'Changed source report')
        raw = load(out/record['source'])
        validate_request(raw, frozen, configs()[0], summary['workload'], expected,
                         instrumented=name == 'traced', output_tokens=17)
        require(memory_observation(raw)['memory_screen_passed'], 'Memory-disturbed profile')
        for row in raw['runs']:
            state = row['after']
            require(state['metal']['residency']['bytes_by_class']['expert'] == state['memory_plan']['expert_bytes'],
                    'Cache residency missing')
    normal, raw = load(out/'normal.json'), load(out/'traced.json')
    commands = load(out/'commands.json')
    deps = [json.loads(line) for line in (out/'dependencies.jsonl').read_text().splitlines()]
    timelines = [profile_row(raw, i, commands, deps) for i in range(2)]
    return dict(kind='resident_decode_analysis_v1', complete=True, status='captured',
        source_seal_sha256=sha(out/'evidence-files.json'),
        analysis_tools={name:sha(Path(__file__).with_name(name)) for name in
                        ('profile_resident_decode.py', 'cache_residency.py', 'measure_buffer_costs.py',
                         'stage200.py', 'submission_timeline.py', 'decode_timeline.py')},
        source_provenance=verify_sources(out.parent, frozen),
        timelines=timelines, driver=driver_analysis(out),
        buffers=[dict(name=r['name'], **buffer_intervals(r)) for r in raw['runs']],
        dependencies=[dependency_details(r, deps, commands['command_groups']) for r in raw['runs']],
        trace_to_normal_decode_ratios=[b['decode_wall_ms']/a['decode_wall_ms']
                                      for a, b in zip(normal['runs'], raw['runs'])],
        normal_request_latency_qualified=False, production_promoted=False,
        limitations=['One short normal/traced conversation; sixteen decode steps per phase.',
            'GPU idle categories are exclusive observations, not causes or guaranteed recoverable latency.',
            'Buffer CPU intervals overlap the timeline and must not be added to its rows.',
            'A faster traced process reflects run variation; no instrumentation adjustment is applied.',
            '1072 slots with core-cache residency. No 2K/4K, sustained-use or larger-cache qualification.'])


def run(out):
    prior = validate_requests(PRIOR/'residency-confirm-01', 5)
    work = [dict(w, max_tokens=17) for w in prior['workload']]
    c = configs()[0]
    exp = Experiment(out, 'resident_decode_capture_v1', [c], work, 360)
    with exp:
        require(exp.report['identity'] == prior['identity'], 'Changed native build or artifact')
        exp.guard.check_resources(initial=True)
        expected = {}
        for name in ('normal', 'traced'):
            extra = ['--decode-diagnostics', '--phase-profile', exp.out/'commands.json',
                     '--profile-decode-only', '1', '--dependency-trace', exp.out/'dependencies.jsonl'] if name == 'traced' else []
            raw = exp.bench(c, name, extra)
            validate_request(raw, exp.frozen, c, work, expected,
                             instrumented=name == 'traced', output_tokens=17)
            memory = memory_observation(raw)
            exp.report[name] = dict(source=name+'.json', sha256=sha(exp.out/(name+'.json')), **memory)
            exp.persist()
            if not memory['memory_screen_passed']:
                raise ResourceBlocked('Memory disturbance during resident decode capture')
            if name == 'normal':
                needed = sum(r['phases']['decode']['after']['metal']['dispatches']-
                             r['phases']['decode']['before']['metal']['dispatches'] for r in raw['runs'])
                require(needed <= 120000, 'Trace would exceed dispatch capacity')
        exp.report['status'] = 'captured'
    return exp.report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('capture', 'analyze'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    if args.mode == 'analyze':
        if args.source is None: parser.error('--source is required for analysis')
        save(args.output, reconstruct(args.source))
    else:
        raise SystemExit(0 if run(args.output)['complete'] else 2)
