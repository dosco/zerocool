#!/usr/bin/env python3
"""Short per-pass diagnostic of resident decode work. Never qualifies latency."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from cache_residency import configs, memory_observation, require
from capture_routes import load
from qualification_evidence import ResourceBlocked, sha, verify_seal
from screen_cache import validate_request
from stage200 import Experiment

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT/'docs/benchmarks/2026-09-14-cache-residency/profile-01'


def split_operations(raw, profile):
    require(profile.get('truncated') is False and profile.get('coverage') == 'all-dispatches'
            and profile.get('timing_kind') ==
            'instrumented per-dispatch compute passes; submission boundaries preserved',
            'Requires complete per-dispatch counter capture')
    groups = profile['command_groups']
    for group in groups:
        ops = group['operations']
        require([o.get('counter_index') for o in ops] == list(range(0, 2*len(ops), 2)),
                'Missing or duplicate dispatch counters')
        require(all(type(o.get('gpu_pass_ns')) is int and o['gpu_pass_ns'] > 0 and
                    o.get('gpu_end_ticks', 0) > o.get('gpu_begin_ticks', 0) for o in ops),
                'Missing or invalid GPU counter timing')
    result = []
    for row in raw['runs']:
        low = row['prompt_tokens']
        steps = len(row['token_latency_ms'])
        require(steps > 0, 'No decode coverage')
        totals = defaultdict(lambda: [0, 0])
        counts = Counter()
        layer_steps = Counter()
        for group in groups:
            selected = [o for o in group['operations'] if o['request_phase'] == 'decode'
                        and low <= o['offset'] < low+steps]
            if not selected:
                continue
            require(len(selected) == len(group['operations']), 'Group crosses decode window')
            before_reads = any(o['stage'] == 'router' for o in selected)
            for op in selected:
                require(op['tokens'] == 1, 'Unexpected batched decode')
                counts[op['kernel']] += 1
                if op['kernel'] == 'route_simd':
                    layer_steps[op['offset'], op['layer']] += 1
                matrix = op.get('matrix', {})
                key = (before_reads, op['stage'], op['kernel'], matrix.get('K'), matrix.get('N'))
                totals[key][0] += op['gpu_pass_ns']
                totals[key][1] += 1
        a, b = (row['phases']['decode'][k]['metal']['kernel_dispatches'] for k in ('before', 'after'))
        expected = Counter({k: b.get(k, 0)-a.get(k, 0) for k in set(a)|set(b)})
        require(counts == expected, 'Counter capture does not cover every decode dispatch')
        require(layer_steps == Counter({(t, l): 1 for t in range(low, low+steps) for l in range(48)}),
                'Missing or duplicate layer/token coverage')
        result.append(dict(name=row['name'], offsets=list(range(low, low+steps)),
            dispatches=sum(counts.values()), operations=[dict(before_expert_reads=k[0],
                stage=k[1], kernel=k[2], K=k[3], N=k[4], gpu_pass_ms_per_token=v[0]/steps/1e6,
                dispatches_per_token=v[1]/steps) for k, v in
                sorted(totals.items(), key=lambda kv: -kv[1][0])]))
    return result


def run(out):
    verify_seal(PRIOR, sha(PRIOR/'evidence-files.json'))
    prior = load(PRIOR/'summary.json')
    require(prior.get('complete') is True, 'Prior profile is incomplete')
    work = [dict(w, max_tokens=5) for w in prior['workload']]
    c = configs()[0]
    exp = Experiment(out, 'resident_operator_capture_v1', [c], work, 360)
    with exp:
        require(exp.report['identity'] == prior['identity'], 'Changed model or native baseline')
        exp.guard.check_resources(initial=True)
        expected = {}
        for name in ('normal', 'traced'):
            extra = ['--dispatch-profile', exp.out/'dispatch.json'] if name == 'traced' else []
            raw = exp.bench(c, name, extra)
            validate_request(raw, exp.frozen, c, work, expected, instrumented=name == 'traced', output_tokens=5)
            memory = memory_observation(raw)
            exp.report[name] = dict(source=name+'.json', sha256=sha(exp.out/(name+'.json')), **memory)
            exp.persist()
            if not memory['memory_screen_passed']:
                raise ResourceBlocked('Memory disturbance during resident operator capture')
            if name == 'normal':
                needed = raw['runs'][-1]['after']['metal']['dispatches']-raw['runs'][0]['before']['metal']['dispatches']
                require(needed <= 100000, 'Request would exceed per-dispatch capture bound')
        normal = load(exp.out/'normal.json')
        exp.report.update(status='captured', operations=split_operations(raw, load(exp.out/'dispatch.json')),
            trace_to_normal_decode_ratios=[b['decode_wall_ms']/a['decode_wall_ms']
                                          for a, b in zip(normal['runs'], raw['runs'])],
            limitations=['Four decode forwards per phase; short initial prompt and retained 128-token append.',
                'Per-dispatch passes perturb execution. Costs are diagnostic, not normal kernel durations.',
                'A before-expert-reads group includes carried reduction/residual work from the preceding layer.',
                'No overhead subtraction, speed prediction, long-context or sustained-use qualification.'])
    return exp.report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    raise SystemExit(0 if run(parser.parse_args().output)['complete'] else 2)
