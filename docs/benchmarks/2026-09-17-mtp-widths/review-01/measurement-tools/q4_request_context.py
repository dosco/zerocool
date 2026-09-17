#!/usr/bin/env python3
"""Screen actual Q4 requests, then capture their dependency context if needed."""
import argparse
import json
import math
from pathlib import Path
import shutil

from cache_residency import memory_observation, require
from capture_routes import load
from combined_q4 import PACKED, configs, freeze, shader_origin, state_proof
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
from q4_request_profile import analyze_trace, compare_traces
from screen_cache import validate_request
from stage200 import Experiment
from verify_stage200 import verify_sources

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'docs/benchmarks/2026-09-15-q4-request-context'
PROTOCOL = BASE/'protocol.md'
SOURCE = ROOT/'docs/benchmarks/2026-09-14-combined-q4/screen-01'
STATE = ROOT/'docs/benchmarks/2026-09-14-combined-q4/state-01'
ORDER = ['normal-A', 'normal-B', 'traced-B', 'traced-A']
CRITERIA = dict(output_tokens=17, decode_forwards_per_phase=16, expert_slots=1072,
    memory_budget_bytes=12*1024**3, stage_seconds=240, inference_process_seconds=90,
    maximum_inference_processes=4, cancellation_drain_seconds=45,
    normal_order=ORDER[:2], conditional_trace_order=ORDER[2:],
    trace_trigger='any packed/reference phase decode_wall_ms ratio > 1',
    validation=False, counters=False, trace_dispatch_limit=120000)
FLAGS = dict(normal_request_latency_qualified=False, production_promoted=False)
LIMITATIONS = [
    'One normal pair is a directional trigger, not a confidence-bounded gain or regression; previous rejected request evidence is unchanged.',
    'The initial prompt has 72 tokens. The 128-token append follows 17 outputs, reuses 88 computed tokens and ingests the pending output. This is not 2K/4K or sustained-use qualification.',
    'Native memory is sampled at lifecycle boundaries and cumulative peaks; generic live guards cover disk and identity, not continuous physical memory or host conditions.',
    'Current requests verify exact output tokens, dispatch work and actual reuse. Persistent-state correctness is reused from the sealed compatible full-model check, not newly hashed here.',
    'Traced and normal timings are reported separately without an instrumentation correction. Mixed GPU commands are not isolated kernel costs; overlapping timing annotations are not additive.',
    'No recurrence ends this attempt without promoting the candidate. Resource blocks or incomplete traces cannot establish a performance result.']


def workload():
    work = load(SOURCE/'workload.json')
    require([len(w['tokens']) for w in work] == [72,128] and
            all(w['max_tokens'] == 33 for w in work), 'Changed source workload')
    return [dict(w, max_tokens=17) for w in work]


def memory(raw):
    result = memory_observation(raw)
    states = [s for row in raw['runs'] for s in [row['before'], row['after'],
        *[p[k] for p in row['phases'].values() for k in ('before','after')],
        *[d[k] for d in row.get('decode_diagnostics',{}).get('samples',[]) for k in ('before','after')]]]
    processes = [s['process'] for s in states]
    valid = bool(processes) and all(type(p.get(k)) is int and p[k] >= 0 for p in processes for k in
        ('physical_footprint_bytes','physical_footprint_peak_bytes','compressed_bytes',
         'compressed_peak_bytes','decompressions','system_swap_used_bytes'))
    bounds = valid and all(0 < p['physical_footprint_bytes'] <= p['physical_footprint_peak_bytes'] <=
        CRITERIA['memory_budget_bytes'] for p in processes)
    clean = (valid and all(p['compressed_bytes'] == p['compressed_peak_bytes'] == 0 for p in processes) and
        len({p['decompressions'] for p in processes}) == 1)
    # Samples appear after their containing boundaries above; compare swap extrema,
    # not that deliberately nonchronological collection's adjacent values.
    stable_swap = valid and len({p['system_swap_used_bytes'] for p in processes}) == 1
    result.update(physical_bounds_passed=bounds, cumulative_compression_clean=clean,
        stable_observed_swap=stable_swap,
        peak_physical_bytes=max(p['physical_footprint_peak_bytes'] for p in processes) if valid else None,
        memory_screen_passed=bool(result['memory_screen_passed'] and bounds and clean and stable_swap))
    return result


def observe(raw, frozen, config, work, expected, traced):
    requests = validate_request(raw, frozen, config, work, expected,
        output_tokens=17, instrumented=traced)
    counts = []
    for row in raw['runs']:
        for state in [row['before'],row['after'],*[p[k] for p in row['phases'].values() for k in ('before','after')]]:
            m = state['metal']; r = m['residency']
            require(m['live_command_groups'] == 0 and m['peak_command_groups'] <= 2 and
                r['mode'] == 'core-cache' and r['pending_retirements'] == 0 and
                m['kernels']['counter_profile'] is False and
                m['kernels']['profile_decode_only'] is traced, 'Changed residency or instrumentation')
        final = row['after']
        require(final['metal']['residency']['bytes_by_class']['expert'] == final['memory_plan']['expert_bytes'],
                'Cache residency enrollment missing')
        a,b = (row['phases']['decode'][k]['metal'] for k in ('before','after'))
        delta = {k:b['kernel_dispatches'].get(k,0)-a['kernel_dispatches'].get(k,0)
                 for k in set(a['kernel_dispatches']) | set(b['kernel_dispatches'])}
        require(all(v >= 0 for v in delta.values()), 'Dispatch counter reset')
        for packed, ref in zip(PACKED, ('q4_gate_up','q4_mm')):
            n = delta.pop(packed,0)
            require(n == (16*480 if config['q4_decode'] == 'packed-r2' else 0), 'Wrong packed work')
            delta[ref] = delta.get(ref,0)+n
            require(delta[ref] == 16*480, 'Missing complete routed projection work')
        delta = {k:v for k,v in delta.items() if v}
        require(expected.setdefault('dispatches_'+row['name'],delta) == delta and
                sum(delta.values()) == b['dispatches']-a['dispatches'], 'Changed normalized decode work')
        counts.append(sum(delta.values()))
    require(sum(counts) <= CRITERIA['trace_dispatch_limit'], 'Trace would exceed dispatch capacity')
    return dict(requests=requests, dispatches_by_phase=counts, **memory(raw))


def normal_decision(rows):
    require([r['stem'] for r in rows] == ORDER[:2], 'Missing ordered normal pair')
    phases = []
    for a,b in zip(rows[0]['requests'], rows[1]['requests']):
        require(a['name'] == b['name'] and all(type(r[k]) in (int,float) and math.isfinite(r[k]) and r[k] > 0
            for r in (a,b) for k in ('decode_wall_ms','request_ms','time_to_first_token_ms')), 'Invalid paired phase')
        phases.append(dict(name=a['name'],
            ratios={k:b[k]/a[k] for k in ('decode_wall_ms','request_ms','time_to_first_token_ms')},
            reference_ms_per_token=a['decode_wall_ms']/16, packed_ms_per_token=b['decode_wall_ms']/16,
            reference_gap_to_5tps_ms=max(0,a['decode_wall_ms']/16-200),
            packed_gap_to_5tps_ms=max(0,b['decode_wall_ms']/16-200)))
    require(len(phases) == 2, 'Missing normal phase')
    return dict(trace_triggered=any(p['ratios']['decode_wall_ms'] > 1 for p in phases),
        phases=phases, confidence_95=None, paired_repetitions=1, prior_pairs_pooled=False)


def reconstruct_trace(out, stem):
    return analyze_trace(load(out/(stem+'.json')), load(out/(stem+'.commands.json')),
        [json.loads(line) for line in (out/(stem+'.dependencies.jsonl')).read_text().splitlines()])


def finish(rows, analyses):
    normal = normal_decision(rows[:2])
    require([r['stem'] for r in rows] == (ORDER if normal['trace_triggered'] else ORDER[:2]),
            'Conditional process order differs')
    require(all(r['memory_screen_passed'] for r in rows), 'Disturbed request memory')
    result = dict(status='context_captured' if normal['trace_triggered'] else 'slowdown_not_reproduced',
                  normal=normal, **FLAGS)
    if normal['trace_triggered']:
        result['comparison'] = compare_traces(analyses['traced-A'],analyses['traced-B'])
        result['trace_to_normal_decode_ratios'] = {
            arm:[t['decode_wall_ms']/n['decode_wall_ms'] for n,t in
                 zip(rows[i]['requests'],rows[3-i]['requests'])] for i,arm in enumerate(('A','B'))}
    else:
        require(not analyses, 'Unplanned trace data')
    return result


def run(out):
    work = workload(); exp = Experiment(out,'q4_request_context_v1',configs()[:2],work,240)
    with exp:
        verify_seal(SOURCE,sha(SOURCE/'evidence-files.json'))
        exp.report.update(criteria=CRITERIA,limitations=LIMITATIONS,shader_origin=shader_origin(),
            correctness=state_proof(STATE,exp.frozen),
            state_source=dict(path=str(STATE),seal_sha256=sha(STATE/'evidence-files.json')),
            workload_source=dict(path=str(SOURCE),seal_sha256=sha(SOURCE/'evidence-files.json'),
                                 sha256=sha(SOURCE/'workload.json')))
        shutil.copyfile(PROTOCOL,exp.out/'protocol.md')
        exp.report['protocol_sha256'] = sha(PROTOCOL)
        freeze(exp,[PROTOCOL,exp.out/'protocol.md',SOURCE/'workload.json',SOURCE/'evidence-files.json',
                    *[p for p in STATE.iterdir() if p.is_file()]])
        exp.persist(); exp.guard.check_resources(initial=True)
        expected = {}; analyses = {}
        for stem in ORDER:
            traced = stem.startswith('traced')
            if traced and not normal_decision(exp.report['measurements'][:2])['trace_triggered']: break
            c = configs()[stem.endswith('B')]
            extra = ['--decode-diagnostics','--phase-profile',exp.out/(stem+'.commands.json'),
                '--profile-decode-only','1','--dependency-trace',exp.out/(stem+'.dependencies.jsonl')] if traced else []
            raw = exp.bench(c,stem,extra,limit=90)
            item = dict(stem=stem,source=stem+'.json',sha256=sha(exp.out/(stem+'.json')),
                        **observe(raw,exp.frozen,c,work,expected,traced))
            exp.report['measurements'].append(item); exp.persist()
            if not item['memory_screen_passed']: raise ResourceBlocked('Request memory is disturbed, over budget or unavailable')
            if traced:
                analyses[stem] = reconstruct_trace(exp.out,stem)
                save(exp.out/(stem+'.analysis.json'),analyses[stem])
        exp.report.update(finish(exp.report['measurements'],analyses))
    return exp.report


def verify(out):
    digest = sha(out/'evidence-files.json'); verify_seal(out,digest)
    saved,frozen = load(out/'summary.json'),load(out/'identity.json')
    require(saved['kind'] == 'q4_request_context_v1' and saved['criteria'] == CRITERIA and
        saved['limitations'] == LIMITATIONS and saved['configurations'] == configs()[:2] and
        saved['time_limit_seconds'] == CRITERIA['stage_seconds'] and
        saved['identity'] == {k:frozen[k] for k in saved['identity']} and
        all(saved.get(k) is False for k in FLAGS), 'Changed request context protocol or identity')
    provenance = verify_sources(out.parent,frozen)
    require(sha(out/'protocol.md') == saved['protocol_sha256'] == frozen['files'][str(PROTOCOL)], 'Protocol changed')
    source = Path(saved['workload_source']['path'])
    verify_seal(source,saved['workload_source']['seal_sha256'])
    require(sha(source/'workload.json') == saved['workload_source']['sha256'] and
        saved['workload'] == [dict(w,max_tokens=17) for w in load(source/'workload.json')] == load(out/'workload.json'),
        'Workload changed')
    state = Path(saved['state_source']['path'])
    require(sha(state/'evidence-files.json') == saved['state_source']['seal_sha256'] and
        state_proof(state,frozen) == saved['correctness'] and saved['shader_origin'] == shader_origin(), 'State proof changed')
    expected = {}; analyses = {}; rows = saved['measurements']
    require(len(rows) <= 4, 'Extra inference processes')
    for i,item in enumerate(rows):
        stem = ORDER[i]; traced = stem.startswith('traced')
        require(item['stem'] == stem and item['source'] == stem+'.json' and
            sha(out/item['source']) == item['sha256'], 'Changed or reordered source')
        if traced: require(normal_decision(rows[:2])['trace_triggered'], 'Untriggered trace')
        result = observe(load(out/item['source']),frozen,configs()[stem.endswith('B')],saved['workload'],expected,traced)
        require(all(item.get(k) == v for k,v in result.items()), 'Changed observation')
        path = out/(stem+'.analysis.json')
        if path.exists():
            require(traced, 'Normal run has trace analysis')
            analyses[stem] = reconstruct_trace(out,stem)
            require(load(path) == analyses[stem], 'Changed trace analysis')
    if saved['complete']:
        result = finish(rows,analyses)
        require(all(saved.get(k) == v for k,v in result.items()), 'Changed final decision')
    else:
        require(saved['status'] in ('resource_blocked','failed','interrupted','time_budget_exhausted'), 'Unfinished disposition')
    return dict(kind='q4_request_context_audit_v1',complete=True,audit_passed=True,
        recorded_complete=saved['complete'],status=saved['status'],source_seal_sha256=digest,
        source_provenance=provenance,**FLAGS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=('run','verify'))
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--source',type=Path)
    args = parser.parse_args()
    if args.action == 'verify':
        if args.source is None: parser.error('--source required')
        save(args.output,verify(args.source.resolve()))
    else: raise SystemExit(0 if run(args.output)['complete'] else 2)
