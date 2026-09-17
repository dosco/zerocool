#!/usr/bin/env python3
"""One bounded reference trace with observed compression; never qualification."""
import argparse
import math
from pathlib import Path
import shutil

from cache_residency import require
from capture_routes import load
from combined_q4 import configs, freeze, shader_origin, state_proof
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
import q4_request_context as context
from request_diagnostic_memory import analyze_memory
from stage200 import Experiment
from verify_stage200 import verify_sources

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'docs/benchmarks/2026-09-15-reference-diagnostic'
PROTOCOL = BASE/'protocol.md'
REFERENCE = context.BASE/'capture-02'
CRITERIA = dict(inference_processes=1, inference_seconds=90, stage_seconds=150,
    cancellation_drain_seconds=45, budget_bytes=12*1024**3,
    compression_peak_limit_bytes=512*1024**2, expert_slots=1072, output_tokens=17,
    decode_forwards=32, dispatches=101600, layer_passes=1536, expert_records=15360,
    profiling='existing decode-only command/dependency and token observations',
    validation=False, counter_profile=False, safety_observation_scope='completed-run boundaries and lifetime peaks')
FLAGS = dict(normal_request_latency_qualified=False, production_promoted=False,
             performance_comparison=False, candidate_qualified=False)
LIMITATIONS = [
    'One instrumented reference conversation generates hypotheses only. There is no packed comparison, clean timing qualification or production promotion.',
    'The 512MiB compression peak limit is a declared operational diagnostic allowance, not an empirically established safe or harmless amount of compression.',
    'Memory and swap gates inspect completed-run boundaries and lifetime peaks; they do not provide live cancellation at the compression threshold. Native allocations stay admitted, and runtime/disk/cancellation guards remain active.',
    'All captured tokens remain included. Compression and decompression annotations do not identify the affected allocation or establish the cause of any stall.',
    'Exclusive token buckets reconcile to forward time. GPU command classes, read service, CPU costs and boundary gaps overlap and are not additional or removable milliseconds.',
    'The previous normal request is used only for exact output/workload checks. Its disturbed timings are not a control, pooled sample or instrumentation correction.',
    'The short initial/retained-append workload does not qualify 2K/4K/7K contexts, coding quality, or sustained operation.']


def rank(analysis):
    """Rank observed populations, without decomposing mixed commands or predicting gain."""
    require(analysis.get('kind') == 'q4_request_profile_v1' and analysis.get('complete') is True and
        analysis.get('variant') == 'reference' and
        analysis.get('coverage') == dict(phases=2,forwards=32,dispatches=101600,
            dependency_passes=1536,selected_experts=15360,explicit_dependency_file=True,omitted_tokens=0),
        'Incomplete reference opportunity coverage')
    phases = analysis.get('phases',[]); require(len(phases) == 2, 'Missing phase')
    expected_buckets = {'gpu_active','gpu_idle_submitted','gpu_idle_ready_expert',
                        'gpu_idle_pending_read','gpu_idle_callback','gpu_idle_other'}
    output = []; means = dict.fromkeys(expected_buckets,0.)
    for p in phases:
        buckets = p['mean_buckets_ms']
        require(p['captured_tokens'] == 16 and len(p['tokens']) == 16 and
            set(buckets) == expected_buckets and
            all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in buckets.values()) and
            math.isfinite(p['mean_forward_ms']) and p['mean_forward_ms'] > 0 and
            math.isclose(sum(buckets.values()),p['mean_forward_ms'],abs_tol=1e-6),
            'Missing, invalid or nonreconciling exclusive timing')
        for k,v in buckets.items(): means[k] += v/2
        classes = p['gpu_command_classes']; layers = p['layer_command_classes']
        for population in (classes,layers):
            require(population and all(r['stages'] and type(r['gpu_command_ms_per_token']) in (int,float) and
                math.isfinite(r['gpu_command_ms_per_token']) and r['gpu_command_ms_per_token'] >= 0
                for r in population), 'Invalid command class population')
            require(math.isclose(sum(r['gpu_command_ms_per_token'] for r in population),
                p['mean_gpu_command_ms'],abs_tol=1e-6), 'Command classes do not reconcile')
        output.append(dict(name=p['name'], observed_forward_ms=p['mean_forward_ms'],
            distance_to_200ms=max(0,p['mean_forward_ms']-200),
            exclusive_buckets=sorted([dict(category=k,ms_per_token=v,
                fraction_of_forward=v/p['mean_forward_ms']) for k,v in buckets.items()],
                key=lambda r:(-r['ms_per_token'],r['category'])),
            gpu_command_classes=sorted(classes,key=lambda r:-r['gpu_command_ms_per_token']),
            layer_command_classes=sorted(layers,key=lambda r:-r['gpu_command_ms_per_token'])))
    ordered = sorted([dict(category=k,ms_per_token=v) for k,v in means.items()],
                     key=lambda r:(-r['ms_per_token'],r['category']))
    return dict(kind='reference_diagnostic_opportunities_v1',complete=True,
        identity=analysis['identity'],phases=output,equal_phase_mean_buckets=ordered,
        largest_observed_category=ordered[0]['category'],
        possible_latency_saving_ms=None,causal_bottleneck_identified=False,
        limitations=LIMITATIONS,**FLAGS)


def annotation(raw):
    result = analyze_memory(raw,budget_bytes=CRITERIA['budget_bytes'],
                          compression_limit_bytes=CRITERIA['compression_peak_limit_bytes'])
    # This measures only explicit observation calls, not every tracing cost.
    result['observation_costs'] = [dict(name=r['name'],
        sample_calls_ms=sum(s[k]['sample_ns'] for s in r['decode_diagnostics']['samples']
                            for k in ('before','after'))/1e6,
        mean_sample_calls_ms_per_token=sum(s[k]['sample_ns'] for s in r['decode_diagnostics']['samples']
                            for k in ('before','after'))/16e6,
        timing_correction_applied=False) for r in raw['runs']]
    return result


def source_proof(exp):
    verify_seal(REFERENCE,sha(REFERENCE/'evidence-files.json'))
    saved = load(REFERENCE/'summary.json'); raw = load(REFERENCE/'normal-A.json')
    require(saved['identity'] == exp.report['identity'] and saved['workload'] == exp.report['workload'] and
        saved['measurements'][0]['stem'] == 'normal-A' and
        saved['measurements'][0]['sha256'] == sha(REFERENCE/'normal-A.json'), 'Changed output reference')
    expected = {}
    context.observe(raw,exp.frozen,configs()[0],exp.report['workload'],expected,False)
    return expected


def run(out):
    exp = Experiment(out,'reference_pressure_diagnostic_v1',[configs()[0]],context.workload(),150)
    with exp:
        exp.report.update(criteria=CRITERIA,limitations=LIMITATIONS,**FLAGS)
        exp.persist()
        exp.report.update(shader_origin=shader_origin(),correctness=state_proof(context.STATE,exp.frozen),
            state_source=dict(path=str(context.STATE),seal_sha256=sha(context.STATE/'evidence-files.json')),
            reference_source=dict(path=str(REFERENCE),seal_sha256=sha(REFERENCE/'evidence-files.json'),
                                  raw_sha256=sha(REFERENCE/'normal-A.json')))
        expected = source_proof(exp)
        shutil.copyfile(PROTOCOL,exp.out/'protocol.md');exp.report['protocol_sha256'] = sha(PROTOCOL)
        freeze(exp,[PROTOCOL,exp.out/'protocol.md',
            *[p for d in (context.STATE,REFERENCE) for p in d.iterdir() if p.is_file()]])
        exp.persist();exp.guard.check_resources(initial=True)
        extra = ['--decode-diagnostics','--phase-profile',exp.out/'reference.commands.json',
                 '--profile-decode-only','1','--dependency-trace',exp.out/'reference.dependencies.jsonl']
        raw = exp.bench(configs()[0],'reference',extra,limit=90)
        memory = annotation(raw); save(exp.out/'memory.json',memory)
        exp.report['memory'] = dict(source='memory.json',sha256=sha(exp.out/'memory.json'),
            hard_limits_passed=memory['hard_limits_passed'],strict_memory_clean=memory['strict_memory_clean'])
        exp.persist()
        if not memory['hard_limits_passed']:
            raise ResourceBlocked('Diagnostic memory allowance exceeded: '+str(memory['reasons']))
        observation = context.observe(raw,exp.frozen,configs()[0],exp.report['workload'],expected,True)
        exp.report['measurements'].append(dict(source='reference.json',sha256=sha(exp.out/'reference.json'),
            exact_outputs_match_reference=True,**observation));exp.persist()
        analysis = context.reconstruct_trace(exp.out,'reference');save(exp.out/'timeline.json',analysis)
        opportunities = rank(analysis);save(exp.out/'opportunities.json',opportunities)
        exp.report.update(status='diagnostic_captured_clean_memory' if memory['strict_memory_clean'] else
            'diagnostic_captured_with_pressure',timeline=dict(source='timeline.json',sha256=sha(exp.out/'timeline.json')),
            opportunities=dict(source='opportunities.json',sha256=sha(exp.out/'opportunities.json')),
            largest_observed_category=opportunities['largest_observed_category'])
    return exp.report


def verify(out):
    digest = sha(out/'evidence-files.json');verify_seal(out,digest)
    saved,frozen = load(out/'summary.json'),load(out/'identity.json')
    require(saved['kind'] == 'reference_pressure_diagnostic_v1' and saved['criteria'] == CRITERIA and
        saved['limitations'] == LIMITATIONS and saved['configurations'] == [configs()[0]] and
        saved['identity'] == {k:frozen[k] for k in saved['identity']} and
        saved['time_limit_seconds'] == CRITERIA['stage_seconds'] and
        all(saved.get(k) is False for k in FLAGS), 'Changed diagnostic scope or identity')
    provenance = verify_sources(out.parent,frozen)
    if 'protocol_sha256' in saved:
        require(sha(out/'protocol.md') == saved['protocol_sha256'] == frozen['files'][str(PROTOCOL)],'Protocol changed')
    expected = {}
    if 'reference_source' in saved:
        src = saved['reference_source'];path = Path(src['path']);verify_seal(path,src['seal_sha256'])
        require(sha(path/'normal-A.json') == src['raw_sha256'], 'Output reference changed')
        prior = load(path/'summary.json')
        require(prior['identity'] == saved['identity'] and prior['workload'] == saved['workload'] == load(out/'workload.json'),
            'Reference identity or workload changed')
        context.observe(load(path/'normal-A.json'),frozen,configs()[0],saved['workload'],expected,False)
        state = saved['state_source'];path = Path(state['path'])
        require(sha(path/'evidence-files.json') == state['seal_sha256'] and
            state_proof(path,frozen) == saved['correctness'] and saved['shader_origin'] == shader_origin(), 'State proof changed')
    require(len(saved['measurements']) <= 1,'Extra inference measurements')
    memory = None; raw = None
    if 'memory' in saved:
        raw = load(out/'reference.json');memory = annotation(raw)
        require(saved['memory']['source'] == 'memory.json' and sha(out/'memory.json') == saved['memory']['sha256'] and
            load(out/'memory.json') == memory and all(saved['memory'][k] == memory[k] for k in
            ('hard_limits_passed','strict_memory_clean')), 'Memory annotation changed')
    if saved['measurements']:
        m = saved['measurements'][0];require(m['source'] == 'reference.json' and
            sha(out/'reference.json') == m['sha256'] and m['exact_outputs_match_reference'] is True, 'Native source changed')
        observation = context.observe(raw,frozen,configs()[0],saved['workload'],expected,True)
        require(all(m.get(k) == v for k,v in observation.items()), 'Work observation changed')
    if saved['complete']:
        require(len(saved['measurements']) == 1 and memory['hard_limits_passed'] and
            saved['status'] == ('diagnostic_captured_clean_memory' if memory['strict_memory_clean'] else
                                'diagnostic_captured_with_pressure'), 'Invalid completed diagnostic')
        analysis = context.reconstruct_trace(out,'reference');opportunities = rank(analysis)
        for field,value in (('timeline',analysis),('opportunities',opportunities)):
            record=saved[field];path=out/(field+'.json')
            require(record['source'] == path.name and sha(path) == record['sha256'] and load(path) == value,
                    'Derived diagnostic changed')
        require(saved['largest_observed_category'] == opportunities['largest_observed_category'],'Ranking changed')
    else:
        require(saved['status'] in ('failed','resource_blocked','interrupted','time_budget_exhausted'), 'Missing terminal status')
    return dict(kind='reference_pressure_diagnostic_audit_v1',complete=True,audit_passed=True,
        recorded_complete=saved['complete'],status=saved['status'],source_seal_sha256=digest,
        source_provenance=provenance,**FLAGS)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('run','verify'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path);args=p.parse_args()
    if args.action == 'verify':
        if args.source is None:p.error('--source required')
        save(args.output,verify(args.source.resolve()))
    else:raise SystemExit(0 if run(args.output)['complete'] else 2)
