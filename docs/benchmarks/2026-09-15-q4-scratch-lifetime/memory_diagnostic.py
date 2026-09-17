#!/usr/bin/env python3
"""One sealed forward trace to observe memory; no timing qualification or retry."""
import argparse
from pathlib import Path
import shutil
import sys

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[3]
sys.path.insert(0, str(ROOT/'scripts/qwen'))
from cache_residency import require
from capture_routes import load
from combined_q4 import freeze, shader_origin, state_proof
import q4_read_arrivals as arrivals
import q4_scratch_lifetime as lifetime
from q4_shared_arrivals import REFERENCE, reference_files, reference_identity, reference_inputs
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
from screen_q4_packed import FIXTURES, verify_records
from stage200 import Experiment
from verify_stage200 import verify_sources

PROTOCOL = SCRIPT.with_name('memory-protocol.md')
BLOCKED = [SCRIPT.with_name(name) for name in ('forward-01', 'forward-02')]
CRITERIA = dict(lifetime.CRITERIA, process_seconds=60, stage_seconds=90,
                modes=['trace'], performance_comparison=False,
                validation=False, command_profile=True, counter_profile=False)
LIMITATIONS = [
    'The two original forward stages remain resource_blocked. This observation neither resumes nor qualifies them.',
    'Original check processes enabled Metal validation and disabled command profiling; this trace disables validation and enables command profiling. The instrumentation changes together, so memory differences cannot be causally attributed to validation.',
    'There is no uninstrumented timing process or performance ratio. A clean trace proves only observed memory and captured lifetime correctness under this diagnostic configuration.',
    'Memory gauges are boundary observations; they can miss brief events. Device counters include other processes.']


def blocked_proof(path, frozen):
    audit = lifetime.verify(path)
    saved = load(path/'summary.json'); raw = load(path/'check.json')
    require(audit['audit_passed'] is True and saved['scope'] == 'forward' and
            saved['complete'] is False and saved['status'] == 'resource_blocked' and
            set(audit['modes']) == {'check'} and saved['check']['clean_memory'] is False and
            saved['check']['clean_host'] is True and
            saved['identity'] == {k: frozen[k] for k in saved['identity']},
            'Requires the unchanged compatible forward checks blocked specifically by observed memory')
    return dict(path=str(path.resolve()), seal_sha256=sha(path/'evidence-files.json'),
        check_sha256=sha(path/'check.json'), status='resource_blocked',
        routed_output_sha256=raw['output_sha256'],
        shared_output_sha256=raw['shared_prelude']['expected_native_sha256'])


def analyze(raw, frozen, records, manifest, gates, blocked):
    require(raw.get('mode') == 'trace' and raw.get('validation') is False,
            'Memory observation requires only the declared validation-off trace')
    result = lifetime.analyze(raw, 'forward', frozen, records, manifest, gates)
    require(all(raw['output_sha256'] == p['routed_output_sha256'] and
                raw['shared_prelude']['expected_native_sha256'] == p['shared_output_sha256'] for p in blocked),
            'Trace outputs differ from either independently sealed blocked check')
    result.update(status='memory_observed' if result['clean_memory'] and result['clean_host'] else 'resource_blocked',
        performance_comparison=False, validation_causal_attribution=False,
        normal_request_latency_qualified=False, production_promoted=False, original_lifetime_stage_qualified=False)
    result['limitations'] += LIMITATIONS
    require(result['results'] == [], 'Memory-only observation unexpectedly produced timing comparisons')
    return result


def run(args):
    exp = Experiment(args.output, 'q4_scratch_memory_diagnostic_v1', [], [], 90)
    with exp:
        exp.report.update(criteria=CRITERIA, shader_origin=shader_origin(), scope='forward',
            limitations=LIMITATIONS, performance_comparison=False, validation_causal_attribution=False,
            original_lifetime_stage_qualified=False)
        for source, target in ((PROTOCOL, 'protocol.md'), (SCRIPT, 'memory_diagnostic.py')):
            shutil.copyfile(source, exp.out/target)
        exp.report.update(protocol_sha256=sha(PROTOCOL), runner_sha256=sha(SCRIPT))
        manifest, gates = reference_inputs(args.shared_fixtures)
        exp.report['shared_reference_source'] = dict(path=str(args.shared_fixtures.resolve()),
            manifest_sha256=sha(args.shared_fixtures/'manifest.json'), manifest=manifest)
        exp.report['correctness'] = state_proof(args.state, exp.frozen)
        exp.report['control'] = arrivals.control_proof(args.control, exp.frozen)
        for key in ('state', 'control'):
            path = getattr(args, key)
            exp.report[key+'_source'] = dict(path=str(path.resolve()), sha256=sha(path/'evidence-files.json'))
        exp.report['blocked_checks'] = [blocked_proof(path, exp.frozen) for path in BLOCKED]
        freeze(exp, [args.binary, SCRIPT, PROTOCOL, exp.out/'protocol.md', exp.out/'memory_diagnostic.py',
            ROOT/'scripts/qwen/probe_q4_packed.metal', ROOT/'kernels/metal/qwen.metal',
            *reference_files(args.shared_fixtures),
            *[p for d in (args.state, args.control, FIXTURES, *BLOCKED) for p in d.iterdir() if p.is_file()]])
        reference_identity(manifest, exp.frozen)
        exp.report['verified_records'] = verify_records(exp.prepared); exp.persist()
        exp.guard.check_resources(initial=True)
        exp.command([args.binary, FIXTURES, exp.out/'trace.json', 'arrivals', exp.model, exp.prepared,
                     '2', 'trace', 'invalidate', args.shared_fixtures, 'scratch-forward'], 'trace', 60, validation=False)
        raw = load(exp.out/'trace.json')
        require(raw['shared_prelude']['reference_manifest_sha256'] == sha(args.shared_fixtures/'manifest.json'),
                'Trace used another CPU reference')
        exp.report['trace'] = analyze(raw, exp.frozen, exp.report['verified_records'], manifest, gates, exp.report['blocked_checks'])
        exp.persist()
        if exp.report['trace']['status'] != 'memory_observed':
            raise ResourceBlocked('Forward trace still has disturbed or unavailable memory/host observations')
        exp.report.update(status='memory_observed', normal_request_latency_qualified=False, production_promoted=False, original_lifetime_stage_qualified=False)
    return exp.report


def verify(directory):
    directory = directory.resolve(); digest = sha(directory/'evidence-files.json'); verify_seal(directory, digest)
    saved, frozen = load(directory/'summary.json'), load(directory/'identity.json')
    require(saved.get('kind') == 'q4_scratch_memory_diagnostic_v1' and saved.get('criteria') == CRITERIA and
            saved.get('scope') == 'forward' and saved.get('configurations') == [] and saved.get('workload') == [] and
            saved['identity'] == {k: frozen[k] for k in saved['identity']} and
            saved.get('normal_request_latency_qualified') is False and saved.get('production_promoted') is False and
            saved.get('performance_comparison') is False and saved.get('validation_causal_attribution') is False and
            saved.get('original_lifetime_stage_qualified') is False and
            saved.get('limitations') == LIMITATIONS, 'Changed memory-only diagnostic identity or scope')
    provenance = verify_sources(directory.parent,
        dict(frozen, files={p:h for p,h in frozen['files'].items() if Path(p).is_relative_to(frozen['root'])}))
    for source, target, key in ((SCRIPT, 'memory_diagnostic.py', 'runner_sha256'), (PROTOCOL, 'protocol.md', 'protocol_sha256')):
        require(sha(directory/target) == saved[key] == frozen['files'][str(source)], 'Changed frozen memory diagnostic source')
    require(saved['shader_origin'] == shader_origin(), 'Changed shader origin')
    for key, checker in (('state', state_proof), ('control', arrivals.control_proof)):
        source=saved[key+'_source']; path=Path(source['path'])
        require(sha(path/'evidence-files.json') == source['sha256'] and
                checker(path, frozen) == saved['correctness' if key == 'state' else key], 'Changed prior '+key+' proof')
    require(saved['blocked_checks'] == [blocked_proof(p, frozen) for p in BLOCKED], 'Changed blocked check provenance')
    source=saved['shared_reference_source']; path=Path(source['path']); manifest,gates=reference_inputs(path)
    reference_identity(manifest, frozen)
    require(manifest == source['manifest'] and sha(path/'manifest.json') == source['manifest_sha256'], 'Changed CPU oracle')
    for file in reference_files(path):
        require(sha(file) == frozen['files'].get(str(file.resolve())), 'Changed frozen reference payload')
    result = None
    if 'trace' in saved:
        raw=load(directory/'trace.json')
        require(raw['shared_prelude']['reference_manifest_sha256'] == source['manifest_sha256'], 'Changed native oracle identity')
        result=analyze(raw,frozen,saved['verified_records'],manifest,gates,saved['blocked_checks'])
        require(result == saved['trace'], 'Changed memory trace analysis')
    require(not (directory/'timing.json').exists() and not (directory/'check.json').exists(),
            'Unexpected performance or validation process in memory observation')
    if saved['complete']:
        require(result is not None and saved['status'] == result['status'] == 'memory_observed', 'Incomplete clean-memory claim')
    else:
        require(saved['status'] in ('resource_blocked','failed','interrupted','time_budget_exhausted'), 'Missing terminal failure disposition')
    return dict(kind='q4_scratch_memory_audit_v1', complete=True, audit_passed=True, source_seal_sha256=digest,
        source_provenance=provenance, recorded_complete=saved['complete'], recomputed_status=saved['status'], trace=result,
        performance_comparison=False, validation_causal_attribution=False,
        normal_request_latency_qualified=False, production_promoted=False, original_lifetime_stage_qualified=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='action',required=True)
    execute=sub.add_parser('run');execute.add_argument('--output',type=Path,required=True)
    execute.add_argument('--binary',type=Path,default=ROOT/'build/qwen/qwen_q4_check')
    execute.add_argument('--shared-fixtures',type=Path,default=REFERENCE)
    execute.add_argument('--state',type=Path,default=arrivals.STATE);execute.add_argument('--control',type=Path,default=arrivals.CONTROL)
    audit=sub.add_parser('verify');audit.add_argument('directory',type=Path);audit.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='verify':save(args.output,verify(args.directory))
    else:
        for field in ('binary','shared_fixtures','state','control'):setattr(args,field,getattr(args,field).resolve())
        raise SystemExit(0 if run(args)['complete'] else 2)
