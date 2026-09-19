#!/usr/bin/env python3
"""Compare reused batch scratch with a retained arena over 48 expert passes."""
import argparse
from pathlib import Path
import shutil

from cache_residency import require
from capture_routes import load
from combined_q4 import freeze, shader_origin, state_proof
import q4_read_arrivals as arrivals
from native_q4_replay import uint
from q4_shared_arrivals import (REFERENCE, reference_files, reference_identity, reference_inputs, shared_proof)
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
from screen_q4_packed import FIXTURES, verify_records
from stage200 import Experiment
from verify_stage200 import verify_sources

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/'docs/benchmarks/2026-09-15-q4-scratch-lifetime/protocol.md'
SCOPES = ('batch', 'forward')
PASSES = 48
SCRATCH_BUFFERS_PER_PASS = 27
SCRATCH_BYTES_PER_PASS = 27*16384
CRITERIA = dict(arrivals.CRITERIA, timing_cycles=6, capture_cycles=6,
    hits_per_batch=2, cache_preparation=arrivals.INVALIDATE_FILES,
    shared_chains_per_batch=1, shared_layer_order=arrivals.SHARED_LAYERS,
    shared_gpu_scope='shared-and-routed-command-intervals', passes=PASSES,
    retained_payload='shared-and-routed-temporaries',
    batch_scratch_bytes=SCRATCH_BYTES_PER_PASS,
    forward_scratch_bytes=PASSES*SCRATCH_BYTES_PER_PASS)


def analyze(raw, scope, frozen=None, expected_records=None, reference_manifest=None, gate_values=None):
    require(scope in SCOPES and raw.get('hits_per_batch') == 2 and
            raw.get('cache_preparation') == arrivals.INVALIDATE_FILES and reference_manifest is not None,
            'Changed scratch condition, hit population, cache preparation or missing CPU reference')
    proof = shared_proof(raw, reference_manifest, gate_values)
    result = arrivals.analyze(raw, frozen, expected_records, shared=True, lifetime=scope)
    result.update(scope=scope, shared_reference=proof, criteria=CRITERIA)
    result['scratch_lifetime'] = lifetime_proof(raw, scope)
    result['limitations'] += [
        'The 48 passes repeat the same eight expert and four shared-layer fixtures six times; they are not 48 distinct model layers or a complete model forward pass.',
        'Both scopes drain GPU work and explicitly prepare file reads between passes. The forward condition retains its scratch arena across those boundaries.',
        'Batch and forward scope timings come from separate processes; packed/reference pairs within each scope are the statistical units. Cross-scope absolute differences are contextual.',
        'The fixed expert cache remains eight slots. This measures temporary-buffer lifetime and does not vary expert-cache capacity.']
    return result


def scratch_snapshot(snapshot, retained_bytes, used_buffers, active):
    require(isinstance(snapshot, dict) and snapshot.get('active_scratch_slot') == active and
            snapshot.get('live_command_groups') == 0 and uint(snapshot.get('live_buffer_bytes')) and
            0 < snapshot['live_buffer_bytes'] <= arrivals.BUDGET,
            'Scratch observation has wrong arena state, outstanding users or invalid live bytes')
    pool = snapshot.get('pool', {})
    require(all(uint(pool.get(k)) for k in ('capacity_bytes', 'allocated_bytes', 'peak_bytes',
            'unused_retained_bytes', 'reuses', 'allocation_count', 'wait_ns')) and
            pool['capacity_bytes'] == 128*1024**2 and pool['allocated_bytes'] == retained_bytes and
            retained_bytes <= pool['peak_bytes'] <= pool['capacity_bytes'] and
            pool['unused_retained_bytes'] == retained_bytes-used_buffers*16384 and
            snapshot.get('active_prefix_bytes') == used_buffers*16384 and
            snapshot.get('inferred_used_buffers') == used_buffers,
            'Scratch cursor, retained bytes or fixed buffer-charge inference differs')
    return pool


def lifetime_proof(raw, scope):
    expected = dict(passes=PASSES, scope=scope, retained_payload='shared-and-routed-temporaries',
                    buffers_per_pass=SCRATCH_BUFFERS_PER_PASS, buffer_charge_bytes=16384)
    require(raw.get('scratch_lifetime') == expected, 'Changed scratch lifetime protocol')
    retained = SCRATCH_BYTES_PER_PASS*(PASSES if scope == 'forward' else 1)
    total_buffers = retained//16384; boundaries = PASSES if scope == 'batch' else 1
    captures = []
    for pair in raw['pairs']:
        for arm in pair['arms']:
            sample = arm['sample']
            require(sample.get('scope_begins') == sample.get('scope_ends') == boundaries and
                    sample.get('retained_scratch_bytes') == retained,
                    'Missing arena boundaries or changed retained scratch payload')
            states = [arm['metal_before'], arm['metal_after']]
            for state in states:
                pools = state.get('scratch_pools')
                require(isinstance(pools, list) and len(pools) == 2 and
                        pools[1].get('allocated_bytes') == 0 and pools[1].get('capacity_bytes') == 0,
                        'Unexpected secondary scratch pool')
                scratch_snapshot(dict(active_scratch_slot=state['active_scratch_slot'],
                    live_command_groups=state['live_command_groups'], live_buffer_bytes=state['live_buffer_bytes'],
                    pool=pools[0], active_prefix_bytes=retained, inferred_used_buffers=total_buffers),
                    retained, total_buffers, -1)
            before, after = (state['scratch_pools'][0] for state in states)
            require(states[0]['live_buffer_bytes'] == states[1]['live_buffer_bytes'] and
                    after['allocation_count'] == before['allocation_count'] and
                    after['reuses']-before['reuses'] == PASSES*SCRATCH_BUFFERS_PER_PASS,
                    'Scratch allocation or reuse changed across the measured arm')
            if raw['mode'] != 'trace':
                require(not sample.get('batches'), 'Untimed scratch instrumentation leaked into timing/check modes')
                continue
            previous = None; summary = []
            for index, batch in enumerate(sample['batches']):
                lifetime = batch.get('lifetime', {})
                require(set(lifetime) == {'before_preparation', 'after_preparation', 'after_work'},
                        'Missing complete scratch lifecycle observations')
                entering = lifetime['before_preparation']; prepared = lifetime['after_preparation']; done = lifetime['after_work']
                entering_used = total_buffers if index == 0 or scope == 'batch' else index*SCRATCH_BUFFERS_PER_PASS
                entering_active = 0 if scope == 'forward' and index else -1
                final_used = (index+1)*SCRATCH_BUFFERS_PER_PASS if scope == 'forward' else SCRATCH_BUFFERS_PER_PASS
                final_active = 0 if scope == 'forward' and index < PASSES-1 else -1
                a = scratch_snapshot(entering, retained, entering_used, entering_active)
                scratch_snapshot(prepared, retained, entering_used, entering_active)
                c = scratch_snapshot(done, retained, final_used, final_active)
                require(entering == prepared, 'Read preparation changed a live scratch arena')
                require(entering['live_buffer_bytes'] == done['live_buffer_bytes'] == states[0]['live_buffer_bytes'] and
                        c['allocation_count'] == a['allocation_count'] and
                        c['reuses']-a['reuses'] == SCRATCH_BUFFERS_PER_PASS and c['wait_ns'] >= a['wait_ns'],
                        'Scratch work allocated, released or failed to retain the declared temporaries')
                if previous is not None:
                    require(entering == previous, 'Scratch ownership changed between consecutive passes')
                else:
                    require(a == before, 'Initial captured scratch pool differs from the native arm boundary')
                previous = done
                summary.append(dict(pass_index=index, cycle=batch['cycle'], batch=batch['batch'],
                    active_after_work=final_active, used_buffers=final_used,
                    active_prefix_bytes=done['active_prefix_bytes'], retained_bytes=retained))
            require(len(summary) == PASSES and previous['pool'] == after,
                    'Scratch trace does not reach the final arm boundary')
            captures.append(dict(pair=pair['pair'], group=pair['group'], variant=arm['variant'], passes=summary))
    return dict(**expected, retained_scratch_bytes=retained, retained_buffers=total_buffers,
        scope_begins_per_arm=boundaries, scope_ends_per_arm=boundaries,
        reused_buffers_per_arm=PASSES*SCRATCH_BUFFERS_PER_PASS, captures=captures,
        complete_model_forward=False, inferred_cursor_basis='All scratch requests are charged 16384 bytes; used prefix is allocated minus unused retained bytes')


def run(args):
    exp = Experiment(args.output, 'q4_scratch_lifetime_diagnostic_v1', [], [], CRITERIA['stage_seconds'])
    with exp:
        exp.report.update(scope=args.scope, hits_per_batch=2, criteria=CRITERIA,
                          shader_origin=shader_origin(), cache_preparation=arrivals.INVALIDATE_FILES)
        shutil.copyfile(PROTOCOL, exp.out/'protocol.md')
        exp.report['protocol_sha256'] = sha(PROTOCOL)
        manifest, gates = reference_inputs(args.shared_fixtures)
        exp.report['shared_reference_source'] = dict(path=str(args.shared_fixtures.resolve()),
            manifest_sha256=sha(args.shared_fixtures/'manifest.json'), manifest=manifest)
        exp.report['correctness'] = state_proof(args.state, exp.frozen)
        exp.report['control'] = arrivals.control_proof(args.control, exp.frozen)
        for key in ('state', 'control'):
            path = getattr(args, key)
            exp.report[key+'_source'] = dict(path=str(path.resolve()), sha256=sha(path/'evidence-files.json'))
        freeze(exp, [args.binary, PROTOCOL, exp.out/'protocol.md',
            ROOT/'scripts/qwen/probe_q4_packed.metal', ROOT/'kernels/metal/qwen.metal',
            *reference_files(args.shared_fixtures),
            *[p for d in (args.state, args.control, FIXTURES) for p in d.iterdir() if p.is_file()]])
        reference_identity(manifest, exp.frozen)
        exp.report['verified_records'] = verify_records(exp.prepared); exp.persist()
        exp.guard.check_resources(initial=True)
        for mode in ('check', 'timing', 'trace'):
            extra = [args.shared_fixtures, 'scratch-'+args.scope]
            exp.command([args.binary, FIXTURES, exp.out/(mode+'.json'), 'arrivals', exp.model, exp.prepared,
                         '2', mode, 'invalidate', *extra], mode, CRITERIA['process_seconds'], validation=mode == 'check')
            raw = load(exp.out/(mode+'.json'))
            require(raw.get('mode') == mode, 'Native shared replay returned wrong mode')
            require(raw['shared_prelude']['reference_manifest_sha256'] == sha(args.shared_fixtures/'manifest.json'),
                    'Native shared replay used another CPU fixture')
            exp.report[mode] = analyze(raw, args.scope, exp.frozen, exp.report['verified_records'], manifest, gates)
            exp.persist()
            if not all(exp.report[mode][k] for k in ('clean_memory', 'clean_host')):
                raise ResourceBlocked('Scratch replay has disturbed or unavailable memory/host observations')
        require(len({exp.report[m]['shared_reference']['expected_native_sha256'] for m in ('check', 'timing', 'trace')}) == 1,
                'Native shared outputs differ between independent mode processes')
        exp.report.update(status=exp.report['timing']['status'], normal_request_latency_qualified=False, production_promoted=False)
    return exp.report


def verify(directory):
    directory = directory.resolve(); seal_digest = sha(directory/'evidence-files.json')
    verify_seal(directory, seal_digest)
    saved, frozen = load(directory/'summary.json'), load(directory/'identity.json')
    require(saved.get('kind') == 'q4_scratch_lifetime_diagnostic_v1' and saved.get('scope') in SCOPES and
            saved.get('hits_per_batch') == 2 and saved.get('cache_preparation') == arrivals.INVALIDATE_FILES and
            saved.get('criteria') == CRITERIA and saved.get('configurations') == [] and saved.get('workload') == [] and
            saved['identity'] == {k: frozen[k] for k in saved['identity']}, 'Changed scratch experiment identity or design')
    sources = dict(frozen, files={p: h for p, h in frozen['files'].items() if Path(p).is_relative_to(frozen['root'])})
    provenance = verify_sources(directory.parent, sources)
    require(sha(directory/'protocol.md') == saved['protocol_sha256'] == frozen['files'][str(PROTOCOL.resolve())],
            'Changed declared shared arrival protocol')
    require(saved.get('shader_origin') == shader_origin(), 'Changed shared integrated shader origin')
    for key, checker in (('state', state_proof), ('control', arrivals.control_proof)):
        if key+'_source' not in saved: continue
        source = saved[key+'_source']; path = Path(source['path'])
        require(sha(path/'evidence-files.json') == source['sha256'] and
                checker(path, frozen) == saved['correctness' if key == 'state' else 'control'], 'Changed prior '+key+' evidence')
    source = saved['shared_reference_source']; fixture_path = Path(source['path'])
    manifest, gates = reference_inputs(fixture_path)
    reference_identity(manifest, frozen)
    require(manifest == source['manifest'] and sha(fixture_path/'manifest.json') == source['manifest_sha256'],
            'Changed shared reference source')
    for file in reference_files(fixture_path):
        require(frozen['files'].get(str(file.resolve())) == sha(file), 'Changed frozen shared reference file')
    results = {}
    for mode in ('check', 'timing', 'trace'):
        if mode not in saved: continue
        raw = load(directory/(mode+'.json'))
        require(raw.get('mode') == mode, 'Wrong captured shared mode')
        require(raw['shared_prelude']['reference_manifest_sha256'] == source['manifest_sha256'], 'Wrong captured shared reference')
        results[mode] = analyze(raw, saved['scope'], frozen, saved['verified_records'], manifest, gates)
        require(saved[mode] == results[mode], 'Changed shared '+mode+' analysis')
    if saved['complete'] is True:
        require(set(results) == {'check', 'timing', 'trace'} and 'state_source' in saved and 'control_source' in saved and
                saved['status'] == results['timing']['status'] and
                all(r[k] for r in results.values() for k in ('clean_memory', 'clean_host')), 'Incomplete or disturbed completed shared stage')
        require(len({r['shared_reference']['expected_native_sha256'] for r in results.values()}) == 1,
                'Changed shared outputs across mode processes')
    else:
        require(saved['status'] in ('failed', 'resource_blocked', 'interrupted', 'time_budget_exhausted'),
                'Unfinished shared stage lacks a terminal disposition')
    return dict(kind='q4_scratch_lifetime_audit_v1', complete=True, audit_passed=True,
        source_seal_sha256=seal_digest, source_provenance=provenance, scope=saved['scope'],
        recorded_complete=saved['complete'], recomputed_status=saved['status'], modes=results,
        normal_request_latency_qualified=False, production_promoted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='mode', required=True)
    execute = sub.add_parser('run'); execute.add_argument('--output', type=Path, required=True)
    execute.add_argument('--scope', choices=SCOPES, required=True)
    execute.add_argument('--shared-fixtures', type=Path, default=REFERENCE)
    execute.add_argument('--binary', type=Path, default=ROOT/'build/qwen/qwen_q4_check')
    execute.add_argument('--state', type=Path, default=arrivals.STATE)
    execute.add_argument('--control', type=Path, default=arrivals.CONTROL)
    audit = sub.add_parser('verify'); audit.add_argument('directory', type=Path)
    audit.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'verify': save(args.output, verify(args.directory))
    else:
        for field in ('binary', 'state', 'control', 'shared_fixtures'): setattr(args, field, getattr(args, field).resolve())
        raise SystemExit(0 if run(args)['complete'] else 2)
