#!/usr/bin/env python3
"""Reconstruct resident-operator evidence without running the model or GPU."""
import argparse
from pathlib import Path

from cache_residency import configs, memory_observation, require
from capture_routes import load
from profile_resident_operators import split_operations
from qualification_evidence import save, sha, verify_seal
from screen_q8_lookahead import analyze
from screen_cache import validate_request
from verify_stage200 import verify_sources


def verify(directory, source_snapshot=None):
    results = []
    for attempt in sorted(directory.iterdir()):
        if not (attempt/'evidence-files.json').is_file():
            continue
        seal = sha(attempt/'evidence-files.json')
        verify_seal(attempt, seal)
        summary, identity = load(attempt/'summary.json'), load(attempt/'identity.json')
        require(summary['identity'] == {k: identity[k] for k in summary['identity']}, 'Changed experiment identity')
        source_identity = dict(identity, files={p: h for p, h in identity['files'].items()
                                               if Path(p).is_relative_to(identity['root'])})
        row = dict(attempt=attempt.name, source_seal_sha256=seal,
            recorded_status=summary['status'], recorded_complete=summary['complete'],
            source_provenance=verify_sources(directory, source_identity, source_snapshot),
            external_binary_hashes={p: h for p, h in identity['files'].items()
                                    if p not in source_identity['files']})
        if summary['complete'] is not True:
            require(summary['status'] in ('failed', 'resource_blocked', 'interrupted', 'time_budget_exhausted'),
                    'Incomplete attempt lacks terminal disposition')
            row.update(recomputed_status=summary['status'], result_qualified=False)
            results.append(row)
            continue
        if summary['kind'] == 'q8_lookahead_screen_v1':
            validation, timing = (load(attempt/(name+'.json')) for name in ('validate', 'timing'))
            require(validation.get('complete') is True and validation.get('validation') is True and
                    validation['cases'] == timing['cases'] and validation['shader_sha256'] == timing['shader_sha256'],
                    'Missing or different validation coverage')
            require([(p['pair'], p['case']) for p in validation['pairs']] == [(0, i) for i in range(7)],
                    'Missing validation execution')
            weight_hashes = [(c['origin']['case']['tensors'][k]['bytes'], c['origin']['case']['tensors'][k]['sha256'])
                             for c in timing['cases'] for k in ('w', 's', 'b')]
            require(weight_hashes == [(e['bytes'], e['sha256']) for e in summary['verified_weight_tensors']],
                    'Captured weights differ from verified source ranges')
            decision = analyze(timing)
            require(all(summary[k] == v for k, v in decision.items()), 'Changed operator decision')
            row.update(recomputed_status=decision['status'], result_qualified=False,
                       median_shape_frequency_projection_ms=decision['median_shape_frequency_projection_ms'])
        elif summary['kind'] == 'resident_operator_capture_v1':
            require(summary['configurations'] == [configs()[0]], 'Changed capture configuration')
            expected = {}
            reports = []
            for name in ('normal', 'traced'):
                record = summary[name]
                require(record['source'] == name+'.json' and sha(attempt/record['source']) == record['sha256'],
                        'Changed request source')
                raw = load(attempt/record['source'])
                validate_request(raw, identity, configs()[0], summary['workload'], expected,
                                 instrumented=name == 'traced', output_tokens=5)
                memory = memory_observation(raw)
                require(memory['memory_screen_passed'] and all(record[k] == v for k, v in memory.items()),
                        'Changed memory observations')
                reports.append(raw)
            ratios = [b['decode_wall_ms']/a['decode_wall_ms'] for a, b in zip(reports[0]['runs'], reports[1]['runs'])]
            require(summary['trace_to_normal_decode_ratios'] == ratios, 'Changed instrumentation ratios')
            operations = split_operations(load(attempt/'traced.json'), load(attempt/'dispatch.json'))
            require(operations == summary['operations'], 'Changed operator attribution')
            row.update(recomputed_status='captured', result_qualified=False, requests_verified=True)
        else:
            raise ValueError('Unknown resident experiment kind')
        results.append(row)
    require(bool(results), 'No resident-operator evidence')
    return dict(kind='resident_operator_audit_v1', complete=True, audit_passed=True, attempts=results,
        normal_request_latency_qualified=False, production_promoted=False,
        limitations=['Reconstructs sealed raw evidence; does not rerun kernels, request/state checks, or rehash checkpoint payloads.',
                    'Archived fixture manifests preserve their old producer builds. Failed attempts stay incomplete.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--source-snapshot', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    save(args.output, verify(args.directory, args.source_snapshot))
