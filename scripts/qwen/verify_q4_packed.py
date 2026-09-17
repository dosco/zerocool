#!/usr/bin/env python3
"""Reconstruct the sealed packed-Q4 operator and locality screens."""
import argparse
from pathlib import Path

from cache_residency import require
from capture_routes import load
from qualification_evidence import save, sha, verify_seal
from screen_q4_packed import analyze
from verify_stage200 import verify_sources


def verify(directory):
    attempts = []
    for path in sorted(directory.iterdir()):
        if not (path/'evidence-files.json').is_file(): continue
        seal = sha(path/'evidence-files.json'); verify_seal(path, seal)
        summary, frozen = load(path/'summary.json'), load(path/'identity.json')
        source = dict(frozen, files={p: h for p, h in frozen['files'].items() if Path(p).is_relative_to(frozen['root'])})
        provenance = verify_sources(directory, source)
        require(summary['kind'] == 'q4_packed_operator_screen_v1' and summary['identity'] ==
                {k: frozen[k] for k in summary['identity']}, 'Wrong screen identity')
        row = dict(attempt=path.name, source_seal_sha256=seal, source_provenance=provenance,
                   recorded_status=summary['status'], recorded_complete=summary['complete'])
        if summary['complete'] is not True:
            require(summary['status'] in ('failed', 'resource_blocked', 'interrupted', 'time_budget_exhausted'),
                    'Unfinished screen lacks disposition')
            row['recomputed_status'] = summary['status']; attempts.append(row); continue
        validation, timing = (load(path/(name+'.json')) for name in ('validate', 'timing'))
        require(validation.get('complete') is True and validation.get('validation') is True and
                validation.get('exact') is True and validation.get('expert_input_cases') == 64 and
                [(p['pair'], p['group']) for p in validation['pairs']] == [(0, 1), (0, 4)],
                'Incomplete Metal validation')
        for key in ('reference_shader_sha256', 'candidate_shader_sha256', 'fixture_manifest', 'output_modes'):
            require(validation[key] == timing[key], 'Validation and timing differ')
        shader = str(Path(frozen['root'])/'scripts/qwen/probe_q4_packed.metal')
        require(timing['candidate_shader_sha256'] == frozen['files'][shader], 'Wrong candidate shader')
        records = [dict(layer=int(layer), expert=e['expert'], sha256=e['record']['sha256'])
                   for layer, item in timing['fixture_manifest']['layers'].items() for e in item['experts']]
        require(sorted(records, key=lambda e: (e['layer'], e['expert'])) ==
                sorted(summary['verified_records'], key=lambda e: (e['layer'], e['expert'])), 'Changed source-record proof')
        result = analyze(timing)
        require(all(summary[k] == v for k, v in result.items()), 'Changed screening decision')
        stream = timing.get('stream_bytes_before_each_group', 0)
        require(validation.get('stream_bytes_before_each_group', 0) == stream ==
                summary.get('stream_bytes_before_each_group', 0), 'Changed locality intervention')
        row.update(recomputed_status=result['status'], stream_bytes_before_each_group=stream,
                   results=result['results']); attempts.append(row)
    require(bool(attempts), 'No Q4 operator evidence')
    return dict(kind='q4_packed_operator_audit_v1', complete=True, audit_passed=True, attempts=attempts,
        normal_request_latency_qualified=False, production_promoted=False,
        limitations=['Reconstructs source-bound raw measurements; no new GPU execution or source-record rehash.',
            'The operator gate remains failed in both runs; results are neither pooled nor promoted.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); save(args.output, verify(args.directory))
