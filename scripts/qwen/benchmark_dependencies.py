#!/usr/bin/env python3
"""Alternate paired layout/schedule replays; never qualifies request latency."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


def measurements(report, expected):
    """Keep every recorded pass and reject changed routes/arithmetic/builds."""
    runs = report['runs']
    if not runs:
        raise ValueError('Replay returned no recorded passes')
    build = report['machine']['build_fingerprint']
    identity = (build, report['artifact_revision'], len(runs))
    if expected.setdefault('identity', identity) != identity:
        raise ValueError('Build, artifact, or recorded pass count changed within paired comparison')
    result = []
    for index, measured in enumerate(runs):
        layers = measured['layers']
        if measured['recorded_pass'] != index or measured['repetition'] != 0 or len(layers) != 48:
            raise ValueError('Incomplete or reordered replay pass')
        for layer, row in enumerate(layers):
            signature = (tuple(row['routes']), row['output_sha256'])
            if row['layer'] != layer or len(row['output_sha256']) != 64:
                raise ValueError('Invalid replay layer output identity')
            key = (index, layer)
            if expected.setdefault(key, signature) != signature:
                raise ValueError(f'Expert output changed at recorded pass {index}, layer {layer}')
        result.append(dict(recorded_pass=index, nanoseconds=measured['expert_dependency_ns'],
                           application_bytes=sum(l['application_read_bytes'] for l in layers), build=build))
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [('source-batched', False, False, 4), ('source-completion', False, True, 4),
             ('prepared-batched', True, False, 4), ('prepared-completion', True, True, 4)]
    cases += [(f'prepared-group-{n}', True, True, n) for n in (1, 2, 8)]
    results, expected = [], {}
    for hits in (0, 5, 10):
        for rep in range(args.pairs):
            for name, prepared, completion, group in (cases if rep % 2 == 0 else cases[::-1]):
                dest = args.output / f'{name}-hits{hits}-rep{rep}.json'
                cmd = [str(args.binary), 'bench', '--model', str(args.model),
                       '--replay-routes', str(args.routes), '--replay-hits', str(hits),
                       '--repetitions', '1', '--ready-group', str(group), '--json', str(dest)]
                if prepared:
                    cmd += ['--prepared', str(args.prepared)]
                if not completion:
                    cmd += ['--legacy-schedule']
                subprocess.run(cmd, check=True, timeout=180)
                report = json.loads(dest.read_text())
                for measured in measurements(report, expected):
                    results.append(dict(case=name, hits=hits, repetition=rep, **measured, file=dest.name,
                                        sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
                    print(f'{name} hits={hits} rep={rep} pass={measured["recorded_pass"]}: '
                          f'{measured["nanoseconds"]/1e6:.2f} ms', flush=True)
                (args.output / 'progress.json').write_text(json.dumps(results, indent=2) + '\n')
    summary = []
    for hits in (0, 5, 10):
        for name, *_ in cases:
            for recorded_pass in range(expected['identity'][2]):
                selected = [r for r in results if r['case'] == name and r['hits'] == hits
                            and r['recorded_pass'] == recorded_pass]
                summary.append(dict(case=name, seeded_hits_per_layer=hits, recorded_pass=recorded_pass,
                                    median_ms=statistics.median(r['nanoseconds'] for r in selected)/1e6,
                                    min_ms=min(r['nanoseconds'] for r in selected)/1e6,
                                    max_ms=max(r['nanoseconds'] for r in selected)/1e6,
                                    application_bytes=selected[0]['application_bytes']))
    output = dict(kind='paired_dependency_replay', complete=True, pairs=args.pairs,
                  normal_request_latency_qualified=False,
                  expert_output_hashes_equal=True,
                  routes_sha256=hashlib.sha256(args.routes.read_bytes()).hexdigest(),
                  summary=summary, measurements=results,
                  limitations='Recorded routes from the five-token fixture, with explicitly seeded 0/5/10 ready hits per layer. Fixed BF16 expert input. Excludes attention, routing, shared work, ngrams, session state, and all cache warmup reads from timing. Device traffic includes other processes.')
    (args.output / 'summary.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary', type=Path, default=root / 'build/qwen/bin/zerocool')
    ap.add_argument('--model', type=Path, default=root / '.cache/models/qwen38-flash-next')
    ap.add_argument('--prepared', type=Path, required=True)
    ap.add_argument('--routes', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--pairs', type=int, default=5)
    args = ap.parse_args()
    if args.pairs < 5:
        ap.error('At least five paired repetitions are required')
    run(args)
