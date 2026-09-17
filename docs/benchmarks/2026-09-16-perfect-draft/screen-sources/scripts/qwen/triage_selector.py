#!/usr/bin/env python3
"""Bounded cached-selector triage using completed, unchanged-build quick checks."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from qualification_evidence import EvidenceGuard, ResourceBlocked, identity, save, sha, verify_seal
from selector_qualification import CONFIG, ROOT, WORKLOAD, check_cached, configurations
from cached_progress import summarize as summarize_progress


def decide(result):
    """A triage gate only. Five fixed pairs; no additional samples on ambiguity."""
    bounds = result['latency_ratio']
    if (result.get('exact') is not True or result.get('pairs') != 5 or
        result.get('application_read_bytes') != 0 or result.get('prompt_tokens') != 4096 or
        result.get('comparison_axis') != 'sparse_selection' or bounds.get('pairs') != 5):
        raise ValueError('Requires five exact, zero-read 4K selector pairs')
    values = [bounds[k] for k in ('low', 'median', 'high')]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('Invalid cached latency bounds')
    if values != sorted(values):
        raise ValueError('Unordered cached latency bounds')
    promising = bounds['median'] <= .99 and bounds['high'] < 1
    status = 'promising' if promising else 'inconclusive' if bounds['low'] < 1 < bounds['high'] else 'not_promising'
    return dict(status=status, advance_to_normal_screen=promising,
                normal_request_latency_qualified=False, full_validation=False,
                production_promoted=False)


def prerequisites(source, output):
    source = Path(source).resolve()
    previous = json.loads((source/'identity.json').read_text())
    # Preserve the old identities. Additional tooling files may exist, but every
    # original native/tooling/asset dependency must remain unchanged.
    EvidenceGuard(previous, output).check_identity()
    current = identity(ROOT, configurations(), ROOT/'.cache/qwen-mixed-reference',
                       ROOT/'.cache/prepared/q4-records-v1', WORKLOAD)
    current['files'][str(CONFIG.resolve())] = sha(CONFIG)
    if ({k: v for k, v in current.items() if k != 'files'} !=
        {k: v for k, v in previous.items() if k != 'files'} or
        any(current['files'].get(k) != v for k, v in previous['files'].items())):
        raise ValueError('Quick-check inputs, configurations or build changed')
    report = json.loads((source/'summary.json').read_text())
    selected = {}
    for name in ('recovery', 'capture'):
        phase = report['phases'][name]
        if phase['status'] != 'passed':
            raise ValueError('Missing completed prerequisite: ' + name)
        verify_seal(source/name, phase['files_sha256'])
        selected[name] = phase
    save(output/'prerequisites.json', dict(source=str(source), identity_sha256=sha(source/'identity.json'),
        phases=selected, note='Reused original evidence for unchanged dependencies; no identities relabeled.'))
    save(output/'identity.json', current)
    return current


def run(args):
    if not 0 < args.time_limit <= 600:
        raise ValueError('Cached triage must have a positive deadline of at most 600 seconds')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(kind='selector_cached_triage', status='running', complete=False,
        time_limit_seconds=args.time_limit, started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        criterion=dict(pairs=5, context=4096, median_ratio_max=.99, upper_95_ratio_below=1),
        advance_to_normal_screen=False, normal_request_latency_qualified=False,
        full_validation=False, production_promoted=False)
    save(output/'summary.json', report)
    try:
        evidence = prerequisites(args.prerequisites, output)
        guard = EvidenceGuard(evidence, output)
        guard.check_resources(initial=True)
        command = [sys.executable, ROOT/'scripts/qwen/benchmark_sparse_attention.py',
            '--track', 'selector', '--phase', 'cached', '--contexts', '4096',
            '--output', output/'cached-run']
        report['command'] = [str(x) for x in command]
        report['build'] = evidence['build']
        save(output/'summary.json', report)
        remaining = args.time_limit - (time.monotonic() - started)
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, args.time_limit)
        env = dict(os.environ)
        for key in ('MTL_DEBUG_LAYER', 'MTL_SHADER_VALIDATION'):
            env.pop(key, None)
        print(f'Cached 4K selector triage: at most {args.time_limit}s, including setup.', flush=True)
        with (output/'launch.log').open('w') as log:
            guard.run(command, stdout=log, timeout=remaining, env=env)
        saved = json.loads((output/'cached-run/summary.json').read_text())['phases']['cached-4096']
        if saved['status'] != 'passed':
            raise ValueError('Cached comparison did not complete')
        directory = output/'cached-run/cached-4096'
        verify_seal(directory, saved['files_sha256'])
        result = check_cached(json.loads((directory/'report.json').read_text()), evidence,
                              json.loads((directory/'tokens.json').read_text()))
        if result != saved['result']:
            raise ValueError('Cached summary differs from original measurements')
        report.update(decide(result), complete=True, cached_result=result)
    except ResourceBlocked as error:
        report.update(status='resource_blocked', error=str(error))
    except subprocess.TimeoutExpired:
        report.update(status='time_budget_exhausted', error='Stopped at the triage deadline; no performance decision.')
    except KeyboardInterrupt:
        report.update(status='interrupted', error='Stopped by the user; no performance decision.')
    except Exception as error:
        report.update(status='failed', error=str(error))
    finally:
        try:
            report['phase_progress']=summarize_progress(output/'cached-run/cached-4096/progress.jsonl',report.get('build'))
        except (ValueError,OSError) as error:
            report['phase_progress']=dict(available=False,error=str(error),limitation='Invalid progress cannot explain the timeout.')
        report['elapsed_seconds'] = time.monotonic() - started
        report['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(output/'summary.json', report)
        action = ('Run a bounded normal-request screen before considering full validation.'
                  if report['advance_to_normal_screen'] else
                  'Do not advance this attempt to full validation.')
        (output/'README.md').write_text('# Cached selector triage\n\n'
            f"Status: **{report['status']}**. Complete: **{report['complete']}**.\n\n"
            f'{action}\n\nNo normal-request or production qualification is claimed.\n')
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report['complete'] else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prerequisites', type=Path, required=True,
                        help='Original sealed recovery/capture evidence for unchanged dependencies')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-limit', type=int, default=600)
    sys.exit(run(parser.parse_args()))
