#!/usr/bin/env python3
"""Five alternating paired normal-request comparisons at equal admitted budgets.

The explicit panel path must first pass real numerical/state checks. This tool
fails on changed output tokens, artifacts, native builds, or admitted budgets.
Diagnostic trunk streaming is never a qualifying schedule.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
from build_identity import build_fingerprint


def validate(report, expected, panel, budget):
    if report.get('complete') is not True or not report.get('runs'):
        raise ValueError('Normal request did not complete')
    if report.get('model_revision') != expected['revision']:
        raise ValueError('Benchmark artifact differs from the requested artifact')
    workload = json.dumps(report['workloads'], sort_keys=True, separators=(',', ':'))
    if expected.setdefault('workload', workload) != workload:
        raise ValueError('Workload token IDs or requested output limits changed')
    if len({r['name'] for r in report['runs']}) != len(report['runs']):
        raise ValueError('Each paired request must have a unique workload name')
    observations = []
    for row in report['runs']:
        state = row['after']
        if state.get('artifact_revision') != expected['revision']:
            raise ValueError('Runtime benchmark artifact differs')
        plan = state['memory_plan']
        machine = state['metal']
        if state['diagnostic_stream_trunk'] or machine['device'] != 'Apple M1 Pro' or machine['physical_bytes'] != 32*1024**3:
            raise ValueError('Requires normal execution on the actual 32GiB M1 Pro')
        if plan['limit_bytes'] != budget or plan['planned_bytes'] > budget:
            raise ValueError('Live admission changed the comparison budget; choose a smaller common budget')
        if plan['expert_slots'] != row['before']['memory_plan']['expert_slots']:
            raise ValueError('Memory pressure resized the expert cache during the request')
        if state['process']['physical_footprint_bytes'] > budget:
            raise ValueError('Observed process footprint exceeded the comparison budget')
        if plan['panel_tokens'] != panel:
            raise ValueError('Requested panel was reduced; compare the admitted configuration explicitly')
        if machine['build_fingerprint'] != expected['build']:
            raise ValueError('Native build changed within the comparison')
        key = row['name']
        signature = (report['model_revision'], state['prepared']['manifest_sha256'],
                     row['prompt_tokens'], row['reused_tokens'], tuple(row['output_token_ids']))
        if expected.setdefault(key, signature) != signature:
            raise ValueError(f'Artifact, workload, reuse, or generated tokens changed for {key}')
        if row['output_tokens'] != 256:
            raise ValueError(f'{key} produced fewer than the required 256 tokens')
        before = row['before']
        def reads(s):
            return s['checkpoint_application_read_bytes'] + s['prepared']['application_read_bytes']
        observations.append(dict(name=key, panel=panel, ttft_ms=row['time_to_first_token_ms'],
                                 tokens_per_second=row['tokens_per_second'],
                                 application_read_bytes=reads(state)-reads(before), expert_slots=plan['expert_slots'],
                                 peak_metal_bytes=machine['peak_buffer_bytes'],
                                 final_process_footprint_bytes=state['process']['physical_footprint_bytes']))
    return observations


def run(args):
    root = Path(__file__).resolve().parents[2]
    if args.pairs < 5:
        raise ValueError('At least five alternating pairs are required')
    if any(os.environ.get(k) not in (None, '', '0') for k in ('MTL_DEBUG_LAYER', 'MTL_SHADER_VALIDATION')):
        raise ValueError('Disable Metal validation for request timing; run correctness checks separately')
    args.output.mkdir(parents=True, exist_ok=False)
    budget = int(args.memory_gb * 1024**3)
    lock=json.loads((root/('models.lock.json' if args.artifact=='q4-control' else 'mixed-models.lock.json')).read_text())
    expected = dict(build=build_fingerprint(root), revision=lock['revision'])
    measurements = []
    identity = dict(kind='normal_request_panel_comparison', build_fingerprint=expected['build'],
                    artifact=args.artifact, artifact_revision=expected['revision'],
                    workload_sha256=hashlib.sha256(args.workload.read_bytes()).hexdigest(),
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    pairs=args.pairs, budget_bytes=budget)
    try:
        for pair in range(args.pairs):
            for panel in ((0, 256, 512, 1024) if pair % 2 == 0 else (1024, 512, 256, 0)):
                output = args.output/f'panel-{panel}-pair-{pair}.json'
                log = output.with_suffix('.log')
                cmd = [str(args.binary), 'bench', '--artifact', args.artifact, '--model', str(args.model), '--prepared', str(args.prepared),
                       '--context', '8192', '--chunk', '128', '--panel', str(panel), '--memory-gb', str(args.memory_gb),
                       '--workload-file', str(args.workload), '--repetitions', '1', '--temperature', '0',
                       '--seed', '0', '--json', str(output)]
                with log.open('w') as stream:
                    admission_file = output.with_suffix('.admission.json')
                    inspect = [str(args.binary), 'inspect', '--artifact', args.artifact, '--model', str(args.model), '--prepared', str(args.prepared),
                               '--context', '8192', '--chunk', '128', '--panel', str(panel), '--memory-gb', str(args.memory_gb),
                               '--json', str(admission_file)]
                    subprocess.run(inspect, check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=60)
                    admission = json.loads(admission_file.read_text())['current_admission']
                    if admission.get('limit_bytes') != budget or admission.get('panel_tokens') != panel:
                        raise ValueError(f'Comparison cannot admit the requested memory/panel: {admission}')
                    subprocess.run(cmd, check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout)
                report = json.loads(output.read_text())
                for row in validate(report, expected, panel, budget):
                    row.update(pair=pair, report=output.name, sha256=hashlib.sha256(output.read_bytes()).hexdigest())
                    measurements.append(row)
                    print(f'{row["name"]} panel={panel} pair={pair}: first token {row["ttft_ms"]/1000:.2f}s, '
                          f'{row["tokens_per_second"]:.3f} tokens/s', flush=True)
                (args.output/'progress.json').write_text(json.dumps(dict(identity, complete=False, measurements=measurements), indent=2)+'\n')
    except Exception as error:
        (args.output/'summary.json').write_text(json.dumps(dict(identity, complete=False, error=str(error), measurements=measurements), indent=2)+'\n')
        raise
    summary = []
    for name in sorted({r['name'] for r in measurements}):
        for panel in (0, 256, 512, 1024):
            rows = [r for r in measurements if r['name'] == name and r['panel'] == panel]
            summary.append(dict(name=name, panel=panel, repetitions=len(rows),
                                median_ttft_ms=statistics.median(r['ttft_ms'] for r in rows),
                                median_tokens_per_second=statistics.median(r['tokens_per_second'] for r in rows),
                                median_application_read_bytes=statistics.median(r['application_read_bytes'] for r in rows)))
    (args.output/'summary.json').write_text(json.dumps(dict(identity, complete=True, generated_tokens_equal=True,
        summary=summary, measurements=measurements, full_acceptance_qualified=False), indent=2)+'\n')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=root/'build/qwen/bin/zerocool')
    parser.add_argument('--model', type=Path, default=root/'.cache/models/qwen38-flash-next')
    parser.add_argument('--artifact', choices=('q4-control','mixed-4_8bit'), default='q4-control')
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--memory-gb', type=float, required=True)
    parser.add_argument('--pairs', type=int, default=5)
    parser.add_argument('--timeout', type=int, default=3600)
    run(parser.parse_args())
