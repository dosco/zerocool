#!/usr/bin/env python3
"""Deadline-limited normal conversation routes; no timing qualification or tuning."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from benchmark_exact import configurations, config_args, fixed_sampling
from cache_simulation import cache_curve
from evidence_index import Index
from qualification_evidence import EvidenceGuard, ResourceBlocked, identity, save, sha, seal
from qualify_exact_sessions import check_configuration
from route_trace import decode, progress, tokens as validate_tokens
from selector_qualification import check_machine

ROOT = Path(__file__).resolve().parents[2]
PROMPT = ('Write a Python function merge_intervals(intervals) that merges overlapping closed intervals. '
          'Return sorted intervals without mutating the input. Handle an empty list and intervals '
          'that touch at an endpoint. Include the implementation and three small assertions testing '
          'empty input, overlapping intervals, and unsorted disjoint intervals. Explain the time complexity.')
APPEND_BODY = ('Now review the function you wrote. Explain whether touching endpoints merge, whether '
    'zero-length intervals are valid, and whether the original input is modified. Add tests for nested '
    'intervals, duplicate intervals, negative endpoints, reversed input order, and multiple chains of '
    'overlaps. If you find a mistake, provide a corrected implementation. Use deterministic examples '
    'and state the expected result for every test. Check empty input and confirm that sorting the result '
    'does not mutate the caller\'s list. Explain the time and space complexity. Consider whether an '
    'invalid interval with its endpoints reversed should be rejected. Keep the examples self-contained.')


def append_tokens(parts):
    """Keep chat framing intact; bound only the user body to 128 input tokens."""
    prefix, body, suffix = (parts[k]['tokens'] for k in ('prefix','body','suffix'))
    for part in (prefix, body, suffix): validate_tokens(part)
    room = 128 - len(prefix) - len(suffix)
    if not 0 < room <= len(body):
        raise ValueError('Append framing or body cannot produce exactly 128 tokens')
    return prefix + body[:room] + suffix


def load(path):
    return json.loads(Path(path).read_text())


def validate_capture(raw, trace, evidence, workload, config, decode_steps=32):
    if (raw.get('complete') is not True or raw.get('workloads') != workload or
        raw.get('model_revision') != evidence['artifact_revision'] or not fixed_sampling(raw.get('sampling')) or
        len(raw.get('runs', [])) != len(workload) or not trace['complete'] or len(trace['requests']) != len(workload)):
        raise ValueError('Missing completed normal request and matching committed trace')
    for key in ('build', 'artifact_revision', 'prepared_manifest_sha256', 'budget_bytes', 'device', 'physical_bytes'):
        if trace['identity'].get(key) != evidence[key]:
            raise ValueError('Route identity differs: ' + key)
    if len(workload) not in (1,2) or workload[0].get('append') or workload[0]['max_tokens'] != decode_steps+1:
        raise ValueError('Invalid capture workload')
    history, results = [], []
    initial_plan = raw['runs'][0]['before']['memory_plan']
    session = None
    for i, (row, task, request) in enumerate(zip(raw['runs'], workload, trace['requests'])):
        target = decode_steps if i==0 else 32
        if i and (task.get('append') is not True or len(task['tokens']) != 128 or task['max_tokens'] != 33):
            raise ValueError('Follow-up requires 128 new tokens and 32 decode steps')
        prompt = history + task['tokens'] if i else task['tokens']
        reused = len(history)-1 if i else 0  # Last sampled output has not yet been forwarded.
        if (row.get('profiling_enabled') is not True or row.get('runtime_cache_state') != ('retained' if i else 'empty_at_process_start') or
            row.get('repetition') != 0 or row.get('name') != task['name'] or row.get('prompt_tokens') != len(prompt) or
            row.get('reused_tokens') != reused or (i and row.get('pending_tokens_ingested') != 1)):
            raise ValueError('Capture must reuse the exact live generated history')
        for boundary in ('before','after'):
            state = row[boundary]
            check_machine(state,evidence);check_configuration(state,config)
            if (state['diagnostic_stream_trunk'] or state['memory_plan'] != initial_plan or
                state['memory_plan']['panel_tokens'] != config['panel'] or
                state['phase_memory']['pressure_resizes'] != raw['runs'][0]['before']['phase_memory']['pressure_resizes']):
                raise ValueError('Diagnostic, changed memory plan or pressure resize')
        if (request['name'] != task['name'] or request['input_token_ids'] != prompt or request['max_tokens'] != task['max_tokens'] or request['prime'] or
            request['result']['output_token_ids'] != row['output_token_ids'] or
            request['result']['finish_reason'] != row['finish_reason'] or request['result']['reused_tokens'] != reused):
            raise ValueError('Trace tokens or finish reason differ from native report')
        forwards = [f for f in trace['committed'] if f['request_id'] == request['request_id']]
        if not forwards: raise ValueError('Missing committed request forwards')
        if session is None: session = forwards[0]['session_id']
        if any(f['session_id'] != session for f in forwards): raise ValueError('Follow-up rebuilt session state')
        generation = [f for f in forwards if f['request_phase'] == 'decode']
        ingestion = [f for f in forwards if f['request_phase'] == ('append' if i else 'prefill')]
        if (len(generation)+len(ingestion) != len(forwards) or sum(f['tokens'] for f in ingestion) != len(prompt)-reused or
            any(f['tokens'] != 1 for f in generation) or len(generation) != row['output_tokens']-1 or
            row['output_tokens'] != len(row['output_token_ids'])):
            raise ValueError('Generated outputs do not match committed forwards')
        results.append(dict(name=task['name'],request_id=request['request_id'],generated_tokens=row['output_tokens'],
            committed_decode_steps=len(generation),target_decode_steps=target,target_steps_met=len(generation)==target,
            finish_reason=row['finish_reason'],prompt_tokens=row['prompt_tokens'],reused_tokens=reused,
            new_input_tokens=len(task['tokens']),pending_tokens_ingested=row.get('pending_tokens_ingested',0)))
        history = prompt + row['output_token_ids']
    return dict(requests=results,target_steps_met=all(r['target_steps_met'] for r in results),trace_complete=True,
                target_32_steps_met=results[0]['committed_decode_steps']==32,
                committed_decode_steps=results[0]['committed_decode_steps'],generated_tokens=results[0]['generated_tokens'])


def run(args):
    if not 0 < args.time_limit <= 300:
        raise ValueError('Route capture deadline must be 1..300 seconds')
    steps = getattr(args,'decode_steps',32)
    append = getattr(args,'append_128',False)
    if type(steps) is not int or steps not in (32,256): raise ValueError('Decode steps must be 32 or 256')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(kind='normal_route_capture_v1', status='running', complete=False,
                  phase='preparation', time_limit_seconds=args.time_limit,
                  started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  target_decode_steps=steps, max_output_tokens=steps+1, append_128=append, normal_request_latency_qualified=False,
                  production_promoted=False)
    save(output/'summary.json', report)
    def remaining():
        value = args.time_limit - (time.monotonic() - started)
        if value <= 0:
            raise subprocess.TimeoutExpired('route-capture', args.time_limit)
        return value
    env = dict(os.environ)
    for key in ('MTL_DEBUG_LAYER', 'MTL_SHADER_VALIDATION'):
        env.pop(key, None)
    try:
        binary = ROOT/'build/qwen/bin/freellm'
        model, prepared = ROOT/'.cache/qwen-mixed-reference', ROOT/'.cache/prepared/q4-records-v1'
        config = configurations()[0]
        save(output/'chat.json', dict(messages=[dict(role='user', content=PROMPT)], enable_thinking=False))
        with (output/'render.log').open('w') as log:
            subprocess.run([str(binary), 'inspect', '--model', str(model), '--render-chat', str(output/'chat.json'),
                            '--json', str(output/'rendered.json')], check=True, stdout=log,
                           stderr=subprocess.STDOUT, timeout=min(30, remaining()), env=env)
        workload = [dict(name=f'coding_routes_{steps}', tokens=load(output/'rendered.json')['tokens'], max_tokens=steps+1)]
        prepared_files = ['chat.json','rendered.json']
        if append:
            pieces = dict(prefix='<|im_end|>\n<|im_start|>user\n', body=APPEND_BODY,
                          suffix='\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
            parts = {}
            for name,text in pieces.items():
                destination = output/f'append-{name}.json'
                with (output/f'append-{name}.log').open('w') as log:
                    subprocess.run([str(binary),'inspect','--model',str(model),'--tokenize',text,'--json',str(destination)],
                        check=True,stdout=log,stderr=subprocess.STDOUT,timeout=min(30,remaining()),env=env)
                parts[name] = load(destination);prepared_files.append(destination.name)
            followup = append_tokens(parts)
            workload.append(dict(name='append_128',tokens=followup,append=True,max_tokens=33))
            save(output/'append-recipe.json',dict(parts=pieces,tokens=followup,
                policy='Keep prefix/suffix framing; take a bounded user-body token prefix to total 128 tokens. '
                       'The fixed closing delimiter is retained even if the first request reaches EOS.'))
            prepared_files.append('append-recipe.json')
        save(output/'workload.json', workload)
        evidence = identity(ROOT, [config], model, prepared, output/'workload.json')
        evidence['files'].update({str((output/name).resolve()):sha(output/name) for name in prepared_files})
        save(output/'identity.json', evidence)
        report.update(build=evidence['build'], artifact_revision=evidence['artifact_revision'],
                      budget_bytes=evidence['budget_bytes'], configuration=config, prompt_tokens=len(workload[0]['tokens']))
        guard = EvidenceGuard(evidence, output)
        common = ['--model', model, '--artifact', 'mixed-4_8bit', '--prepared', prepared,
                  '--memory-gb', '12', '--context', '8192', *config_args(config)]
        with (ROOT/'.cache/qwen-qualification.lock').open('a') as lock:
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: raise ResourceBlocked('Another workload owns the GPU lease')
            guard.check_resources(initial=True)
            report['phase'] = 'admission'; save(output/'summary.json', report)
            with (output/'launch.log').open('w') as log:
                guard.run([binary, 'inspect', *common, '--json', output/'admission.json'],
                          stdout=log, timeout=remaining(), env=env)
                admitted = load(output/'admission.json')
                if (admitted['current_admission'].get('limit_bytes') != evidence['budget_bytes'] or
                    admitted['current_admission'].get('panel_tokens') != config['panel']):
                    raise ResourceBlocked('The fixed 12GiB route capture budget is not currently admitted')
                for key, field in [('device', 'device'), ('physical_bytes', 'physical_bytes'), ('build', 'build_fingerprint')]:
                    if admitted['machine'][field] != evidence[key]:
                        raise ValueError('Admission identity differs: ' + key)
                report['phase'] = 'normal_inference'; save(output/'summary.json', report)
                guard.run([binary, 'bench', *common, '--workload-file', output/'workload.json', '--repetitions', '1',
                           '--temperature', '0', '--seed', '0', '--route-trace', output/'routes.jsonl',
                           '--json', output/'native.json'], stdout=log, timeout=remaining(), env=env)
        trace = decode((output/'routes.jsonl').read_bytes())
        result = validate_capture(load(output/'native.json'), trace, evidence, workload, config,steps)
        report.update(complete=True, phase='finished', status='captured' if result['target_steps_met'] else 'early_stop', result=result)
    except ResourceBlocked as error:
        report.update(status='resource_blocked', error=str(error))
    except subprocess.TimeoutExpired:
        report.update(status='time_budget_exhausted', error='Stopped at total deadline; partial routes do not establish the target.')
    except KeyboardInterrupt:
        report.update(status='interrupted', error='User interrupted the capture.')
    except Exception as error:
        report.update(status='failed', error=str(error))
    finally:
        if (output/'routes.jsonl').exists():
            try:
                report['route_progress'] = progress(output/'routes.jsonl')
                index = Index(output/'analysis.sqlite')
                try:
                    imported = index.import_paths([output/'routes.jsonl'])
                    if imported['issues']: raise ValueError(str(imported['issues']))
                    save(output/'cache-curve.json', cache_curve(index, str(output/'routes.jsonl')))
                finally: index.close()
            except Exception as error:
                report['trace_analysis_error'] = str(error)
        report.update(elapsed_seconds=time.monotonic()-started,
                      finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(output/'summary.json', report)
        report['evidence_seal'] = seal(output)
        # The seal includes summary.json; do not rewrite it after sealing.
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report['complete'] else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-limit', type=float, default=180)
    parser.add_argument('--decode-steps', type=int, choices=[32,256], default=32)
    parser.add_argument('--append-128', action='store_true')
    raise SystemExit(run(parser.parse_args()))
