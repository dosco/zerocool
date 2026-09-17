"""Conservative queries over original reports, with explicit evidence gaps."""
import math
import statistics

from benchmark_exact import validate, paired_interval
from evidence_index import MAX_LINES, encoded, parse, terminal_result
from summarize_cached_comparison import summarize as cached_comparison


def result(sources, limitations, **values):
    return dict(**values, sources=sources, limitations=limitations, production_promoted=False)


def positive(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Missing or invalid positive measurement')
    return value


def compare(index, selector, control, candidate, changes, case=None):
    data, source = index.json(selector)
    sources = [source]
    limits = ['Comparisons use one recorded experiment; unrelated runs are never assigned synthetic pairs.',
              'Numerical correctness and generated-token agreement do not establish coding quality.']
    try:
        if not isinstance(data, dict): raise ValueError('Select a report object, not a workload array')
        if not terminal_result(data): raise ValueError('Report is unfinished, failed or lacks a completed result')
        if data.get('kind') in ('mtp_direct_output_screen_v1', 'mtp_target_recovery_screen_v1', 'mtp_width_screen_v1'):
            from mtp_evidence import compare as mtp_compare
            return mtp_compare(index, selector, control, candidate, changes, case)
        if data.get('kind') in ('expert_tail_screen_v1','decode_scratch_screen_v1','decode_scratch_confirmation_v1'):
            confirmation=data['kind']=='decode_scratch_confirmation_v1'
            scratch=data['kind']!='expert_tail_screen_v1'
            if confirmation:
                from confirm_decode_scratch import revalidate
            elif scratch:
                from screen_decode_scratch import revalidate
            else:
                from screen_expert_tail import revalidate
            axis='decode_scratch' if scratch else 'expert_tail'
            if case is not None or (control,candidate)!=('control','candidate') or set(changes)!={axis}:
                raise ValueError('Comparison requires control/candidate and only '+axis)
            def resolve(digest):
                raw,original=index.json(digest);sources.append(original);return raw
            decision=revalidate(data,resolve)
            return result(sources,limits+['Five fresh pairs use paired log-ratio Student-t bounds; temporal effects can violate their assumptions.' if confirmation else
                'Two short pairs have no confidence bounds; GPU-speed variation is not normalized away.',
                'A failed screen does not run the later full-model state qualification.',
                'This experiment does not qualify 2K/4K context or the 5 tokens/s target.'],
                comparable=True,status='measured',scope=axis+('_confirmation' if confirmation else '_conversation'),comparison=decision,
                controlled_change={axis:['none','reuse'] if scratch else ['wait','overlap']},normal_request_latency_qualified=False)
        if data.get('kind') in ('route_selection_screen_v1','route_selection_confirmation_v1'):
            confirmation=data['kind']=='route_selection_confirmation_v1'
            if confirmation:
                from confirm_route_selection import revalidate
            else:
                from screen_route_selection import revalidate
            if case is not None or (control,candidate)!=('control','candidate') or set(changes)!={'route_selection'}:
                raise ValueError('Router comparison requires control/candidate and only route_selection')
            def resolve(digest):
                raw,original=index.json(digest);sources.append(original);return raw
            decision=revalidate(data,resolve)
            return result(sources,limits+['Short workload comparisons cannot promote production or qualify the original target.',
                'Both timing arms use experimental packed Q8; its earlier first-token guard remains inconclusive.'],
                comparable=True,status='measured',scope='router_selection_confirmation' if confirmation else 'router_selection_conversation',comparison=decision,
                controlled_change={'route_selection':['serial','simd']},normal_request_latency_qualified=False)
        if data.get('kind') == 'q8_memory_budget_screen_v1':
            from screen_memory_budget import revalidate
            if case is not None or (control,candidate)!=('control','candidate') or set(changes)!={'memory_gb','expert_slots'}:
                raise ValueError('Budget comparison requires control/candidate and exactly memory_gb and expert_slots')
            def resolve(digest):
                raw,original=index.json(digest);sources.append(original);return raw
            decision=revalidate(data,resolve)
            return result(sources,limits+['Candidate uses six additional GiB of engine budget for expert cache.',
                'Two short pairs have no confidence bounds and do not qualify sustained or long-context performance.',
                'Both arms explicitly use experimental packed Q8; this does not change its prior inconclusive first-token guard.'],
                comparable=True,status='measured',scope='memory_budget_conversation',comparison=decision,
                controlled_change={'memory_gb':[12,18],'expert_slots':[1848,4175]},normal_request_latency_qualified=False)
        if data.get('kind') in ('q8_steady_short_screen_v1','q8_steady_confirmation_v1'):
            confirmation=data['kind']=='q8_steady_confirmation_v1'
            if confirmation:
                from confirm_q8_steady import revalidate
            else:
                from screen_q8_steady import revalidate
            if case is not None or (control,candidate)!=('control','candidate') or set(changes)!={'kernel_policy','q8_decode_rows'}:
                raise ValueError('Q8 conversation requires control/candidate and the kernel_policy and q8_decode_rows changes')
            def resolve(digest):
                raw, original = index.json(digest); sources.append(original); return raw
            decision=revalidate(data,resolve)
            return result(sources,limits+['Five fresh pairs use a paired log-ratio Student-t interval; temporal effects can violate its assumptions.' if confirmation else
                'Two short conversation pairs are screening evidence without confidence bounds.',
                'Operator timing compares against the previous two-row kernel; request timing compares against the original reference.',
                'Short exact state checks do not qualify 7K context or sustained coding.'],
                comparable=True,status='measured',scope='q8_steady_conversation',comparison=decision,
                controlled_change={'kernel_policy':['reference','candidate'],'q8_decode_rows':[0,2]},
                normal_request_latency_qualified=False)
        if data.get('kind') in ('cache_capacity_screen_v1','cache_capacity_confirmation_v1'):
            from capacity_experiment import revalidate
            if case is not None or (control,candidate)!=('control','candidate') or set(changes)!={'expert_slots'}:
                raise ValueError('Capacity conversation requires control/candidate and exactly --change expert_slots')
            def resolve(digest):
                raw, original = index.json(digest); sources.append(original); return raw
            decision=revalidate(data,resolve)
            return result(sources,limits+['The fixed ceiling is equal; actual expert allocations intentionally differ.',
                'Boundary probes can warm hardware; diagnostic screens cannot qualify ordinary latency.',
                'Short histories do not establish long-context or sustained coding performance.'],
                comparable=True,status='measured',scope='capacity_conversation',comparison=decision,
                controlled_change={'expert_slots':[1848,1460]},gpu_reference_mode=data['gpu_reference_mode'],
                normal_request_latency_qualified=False)
        if data.get('kind') == 'cached_full_token_replay':
            if case is not None: raise ValueError('--case selects normal workloads, not cached context')
            summary = cached_comparison(data)
            axis = summary['comparison_axis']
            if set(changes) != {axis} or (control, candidate) != ('control', 'candidate'):
                raise ValueError('Declare the recorded cached axis and control/candidate arms')
            identity = None
            for row in data['runs']:
                for state in (row['before'], row['after']):
                    current = (state['artifact_revision'], state['prepared']['manifest_sha256'],
                               state['metal']['device'], state['metal']['physical_bytes'],
                               state['metal']['build_fingerprint'], state['memory_plan']['limit_bytes'])
                    if any(v is None for v in current) or (identity is not None and current != identity):
                        raise ValueError('Cached artifact, machine, build or budget differs')
                    identity = current
            if not data.get('input_sha256'):
                raise ValueError('Missing cached input identity')
            return result(sources, limits + ['Zero-read cached timing is not normal-request latency.',
                          'Within this report, native exact-state assertions are recorded evidence, not an independent oracle.'],
                          comparable=True, status='measured', scope='cached', comparison=summary,
                          controlled_change={axis: [summary['control_value'], summary['candidate_value']]},
                          normal_request_latency_qualified=False)
        if data.get('kind') != 'exact_kernel_normal_requests':
            raise ValueError('Select an exact_kernel_normal_requests summary or paired cached raw report')
        if data.get('complete') is not True:
            raise ValueError('Experiment is unfinished; partial pairs cannot answer whether it helped')
        configs = {c['name']: c for c in data['configurations']}
        if len(configs) != len(data['configurations']) or control == candidate:
            raise ValueError('Duplicate configurations or identical arms')
        a, b = configs[control], configs[candidate]
        difference = {k: [a.get(k), b.get(k)] for k in a.keys() | b.keys() if k != 'name' and a.get(k) != b.get(k)}
        if not difference or set(changes) != set(difference):
            raise ValueError('Declared changes must exactly match configuration differences: ' + encoded(difference))
        rows = [r for r in data['measurements'] if r['configuration'] in (control, candidate) and (case is None or r['name'] == case)]
        if not rows: raise ValueError('No measurements for the selected workload')
        cases = sorted({r['name'] for r in rows})
        if 'cases' in data:
            declared = data['cases']
            if (not isinstance(declared, list) or not declared or len(set(declared)) != len(declared) or
                (case is None and set(cases) != set(declared)) or (case is not None and case not in declared)):
                raise ValueError('Missing or inconsistent declared workload coverage')
        pairs = sorted({r['pair'] for r in rows})
        if any(isinstance(p, bool) or not isinstance(p, int) for p in pairs) or pairs != list(range(len(pairs))):
            raise ValueError('Pair indices must be contiguous from zero')
        if 'pairs' in data and data['pairs'] != len(pairs):
            raise ValueError('Missing recorded pairs')
        keys = [(r['pair'], r['configuration'], r['name']) for r in rows]
        if len(keys) != len(set(keys)) or set(keys) != {(p,c,n) for p in pairs for c in (control,candidate) for n in cases}:
            raise ValueError('Missing or duplicate paired observations')
        # The actual summary order must show alternating arms. Never sort timings
        # into an invented experimental order.
        for p in pairs:
            observed = list(dict.fromkeys(r['configuration'] for r in rows if r['pair'] == p))
            order = [c['name'] for c in (data['configurations'] if p % 2 == 0 else data['configurations'][::-1]) if c['name'] in (control,candidate)]
            if observed != order: raise ValueError('Recorded arms are not alternating')
        expected = dict(build=data['build'], revision=data['artifact_revision'])
        workloads = {}; observations = {}; instrumentation = None; common_identity = None; used_reports = set()
        for row in rows:
            report_hash = row['report_sha256']
            if not isinstance(report_hash, str) or len(report_hash) != 64 or any(c not in '0123456789abcdef' for c in report_hash):
                raise ValueError('Raw report reference requires a full SHA256, not a prefix or path')
            if report_hash in used_reports: raise ValueError('The same raw run cannot count as another pair or workload')
            used_reports.add(report_hash)
            raw, ref = index.json(report_hash)
            sources.append(dict(ref, pointer='/runs'))
            if not isinstance(raw, dict) or not terminal_result(raw): raise ValueError('Native report did not complete successfully')
            if raw.get('workloads') is None:
                raise ValueError('Missing original token workload')
            if len(raw['workloads']) != len(raw['runs']):
                raise ValueError('Missing workload/continuation evidence')
            history = 0
            for work, native in zip(raw['workloads'], raw['runs']):
                tokens = work.get('tokens')
                if not isinstance(tokens, list) or not tokens or any(type(t) is not int for t in tokens):
                    raise ValueError('Missing or invalid original token IDs')
                prompt = history + len(tokens) if work.get('append') else len(tokens)
                if native['name'] != work['name'] or native['prompt_tokens'] != prompt:
                    raise ValueError('Recorded workload does not match executed prompt')
                if work.get('prime'):
                    if native.get('finish_reason') != 'primed' or native['output_tokens'] != 0:
                        raise ValueError('History prime did not complete')
                elif work.get('max_tokens') != native['output_tokens']:
                    raise ValueError('Output length differs from original workload')
                history = prompt + native['output_tokens']
            workload = encoded(raw['workloads'])
            name = row['name']
            if workloads.setdefault(name, workload) != workload:
                raise ValueError('Original input tokens, history or output limits differ')
            output = raw['runs'][-1]['output_tokens']
            if output < 2: raise ValueError('Decode timing requires at least two output tokens')
            observation = validate(raw, configs[row['configuration']], expected, data['budget_bytes'], output)
            if observation['name'] != name: raise ValueError('Summary workload label differs from executed workload')
            for native_row in raw['runs']:
                before, after = native_row['before'], native_row['after']
                for k in ('execution', 'completion_pipeline', 'ready_group', 'chunk_tokens', 'io_workers'):
                    if before.get(k) != after.get(k): raise ValueError('Runtime configuration changed during request')
                if before['metal']['kernels'] != after['metal']['kernels']:
                    raise ValueError('Kernels changed during request')
                for state in (native_row['before'], native_row['after']):
                    actual = (state['artifact_revision'], state['prepared']['manifest_sha256'],
                              state['metal']['device'], state['metal']['physical_bytes'],
                              state['metal']['build_fingerprint'], state['memory_plan']['limit_bytes'])
                    if any(v is None for v in actual) or (common_identity is not None and common_identity != actual):
                        raise ValueError('Artifact, prepared bytes, build, machine or memory budget differs')
                    common_identity = actual
                    mode = (native_row.get('profiling_enabled'), state['metal']['kernels'].get('profile'),
                            state['metal']['kernels'].get('counter_profile'))
                    if mode[0] is not False or mode[1] is not False or mode[2] not in (None, False):
                        raise ValueError('Instrumented or missing profiling mode cannot qualify normal timing')
                    if instrumentation is not None and instrumentation != mode:
                        raise ValueError('Instrumentation modes differ')
                    instrumentation = mode
            for metric in ('ttft_ms', 'decode_ms_per_token', 'request_ms'):
                positive(observation[metric])
                if row.get(metric) != observation[metric]:
                    raise ValueError('Summary metric differs from original native report: ' + metric)
            observations[row['pair'], row['configuration'], name] = observation
        metrics = []
        for name in cases:
            for metric in ('ttft_ms', 'decode_ms_per_token', 'request_ms'):
                left = [observations[p,control,name][metric] for p in pairs]
                right = [observations[p,candidate,name][metric] for p in pairs]
                ratios = [y/x for x,y in zip(left,right)]
                metrics.append(dict(case=name, metric=metric, pairs=len(pairs), control_median=statistics.median(left),
                                    candidate_median=statistics.median(right), ratios=ratios,
                                    median_ratio=statistics.median(ratios),
                                    confidence_95=paired_interval(ratios) if len(pairs) >= 5 else None))
        if len(pairs) < 5: limits.append('Fewer than five pairs: descriptive screen only; no confidence bounds or latency qualification.')
        if instrumentation[2] is None: limits.append('Legacy reports record profiling disabled but omit a separate counter_profile field.')
        limits.append('No separate Metal validation environment receipt is available in legacy normal summaries.')
        return result(sources, limits, comparable=True, status='measured', scope='normal_requests',
                      controlled_change=difference, correctness=dict(generated_tokens_equal=True,
                      full_model_logits_and_state='not established by this timing query'), metrics=metrics,
                      normal_request_latency_qualified=False)
    except (ValueError, KeyError, TypeError, IndexError, OSError) as error:
        return result(sources, limits, comparable=False, status='insufficient_or_incompatible_evidence', reason=str(error))


def snapshots(data):
    """Visit recorded state objects; pointers distinguish planning from allocation."""
    found = []
    def optional_object(value):
        if value is None: return {}
        if not isinstance(value, dict): raise ValueError('Invalid optional memory snapshot object')
        return value
    def visit(value, pointer):
        if isinstance(value, dict):
            if value.get('kind')=='memory_boundary_v1':
                process=optional_object(value.get('process'));metal=optional_object(value.get('metal'))
                costs=optional_object(metal.get('buffer_costs'));held=None
                classes=costs.get('classes')
                if isinstance(classes,dict):
                    held={}
                    for name,c in classes.items():
                        if not isinstance(c,dict):raise ValueError('Invalid buffer class counters')
                        a,b=c.get('allocated_bytes'),c.get('owner_released_bytes')
                        held[name]=a-b if type(a) is int and type(b) is int and 0<=b<=a else None
                found.append(dict(pointer=pointer,where=value.get('where'),lifecycle=value.get('lifecycle'),
                    monotonic_ns=value.get('monotonic_ns'),process=process,system=value.get('system'),
                    engine_owner_held_bytes=held,live_buffer_bytes=metal.get('live_buffer_bytes'),
                    peak_buffer_bytes=metal.get('peak_buffer_bytes'),device_allocated_bytes=metal.get('device_allocated_bytes'),
                    live_command_groups=metal.get('live_command_groups'),encoded_buffer_references=metal.get('encoded_buffer_references')))
                return
            if isinstance(value.get('metal'), dict) and isinstance(value.get('memory_plan'), dict):
                metal = value['metal']
                process = optional_object(value.get('process'))
                residency = optional_object(metal.get('residency'))
                found.append(dict(pointer=pointer, memory_plan=value['memory_plan'],
                    physical_footprint_bytes=process.get('physical_footprint_bytes'),
                    live_buffer_bytes=metal.get('live_buffer_bytes'), peak_buffer_bytes=metal.get('peak_buffer_bytes'),
                    live_command_groups=metal.get('live_command_groups'),
                    pending_retirements=residency.get('pending_retirements'),
                    residency_bytes_by_class=residency.get('bytes_by_class'),
                    scratch_pools=metal.get('scratch_pools'), expert_cache=value.get('expert_cache'),
                    phase_memory=value.get('phase_memory')))
                return
            for k,v in value.items(): visit(v, pointer+'/'+str(k).replace('~','~0').replace('/','~1'))
        elif isinstance(value, list):
            for n,v in enumerate(value): visit(v, pointer+'/'+str(n))
    visit(data, '')
    return found


def memory(index, selector, limit=10, offset=0):
    raw, source = index.source(selector);coverage=None
    if source['path'].endswith('.jsonl'):
        lines=raw.splitlines();data=[parse(line) if line.strip() else None for line in lines[:MAX_LINES]]
        coverage=dict(file_lines=len(lines),returned_to_parser=min(len(lines),MAX_LINES),query_truncated=len(lines)>MAX_LINES)
    else:data=parse(raw)
    rows = snapshots(data)
    if coverage:
        for row in rows:row['pointer']='line:'+str(int(row['pointer'].split('/')[1])+1)
    return result([source], ['Plans are capacity budgets, not measured allocations; nested categories overlap and are not added.',
        'Physical footprint includes more than Metal allocations; missing values remain null.',
        'Owner-held bytes and device resource sizes are not physical residency; system VM categories overlap.',
        'A memory trace query shows recorded samples; it does not establish complete capture or qualify latency.',
        'Only recorded lifecycle boundaries are shown. Model destruction or final drain must not be inferred.'],
        status='recorded_snapshots' if rows else 'missing_measurements', snapshots=rows[offset:offset+limit],
        total=len(rows), coverage=coverage,next_offset=offset+limit if offset+limit<len(rows) else None)


def timeline(index, selector, phase=None, layer=None, token=None, limit=10, offset=0):
    raw, source = index.source(selector)
    if source['path'].endswith('.jsonl'):
        lines = raw.splitlines()
        data = dict(query_line_limit=MAX_LINES, query_truncated=len(lines)>MAX_LINES)
        events = [(f'line:{n}', parse(line)) for n,line in enumerate(lines[:MAX_LINES],1) if line.strip()]
    else:
        data = parse(raw)
        if not isinstance(data, dict): raise ValueError('Select a profile object or JSONL trace')
        events = [(f'/expert_dependencies/{n}', e) for n,e in enumerate(data.get('expert_dependencies', []))]
    selected = []
    total_records = 0; records_missing = False
    for pointer, event in events:
        if not isinstance(event, dict): continue
        if phase is not None and event.get('request_phase') != phase: continue
        if layer is not None and event.get('layer') != layer: continue
        if token is not None and not (isinstance(event.get('offset'), int) and isinstance(event.get('tokens'), int)
                                     and event['offset'] <= token < event['offset']+event['tokens']): continue
        records = event.get('records')
        if records is None: records_missing = True
        elif not isinstance(records, list): raise ValueError('Invalid expert lifecycle records')
        else: total_records += len(records)
        fields = ('layer','offset','tokens','request_phase','duration_ns','coordinator_wait_ns','last_required_read_ns',
                  'read_queue_sum_ns','read_service_sum_ns','ready_to_gpu_sum_ns','gpu_execution_sum_ns',
                  'ready_hits','loading_joins','new_misses','peak_leases','peak_gpu_groups')
        selected.append(dict(pointer=pointer, **{k:event.get(k) for k in fields},
                             records=records[:8] if records is not None else None,
                             recorded_records=len(records) if records is not None else None,
                             returned_records=min(8,len(records)) if records is not None else None))
    return result([source], ['Times can overlap; sums are not a critical-path decomposition.',
        'Expert lifecycle timestamps are joined only within the recorded entry. No cross-file command/request IDs are invented.',
        'Blocking reasons (data, GPU capacity, free buffer, prior computation) are not separately recorded.',
        'Detailed capture may omit later tokens/panels, even when the command trace reports truncated=false.',
        'At most eight expert lifecycles per returned pass; source pointers retain the full captured records.'],
        status='recorded_partial_timeline' if selected else 'no_matching_captured_events',
        coverage=dict(capture_limits=data.get('dependency_capture_limits'), command_trace_truncated=data.get('truncated'),
                      query_line_limit=data.get('query_line_limit'), query_truncated=data.get('query_truncated'),
                      matching_passes=len(selected), matching_read_records=None if records_missing else total_records, whole_request_covered=None),
        passes=selected[offset:offset+limit], next_offset=offset+limit if offset+limit<len(selected) else None)


def next_experiment(index, selector):
    data, source = index.json(selector)
    if not isinstance(data, dict): raise ValueError('Select a report object')
    if data.get('kind') in ('native_mtp_continuation_v1', 'native_mtp_continuation_v2', 'native_mtp_width_v1',
                            'mtp_direct_output_screen_v1', 'mtp_target_recovery_screen_v1', 'mtp_width_screen_v1'):
        from mtp_evidence import next_experiment as mtp_next
        return mtp_next(index, selector)
    limits = ['No request speedup is inferred from cached kernels or summed overlapping waits.',
              'This is a proposed measurement, not an accepted optimization or automatic execution.']
    if data.get('complete') is False or data.get('status') in ('failed','resource_blocked','interrupted','time_budget_exhausted'):
        return result([source], limits, hypothesis='No completed result supports choosing a winner.',
                      possible_request_benefit=None, missing_evidence=['completed comparable short timing screen'],
                      smallest_experiment='Resolve the recorded blocker, then run the capped screen; preserve this unfinished attempt.')
    if data.get('kind') == 'instrumented_cached_decode_attribution':
        return result([source], limits, hypothesis='The largest cached GPU operation may be hidden by normal-request dependencies.',
                      cached_hotspots=data.get('kernels', [])[:3], possible_request_benefit=None,
                      missing_evidence=['matched normal-request dependency trace and explicit blocking reasons'],
                      smallest_experiment='Capture one bounded normal decode/append trace on the same build and artifact before ranking kernel work.')
    deps = [s for s in snapshots(data) if isinstance(s.get('expert_cache'), dict)]
    return result([source], limits, hypothesis='Existing aggregates cannot identify exposed critical-path savings.',
                  possible_request_benefit=None, recorded_memory_boundaries=len(deps),
                  missing_evidence=['joined dependency events with blocking reasons', 'comparable paired request outcome'],
                  smallest_experiment='Use compare on the recorded experiment first; add only the event linkage needed for an unresolved dependency.')
