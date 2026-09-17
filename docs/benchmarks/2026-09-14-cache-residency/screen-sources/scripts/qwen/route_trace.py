"""Read committed route capture and live progress without starting inference."""
from evidence_index import parse

KIND = 'qwen_route_trace_v1'
MAX_BYTES = 16 * 1024**2


def decode(raw):
    if len(raw) > MAX_BYTES:
        raise ValueError('Route trace exceeds 16MiB')
    lines = raw.splitlines(keepends=True)
    partial = bool(lines and not lines[-1].endswith(b'\n'))
    if partial:
        lines.pop()
    if len(lines) > 20000:
        raise ValueError('Route event count exceeded')
    events = [(f'line:{i}', parse(line)) for i, line in enumerate(lines, 1)]
    if not events:
        raise ValueError('Missing route trace header')
    rows, committed, requests = [], [], []
    request = active = None
    request_id = forward_id = timestamp = 0
    header = terminal = last = None
    aborted = 0
    epoch = 0
    states = {}
    current_session = None
    for sequence, (pointer, row) in enumerate(events, 1):
        if (not isinstance(row, dict) or row.get('kind') != KIND or
            type(row.get('sequence')) is not int or row['sequence'] != sequence or terminal is not None):
            raise ValueError('Invalid route sequence or events after termination')
        now = row.get('monotonic_ns')
        if type(now) is not int or now < timestamp:
            raise ValueError('Route timestamps must be monotonic integers')
        timestamp = now
        event, d = row.get('event'), row.get('details')
        if not isinstance(d, dict):
            raise ValueError('Route event details must be an object')
        last = dict(event=event, source_pointer=pointer, monotonic_ns=now)
        if event == 'trace_begin':
            if sequence != 1 or not isinstance(d.get('identity'), dict):
                raise ValueError('Invalid route header')
            header = d['identity']
        elif header is None:
            raise ValueError('Route header must be first')
        elif event == 'request_begin':
            if request is not None or active is not None:
                raise ValueError('Overlapping route requests')
            request_id += 1
            request = dict(d, request_id=request_id, source_pointer=pointer, aborted=False)
            request['_initial_sizes'] = {k:len(v) for k,v in states.items()}
            tokens(d.get('input_token_ids'))
            if type(d.get('max_tokens')) is not int or d['max_tokens'] < 1 or type(d.get('prime')) is not bool:
                raise ValueError('Invalid request output limit or prime flag')
        elif event == 'forward_begin':
            if request is None or active is not None:
                raise ValueError('Forward needs one active request')
            forward_id += 1
            tokens(d.get('input_token_ids'))
            if (type(d.get('offset')) is not int or d['offset'] < 0 or type(d.get('tokens')) is not int or
                d['tokens'] != len(d['input_token_ids']) or d['offset'] + d['tokens'] > 8192 or
                not isinstance(d.get('session_id'), str) or not d['session_id'] or
                not isinstance(d.get('request_phase'), str) or not d['request_phase'] or
                type(d.get('expert_slots')) is not int or d['expert_slots'] < 1):
                raise ValueError('Invalid forward identity or geometry')
            active = dict(d, forward_id=forward_id, request_id=request_id,
                          source_pointer=pointer, rows=[], epoch=epoch)
            if d['offset'] != len(states.get(d['session_id'], [])):
                raise ValueError('Forward offset does not continue the recorded session')
        elif event == 'routes':
            if active is None or type(d.get('layer')) is not int or d['layer'] != len(active['rows']) or d['layer'] >= 48:
                raise ValueError('Missing, repeated or reordered route layer')
            routes = d.get('routes')
            if not isinstance(routes, list) or len(routes) != active['tokens'] * 10:
                raise ValueError('Missing top-10 routes')
            for start in range(0, len(routes), 10):
                selected = routes[start:start+10]
                if any(type(e) is not int or not 0 <= e < 512 for e in selected) or len(set(selected)) != 10:
                    raise ValueError('Invalid selected experts')
            active['rows'].append((pointer, dict(layer=d['layer'], routes=routes, offset=active['offset'],
                tokens=active['tokens'], request_phase=active['request_phase'], request_id=active['request_id'],
                session_id=active['session_id'] + ':' + str(epoch), build=header.get('build'),
                artifact_revision=header.get('artifact_revision'))))
        elif event == 'forward_commit':
            if (active is None or len(active['rows']) != 48 or type(d.get('position')) is not int or
                d['position'] != active['offset'] + active['tokens']):
                raise ValueError('Commit needs all 48 layers and the exact resulting position')
            rows.extend(active['rows'])
            current_session = active['session_id']
            states.setdefault(current_session, []).extend(active['input_token_ids'])
            committed.append({k:v for k,v in active.items() if k != 'rows'})
            active = None
        elif event == 'forward_abort':
            if active is None or type(d.get('captured_layers')) is not int or d['captured_layers'] != len(active['rows']):
                raise ValueError('Invalid aborted forward')
            aborted += 1
            rows.append((pointer, dict(cache_boundary='forward_abort')))
            request['aborted'] = True
            active = None
        elif event == 'request_end':
            if request is None or active is not None or request['aborted']:
                raise ValueError('Cannot complete a missing, aborted or active request')
            tokens(d.get('output_token_ids'), empty=True)
            if (d.get('finish_reason') not in ('length', 'stop', 'primed') or
                type(d.get('reused_tokens')) is not int or not 0 <= d['reused_tokens'] <= len(request['input_token_ids'])):
                raise ValueError('Invalid request result')
            output = d['output_token_ids']
            if (len(output) > request['max_tokens'] or
                request['prime'] != (d['finish_reason'] == 'primed') or
                (request['prime'] and output) or (not request['prime'] and not output) or
                (d['finish_reason'] == 'length' and len(output) != request['max_tokens']) or
                (d['finish_reason'] == 'stop' and output[-1] not in (248044, 248046))):
                raise ValueError('Output does not match the request stopping rule')
            expected = request['input_token_ids'] + output[:-1]
            if (states.get(current_session) != expected or
                request['_initial_sizes'].get(current_session, 0) != d['reused_tokens']):
                raise ValueError('Committed inputs or reused computation do not match the request result')
            request.pop('_initial_sizes')
            requests.append(dict(request, result=d))
            request = None
        elif event == 'cache_reset':
            if active is not None:
                raise ValueError('Cannot reset cache in a forward')
            epoch += 1
            rows.append((pointer, dict(cache_boundary='cache_reset')))
        elif event == 'trace_end':
            terminal = d.get('status')
            if terminal not in ('complete', 'incomplete') or (terminal == 'complete' and (request or active)):
                raise ValueError('Invalid route termination')
        else:
            raise ValueError('Unknown route event: ' + str(event))
        if any(type(row.get(k)) is not int or row[k] != value for k, value in
               [('request_id', request_id), ('forward_id', forward_id)]):
            raise ValueError('Mismatched route request or forward ID')
    if terminal and partial:
        raise ValueError('Trailing bytes after route termination')
    return dict(rows=rows, committed=committed, requests=requests, identity=header,
                status=terminal or 'unfinished', complete=terminal == 'complete',
                events=len(events), partial_final_line=partial, aborted_forwards=aborted,
                active_request=None if request is None else {k:v for k,v in request.items() if k not in ('input_token_ids','_initial_sizes')},
                active_forward=None if active is None else {k:v for k,v in active.items() if k not in ('rows','input_token_ids')},
                uncommitted_layer_passes=len(active['rows']) if active else 0, last_event=last)


def tokens(value, empty=False):
    if (not isinstance(value, list) or (not value and not empty) or len(value) > 8192 or
        any(type(t) is not int or not 0 <= t < 248320 for t in value)):
        raise ValueError('Invalid token IDs')


def progress(path):
    with open(path, 'rb') as stream:
        report = decode(stream.read(MAX_BYTES + 1))
    return {k:v for k,v in report.items() if k not in ('rows', 'committed', 'requests')}
