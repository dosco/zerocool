#!/usr/bin/env python3
"""Read cached-replay phase progress, including unfinished and cancelled runs."""
import argparse
import hashlib
import json
from pathlib import Path

from evidence_index import parse


def summarize(path, expected_build=None):
    path=Path(path)
    if not path.exists():
        return dict(available=False,source=str(path),limitation='No phase progress was recorded.')
    with path.open('rb') as stream: raw=stream.read(40*1024**2+1)
    if len(raw)>40*1024**2: raise ValueError('Progress trace exceeds bounded size')
    # A live writer or a killed process can leave a partial final line. Never
    # accept that fragment, even if it happens to be parseable JSON.
    lines=raw.split(b'\n');partial=bool(lines.pop())
    active=None;phase_started=0;previous=-1;identity=None;last=None;details={};durations={};terminal=None
    for sequence,line in enumerate(lines,1):
        if sequence>10000 or len(line)>4096: raise ValueError('Progress record limit exceeded')
        row=parse(line)
        if not isinstance(row,dict) or row.get('kind')!='cached_replay_progress_v1' or type(row.get('sequence')) is not int or row.get('sequence')!=sequence:
            raise ValueError('Missing or reordered progress events')
        now=row.get('elapsed_ns');elapsed=row.get('phase_elapsed_ns');stamp=row.get('monotonic_ns')
        if any(type(v) is not int or v<0 for v in (now,elapsed,stamp)) or now<previous or elapsed>now:
            raise ValueError('Invalid progress timestamps')
        if last and (stamp<last['monotonic_ns'] or stamp-last['monotonic_ns']!=now-previous):
            raise ValueError('Inconsistent progress clocks')
        previous=now
        if not isinstance(row.get('identity'),dict) or not isinstance(row.get('details'),dict):
            raise ValueError('Missing progress identity or details')
        if identity is None: identity=row['identity']
        if row['identity']!=identity or (expected_build and identity.get('build')!=expected_build):
            raise ValueError('Progress build or identity changed')
        event=row.get('event');phase=row.get('phase')
        if not isinstance(phase,str) or not phase or terminal: raise ValueError('Invalid progress phase or terminal sequence')
        if event=='begin':
            if active: raise ValueError('Overlapping progress phases')
            active=phase;phase_started=now-elapsed;details={}
        elif event in ('progress','end'):
            if phase!=active or now-phase_started!=elapsed: raise ValueError('Progress phase mismatch')
        elif event in ('complete','interrupted','failed'):
            if (active and (phase!=active or now-phase_started!=elapsed)) or (not active and last and phase!=last['phase']):
                raise ValueError('Terminal progress phase mismatch')
            if row['details'].get('phase_incomplete') is not bool(active) or (event=='complete' and active):
                raise ValueError('Unfinished phase cannot complete')
            terminal=event
        else: raise ValueError('Unknown progress event')
        details.update(row['details'])
        if event=='end':
            durations[phase]=durations.get(phase,0)+elapsed;active=None
        last=row
    if terminal and partial: raise ValueError('Data follows terminal progress event')
    return dict(available=last is not None,source=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest(),
        identity=identity,events=len(lines),status=terminal or 'unfinished',complete=terminal=='complete',
        last_phase=last['phase'] if last else None,unfinished_phase=active,
        last_elapsed_ns=last['elapsed_ns'] if last else None,
        unfinished_phase_elapsed_ns=last['phase_elapsed_ns'] if last and active else None,
        completed_phase_ns=durations,latest_details=details,partial_final_line_ignored=partial,
        limitations=['Progress is diagnostic; partial pairs never establish performance or correctness.',
                     'Times end at the last flushed event; silence after it is not measured.',
                     'Interruption/failure timestamps can include resource cleanup.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace',type=Path)
    args=parser.parse_args()
    try: print(json.dumps(summarize(args.trace),indent=2))
    except (ValueError,OSError) as error: parser.exit(2,str(error)+'\n')
