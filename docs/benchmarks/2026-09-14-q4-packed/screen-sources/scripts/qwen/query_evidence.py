#!/usr/bin/env python3
"""Offline benchmark queries for coding agents. JSON stdout; never runs inference."""
import argparse
import datetime
from pathlib import Path
import sqlite3
import sys

from evidence_index import Index, digest, encoded, metadata, parse, terminal_result
from evidence_queries import compare, memory, next_experiment, timeline
from cache_simulation import cache_curve

ROOT = Path(__file__).resolve().parents[2]


def record(index, input_path, directory):
    entry = parse(Path(input_path).read_bytes())
    required = ('hypothesis', 'expected_effect', 'controlled_change', 'correctness',
                'outcome', 'decision', 'smallest_experiment', 'limitations', 'evidence')
    if not isinstance(entry, dict) or any(k not in entry for k in required):
        raise ValueError('Ledger entry requires: ' + ', '.join(required))
    for key in required[:7]:
        if not isinstance(entry[key], str) or not entry[key].strip():
            raise ValueError('Ledger text must be nonempty: ' + key)
    if entry['decision'] not in ('proposed','blocked','rejected','inconclusive','promising','adopted'):
        raise ValueError('Unknown ledger decision')
    if not isinstance(entry['limitations'], list) or not all(isinstance(x,str) for x in entry['limitations']):
        raise ValueError('Ledger limitations must be a list of strings')
    if not isinstance(entry['evidence'], list) or not entry['evidence']:
        raise ValueError('Ledger needs original evidence references')
    refs = []; terminal = []
    for selector in entry['evidence']:
        data, source = index.json(selector)
        if not isinstance(data, dict): raise ValueError('Ledger evidence must reference report objects')
        refs.append(dict(source, recorded_status=metadata(data)['status'], recorded_complete=data.get('complete')))
        terminal.append(terminal_result(data, allow_failure=entry['decision']=='rejected'))
    if not any(terminal) and entry['decision'] not in ('blocked','proposed','inconclusive'):
        raise ValueError('Unfinished or unknown evidence cannot establish rejection, benefit or adoption')
    entry = {k: entry[k] for k in required}
    entry.update(kind='experiment_ledger_v1', evidence=refs,
                 recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 decision_origin='authored interpretation; original report statuses retained', production_promoted=False)
    raw = (encoded(entry)+'\n').encode()
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory / (digest(raw)+'.json')
    with path.open('xb') as f: f.write(raw)
    index.import_paths([path])
    return dict(path=str(path.resolve()), id=digest(raw), sources=refs,
                limitations=['Ledger decisions are authored interpretations, not new measurements.',
                             'Ledger JSON is durable; SQLite can be deleted and rebuilt from it.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=ROOT/'.cache/evidence/index.sqlite')
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('import', help='Index JSON/JSONL evidence; raw files are read-only')
    p.add_argument('paths', nargs='+', type=Path); p.add_argument('--rebuild', action='store_true')
    p = subs.add_parser('history', help='Search runs, failures and ledger entries')
    p.add_argument('--search', default=''); p.add_argument('--limit', type=int, default=20); p.add_argument('--offset', type=int, default=0)
    p = subs.add_parser('show', help='Compact report identity and status'); p.add_argument('source')
    p = subs.add_parser('compare', help='Revalidate paired raw evidence with declared changes')
    p.add_argument('source'); p.add_argument('--control', required=True); p.add_argument('--candidate', required=True)
    p.add_argument('--change', action='append', required=True); p.add_argument('--case')
    for name in ('memory', 'timeline'):
        p = subs.add_parser(name); p.add_argument('source'); p.add_argument('--limit', type=int, default=10); p.add_argument('--offset', type=int, default=0)
        if name == 'timeline':
            p.add_argument('--phase'); p.add_argument('--layer', type=int); p.add_argument('--token', type=int)
    p = subs.add_parser('next', help='Smallest useful measurement, with explicit evidence gaps'); p.add_argument('source')
    p = subs.add_parser('cache', help='Offline cache-size curves; simulated reads, never latency')
    p.add_argument('source'); p.add_argument('--budget-mib', type=int, action='append')
    p.add_argument('--phase'); p.add_argument('--per-layer', action='store_true')
    p = subs.add_parser('record', help='Append a durable, source-bound experiment ledger entry')
    p.add_argument('input', type=Path); p.add_argument('--ledger', type=Path, default=ROOT/'docs/experiments')
    args = parser.parse_args(argv)
    index = None
    try:
        if not 1 <= getattr(args,'limit',1) <= 100 or getattr(args,'offset',0) < 0:
            raise ValueError('Query limit must be 1..100 and offset nonnegative')
        index = Index(args.db)
        if args.command == 'import': answer = index.import_paths(args.paths, args.rebuild)
        elif args.command == 'history': answer = index.history(args.search, args.limit, args.offset)
        elif args.command == 'show':
            report, source = index.describe(args.source)
            answer = dict(report=report, sources=[source], limitations=['Report claims are preserved, not independently rerun.'])
        elif args.command == 'compare': answer = compare(index,args.source,args.control,args.candidate,args.change,args.case)
        elif args.command == 'memory': answer = memory(index,args.source,args.limit,args.offset)
        elif args.command == 'timeline': answer = timeline(index,args.source,args.phase,args.layer,args.token,args.limit,args.offset)
        elif args.command == 'next': answer = next_experiment(index,args.source)
        elif args.command == 'cache': answer = cache_curve(index,args.source,args.budget_mib,args.phase,args.per_layer)
        else: answer = record(index,args.input,args.ledger)
        print(encoded(answer))
        return 2 if answer.get('comparable') is False else 0
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error) as error:
        print(encoded(dict(status='error', error=str(error), sources=[],
                           limitations=['No conclusion; select valid, current raw evidence and reimport if necessary.'])))
        return 2
    finally:
        if index is not None: index.close()


if __name__ == '__main__': sys.exit(main())
