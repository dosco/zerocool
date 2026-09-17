"""Disposable, content-addressed index of local benchmark evidence (stdlib only)."""
import hashlib
import json
import math
from pathlib import Path
import sqlite3

SCHEMA = 1
MAX_BYTES = 64 * 1024**2
MAX_LINES = 100_000
UNFINISHED_STATUSES = {'running', 'pending', 'unfinished', 'resource_blocked',
                       'interrupted', 'time_budget_exhausted'}


def terminal_result(data, allow_failure=False):
    """A passing subphase, metadata, or unknown completion is not a result."""
    if data.get('complete') is False or data.get('status') in UNFINISHED_STATUSES:
        return False
    if allow_failure:
        return data.get('complete') is True or type(data.get('passed')) is bool
    return (data.get('status') not in ('failed', 'error') and data.get('passed') is not False and
            (data.get('complete') is True or data.get('passed') is True))


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def parse(raw):
    def invalid(value):
        raise ValueError('Non-finite JSON number: ' + value)
    def finite(value):
        number = float(value)
        if not math.isfinite(number): invalid(value)
        return number
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(raw, parse_constant=invalid, parse_float=finite, object_pairs_hook=unique)


def read(path):
    path = Path(path)
    before = path.stat()
    if before.st_size > MAX_BYTES:
        raise ValueError('File exceeds 64MiB import/query limit')
    # Bound the read itself: a growing producer must not bypass the stat limit.
    with path.open('rb') as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES: raise ValueError('File exceeds import/query read limit')
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('File changed during read; retry after it settles')
    return raw


def metadata(data):
    if not isinstance(data, dict):
        return dict(kind='unrecognized', status='unknown', complete=None)
    kind = data.get('kind') or data.get('protocol')
    if not kind:
        kind = ('native_requests' if 'runs' in data and 'model_revision' in data else
                'gpu_profile' if 'command_groups' in data else
                'admission' if 'current_admission' in data else 'report')
    keys = ('status', 'complete', 'passed', 'decision', 'error', 'note', 'build',
            'build_fingerprint', 'artifact_revision', 'model_revision', 'budget_bytes',
            'comparison_axis', 'mode', 'pairs', 'prompt_tokens', 'profiling_enabled',
            'normal_request_latency_qualified', 'production_promoted', 'phases',
            'hypothesis', 'expected_effect', 'controlled_change', 'correctness',
            'outcome', 'smallest_experiment', 'limitations', 'evidence',
            'decision_origin', 'recorded_at', 'started_at', 'finished_at')
    info = {k: data[k] for k in keys if k in data and k != 'phases'}
    if isinstance(data.get('phases'), dict):
        info['phases'] = {k: v.get('status') for k, v in data['phases'].items() if isinstance(v, dict)}
    info.update(kind=kind)
    if 'status' not in info:
        info['status'] = ('complete' if data.get('complete') is True else
                          'unfinished' if data.get('complete') is False else
                          'passed' if data.get('passed') is True else
                          'failed' if data.get('passed') is False else 'unknown')
    info.setdefault('complete', None)
    info['configurations'] = [c.get('name') for c in data.get('configurations', []) if isinstance(c, dict)]
    return info


def projection(raw, suffix):
    if suffix != '.jsonl': return metadata(parse(raw)), []
    lines = raw.splitlines()
    info = dict(kind='jsonl_trace', status='unknown', complete=None,
                lines=len(lines), indexed_lines=min(len(lines), MAX_LINES),
                import_truncated=len(lines) > MAX_LINES)
    events = []
    for n, line in enumerate(lines[:MAX_LINES], 1):
        if not line.strip(): continue
        row = parse(line)
        if not isinstance(row, dict): continue
        keys = ('layer', 'offset', 'tokens', 'request_phase', 'build',
                'artifact_revision', 'duration_ns', 'coordinator_wait_ns',
                'ready_hits', 'loading_joins', 'new_misses', 'event', 'phase',
                'timestamp', 'physical_footprint_bytes')
        events.append((n, encoded({k: row[k] for k in keys if k in row})))
    return info, events


class Index:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        version = self.db.execute('PRAGMA user_version').fetchone()[0]
        tables = self.db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        expected_tables = {'documents', 'paths', 'events', 'issues'}
        if (version not in (0, SCHEMA) or (version == 0 and tables) or
            (version == SCHEMA and {r[0] for r in tables} != expected_tables)):
            self.db.close()
            raise ValueError('Not a compatible evidence index; use a new database path')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS documents (sha TEXT PRIMARY KEY, format TEXT, info TEXT);
            CREATE TABLE IF NOT EXISTS paths (path TEXT PRIMARY KEY, sha TEXT);
            CREATE INDEX IF NOT EXISTS paths_sha ON paths(sha);
            CREATE TABLE IF NOT EXISTS events (sha TEXT, ordinal INTEGER, info TEXT,
                PRIMARY KEY(sha, ordinal));
            CREATE TABLE IF NOT EXISTS issues (path TEXT PRIMARY KEY, error TEXT);
            PRAGMA user_version=1;
        ''')

    def close(self):
        self.db.close()

    def import_paths(self, roots, rebuild=False):
        files = set()
        for root in roots:
            root = Path(root).resolve()
            if not root.exists():
                raise ValueError('Import path does not exist: ' + str(root))
            for p in ([root] if root.is_file() else root.rglob('*')):
                if p.suffix in ('.json', '.jsonl') and p.is_file() and not p.is_symlink():
                    files.add(p.resolve())
        with self.db:
            if rebuild:
                for table in ('paths', 'documents', 'events', 'issues'):
                    self.db.execute('DELETE FROM ' + table)
            for path in sorted(files):
                try:
                    raw = read(path); sha = digest(raw)
                    # Recompute projections even for an existing hash: parser and
                    # extraction fixes must apply on incremental reimport too.
                    info, events = projection(raw, path.suffix)
                    self.db.execute('DELETE FROM events WHERE sha=?', (sha,))
                    self.db.executemany('INSERT INTO events VALUES (?,?,?)', [(sha,n,e) for n,e in events])
                    self.db.execute('INSERT OR REPLACE INTO documents VALUES (?,?,?)', (sha, path.suffix, encoded(info)))
                    self.db.execute('INSERT OR REPLACE INTO paths VALUES (?,?)', (str(path), sha))
                    self.db.execute('DELETE FROM issues WHERE path=?', (str(path),))
                except (ValueError, OSError, TypeError, KeyError) as error:
                    # A changed or malformed file cannot leave a usable stale alias.
                    self.db.execute('DELETE FROM paths WHERE path=?', (str(path),))
                    self.db.execute('INSERT OR REPLACE INTO issues VALUES (?,?)', (str(path), str(error)))
            self.db.execute('DELETE FROM events WHERE sha NOT IN (SELECT sha FROM paths)')
            self.db.execute('DELETE FROM documents WHERE sha NOT IN (SELECT sha FROM paths)')
        return dict(documents=self.db.execute('SELECT count(*) FROM documents').fetchone()[0],
                    paths=self.db.execute('SELECT count(*) FROM paths').fetchone()[0],
                    issues=[dict(r) for r in self.db.execute('SELECT * FROM issues ORDER BY path')],
                    sources=[str(Path(p).resolve()) for p in roots],
                    limitations=['Import is a snapshot; queries verify selected raw files again.',
                                 'Files over 64MiB and malformed files are reported as issues; JSONL indexing is bounded.'])

    def resolve(self, selector):
        exact = self.db.execute('SELECT sha FROM paths WHERE path=?', (str(Path(selector).resolve()),)).fetchone()
        rows = [exact] if exact else self.db.execute('SELECT sha FROM documents WHERE substr(sha,1,?)=?', (len(selector), selector)).fetchall()
        if len(rows) != 1:
            raise ValueError('Select one indexed path or unambiguous content hash')
        return rows[0]['sha']

    def source(self, selector):
        sha = self.resolve(selector)
        paths = [r['path'] for r in self.db.execute('SELECT path FROM paths WHERE sha=? ORDER BY path', (sha,))]
        requested = str(Path(selector).resolve())
        if requested in paths:
            paths.remove(requested); paths.insert(0, requested)
        failures = []
        for name in paths:
            try:
                raw = read(name)
                if digest(raw) != sha: raise ValueError('Content changed since import')
                return raw, dict(path=name, sha256=sha, aliases=paths, unavailable_aliases=failures)
            except (ValueError, OSError) as error:
                failures.append(name + ': ' + str(error))
        raise ValueError('Stale or missing raw evidence; reimport. ' + '; '.join(failures))

    def json(self, selector):
        raw, source = self.source(selector)
        return parse(raw), source

    def describe(self, selector):
        raw, source = self.source(selector)
        info, _ = projection(raw, Path(source['path']).suffix)
        return info, source

    def history(self, search='', limit=20, offset=0):
        matches = []
        for row in self.db.execute('SELECT * FROM documents ORDER BY sha'):
            paths = [r[0] for r in self.db.execute('SELECT path FROM paths WHERE sha=? ORDER BY path', (row['sha'],))]
            if search.casefold() in (' '.join(paths) + row['info']).casefold():
                matches.append((row, paths))
        results = []
        for row, paths in matches[offset:offset+limit]:
            try:
                info, source = self.describe(row['sha']); stale = False
            except ValueError:
                source = dict(sha256=row['sha'], aliases=paths); stale = True; info = parse(row['info'])
            results.append(dict(id=row['sha'], **info, stale=stale, sources=[source]))
        return dict(results=results, total=len(matches), offset=offset,
                    next_offset=offset+limit if offset+limit < len(matches) else None,
                    limitations=['Status is copied from each report, never inferred from a filename or passing subphase.',
                                 'History preserves report and authored-ledger claims; use compare to revalidate their timing dependencies.',
                                 'Identical raw copies share one ID; different summaries are not pooled as repetitions.'])
