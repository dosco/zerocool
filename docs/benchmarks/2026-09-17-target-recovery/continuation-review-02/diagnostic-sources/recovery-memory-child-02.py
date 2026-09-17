import json
import subprocess
import sys
import time
from pathlib import Path

report = Path(sys.argv[5])
captures = []
seen = set()
process = subprocess.Popen(sys.argv[1:])
while process.poll() is None:
    try:
        phase = json.loads(report.with_suffix('.progress.jsonl').read_text().splitlines()[-1])
    except (OSError, ValueError, IndexError):
        phase = {}
    name = phase.get('phase')
    if name in ('prime_target', 'draft_verify') and name not in seen:
        seen.add(name)
        path = report.with_suffix('.' + name + '.vmmap.txt')
        start = time.monotonic_ns()
        with path.open('x') as out:
            result = subprocess.run(['/usr/bin/vmmap', '-pages', '-w', '-noCoalesce', str(process.pid)],
                                    stdout=out, stderr=subprocess.STDOUT, timeout=15)
        captures.append(dict(phase=phase, source=path.name, pid=process.pid,
                             returncode=result.returncode, capture_ns=time.monotonic_ns()-start))
        report.with_suffix('.observer.json').write_text(json.dumps(dict(
            kind='owned_process_memory_observer_v2', performance_measurement=False,
            captures=captures))+'\n')
    time.sleep(.05)
raise SystemExit(process.wait())
