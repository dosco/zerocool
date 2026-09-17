"""Pause only this wrapper's native child for bounded heap attribution."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

report = Path(sys.argv[5])
env = dict(os.environ, MallocStackLogging='1')
process = subprocess.Popen(sys.argv[1:], env=env)
captures = []
try:
    deadline = time.monotonic() + 190
    while process.poll() is None:
        if time.monotonic() > deadline:
            raise TimeoutError('Allocation diagnostic native deadline exceeded')
        try:
            phase = json.loads(report.with_suffix('.progress.jsonl').read_text().splitlines()[-1])
        except (OSError, ValueError, IndexError):
            phase = {}
        if phase.get('phase') == 'draft_verify' and not captures:
            os.kill(process.pid, signal.SIGSTOP)
            try:
                commands = [
                    ('vmmap', ['/usr/bin/vmmap', '-pages', '-w', '-noCoalesce', str(process.pid)]),
                    ('heap', ['/usr/bin/heap', '--noContent', '--addresses=malloc[32k-]', '-s', str(process.pid)]),
                    ('stacks', ['/usr/bin/malloc_history', str(process.pid), '-callTree', '-invert', '-ignoreThreads', '-noContent', 'malloc[32k-]']),
                ]
                for name, cmd in commands:
                    path = report.with_suffix('.' + name + '.txt')
                    started = time.monotonic_ns()
                    with path.open('x') as out:
                        result = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, timeout=30)
                    captures.append(dict(phase=phase, source=path.name, pid=process.pid,
                        returncode=result.returncode, capture_ns=time.monotonic_ns()-started, command=cmd))
                    report.with_suffix('.observer.json').write_text(json.dumps(dict(
                        kind='owned_process_heap_observer_v1', performance_measurement=False,
                        stack_logging=True, captures=captures))+'\n')
                    if result.returncode != 0:
                        raise RuntimeError(name+' allocation observer failed')
            finally:
                if process.poll() is None:
                    os.kill(process.pid, signal.SIGCONT)
        time.sleep(.02)
    raise SystemExit(process.wait())
finally:
    if process.poll() is None:
        os.kill(process.pid, signal.SIGCONT)
        for sig, seconds in ((signal.SIGINT,10),(signal.SIGTERM,5),(signal.SIGKILL,5)):
            if process.poll() is not None: break
            process.send_signal(sig)
            try: process.wait(timeout=seconds)
            except subprocess.TimeoutExpired: pass
