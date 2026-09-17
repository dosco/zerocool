"""One bounded CPU-only check of per-range file-cache invalidation."""
from pathlib import Path
import fcntl, hashlib, json, subprocess, time

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
OUT = BASE/'cache-preflight-01'
FIXTURES = ROOT/'docs/benchmarks/2026-09-13-stage200/q3-02/tuning'
PREPARED = ROOT/'.cache/prepared/q4-records-v1'
BINARY = ROOT/'build/qwen/qwen_q4_check'

def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()

def metadata(path):
    s = path.stat()
    return dict(path=str(path), size=s.st_size, mtime_ns=s.st_mtime_ns,
                ctime_ns=s.st_ctime_ns, inode=s.st_ino, device=s.st_dev)

OUT.mkdir(exist_ok=False)
manifest=json.loads((PREPARED/'manifest.json').read_text())
paths=[PREPARED/manifest['experts'][i]['file'] for i in (0,16,32,47)]
sources=[BINARY, Path(__file__), ROOT/'scripts/qwen/check_q4.cpp',
         ROOT/'scripts/qwen/replay_q4_reads.hpp', ROOT/'scripts/qwen/replay_q4.hpp',
         FIXTURES/'manifest.json', PREPARED/'manifest.json']
identity={str(p):sha(p) for p in sources}
before=[metadata(p) for p in paths]
report=dict(complete=False, kind='q4_file_cache_preflight_driver_v1', files=identity,
            prepared_files_before=before, process_limit_seconds=30, gpu_used=False)
(OUT/'identity.json').write_text(json.dumps(report, indent=2)+'\n')
with (ROOT/'.cache/qwen-qualification.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic()
    try:
        with (OUT/'probe.log').open('w') as log:
            subprocess.run([BINARY,FIXTURES,OUT/'probe.json','cache_probe',PREPARED],
                           stdout=log,stderr=subprocess.STDOUT,timeout=30,check=True)
        report['prepared_files_after']=[metadata(p) for p in paths]
        assert report['prepared_files_after']==before, 'Prepared source metadata changed'
        assert {str(p):sha(p) for p in sources}==identity, 'Probe sources changed'
        report.update(complete=True, status='measured')
    except Exception as error:
        report.update(status='failed',error=str(error))
    report['elapsed_seconds']=time.monotonic()-start
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
(OUT/'evidence-files.json').write_text(json.dumps({p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file()},indent=2)+'\n')
print(json.dumps(report))
if not report['complete']:raise SystemExit(2)
