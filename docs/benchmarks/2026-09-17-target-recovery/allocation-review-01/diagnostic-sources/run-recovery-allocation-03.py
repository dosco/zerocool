import json
import sys
from pathlib import Path

ROOT = Path('/repo')
SNAPSHOT = ROOT/'.cache/experiment-sources/target-recovery-06'
sys.path.insert(0, str(SNAPSHOT/'scripts/qwen'))
import trial_recovery as trial
from combined_q4 import freeze
from qualification_evidence import ResourceBlocked, save, sha
from stage200 import Experiment
from trace_mtp_memory import allocation_categories

out = ROOT/'docs/benchmarks/2026-09-17-target-recovery/allocation-diagnostic-03'
previous = ROOT/'docs/benchmarks/2026-09-17-target-recovery/trial-09'
old = trial.read(previous/'summary.json')
work = trial.read(previous/'correctness/case-0.json')
recipe = SNAPSHOT/'scripts/qwen/recipes/target_recovery.json'
wrapper = ROOT/'.cache/recovery-allocation-child-03.py'
exp = Experiment(out, 'mtp_recovery_allocation_diagnostic_v1',
                 [trial.configuration(4, expert_slots=1460)], work, 280)
with exp:
    cfg, host = trial.setup(exp, old['build_directory'], recipe)
    reference = previous/'correctness/case-0-pair-0-full-replay.json'
    freeze(exp, [Path(__file__), wrapper, reference])
    trial.host_check(exp, host, 'diagnostic')
    exp.env['FREELLM_TARGET_RECOVERY'] = 'full-replay'
    exp.report.update(performance_measurement=False, observer='Malloc stack logging and heap/VM inspection of owned paused child; diagnostic only', allocation_ownership_identified=False,
                      compression_limit_bytes=512*1024**2, reference=str(reference))
    exp.command([sys.executable, wrapper, cfg['binary'], exp.model, trial.PREPARED,
                 out/'workload.json', out/'native.json', 'fast-validate'], 'native',
                limit=240, validation=True)
    raw = trial.read(out/'native.json')
    observed = trial.observe(raw, work, sha(out/'workload.json'), True)
    baseline = trial.read(reference)
    fields = ('prime_logits_sha256', 'committed_token_ids', 'row_logits_sha256', 'boundaries',
              'final_target_state', 'final_draft_state', 'next_id', 'stop_reason', 'admission')
    trial.require(all(raw[k] == baseline[k] for k in fields), 'Diagnostic changed numerical outputs')
    observer = trial.read(out/'native.observer.json')
    trial.require(len(observer['captures']) == 3 and all(c['returncode'] == 0 for c in observer['captures']),
                  'Incomplete allocation observer')
    for capture in observer['captures']:
        capture['sha256'] = sha(out/capture['source'])
    save(out/'observer-audit.json', observer)
    save(out/'allocation-map.json', allocation_categories((out/'native.vmmap.txt').read_text()))
    memory = [raw['before_load'], raw['before']['process'], raw['after']['process'], raw['after_destroy'],
              *[c[k] for c in raw['cycles'] for k in ('memory_before', 'memory_after')]]
    peak = max(m['compressed_peak_bytes'] for m in memory)
    exp.report.update(status='diagnostic_complete', observations=observed, compression_peak_bytes=peak,
                      exact_reference=True, maps_source='allocation-map.json', source=sha(out/'native.json'))
    if not observed['clean_host'] or observed['peak_physical_bytes'] > 12*1024**3 or peak > 512*1024**2 or len({m['system_swap_used_bytes'] for m in memory}) != 1:
        raise ResourceBlocked('Diagnostic memory or host bounds exceeded')
