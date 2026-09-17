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
from screen_mtp_recovery import equivalent

out = ROOT/'docs/benchmarks/2026-09-17-target-recovery/prewarm-01'
previous = ROOT/'docs/benchmarks/2026-09-17-target-recovery/trial-09'
old = trial.read(previous/'summary.json')
work = trial.read(previous/'correctness/case-0.json')
recipe = SNAPSHOT/'scripts/qwen/recipes/target_recovery.json'
candidate = ROOT/'.cache/recovery-prewarm-build-01'
proof = trial.read(candidate/'producer.json')
exp = Experiment(out, 'mtp_pipeline_prewarm_diagnostic_v1',
                 [trial.configuration(4, expert_slots=1460)], work, 240)
with exp:
    _, host = trial.setup(exp, old['build_directory'], recipe)
    trial.require(proof['complete'] and all(sha(p)==h for key in ('inputs','generated','objects')
                  for p,h in proof[key].items()) and sha(proof['binary'])==proof['binary_sha256'],
                  'Changed diagnostic producer')
    save(out/'producer.json', proof)
    freeze(exp, [Path(__file__), candidate/'producer.json', Path(proof['binary']),
                 *[Path(p) for key in ('inputs','generated','objects') for p in proof[key]]])
    exp.env['FREELLM_TARGET_RECOVERY'] = 'full-replay'
    exp.report.update(performance_measurement=False, compression_limit_bytes=512*1024**2,
                      controlled_change='Prepare only observed kernels; same kernel code and arithmetic', samples=[])
    values=[]
    for arm in ('all','observed'):
        trial.host_check(exp, host, arm)
        exp.env['FREELLM_PIPELINE_PREWARM'] = arm
        exp.command([proof['binary'], exp.model, trial.PREPARED, out/'workload.json', out/(arm+'.json'),
                     'fast-validate'], arm, limit=120, validation=True)
        raw=trial.read(out/(arm+'.json'));observed=trial.observe(raw,work,sha(out/'workload.json'),True)
        pipeline=raw['after']['metal']['pipeline_preparation']
        trial.require(pipeline['mode']==arm and pipeline['prepared'] and pipeline['late_creations']==0,
                      'Pipeline compilation occurred after preparation')
        memory=[raw['before_load'],raw['before']['process'],raw['after']['process'],raw['after_destroy'],
                *[c[k] for c in raw['cycles'] for k in ('memory_before','memory_after')]]
        peak=max(m['compressed_peak_bytes'] for m in memory)
        exp.report['samples'].append(dict(arm=arm,source=arm+'.json',sha256=sha(out/(arm+'.json')),
            strict_clean_memory=observed['clean_memory'],clean_host=observed['clean_host'],
            peak_physical_bytes=observed['peak_physical_bytes'],compression_peak_bytes=peak,
            pipeline_preparation=pipeline))
        exp.persist();values.append(raw)
        if not observed['clean_host'] or observed['peak_physical_bytes']>12*1024**3 or peak>512*1024**2 or len({m['system_swap_used_bytes'] for m in memory})!=1:
            raise ResourceBlocked('Diagnostic exceeded memory/host bounds')
    equivalent(*values)
    trial.require(all(values[0][k]==values[1][k] for k in ('row_logits_sha256','committed_token_ids','final_draft_state')),
                  'Pipeline preparation changed exact computation')
    exp.report.update(status='diagnostic_complete',exact_outputs_and_state=True,
                      qualifying_performance_evidence=False)
