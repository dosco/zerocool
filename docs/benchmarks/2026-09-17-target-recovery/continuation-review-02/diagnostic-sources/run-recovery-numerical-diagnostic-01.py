"""Numerical diagnostics only; never substitutes for clean release qualification."""
import sys
from pathlib import Path

ROOT=Path('/repo')
SNAPSHOT=ROOT/'.cache/experiment-sources/target-recovery-06'
sys.path.insert(0,str(SNAPSHOT/'scripts/qwen'))
import trial_recovery as trial
from qualification_evidence import ResourceBlocked,save,sha

out=ROOT/'docs/benchmarks/2026-09-17-target-recovery/numerical-diagnostic-01'
prior=ROOT/'docs/benchmarks/2026-09-17-target-recovery/trial-06'
old=trial.read(prior/'summary.json')
recipe=SNAPSHOT/'scripts/qwen/recipes/target_recovery.json'
cases=trial.correctness_cases()
exp=trial.Experiment(out,'mtp_target_recovery_numerical_diagnostic_v1',
    [trial.configuration(4,expert_slots=1460)],cases,1800)
with exp:
    cfg,host=trial.setup(exp,old['build_directory'],recipe)
    trial.prerequisite(exp,prior/'fixtures','fixture_validated')
    trial.freeze(exp,[Path(__file__).resolve()])
    exp.report.update(performance_measurement=False,resource_qualified=False,qualification_eligible=False,
        diagnostic_bounds=dict(physical_bytes=12*1024**3,compression_peak_bytes=512*1024**2,
                               observed_swap_growth_allowed=False),
        limitations=['Exact numerical evidence only; compressed runs do not qualify latency or the clean correctness stage.',
                     'Every release/performance gate remains unchanged.'],pairs=[],samples=[])
    for case,work in enumerate(cases):
        input_path=out/f'case-{case}.json';save(input_path,work);trial.freeze(exp,[input_path]);values={}
        for arm in (trial.ARMS if case%2==0 else trial.ARMS[::-1]):
            stem=f'case-{case}-{arm}';trial.host_check(exp,host,stem)
            exp.env['FREELLM_TARGET_RECOVERY']=arm
            exp.command([cfg['binary'],exp.model,trial.PREPARED,input_path,out/(stem+'.json'),'fast-validate'],
                        stem,limit=300,validation=True)
            raw=trial.read(out/(stem+'.json'));observed=trial.observe(raw,work,sha(input_path),True)
            memory=[raw['before_load'],raw['before']['process'],
                *[c[k] for c in raw['cycles'] for k in ('memory_before','memory_after')],
                raw['after']['process'],raw['after_destroy']]
            peak=max(m['compressed_peak_bytes'] for m in memory)
            swap_growth=any(b['system_swap_used_bytes']>a['system_swap_used_bytes'] for a,b in zip(memory,memory[1:]))
            exp.report['samples'].append(dict(case=case,arm=arm,source=stem+'.json',sha256=sha(out/(stem+'.json')),
                strict_clean_memory=observed['clean_memory'],clean_host=observed['clean_host'],
                peak_physical_bytes=observed['peak_physical_bytes'],compression_peak_bytes=peak,
                observed_swap_growth=swap_growth,generated_tokens=observed['generated_tokens']))
            exp.persist()
            if not observed['clean_host'] or observed['peak_physical_bytes']>12*1024**3 or peak>512*1024**2 or swap_growth:
                raise ResourceBlocked('Numerical diagnostic exceeded its physical/host/compression/no-swap-growth bounds')
            values[arm]=raw
        compared=trial.comparison(*(values[a] for a in trial.ARMS))
        exp.report['pairs'].append(dict(case=case,name=work['name'],**{k:v for k,v in compared.items()
            if k not in ('ratio','control_tps','candidate_tps')}))
        exp.persist()
    exp.report.update(status='numerical_diagnostic_complete',checked_prefixes=[1,2,3,4],eos_checked=True)
