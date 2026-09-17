"""Fresh fixed-width comparison on the two remaining registered coding prompts."""
import math
import sys
from pathlib import Path

ROOT = Path('/repo')
sys.path.insert(0, str(ROOT/'scripts/qwen'))
import screen_mtp_widths as w
from evidence_index import Index
from mtp_evidence import compare

base = ROOT/'docs/benchmarks/2026-09-17-mtp-widths'
build = ROOT/'.cache/mtp-widths-build-04'
validation = base/'validation-04'
previous = base/'long-lru-01'
out = base/'other-coding-01'

# Python evidence checking changed after the previous experiment. Recheck all
# sealed, clean numerical runs against today's checker, without reusing timing.
result = w.validate(validation, build, resume=base/'validation-03')
w.require(result['complete'] and result['resource_qualified'], 'Clean numerical validation is required')
index = Index(ROOT/'.cache/evidence/index.sqlite')
try:
    checked = compare(index, str(previous/'summary.json'), '4', '1', ['requested_width'])
finally:
    index.close()
w.require(checked['full_correctness_stage_passed'] and len(checked['pairs'])==2 and
          all(p['ratio']<1 for p in checked['pairs']) and
          checked['cases'][0]['geometric_mean_ratio']<=.97, 'A clean promising LRU screen is required')

cases = [v for v in w.workloads(ROOT/'.cache/qwen-mixed-reference',128)
         if v['name'] in ('merge_intervals','retry_backoff')]
w.require([v['name'] for v in cases]==['merge_intervals','retry_backoff'], 'Unexpected workload inventory')
exp = w.Experiment(out,'mtp_width_screen_v1',[w.configuration(4,expert_slots=1460)],cases,1500)
with exp:
    cfg,host=w.setup(exp,build)
    proof=w.read(exp.out/'producer.json')
    w.require(w.read(previous/'producer.json')==proof, 'Previous screen used another producer')
    clean,files=w.validation_prerequisite(validation,proof)
    w.require(clean, 'Clean full-model numerical validation is required')
    w.freeze(exp,[*files,Path(__file__),*[p for p in previous.iterdir() if p.is_file()]])
    exp.report.update(stage='other-coding-diagnostic',performance_measurement=True,
        preliminary=True,advancement_allowed=False,full_clean_correctness=True,
        controlled_change='requested_width',recovery='full-replay',control_width=4,
        candidate_width=1,samples=[],pairs=[],historical_timing_reused=False,
        prerequisite=str(previous/'summary.json'),case_results=[],
        limitations=['Two fresh alternating pairs per 128-token coding workload; no confidence bound.',
                     'Workloads remain separate. Previous LRU timing is not pooled into this stage.',
                     'This measures workload crossover and cannot qualify an adaptive policy or production.',
                     'Five fresh pairs, long-context and sustained-session qualification remain outstanding.'])
    for case,work in enumerate(cases):
        w.save(exp.out/f'case-{case}.json',work)
        w.freeze(exp,[exp.out/f'case-{case}.json'])
        for pair in range(2):
            order=(4,1) if (case+pair)%2==0 else (1,4)
            values={}
            for width in order:
                stem=f'case-{case}-pair-{pair}-width-{width}'
                raw,result=w.sample(exp,cfg,host,width,work,stem,False)
                exp.report['samples'][-1].update(case=case,pair=pair,arm=str(width))
                exp.persist()
                if not result['clean_memory'] or not result['clean_host']:
                    raise w.ResourceBlocked(w.resource_failure(raw,'Other coding width resources disturbed'))
                values[width]=raw
            compared=w.compare_widths(values[4],values[1])
            w.require(values[4]['input_sha256']==values[1]['input_sha256'],'Different paired workload')
            exp.report['pairs'].append(dict(case=case,pair=pair,**compared))
            exp.persist()
        ratios=[p['ratio'] for p in exp.report['pairs'] if p['case']==case]
        ratio=math.exp(sum(map(math.log,ratios))/len(ratios))
        exp.report['case_results'].append(dict(name=work['name'],geometric_mean_ratio=ratio,
            candidate_regresses=ratio>1.02,candidate_passes_three_percent_gate=ratio<=.97 and all(r<1 for r in ratios)))
        exp.persist()
    exp.report.update(status='completed_other_coding_diagnostic_screen')
