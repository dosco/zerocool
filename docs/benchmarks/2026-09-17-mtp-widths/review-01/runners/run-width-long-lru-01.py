"""Fresh 128-token LRU follow-up for the surviving width-one screen."""
import math
import sys
from pathlib import Path
ROOT=Path('/repo')
sys.path.insert(0,str(ROOT/'scripts/qwen'))
import screen_mtp_widths as w
from mtp_evidence import compare
from evidence_index import Index

prior=ROOT/'docs/benchmarks/2026-09-17-mtp-widths/screen-width-1-01'
source=ROOT/'docs/benchmarks/2026-09-17-mtp-widths/validation-03'
out=ROOT/'docs/benchmarks/2026-09-17-mtp-widths/long-lru-01'
build=ROOT/'.cache/mtp-widths-build-04'
w.verify_seal(prior,w.sha(prior/'evidence-files.json'))
short=w.read(prior/'summary.json')
w.require(short['kind']=='mtp_width_screen_v1' and short['control_width']==4 and short['candidate_width']==1,
          'Different exploratory width source')
short_work=w.read(prior/'case-0.json');values={}
for width in (4,1):
    path=prior/f'pair-0-width-{width}.json'
    row=next(v for v in short['samples'] if v['source']==path.name)
    w.require(w.sha(path)==row['sha256'],'Changed short raw report')
    raw=w.read(path);result=w.observed(raw,short_work,w.sha(prior/'case-0.json'),False)
    w.require(result['clean_host'] and result['clean_memory'],'Short first pair was not clean')
    values[width]=raw
first=w.compare_widths(values[4],values[1])
w.require(first['ratio']<=.97,'Short first pair does not justify a longer diagnostic screen')
# The original reverse pair remains resource-blocked. This is a new workload
# length and fresh preliminary comparison, not completion of that short stage.
work=next(v for v in w.workloads(ROOT/'.cache/qwen-mixed-reference',128) if v['name']=='lru_cache')
exp=w.Experiment(out,'mtp_width_screen_v1',[w.configuration(4,expert_slots=1460)],work,840)
with exp:
    cfg,host=w.setup(exp,build)
    proof=w.read(exp.out/'producer.json')
    w.require(w.read(prior/'producer.json')==proof,'Short screen producer changed')
    clean,files=w.validation_prerequisite(source,proof)
    w.require(clean,'Complete clean numerical validation is required')
    w.freeze(exp,[*files,Path(__file__),prior/'summary.json',prior/'evidence-files.json',
                  *[prior/p['source'] for p in w.read(prior/'summary.json')['samples']]])
    exp.report.update(stage='long-lru-diagnostic',performance_measurement=True,preliminary=True,
        advancement_allowed=False,full_clean_correctness=True,controlled_change='requested_width',
        recovery='full-replay',control_width=4,candidate_width=1,samples=[],pairs=[],
        historical_timing_reused=False,first_pair_required_gain=.03,geometric_required_gain=.03,
        prerequisite=str(prior/'summary.json'),short_stage_qualified=False,
        investigation_basis='One complete clean short pair; its later reverse remains resource-blocked',
        limitations=['One coding workload, 128 output tokens and two fresh alternating pairs.',
                     'The short reverse remains unqualified. Other coding cases, five-pair confidence, 2K/4K/7K and sustained use remain unqualified.',
                     'Width one includes draft-state maintenance; all cache and checkpoint budgets are equal.'])
    w.save(exp.out/'case-0.json',work);w.freeze(exp,[exp.out/'case-0.json'])
    for pair,order in enumerate(((4,1),(1,4))):
        values={}
        for width in order:
            raw,result=w.sample(exp,cfg,host,width,work,f'pair-{pair}-width-{width}',False)
            exp.report['samples'][-1].update(case=0,pair=pair,arm=str(width));exp.persist()
            if not result['clean_memory'] or not result['clean_host']:
                raise w.ResourceBlocked(w.resource_failure(raw,'Long width timing resources disturbed'))
            values[width]=raw
        result=w.compare_widths(values[4],values[1])
        w.require(values[4]['input_sha256']==values[1]['input_sha256'],'Different long timing workload')
        exp.report['pairs'].append(dict(case=0,pair=pair,**result));exp.persist()
        ratios=[p['ratio'] for p in exp.report['pairs']]
        if ratios[0]>.97 or any(r>=1 for r in ratios) or (len(ratios)==2 and math.prod(ratios)**.5>.97):
            exp.report.update(status='insufficient_long_gain');break
    else:exp.report.update(status='promising_long_lru_diagnostic_screen')
