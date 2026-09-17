#!/usr/bin/env python3
"""Two alternating normal-request pairs for fewer single-token submissions."""
import argparse
import statistics
from pathlib import Path

from capacity_experiment import decide as timing_decision
from capture_routes import load
from qualification_evidence import ResourceBlocked, save, sha
from screen_cache import validate_request
from stage200 import Experiment, ORDER, SOURCE, clean_memory, configuration


def configs():
    # The preceding 1460-slot diagnostic compressed. Keep both arms at the
    # same smaller allocation, and reject a disturbed run before proceeding.
    return [dict(configuration(1072,name=name),decode_submission=mode)
            for name,mode in (('control','immediate'),('candidate','coalesced'))]


def observe(raw,identity,config,work,expected):
    result=validate_request(raw,identity,config,work,expected)
    for row,summary in zip(raw['runs'],result):
        a,b=(row['phases']['decode'][k]['metal'] for k in ('before','after'))
        counts={k:b['kernel_dispatches'].get(k,0)-a['kernel_dispatches'].get(k,0) for k in b['kernel_dispatches']}
        if expected.setdefault('dispatches_'+row['name'],counts)!=counts:raise ValueError('Arithmetic dispatch counts changed')
        summary['submissions_per_token']=(b['submissions']-a['submissions'])/32
        if config['decode_submission']=='coalesced' and summary['submissions_per_token']>160:
            raise ValueError('Expected submission reduction did not execute')
        for state in (row['before'],row['after']):
            if state['metal']['live_command_groups']!=0 or state['metal']['peak_command_groups']>2:
                raise ValueError('Unbounded or undrained GPU work')
    return result


def decide(rows):
    result=timing_decision(rows,2)
    by={(r['pair'],r['configuration']):r for r in rows}
    savings=[[(by[p,'control']['requests'][i]['decode_wall_ms']-by[p,'candidate']['requests'][i]['decode_wall_ms'])/32
              for p in range(2)] for i in range(2)]
    advance=(all(r.get('clean_memory') is True for r in rows) and result['advance_to_confirmation'] and all(v>0 for phase in savings for v in phase) and
             all(statistics.median(phase)>=20 for phase in savings))
    return dict(result,status='promising' if advance else 'improvement_not_demonstrated',
        advance_to_confirmation=advance,decode_savings_ms_per_token=savings,
        minimum_median_saving_ms=20,full_model_state_qualified=False)


def run(out):
    work=load(SOURCE/'workload.json');cs=configs();exp=Experiment(out,'coalesced_decode_screen_v1',cs,work,600)
    with exp:
        exp.guard.check_resources(initial=True)
        expected={r['name']:r['output_token_ids'] for r in load(SOURCE/'pair-0-control.json')['runs']}
        for pair,arm in ORDER:
            config=cs[arm=='candidate'];stem=f'pair-{pair}-{arm}';raw=exp.bench(config,stem)
            requests=observe(raw,exp.frozen,config,work,expected)
            exp.report['measurements'].append(dict(pair=pair,configuration=arm,requests=requests,
                clean_memory=clean_memory(raw),source=stem+'.json',sha256=sha(exp.out/(stem+'.json'))))
            exp.persist()
            if not clean_memory(raw):raise ResourceBlocked('Native process compressed during the screen; preserve this run and stop comparison')
        exp.report.update(decide(exp.report['measurements']))
    return 0 if exp.report['complete'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(p.parse_args().output))
