#!/usr/bin/env python3
"""Paired core-plus-expert residency screen with immediate expert execution."""
import argparse
from pathlib import Path

from capture_routes import load
from qualification_evidence import ResourceBlocked, sha
from screen_cache import validate_request
from screen_coalesced import decide
from stage200 import Experiment, ORDER, SOURCE, clean_memory, configuration


def configs():
    return [dict(configuration(1072,name=name),decode_submission='immediate',residency=mode)
            for name,mode in (('control','off'),('candidate','core-cache'))]


def observe(raw,identity,config,work,expected):
    results=validate_request(raw,identity,config,work,expected)
    for r in raw['runs']:
        a,b=(r['phases']['decode'][k]['metal'] for k in ('before','after'))
        counts={k:b['kernel_dispatches'].get(k,0)-a['kernel_dispatches'].get(k,0) for k in b['kernel_dispatches']}
        if expected.setdefault('dispatches_'+r['name'],counts)!=counts:raise ValueError('Arithmetic dispatch counts changed')
        s=r['after'];residency=s['metal']['residency']
        if residency['mode']!=config['residency'] or residency['pending_retirements']!=0:
            raise ValueError('Requested residency did not execute or was not drained')
        if config['residency']=='core-cache':
            if residency['bytes_by_class'].get('expert')!=s['memory_plan']['expert_bytes']:
                raise ValueError('Expert cache is not fully enrolled')
        elif residency['registered_bytes']!=0:raise ValueError('Control has additional residency')
    return results


def run(out):
    work=load(SOURCE/'workload.json');cs=configs();exp=Experiment(out,'expert_residency_screen_v1',cs,work,600)
    with exp:
        exp.guard.check_resources(initial=True)
        expected={r['name']:r['output_token_ids'] for r in load(SOURCE/'pair-0-control.json')['runs']}
        for pair,arm in ORDER:
            config=cs[arm=='candidate'];stem=f'pair-{pair}-{arm}';raw=exp.bench(config,stem)
            requests=observe(raw,exp.frozen,config,work,expected)
            exp.report['measurements'].append(dict(pair=pair,configuration=arm,requests=requests,
                clean_memory=clean_memory(raw),source=stem+'.json',sha256=sha(exp.out/(stem+'.json'))));exp.persist()
            if not clean_memory(raw):raise ResourceBlocked('Native process compressed during residency screen; stop comparison')
        exp.report.update(decide(exp.report['measurements']))
    return 0 if exp.report['complete'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(p.parse_args().output))
