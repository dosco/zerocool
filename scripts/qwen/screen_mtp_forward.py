#!/usr/bin/env python3
"""Small staged trained-MTP correctness and actual-cost screen; no promotion."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

import build_mtp_forward as builder
from benchmark_host import build_probe, preflight
from combined_q4 import freeze
from perfect_draft import source_input, configuration
from prepare_mtp import verify as verify_artifact, rounded_bf16
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
from stage200 import Experiment
from cache_residency import require

ROOT=Path(__file__).resolve().parents[2]
PREPARED=ROOT/'.cache/prepared/mtp-q4-q8-v1'
BASE=ROOT/'docs/benchmarks/2026-09-16-mtp-forward'


def clean(raw):
    memory=[raw['before_load'],raw['after_destroy']]
    if raw['mode']=='fixture':memory.append(raw['after'])
    else:
        memory += [raw[k]['process'] for k in ('before','after')]
        memory += [c[k] for c in raw['cycles'] for k in ('memory_before','memory_after')]
    fields=('physical_footprint_bytes','physical_footprint_peak_bytes','compressed_bytes','compressed_peak_bytes','decompressions','system_swap_used_bytes')
    require(all(type(m.get(k)) is int and m[k]>=0 for m in memory for k in fields),'missing MTP memory observation')
    limit=3*1024**3 if raw['mode']=='fixture' else 12*1024**3
    require(all(0<m['physical_footprint_bytes']<=m['physical_footprint_peak_bytes']<=limit for m in memory),'MTP process exceeded memory budget')
    host=[raw[k] for k in ('host_before','host_after')]
    return dict(clean_memory=all(m['compressed_bytes']==m['compressed_peak_bytes']==0 for m in memory) and
        len({m['decompressions'] for m in memory})==len({m['system_swap_used_bytes'] for m in memory})==1,
        clean_host=all(h['thermal_state']==0 and h['low_power_mode'] is False and h['power_source']=='AC Power' for h in host),
        peak_physical_bytes=max(m['physical_footprint_peak_bytes'] for m in memory))


def compare_joint(serial,candidate):
    require(serial.get('complete') is True and candidate.get('complete') is True and
        serial['mode']=='serial' and candidate['mode']=='timing' and
        serial['input_sha256']==candidate['input_sha256'] and serial['draft_manifest_sha256']==candidate['draft_manifest_sha256'] and
        serial['admission']==candidate['admission'] and serial['generated_tokens']==candidate['generated_tokens']==16,
        'incompatible or incomplete joint comparison')
    require(serial['final_target_state']==candidate['final_target_state'],'MTP final target state differs from serial')
    for r in (serial,candidate):
        require(r['decode_wall_ns']>0 and r['decode_wall_ns']==sum(c['wall_ns'] for c in r['cycles']) and
            sum(c['committed_tokens'] for c in r['cycles'])==16 and
            not any(c.get('forced_rejection') for c in r['cycles']),'invalid MTP measured coverage')
    return candidate['decode_wall_ns']/serial['decode_wall_ns']


def setup(exp,directory):
    cfg,proof=builder.verify(directory);require(proof['base_native_fingerprint']==exp.frozen['build'],'base native changed')
    save(exp.out/'producer.json',proof);save(exp.out/'draft-audit.json',verify_artifact(PREPARED))
    host=build_probe(ROOT/'.cache/benchmark-host'/hashlib.sha256(str(exp.out).encode()).hexdigest())
    files=[*builder.inputs(cfg),*builder.generated(cfg['output']),*cfg['objects'],cfg['binary'],
        *host['files'],PREPARED/'manifest.json',PREPARED/'dense.bin',PREPARED/'experts.bin',BASE/'protocol.md']
    freeze(exp,files);exp.env['ZEROCOOL_Q8_EXPANDED']='packed';exp.guard.check_resources(initial=True)
    return cfg,host


def host_check(exp,host,stem):
    observed=preflight(exp,host,stem)
    if observed['host']['power_source']!='AC Power':raise ResourceBlocked('AC power required')


def fixture(output,directory):
    exp=Experiment(output,'mtp_forward_fixture_v1',[],dict(kind='four_row_mtp_fixture'),240)
    with exp:
        cfg,host=setup(exp,directory)
        rng=np.random.default_rng(837);hidden=rng.normal(0,.5,(4,10240)).astype('<f4')
        hidden[:,4096:]*=3 # expose the old <=4096 RMS reduction truncation
        hidden=rounded_bf16(hidden);hidden.tofile(exp.out/'hidden.f32')
        data=dict(ids=[248044,77091,198,248045],hidden_file=str(exp.out/'hidden.f32'))
        save(exp.out/'input.json',data);freeze(exp,[exp.out/'input.json',exp.out/'hidden.f32']);host_check(exp,host,'fixture')
        exp.command([cfg['binary'],exp.model,PREPARED,exp.out/'input.json',exp.out/'native.json','fixture'],'native',limit=90,validation=True)
        raw=json.loads((exp.out/'native.json').read_text());require(raw.get('complete') is True,'incomplete MTP fixture')
        exp.report['native']=clean(raw);exp.persist()
        exp.command([sys.executable,ROOT/'scripts/qwen/reference_mtp_forward.py','--prepared',PREPARED,'--model',exp.model,
            '--input',exp.out/'input.json','--native',exp.out/'native.json','--output',exp.out/'reference.json'],'reference',limit=100)
        ref=json.loads((exp.out/'reference.json').read_text());require(ref['passed'] is True,'MTP reference failed')
        exp.report['reference']=ref
        if not all(exp.report['native'][k] for k in ('clean_memory','clean_host')):raise ResourceBlocked('MTP fixture memory or host disturbed')
        exp.report.update(status='forward_fixture_validated',acceptance_qualified=False)
    return exp.report


def joint(output,directory,fixture_source,draft_slots=128):
    verify_seal(fixture_source,sha(fixture_source/'evidence-files.json'))
    fixture_report=json.loads((fixture_source/'summary.json').read_text())
    require(fixture_report['status']=='forward_fixture_validated' and fixture_report['complete'] is True,'MTP fixture not qualified')
    require(draft_slots in (32,128),'Unsupported draft cache experiment')
    work,_=source_input();work['target_prepared']=str(ROOT/'.cache/prepared/q4-records-v1');work['draft_slots']=draft_slots
    exp=Experiment(output,'mtp_joint_screen_v1',[configuration(4,expert_slots=1460)],work,480)
    with exp:
        cfg,host=setup(exp,directory)
        require(json.loads((fixture_source/'producer.json').read_text())==json.loads((exp.out/'producer.json').read_text()),'MTP producer changed since fixture')
        exp.report['fixture_source']=str(fixture_source);exp.persist()
        for mode in ('validate','serial','timing'):
            host_check(exp,host,mode)
            exp.command([cfg['binary'],exp.model,PREPARED,exp.out/'workload.json',exp.out/(mode+'.json'),mode],mode,limit=150,validation=mode=='validate')
            raw=json.loads((exp.out/(mode+'.json')).read_text());require(raw.get('complete') is True,'incomplete MTP joint run')
            observed=clean(raw);exp.report[mode]=dict(observed,tokens_per_second=raw['tokens_per_second'],
                generated_tokens=raw['generated_tokens'],proposed_tokens=raw['proposed_tokens'],accepted_proposals=raw['accepted_proposals'])
            exp.persist()
            if mode=='validate':
                require(any(c['forced_rejection'] and c['accepted_proposals']==0 for c in raw['cycles']),'no forced joint rejection')
                real=exp.out/'real-fixture';real.mkdir();freeze(exp,[exp.out/'real-hidden.f32',exp.out/'real-input.json'])
                exp.command([cfg['binary'],exp.model,PREPARED,exp.out/'real-input.json',real/'native.json','fixture'],
                    'real-fixture-native',limit=90,validation=True)
                real_raw=json.loads((real/'native.json').read_text());real_clean=clean(real_raw)
                exp.report['real_fixture']=real_clean;exp.persist()
                exp.command([sys.executable,ROOT/'scripts/qwen/reference_mtp_forward.py','--prepared',PREPARED,'--model',exp.model,
                    '--input',exp.out/'real-input.json','--native',real/'native.json','--output',real/'reference.json'],
                    'real-fixture-reference',limit=100)
                if not real_clean['clean_memory'] or not real_clean['clean_host']:raise ResourceBlocked('Real MTP fixture memory disturbed')
            # A disturbed target process can supply verified fixture bytes to a
            # separate small correctness run. Its timing never passes this gate.
            if not observed['clean_memory'] or not observed['clean_host']:raise ResourceBlocked('MTP joint memory or host disturbed')
        serial=json.loads((exp.out/'serial.json').read_text());candidate=json.loads((exp.out/'timing.json').read_text())
        ratio=compare_joint(serial,candidate)
        exp.report.update(status='promising_short_screen' if ratio<1 else 'slower_than_serial',candidate_to_serial_latency_ratio=ratio,
            throughput_target_met=candidate['tokens_per_second']>=5,actual_draft_cost_included=True,
            acceptance_qualified=False,limitations=['Single short coding continuation; one timing pair is a screen, not a confidence claim.',
                'Greedy only. Long context, full coding quality, sustained sessions and production sampling remain unqualified.'])
    return exp.report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['fixture','joint']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--build',type=Path,required=True);p.add_argument('--fixture',type=Path)
    p.add_argument('--draft-slots',type=int,choices=[32,128],default=128);a=p.parse_args()
    if a.mode=='fixture':fixture(a.output,a.build)
    else:
        require(a.fixture is not None,'--fixture is required');joint(a.output,a.build,a.fixture,a.draft_slots)
