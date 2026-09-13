#!/usr/bin/env python3
"""One bounded control/reuse memory diagnostic; never qualifies request latency."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from benchmark_exact import config_args,inspect_admission
from capture_routes import load
from diagnose_decode_startup import analyze_steps
from measure_buffer_costs import CLASSES,FIELDS
from qualification_evidence import EvidenceGuard,ResourceBlocked,identity,save,seal,sha
from screen_cache import validate_request
from screen_decode_scratch import configs

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'docs/benchmarks/2026-09-13-scratch-confirmation/settled-screen'
LIFECYCLE=('before_model_load','model_ready','request_end','request_end','users_drained',
           'session_destroyed','session_users_drained','model_destroyed')


def expected_boundaries(requests,delayed=False):
    result=[(True,dict(event=e)) for e in LIFECYCLE[:2]]
    for request in requests:
        offset=request['reused_tokens'];length=request['prompt_tokens']-offset
        if type(offset) is not int or offset<0 or not 1<length<=512:
            raise ValueError('Diagnostic requires one bounded multi-token ingestion per request')
        def add(event,layer=-1,tokens=length,at=offset):
            result.append((False,dict(event=event,phase='prefill' if not offset else 'append',layer=layer,tokens=tokens,offset=at)))
        add('forward_begin')
        for layer in range(48):
            add('layer_begin',layer)
            if length>128:
                for at in range(0,length,128):add('microchunk_end',layer,min(128,length-at),offset+at)
            add('attention_encoded',layer);add('layer_encoded',layer)
        add('forward_end');result.append((True,dict(event='request_end')))
    result.extend((True,dict(event=e)) for e in LIFECYCLE[4:])
    if delayed:result.extend((True,dict(event=f'after_destroy_{ms}ms')) for ms in (250,1000))
    return result


def analyze_trace(rows,coverage,requests,delayed=False):
    expected=expected_boundaries(requests,delayed);detail=sum(not life for life,_ in expected)
    if (coverage.get('kind')!='memory_trace_coverage_v1' or coverage.get('limit')!=512 or
        coverage.get('observed')!=detail or coverage.get('captured')!=detail or coverage.get('omitted')!=0 or
        coverage.get('lifecycle_records')!=(10 if delayed else 8) or coverage.get('records')!=len(rows) or len(rows)!=len(expected)):
        raise ValueError('Incomplete memory trace coverage or cleanup')
    previous=-1;points=[]
    for sequence,(row,(life,where)) in enumerate(zip(rows,expected),1):
        if (row.get('kind')!='memory_boundary_v1' or row.get('sequence')!=sequence or row.get('where')!=where or
            row.get('lifecycle') is not life or type(row.get('monotonic_ns')) is not int or row['monotonic_ns']<=previous or
            type(row.get('sample_ns')) is not int or row['sample_ns']<0):
            raise ValueError('Memory event identity/order changed')
        previous=row['monotonic_ns'];metal=row.get('metal');held=None
        if where['event'] in ('before_model_load','model_destroyed','after_destroy_250ms','after_destroy_1000ms'):
            if metal is not None:raise ValueError('No Metal owner exists at this lifecycle boundary')
        else:
            if not isinstance(metal,dict):raise ValueError('Missing live Metal observation')
            costs=metal.get('buffer_costs',{});classes=costs.get('classes',{})
            if costs.get('kind')!='buffer_costs_v1' or set(classes)!=CLASSES:
                raise ValueError('Missing per-class ownership counters')
            held={}
            for name,c in classes.items():
                if set(c)!=FIELDS or any(type(v) is not int or v<0 for v in c.values()):raise ValueError('Invalid ownership counter')
                n=c['allocated_bytes']-c['owner_released_bytes']
                if n<0:raise ValueError('Ownership counter reset')
                held[name]=n
            for key in ('live_buffer_bytes','peak_buffer_bytes','device_allocated_bytes','encoded_buffer_references','live_command_groups'):
                if type(metal.get(key)) is not int or metal[key]<0:raise ValueError('Missing allocation/ownership gauge')
            if metal['live_command_groups']>2:raise ValueError('Unbounded command groups')
            if where['event'] in ('users_drained','session_users_drained') and (
                metal['live_command_groups'] or metal['encoded_buffer_references']):raise ValueError('Outstanding users at drain boundary')
        process=row.get('process',{})
        points.append(dict(sequence=sequence,where=where,monotonic_ns=row['monotonic_ns'],process=process,
            engine_owner_held_bytes=held,metal_live_bytes=metal.get('live_buffer_bytes') if metal else None,
            metal_peak_bytes=metal.get('peak_buffer_bytes') if metal else None,
            device_allocated_bytes=metal.get('device_allocated_bytes') if metal else None))
    def first_compressed(phase):
        eligible=[(i,r) for i,r in enumerate(rows) if r['where'].get('phase')==phase]
        for i,row in eligible:
            value=row['process'].get('compressed_bytes')
            if type(value) is int and value>0:
                return dict(sequence=row['sequence'],where=row['where'],compressed_bytes=value,
                    previous_boundary=rows[i-1]['where'] if i else None)
        return None
    process_fields=('physical_footprint_bytes','physical_footprint_peak_bytes','compressed_bytes','decompressions')
    if delayed:
        destroyed=rows[-3]['monotonic_ns']
        if any(rows[-2+i]['monotonic_ns']-destroyed<delay*1000000 for i,delay in enumerate((250,1000))):
            raise ValueError('Delayed cleanup observations were captured too early')
    complete=all(type(r['process'].get(k)) is int and r['process'][k]>=0 for r in rows for k in process_fields)
    peaks={k:max(r['process'][k] for r in rows) if all(type(r['process'].get(k)) is int for r in rows) else None
           for k in ('physical_footprint_bytes','physical_footprint_peak_bytes','compressed_bytes')}
    return dict(points=points,memory_values_complete=complete,peaks=peaks,
        first_observed_compression={p:first_compressed(p) for p in ('prefill','append')},
        lifecycle=[p for p in points if 'phase' not in p['where']],
        normal_request_latency_qualified=False,production_promoted=False,
        limitations=['Layer/microchunk events observe encoding boundaries; no GPU wait is added. Compression may occur between observations.',
            'Process peaks cover the process lifetime; no per-buffer compressed-page attribution.',
            'Engine owner-held bytes and device allocations are not physical residency. System VM categories overlap.',
            'Memory observation includes JSON writes and per-token diagnostics; timings cannot qualify speed.',
            'Post-destruction process footprint still includes this executable, reports and system/driver runtime state.'])


def validate(raw,events,evidence,config,workload,expected):
    if raw.get('kind') not in ('ingestion_memory_diagnostic_v1','ingestion_memory_diagnostic_v2') or raw.get('status')!='captured':raise ValueError('Missing diagnostic result')
    summaries=validate_request(raw,evidence,config,workload,expected,instrumented=True)
    decode=[]
    for row in raw['runs']:
        steps=analyze_steps(row)
        if steps['captured_steps']!=32 or steps['omitted_steps']:raise ValueError('Missing decode samples')
        decode.append(dict(name=row['name'],**steps))
        a,b=(row['phases']['decode'][k]['metal'] for k in ('before','after'))
        counts={k:b['kernel_dispatches'].get(k,0)-a['kernel_dispatches'].get(k,0) for k in b['kernel_dispatches']}
        if expected.setdefault('dispatches_'+row['name'],counts)!=counts:raise ValueError('Arithmetic dispatch counts changed')
    return dict(trace=analyze_trace(events,raw['trace_coverage'],raw['runs'],raw['kind'].endswith('_v2')),decode=decode,requests=summaries)


def run(out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    report=dict(kind='ingestion_memory_capture_v2',complete=False,status='running',phase='prepare',measurements=[],
        normal_request_latency_qualified=False,production_promoted=False,time_limit_seconds=360,
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        limitations=['One instrumented control/reuse pair, fixed order, for memory diagnosis only.',
                     'No reference speed claim, no production promotion, and no full-logit/state qualification.'])
    def remaining():
        value=360-(time.monotonic()-started)
        if value<=0:raise subprocess.TimeoutExpired('ingestion-memory',360)
        return value
    try:
        for name in ('workload.json','pair-1-control.json'):shutil.copyfile(SOURCE/name,out/name)
        work=load(out/'workload.json');prior=load(out/'pair-1-control.json')
        expected={r['name']:r['output_token_ids'] for r in prior['runs']}
        model,prepared=ROOT/'.cache/qwen-mixed-reference',ROOT/'.cache/prepared/q4-records-v1'
        frozen=identity(ROOT,configs(),model,prepared,out/'workload.json')
        for p in (ROOT/'build/qwen/qwen_memory_check',out/'pair-1-control.json'):frozen['files'][str(p.resolve())]=sha(p)
        save(out/'identity.json',frozen);guard=EvidenceGuard(frozen,out)
        report.update(identity={k:frozen[k] for k in ('build','artifact_revision','prepared_manifest_sha256','budget_bytes','device','physical_bytes')},
            configurations=configs(),workload=work,prior_output_control=dict(source='pair-1-control.json',sha256=sha(out/'pair-1-control.json')))
        env=dict(os.environ)
        for k in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION'):env.pop(k,None)
        with (ROOT/'.cache/qwen-qualification.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ResourceBlocked('Another workload owns the GPU lease')
            guard.check_resources(initial=True)
            for c in configs():
                name=c['name'];stem=out/name;report['phase']=name;save(out/'summary.json',report);print(name,flush=True)
                common=['--model',model,'--artifact','mixed-4_8bit','--prepared',prepared,'--memory-gb','12','--context','8192',*config_args(c)]
                with stem.with_suffix('.log').open('w') as log:
                    p=inspect_admission(ROOT/'build/qwen/bin/freellm',common,stem,12*1024**3,512,log,guard,remaining)
                    if load(p)['current_admission']['expert_slots']!=1848:raise ResourceBlocked('Fixed expert capacity not admitted')
                    guard.run([ROOT/'build/qwen/qwen_memory_check',model,prepared,out/'workload.json',c['decode_scratch'],
                        stem.with_suffix('.json'),stem.with_suffix('.jsonl')],stdout=log,timeout=min(150,remaining()),env=env)
                raw=load(stem.with_suffix('.json'));events=[json.loads(line) for line in stem.with_suffix('.jsonl').read_text().splitlines()]
                if raw['trace_coverage']['bytes']!=stem.with_suffix('.jsonl').stat().st_size:raise ValueError('Trace byte coverage differs')
                result=validate(raw,events,frozen,c,work,expected);save(out/(name+'-analysis.json'),result)
                report['measurements'].append(dict(configuration=name,sources=[dict(source=p.name,sha256=sha(p)) for p in
                    (stem.with_suffix('.json'),stem.with_suffix('.jsonl'),out/(name+'-analysis.json'))]))
            report.update(complete=True,status='captured',phase='finished')
    except ResourceBlocked as e:report.update(status='resource_blocked',error=str(e))
    except subprocess.TimeoutExpired:report.update(status='time_budget_exhausted')
    except KeyboardInterrupt:report.update(status='interrupted')
    except Exception as e:report.update(status='failed',error=str(e))
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(out/'summary.json',report);seal(out);print(json.dumps({k:report.get(k) for k in ('status','complete','error')}),flush=True)
    return 0 if report['complete'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(p.parse_args().output))
