#!/usr/bin/env python3
"""One unchanged normal request plus bounded timestamp capture toward 200ms/token."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from benchmark_exact import config_args,inspect_admission
from capture_routes import load
from confirm_route_selection import revalidate as confirm,screen_files
from decode_timeline import summarize
from qualification_evidence import EvidenceGuard,ResourceBlocked,identity,save,seal,sha,import_sealed
from screen_route_selection import configs
from screen_cache import validate_request

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'docs/benchmarks/2026-09-12-route-five-pairs/raw'


def run(out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    report=dict(kind='decode_target_capture_v1',complete=False,status='running',phase='prepare',
        normal_request_latency_qualified=False,production_promoted=False,time_limit_seconds=300,
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    def remaining():
        left=300-(time.monotonic()-start)
        if left<=0:raise subprocess.TimeoutExpired('decode-target',300)
        return left
    try:
        source=out/'baseline';import_sealed(SOURCE,source,sha(SOURCE/'evidence-files.json'))
        prior=load(source/'summary.json');_,originals=screen_files(source/'screen')
        for m in prior['measurements']:
            path=source/m['source']
            if sha(path)!=m['sha256']:raise ValueError('Changed baseline measurement')
            originals[m['sha256']]=path
        if not confirm(prior,lambda h:load(originals[h]))['candidate_for_later_qualification']:
            raise ValueError('Requires confirmed selector baseline')
        config=configs()[1]
        work=[dict(load(source/'workload.json')[0],name='coding_decode_target_16',max_tokens=17)]
        save(out/'workload.json',work)
        model,prepared=ROOT/'.cache/qwen-mixed-reference',ROOT/'.cache/prepared/q4-records-v1'
        frozen=identity(ROOT,[config],model,prepared,out/'workload.json')
        if {k:frozen[k] for k in prior['identity']}!=prior['identity']:raise ValueError('Confirmed baseline identity changed')
        frozen['files'].update({str(p.resolve()):sha(p) for p in source.rglob('*') if p.is_file()})
        save(out/'identity.json',frozen);guard=EvidenceGuard(frozen,out)
        report.update(identity={k:frozen[k] for k in prior['identity']},configuration=config,workload=work)
        expected={work[0]['name']:load(source/'pair-0-candidate.json')['runs'][0]['output_token_ids'][:17]}
        common=['--model',model,'--artifact','mixed-4_8bit','--prepared',prepared,'--memory-gb','12','--context','8192',*config_args(config)]
        env=dict(os.environ)
        for k in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION'):env.pop(k,None)
        with (ROOT/'.cache/qwen-qualification.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ResourceBlocked('Another workload owns the GPU lease')
            guard.check_resources(initial=True)
            for name in ('normal','traced'):
                report['phase']=name;save(out/'summary.json',report);stem=out/name
                with stem.with_suffix('.log').open('w') as log:
                    inspect_admission(ROOT/'build/qwen/bin/freellm',common,stem,12*1024**3,512,log,guard,remaining)
                    extra=['--decode-diagnostics','--phase-profile',out/'commands.json','--dependency-trace',out/'dependencies.jsonl'] if name=='traced' else []
                    guard.run([ROOT/'build/qwen/bin/freellm','bench',*common,'--workload-file',out/'workload.json',
                        '--repetitions','1','--temperature','0','--seed','0','--bench-progress',stem.with_suffix('.progress.jsonl'),
                        '--json',stem.with_suffix('.json'),*extra],stdout=log,timeout=min(150,remaining()),env=env)
                raw=load(stem.with_suffix('.json'))
                validate_request(raw,frozen,config,work,expected,instrumented=name=='traced',output_tokens=17)
                report[name]=dict(source=stem.with_suffix('.json').name,sha256=sha(stem.with_suffix('.json')))
                print(name+' complete',flush=True)
            timeline=summarize(load(out/'traced.json'),load(out/'commands.json'),
                [json.loads(line) for line in (out/'dependencies.jsonl').read_text().splitlines()])
            save(out/'timeline.json',timeline)
            a,b=[load(out/(n+'.json'))['runs'][0] for n in ('normal','traced')]
            report.update(complete=True,status='captured',phase='finished',normal_ms_per_token=a['decode_wall_ms']/16,
                traced_ms_per_token=b['decode_wall_ms']/16,trace_to_normal_ratio=b['decode_wall_ms']/a['decode_wall_ms'],
                timeline=dict(source='timeline.json',sha256=sha(out/'timeline.json')))
    except ResourceBlocked as e:report.update(status='resource_blocked',error=str(e))
    except subprocess.TimeoutExpired:report.update(status='time_budget_exhausted')
    except KeyboardInterrupt:report.update(status='interrupted')
    except Exception as e:report.update(status='failed',error=str(e))
    finally:
        report.update(elapsed_seconds=time.monotonic()-start,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(out/'summary.json',report);seal(out)
        print(json.dumps({k:report.get(k) for k in ('status','complete','error','elapsed_seconds')}),flush=True)
    return 0 if report['complete'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(p.parse_args().output))
