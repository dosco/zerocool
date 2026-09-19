#!/usr/bin/env python3
"""Guarded old/new API screens and retained-history soak. Raw reports are authoritative."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import threading
import time
import urllib.request

from benchmark_host import build_probe, observe
from qualification_evidence import ResourceBlocked, save, sha

GiB=1024**3
ROOT=Path(__file__).resolve().parents[2]


def require(value,message):
    if not value:raise ValueError(message)


def compare(rows,pairs):
    expected=[(i,arm) for i in range(pairs) for arm in (('old','new') if i%2==0 else ('new','old'))]
    require([(r['pair'],r['arm']) for r in rows]==expected,'Incomplete or reordered pairs')
    grouped={(r['pair'],r['arm']):r for r in rows};ratios=[]
    for i in range(pairs):
        a,b=grouped[i,'old'],grouped[i,'new']
        require(a['complete'] is True and b['complete'] is True,'Incomplete request')
        require(a['memory_plan']==b['memory_plan'],'Different admitted memory plans')
        require(a['response']['freellm']['output_token_ids']==b['response']['freellm']['output_token_ids'],'Generated tokens differ')
        require(a['response']['usage']['prompt_tokens']==b['response']['usage']['prompt_tokens'],'Different prompt tokenization')
        require(all(type(r['elapsed_ms']) in (int,float) and math.isfinite(r['elapsed_ms']) and r['elapsed_ms']>0 for r in (a,b)),'Missing or invalid request latency')
        ratios.append(b['elapsed_ms']/a['elapsed_ms'])
    return dict(ratios=ratios,median_ratio=statistics.median(ratios),
        decision='investigate_or_extend' if max(ratios)>1.03 else 'short_screen_within_3_percent',
        qualified=False,limitations=['Short paired screen; no confidence or sustained-use claim. TUI overhead is a separate measurement.'])


def clean_memory(observations):
    keys=('physical_footprint_bytes','physical_footprint_peak_bytes','compressed_bytes','compressed_peak_bytes','decompressions','system_swap_used_bytes')
    require(observations and all(all(type(m.get(k)) is int and m[k]>=0 for k in keys) for m in observations),'Missing memory observations')
    return (all(m['compressed_bytes']==m['compressed_peak_bytes']==0 and 0<m['physical_footprint_bytes']<=m['physical_footprint_peak_bytes']<=12*GiB for m in observations)
        and len({m['decompressions'] for m in observations})==1 and len({m['system_swap_used_bytes'] for m in observations})==1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['paired','soak'])
    p.add_argument('--binary',type=Path,default=ROOT/'build/qwen/bin/freellm')
    p.add_argument('--baseline',type=Path)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--artifact',choices=['4bit','mixed-4_8bit'],default='4bit')
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--workload',type=Path,required=True,help='JSON with messages, optional followups, max_tokens (default 64)')
    p.add_argument('--pairs',type=int,choices=[2,5],default=2)
    p.add_argument('--seconds',type=int,default=1200)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    require(a.mode!='paired' or a.baseline,'Paired mode requires the preserved old binary')
    require(a.seconds>=1200,'Soak must cover at least 1200 seconds')
    require(not any(k.startswith(('FREELLM_','MTL_')) for k in os.environ),'Unset experimental and Metal validation environment variables')
    work=json.loads(a.workload.read_text());require(isinstance(work.get('messages'),list) and work['messages'],'Workload needs messages')
    count=work.get('max_tokens',64);require(type(count) is int and 1<=count<=256,'Use 1..256 output tokens')
    binaries=dict(new=a.binary.resolve())
    if a.baseline:binaries['old']=a.baseline.resolve()
    model=a.model.resolve();prepared=a.prepared.resolve()
    files=[Path(__file__),a.workload.resolve(),model/'freellm-verification.json',prepared/'manifest.json',prepared/'verification.json',*binaries.values()]
    frozen={str(path):sha(path) for path in files}
    report=dict(kind='api_usability_performance_v1',complete=False,status='running',qualified=False,mode=a.mode,files=frozen,
        configuration=dict(artifact=a.artifact,memory_gib=12,context=8192,max_tokens=count,temperature=0,seed=0,thinking=False),measurements=[])
    def persist():save(out/'summary.json',report)
    def identity():require(frozen=={path:sha(path) for path in frozen},'Frozen inputs changed')
    persist();probe=build_probe(out/'host-probe')
    common=['--model',str(model),'--artifact',a.artifact,'--prepared',str(prepared),'--memory-gb','12','--context','8192']
    def host(stem,admit=False):
        path=out/(stem+'-host.json');subprocess.run([str(probe['binary']),str(path)],check=True,stdout=subprocess.DEVNULL,timeout=10)
        raw=json.loads(path.read_text());observed=observe(raw,probe['producer']['base_native_fingerprint'])
        if not observed['clean_host'] or observed['host']['power_source']!='AC Power':raise ResourceBlocked('Nominal thermal state, AC power and Low Power Mode off are required')
        if admit and raw['process']['reclaimable_bytes']<int(13.5*GiB):raise ResourceBlocked('Fixed 12GiB comparison requires 13.5GiB currently available memory')
        return raw
    def get(url,path):
        with urllib.request.urlopen(url+path,timeout=5) as f:return json.load(f)
    def launch(arm,stem):
        identity();host(stem,True)
        inspected=subprocess.run([str(binaries[arm]),'inspect',*common],text=True,capture_output=True,check=True,timeout=30)
        info=json.loads(inspected.stdout);save(out/(stem+'-inspect.json'),info)
        plan=info['current_admission'];require(plan==info['memory_plan'],'Requested budget is not fully admitted')
        # The old executable predates the private port handshake. Reserve a free
        # port briefly, then verify that the child (not an unrelated listener) owns it.
        with socket.socket() as reservation:reservation.bind(('127.0.0.1',0));port=reservation.getsockname()[1]
        log=(out/(stem+'.log')).open('w');child=subprocess.Popen([str(binaries[arm]),'serve',*common,'--port',str(port)],stdout=log,stderr=subprocess.STDOUT)
        url=f'http://127.0.0.1:{port}'
        try:
            deadline=time.monotonic()+180
            while time.monotonic()<deadline:
                require(child.poll() is None,'Server exited during startup')
                owned=subprocess.run(['lsof','-nP','-a','-p',str(child.pid),f'-iTCP:{port}','-sTCP:LISTEN'],capture_output=True).returncode==0
                if owned:
                    try:
                        if get(url,'/health')['status']=='ready':return child,log,url,plan
                    except OSError:pass
                time.sleep(.2)
            raise TimeoutError('Server startup deadline exceeded')
        except BaseException:stop(child,log);raise
    def stop(child,log):
        if child.poll() is None:child.send_signal(signal.SIGINT)
        try:child.wait(timeout=120)
        except subprocess.TimeoutExpired:
            child.kill();child.wait();raise RuntimeError('Server failed to drain during shutdown')
        finally:log.close()
    def request(url,body,stem,arm):
        quit=threading.Event();samples=[];latency=[];errors=[]
        def poll():
            while not quit.is_set():
                began=time.monotonic()
                try:samples.append(get(url,'/freellm/status'));latency.append((time.monotonic()-began)*1000)
                except Exception as error:errors.append(str(error))
                quit.wait(1)
        thread=threading.Thread(target=poll)
        if arm=='new':thread.start()
        try:
            began=time.monotonic()
            req=urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=900) as f:response=json.load(f)
            elapsed=(time.monotonic()-began)*1000
        finally:
            quit.set()
            if thread.is_alive():thread.join()
            save(out/(stem+'-status.json'),dict(samples=samples,latency_ms=latency,errors=errors))
        require(not errors,'Status polling failed during inference')
        diagnostics=response['freellm'];mem=[diagnostics['after']['process']]
        for phase in diagnostics['phases'].values():mem.extend([phase['before']['process'],phase['after']['process']])
        mem.extend(s['process'] for s in samples if s.get('process'))
        row=dict(complete=True,elapsed_ms=elapsed,response=response,peak_physical_bytes=max(m['physical_footprint_peak_bytes'] for m in mem),
            memory_clean=clean_memory(mem),status_p95_ms=sorted(latency)[min(len(latency)-1,int(.95*len(latency)))] if latency else None)
        save(out/(stem+'-response.json'),row)
        if not row['memory_clean']:raise ResourceBlocked('Compression, decompression, swap growth or footprint disturbance; see raw request')
        require(not latency or row['status_p95_ms']<500,'Status response p95 exceeded 500ms')
        return row
    try:
        order=[(i,arm) for i in range(a.pairs) for arm in (('old','new') if i%2==0 else ('new','old'))] if a.mode=='paired' else [(0,'new')]
        for pair,arm in order:
            stem=f'{pair}-{arm}';child,log,url,plan=launch(arm,stem)
            try:
                model_id=get(url,'/v1/models')['data'][0]['id'];messages=list(work['messages'])
                start=time.monotonic();turn=0;warm_peaks=[]
                while True:
                    name=stem+'-'+str(turn);body=dict(model=model_id,messages=messages,max_tokens=count,temperature=0,seed=0,enable_thinking=False)
                    row=request(url,body,name,arm);row.update(pair=pair,arm=arm,memory_plan=plan,source=name+'-response.json')
                    report['measurements'].append(row);persist();host(name+'-after');identity()
                    if a.mode=='paired':break
                    warm_peaks.append(row['peak_physical_bytes'])
                    if time.monotonic()-start>=a.seconds:break
                    messages.extend([row['response']['choices'][0]['message'],dict(role='user',content=work.get('followups',['Explain the previous answer briefly.'])[turn%len(work.get('followups',['Explain the previous answer briefly.']))])])
                    turn+=1
                if a.mode=='soak':
                    report['duration_seconds']=time.monotonic()-start
                    report['memory_growth_requires_review']=len(warm_peaks)<3 or max(warm_peaks[2:])-warm_peaks[1]>128*1024**2
                    report['limitations']=['Scripted retained conversation only. Independently checked coding workflow and actual TUI overhead remain separate gates. Context overflow fails; no automatic compaction.']
            finally:stop(child,log)
        report.update(complete=True,status='measured')
        if a.mode=='paired':report['comparison']=compare(report['measurements'],a.pairs)
    except ResourceBlocked as error:report.update(complete=False,status='resource_blocked',error=str(error))
    except Exception as error:report.update(complete=False,status='failed',error=str(error))
    persist();print(json.dumps({k:report[k] for k in ('complete','status','qualified')},indent=2))
    return 0 if report['complete'] and report['status']=='measured' else 1


if __name__=='__main__':raise SystemExit(main())
