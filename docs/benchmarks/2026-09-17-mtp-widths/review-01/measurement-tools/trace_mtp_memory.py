#!/usr/bin/env python3
"""Bounded VM-map attribution of an owned MTP process; never performance qualification."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import screen_mtp_ngram_init as stage
from cache_residency import require
from combined_q4 import freeze
from qualification_evidence import ResourceBlocked,save,sha,verify_seal


def allocation_categories(text):
    """Read vmmap's page counts without interpreting regions as C++ ownership."""
    page=re.search(r'^VM page size:\s+(\d+) bytes$',text,re.M)
    require(page is not None and int(page[1])>0,'Missing VM page size')
    require(text.count('==== Summary for process ')==1,'Missing or ambiguous VM summary')
    summary=re.split(r'^\s*MALLOC ZONE\s',text.split('==== Summary for process ',1)[1],maxsplit=1,flags=re.M)[0]
    rows={}
    pattern=r'^(.+?)\s+((?:\d+(?:\.\d+)?\s+){7}\d+)(?:\s+.*)?$'
    for line in summary.splitlines():
        m=re.match(pattern,line)
        if not m:continue
        name=m[1].strip();values=m[2].split()
        require(name not in rows,'Duplicate VM category')
        row={k:round(float(v)*int(page[1])) for k,v in zip(
            ('virtual_bytes','resident_bytes','dirty_bytes','swapped_bytes','volatile_bytes','nonvolatile_bytes','empty_bytes'),values[:7])}
        row['regions']=int(values[7]);rows[name]=row
    require('TOTAL' in rows and 'IOAccelerator (graphics)' in rows,'Incomplete VM categories')
    total=rows.pop('TOTAL')
    # vmmap may append a second aggregate after reporting reserved address space.
    # It is a total, not another allocation category.
    without_reserved=rows.pop('TOTAL, minus reserved VM space',None)
    require(all(r['swapped_bytes']<=r['virtual_bytes'] for r in rows.values()),'Invalid VM page accounting')
    return dict(page_bytes=int(page[1]),categories=rows,total=total,total_without_reserved=without_reserved,
        swapped_categories={k:v for k,v in rows.items() if v['swapped_bytes']},
        allocation_ownership_identified=False,performance_measurement=False)


def run(output,directory,validation=False,after_tokens=None):
    require(after_tokens is None or validation and 1<=after_tokens<8,'Later capture requires a validation boundary from 1 to 7')
    stage.configure();s=stage.screen
    if validation:
        work,_=s.source_input();work.update(eos_ids=[248046,248044],max_tokens=16,draft_slots=32,
            target_prepared=str(s.ROOT/'.cache/prepared/q4-records-v1'))
    else:work=s.select_cases(s.workloads(s.ROOT/'.cache/qwen-mixed-reference',16),['merge_intervals'])[0]
    mode='fast-validate' if validation else 'fast-timing'
    exp=s.Experiment(output,'mtp_startup_memory_attribution_v1',[s.configuration(4,expert_slots=1460)],work,220)
    with exp:
        cfg,host=stage.setup(exp,directory);freeze(exp,[stage.BASE/'memory-protocol.md'])
        s.host_check(exp,host,'native');exp.env['FREELLM_MTP_EXPERT_SCRATCH']='off'
        reference=stage.REFERENCE if validation else s.ROOT/'docs/benchmarks/2026-09-16-mtp-continuation/short-reference-01'
        verify_seal(reference,sha(reference/'evidence-files.json'))
        refpath=reference/('case-0-pair-0-off.json' if validation else 'case-0-fast-timing.json')
        freeze(exp,[refpath,reference/'evidence-files.json'])
        exp.report.update(performance_measurement=False,normal_request_latency_qualified=False,
            compression_limit_bytes=512*1024**2,reference=str(refpath),native_mode=mode,metal_validation=validation,
            capture_after_tokens=after_tokens,maps=[])
        exp.guard.check_identity();exp.guard.check_resources();out=exp.out/'native.json';progress=out.with_suffix('.progress.jsonl')
        cmd=[str(p) for p in (cfg['binary'],exp.model,s.PREPARED,exp.out/'workload.json',out,mode)]
        if validation:exp.env.update(MTL_DEBUG_LAYER='1',MTL_SHADER_VALIDATION='1')
        with (exp.out/'native.log').open('w') as log:
            child=subprocess.Popen(cmd,cwd=s.ROOT,env=exp.env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                exp.report.update(native_pid=child.pid,command=cmd);exp.persist();deadline=time.monotonic()+min(160,exp.left());last=None
                while child.poll() is None:
                    if time.monotonic()>deadline:raise subprocess.TimeoutExpired(cmd,160)
                    try:current=json.loads(progress.read_text().splitlines()[-1])
                    except (OSError,ValueError,IndexError):current={}
                    phase=current.get('phase')
                    if phase!=last:print('native phase: '+str(phase),flush=True);last=phase
                    ready=(phase in ('prime_draft','draft_verify') if after_tokens is None else
                        phase=='draft_verify' and current.get('consumed_tokens',0)>=after_tokens)
                    if ready and not exp.report['maps']:
                        path=exp.out/'vmmap.txt';started=time.monotonic_ns()
                        with path.open('w') as f:
                            result=subprocess.run(['/usr/bin/vmmap','-pages','-w','-noCoalesce',str(child.pid)],
                                stdout=f,stderr=subprocess.STDOUT,timeout=10)
                        exp.report['maps'].append(dict(pid=child.pid,phase=current,source=path.name,
                            sha256=sha(path),returncode=result.returncode,capture_ns=time.monotonic_ns()-started));exp.persist()
                        require(result.returncode==0,'VM map capture failed')
                    time.sleep(.05)
                require(child.returncode==0,'Native attribution process failed')
            finally:
                if child.poll() is None:
                    for sig,seconds in ((signal.SIGINT,10),(signal.SIGTERM,5),(signal.SIGKILL,5)):
                        if child.poll() is not None:break
                        os.killpg(child.pid,sig)
                        try:child.wait(timeout=seconds)
                        except subprocess.TimeoutExpired:pass
        exp.guard.check_identity();exp.guard.check_resources();require(len(exp.report['maps'])==1,'Missing VM map phase')
        categories=allocation_categories((exp.out/'vmmap.txt').read_text())
        save(exp.out/'allocation-map.json',dict(categories,source='vmmap.txt',source_sha256=sha(exp.out/'vmmap.txt')))
        raw=json.loads(out.read_text());observed=s.observe(raw,work,sha(exp.out/'workload.json'),mode)
        old=json.loads(refpath.read_text())
        fields=('prime_logits_sha256','committed_token_ids','row_logits_sha256','final_target_state','final_draft_state','next_id','stop_reason')
        if validation:fields+=('boundaries','admission')
        require(all(old[k]==raw[k] for k in fields),'Memory diagnostic changed exact outputs/state')
        memory=[raw['before_load'],raw['before']['process'],raw['after']['process'],raw['after_destroy'],
            *[c[k] for c in raw['cycles'] for k in ('memory_before','memory_after')]]
        exp.report.update(status='memory_captured',strict_clean_memory=observed['clean_memory'],
            clean_host=observed['clean_host'],peak_physical_bytes=observed['peak_physical_bytes'],
            compression_peak_bytes=max(m['compressed_peak_bytes'] for m in memory),exact_reference=True)
        if not observed['clean_host'] or len({m['system_swap_used_bytes'] for m in memory})!=1 or exp.report['compression_peak_bytes']>512*1024**2:
            raise ResourceBlocked('Memory attribution exceeded its host/swap/compression bounds')
    return exp.report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('output','build'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--validation',action='store_true',help='Inspect forced-rejection validation with Metal validation enabled')
    p.add_argument('--after-tokens',type=int,help='Capture a later validation boundary after at least this many committed tokens')
    a=p.parse_args();run(a.output,a.build,a.validation,a.after_tokens)
