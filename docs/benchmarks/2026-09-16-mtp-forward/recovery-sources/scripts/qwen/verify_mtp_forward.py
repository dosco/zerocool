#!/usr/bin/env python3
"""Reconstruct MTP screen evidence without running inference or accepting gaps."""
import argparse
import json
from pathlib import Path
from cache_residency import require
from qualification_evidence import sha,save,verify_seal
from screen_mtp_forward import clean,compare_joint


def verify(directory,snapshot=None):
    directory=Path(directory).resolve();verify_seal(directory,sha(directory/'evidence-files.json'))
    saved=json.loads((directory/'summary.json').read_text());frozen=json.loads((directory/'identity.json').read_text())
    producer=json.loads((directory/'producer.json').read_text());root=Path(frozen['root'])
    require(producer['complete'] is True and producer['base_native_fingerprint']==frozen['build'],'missing producer identity')
    checked=[]
    for name,digest in {**producer['inputs'],**producer['generated'],**producer['objects'],producer['binary']:producer['binary_sha256']}.items():
        p=Path(name);candidates=[p]
        if snapshot is not None:candidates.append(Path(snapshot)/p.relative_to(root))
        match=next((c for c in candidates if c.is_file() and sha(c)==digest),None)
        require(match is not None,'missing immutable build input: '+name);checked.append(str(match))
    if saved['complete'] is not True:
        require(saved['status'] in ('resource_blocked','failed','time_budget_exhausted','interrupted'),'unfinished status missing')
        return dict(passed=True,recorded_status=saved['status'],timing_qualified=False,build_files=len(checked))
    if saved['kind']=='mtp_forward_fixture_v1':
        native=json.loads((directory/'native.json').read_text());ref=json.loads((directory/'reference.json').read_text())
        require(native['complete'] is True and native['input_sha256']==sha(directory/'input.json') and
            native['draft_manifest_sha256']==json.loads((directory/'draft-audit.json').read_text())['manifest_sha256'],'fixture identity mismatch')
        require(saved['native']==clean(native) and all(saved['native'][k] for k in ('clean_memory','clean_host')),'fixture memory differs')
        require(ref['passed'] is True and ref['native_sha256']==sha(directory/'native.json') and ref['input_sha256']==sha(directory/'input.json') and
            saved['reference']==ref and len(ref['cases'])==22 and all(c['passed'] for c in ref['cases']),'reference proof mismatch')
        for name,t in native['fixture_tensors'].items():require(sha(directory/'trace'/(name+'.bin'))==t['sha256'],'changed fixture bytes')
        return dict(passed=True,recorded_status=saved['status'],independent_reference_passed=True,build_files=len(checked),timing_qualified=False)
    require(saved['kind']=='mtp_joint_screen_v1','unknown MTP evidence kind')
    serial=json.loads((directory/'serial.json').read_text());candidate=json.loads((directory/'timing.json').read_text())
    ratio=compare_joint(serial,candidate)
    require(saved['candidate_to_serial_latency_ratio']==ratio,'changed comparison ratio')
    for mode in ('validate','serial','timing'):
        raw=json.loads((directory/(mode+'.json')).read_text());observed=clean(raw)
        require(raw['complete'] is True and raw['input_sha256']==sha(directory/'workload.json') and
            all(observed[k] for k in ('clean_memory','clean_host')) and all(saved[mode][k]==v for k,v in observed.items()),'changed joint observation')
        require(saved[mode]['tokens_per_second']==raw['tokens_per_second']==raw['generated_tokens']*1e9/raw['decode_wall_ns'],'changed throughput')
        require(raw['accepted_proposals']==sum(c['accepted_proposals'] for c in raw['cycles']) and
            raw['proposed_tokens']==sum(c['width']-1 for c in raw['cycles']),'changed acceptance coverage')
    raw=json.loads((directory/'validate.json').read_text());require(any(c['forced_rejection'] and c['accepted_proposals']==0 for c in raw['cycles']),'missing forced rejection')
    require(saved['status']==('promising_short_screen' if ratio<1 else 'slower_than_serial'),'changed screen decision')
    ref=json.loads((directory/'real-fixture/reference.json').read_text())
    require(ref['passed'] is True and ref['native_sha256']==sha(directory/'real-fixture/native.json') and
        ref['input_sha256']==sha(directory/'real-input.json'),'missing real-hidden reference')
    return dict(passed=True,recorded_status=saved['status'],ratio=ratio,actual_cost_counted=True,
        build_files=len(checked),paired_confidence_qualified=False,production_promoted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--snapshot',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=verify(a.directory,a.snapshot);save(a.output,r);print(json.dumps(r))
