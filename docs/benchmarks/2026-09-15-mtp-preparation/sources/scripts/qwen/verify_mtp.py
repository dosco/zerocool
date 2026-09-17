#!/usr/bin/env python3
"""Run one bounded native sidecar format validation, without a full model."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess

from build_identity import build_fingerprint
from mtp_source import ROOT, require
from prepare_mtp import verify
from qualification_evidence import sha, save, seal


def run(prepared,binary,output):
    prepared,binary,output=map(lambda p:Path(p).resolve(),(prepared,binary,output))
    output.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((binary.parent/'producer.json').read_text())
    require(receipt.get('complete') is True and receipt['binary']==str(binary) and sha(binary)==receipt['binary_sha256'] and
        receipt['base_native_fingerprint']==build_fingerprint(ROOT),'Native MTP check producer differs')
    require(all(sha(Path(p))==h for p,h in receipt['files'].items()),'Native MTP check build inputs changed')
    audit=verify(prepared);save(output/'preparation-audit.json',audit)
    report=dict(kind='native_mtp_preparation_validation_v1',complete=False,status='running',
        preparation_audit=audit,producer=receipt,native_draft_implemented=False,
        acceptance_qualified=False,production_promoted=False,normal_request_latency_qualified=False)
    files=[Path(__file__),ROOT/'scripts/qwen/prepare_mtp.py',ROOT/'scripts/qwen/mtp_source.py',ROOT/'mtp-models.lock.json',binary,
        *[prepared/p for p in ('manifest.json','source-inventory.json','source-download.json','experts.bin','dense.bin')]]
    frozen={str(p):sha(p) for p in files};save(output/'identity.json',frozen)
    for name in ('manifest.json','source-inventory.json','source-download.json'):shutil.copyfile(prepared/name,output/name)
    save(output/'summary.json',report)
    try:
        with (ROOT/'.cache/qwen-qualification.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with (output/'native.log').open('w') as log:
                subprocess.run([str(binary),str(prepared),str(output/'native.json')],stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,MTL_DEBUG_LAYER='1',MTL_SHADER_VALIDATION='1'),timeout=60,check=True)
        raw=json.loads((output/'native.json').read_text());metal=raw['metal'];memory=[raw[k] for k in ('before','after','after_destroy')]
        require(raw.get('complete') is True and raw.get('validation') is True and len(raw['cases'])==25 and
            raw['manifest_sha256']==audit['manifest_sha256'] and metal['build_fingerprint']==receipt['base_native_fingerprint'] and
            metal['live_command_groups']==0 and metal['peak_buffer_bytes']<=192*1024**2,'Incomplete native MTP format check')
        expected_dense={k for k,t in json.loads((prepared/'manifest.json').read_text())['tensors'].items() if t['format']=='affine-Q8'}
        names={c['name'] for c in raw['cases']}
        require(names==expected_dense|{f'expert{e}/projection{p}' for e in (0,256,511) for p in range(3)},'Changed MTP matrix coverage')
        for matrix in raw['cases']:
            cases=matrix['cases'];require([(c['tokens'],c['column_start']) for c in cases]==[(1,0),(1,1),(1,2),(1,3),(4,0)] and
                all(c['exact_decoded_columns'] is True and c['output_values']==matrix['output']*c['tokens'] for c in cases),'Missing native column checks')
        require(all(0<m['physical_footprint_bytes']<=m['physical_footprint_peak_bytes']<=1024**3 for m in memory),'Unexpected MTP check footprint')
        clean=all(m['compressed_bytes']==m['compressed_peak_bytes']==0 for m in memory) and len({m['system_swap_used_bytes'] for m in memory})==1
        require(frozen=={str(p):sha(p) for p in files} and all(sha(Path(p))==h for p,h in receipt['files'].items()),'MTP sources changed during validation')
        report.update(complete=True,status='format_validated',native_sha256=sha(output/'native.json'),matrices=25,
            one_token_cases=100,four_token_cases=25,clean_memory=clean,
            peak_physical_bytes=max(m['physical_footprint_peak_bytes'] for m in memory),
            limitations=['One-hot columns validate packed storage and native affine kernels, not full draft outputs or token acceptance.',
                'Three of512 experts are exercised on Metal; all512 records are independently hashed and decoded during preparation.',
                'RMS weights remain original zero-centred BF16; future draft execution must apply the documented1+w convention.',
                'No target model, draft session, rollback integration or request performance is qualified.'])
    except BaseException as error:
        report.update(status='failed',error=str(error));raise
    finally:save(output/'summary.json',report);seal(output)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',type=Path,required=True);p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=run(a.prepared,a.binary,a.output);print(json.dumps({k:r[k] for k in ('complete','status','matrices','clean_memory')}))
