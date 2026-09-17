#!/usr/bin/env python3
"""Build the developer MTP format check with unchanged native compiler flags."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
from build_identity import build_fingerprint
from qualification_evidence import sha, save
from mtp_source import ROOT, require


def build(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    native=ROOT/'build/qwen';harness=ROOT/'scripts/qwen/check_mtp.cpp'
    db=native/'compile_commands.json';link_file=native/'CMakeFiles/freellm.dir/link.txt'
    entries=[e for e in json.loads(db.read_text()) if Path(e['file']).resolve()==ROOT/'src/qwen/model.cpp']
    require(len(entries)==1,'Missing native compiler command')
    command=shlex.split(entries[0]['command']);command[command.index('-o')+1]=str(output/'check.o');command[-1]=str(harness)
    link=shlex.split(link_file.read_text());main='CMakeFiles/freellm.dir/src/qwen/main.cpp.o'
    require(link.count(main)==1,'Unexpected native link command');link[link.index(main)]=str(output/'check.o')
    binary=output/'check-mtp';link[link.index('-o')+1]=str(binary)
    paths=[harness,Path(__file__),db,link_file,native/'libfreellm_lib.a',native/'bin/freellm',
        *sorted((ROOT/'include/qwen').glob('*.hpp'))]
    frozen={str(p.resolve()):sha(p) for p in paths}
    report=dict(kind='native_mtp_check_producer_v1',complete=False,base_native_fingerprint=build_fingerprint(ROOT),
        files=frozen,compiler=command,linker=link,binary=str(binary),production_promoted=False)
    save(output/'producer.json',report)
    with (output/'build.log').open('w') as log:
        for c in (command,link):subprocess.run(c,cwd=native,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    require(frozen=={str(p.resolve()):sha(p) for p in paths},'Native inputs changed while building MTP check')
    report.update(complete=True,binary_sha256=sha(binary),object_sha256=sha(output/'check.o'))
    save(output/'producer.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(build(a.output)))
