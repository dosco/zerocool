"""Native host preflight for experiments; no model, Metal device or OS changes."""
import json
from pathlib import Path
import shlex
import subprocess

from build_identity import build_fingerprint
from qualification_evidence import ResourceBlocked, save, sha

ROOT=Path(__file__).resolve().parents[2]


def require(ok,message):
    if not ok:raise ValueError(message)


def build_probe(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    native=ROOT/'build/qwen';source=ROOT/'scripts/qwen/benchmark_host.cpp'
    db=native/'compile_commands.json';link_path=native/'CMakeFiles/zerocool.dir/link.txt'
    entries=[e for e in json.loads(db.read_text()) if Path(e['file']).resolve()==ROOT/'src/engine/model.cpp']
    require(len(entries)==1 and Path(entries[0]['directory']).resolve()==native,'Missing native compiler command')
    compile=shlex.split(entries[0]['command']);obj=output/'host.o';binary=output/'benchmark-host'
    require(compile.count('-o')==compile.count('-c')==1 and Path(compile[-1]).resolve()==ROOT/'src/engine/model.cpp',
        'Unexpected native compiler command')
    compile[compile.index('-o')+1]=str(obj);compile[-1]=str(source)
    link=shlex.split(link_path.read_text());main='CMakeFiles/zerocool.dir/src/engine/main.cpp.o'
    require(link.count(main)==link.count('-o')==1,'Unexpected native linker command')
    link[link.index(main)]=str(obj);link[link.index('-o')+1]=str(binary)
    inputs=[source,Path(__file__),ROOT/'scripts/qwen/build_identity.py',ROOT/'scripts/qwen/qualification_evidence.py',
        db,link_path,native/'libzerocool_lib.a',*sorted((ROOT/'include/engine').glob('*.hpp'))]
    frozen={str(p.resolve()):sha(p) for p in inputs};base=build_fingerprint(ROOT)
    report=dict(kind='benchmark_host_producer_v1',complete=False,base_native_fingerprint=base,
        files=frozen,compiler=compile,linker=link,binary=str(binary),model_loaded=False,gpu_used=False)
    save(output/'producer.json',report)
    with (output/'build.log').open('w') as log:
        for command in (compile,link):subprocess.run(command,cwd=native,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
    require(frozen=={str(p.resolve()):sha(p) for p in inputs} and build_fingerprint(ROOT)==base,'Host probe build inputs changed')
    report.update(complete=True,binary_sha256=sha(binary),object_sha256=sha(obj));save(output/'producer.json',report)
    return dict(binary=binary,producer=report,files=[*inputs,obj,binary,output/'producer.json'])


def observe(raw,expected_build):
    require(raw.get('kind')=='benchmark_host_preflight_v1' and raw.get('complete') is True and
        raw.get('build_fingerprint')==expected_build and raw.get('model_loaded') is False and raw.get('gpu_used') is False,
        'Invalid native host preflight identity')
    host=raw.get('host',{});reasons=[]
    thermal=host.get('thermal_state');low=host.get('low_power_mode');power=host.get('power_source');time=host.get('monotonic_ns')
    require(type(thermal) is int and 0<=thermal<=3 and type(low) is bool and isinstance(power,str) and power and
        type(time) is int and time>0,'Missing or invalid native power/thermal observation')
    if thermal!=0:reasons.append('thermal state is not nominal')
    if low:reasons.append('Low Power Mode is enabled')
    return dict(clean_host=not reasons,host=host,reasons=reasons)


def preflight(exp,probe,stem):
    name=stem+'-host';exp.command([probe['binary'],exp.out/(name+'.json')],name,limit=5)
    path=exp.out/(name+'.json');observed=observe(json.loads(path.read_text()),exp.frozen['build'])
    exp.report.setdefault('host_preflight',[]).append(dict(source=path.name,sha256=sha(path),observation=observed))
    exp.persist()
    if not observed['clean_host']:
        raise ResourceBlocked('Before '+stem+': '+', '.join(observed['reasons'])+' ('+observed['host']['power_source']+')')
    return observed
