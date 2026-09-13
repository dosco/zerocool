import sys,os,fcntl,time,json,subprocess
from pathlib import Path
ROOT=Path('/repo');sys.path.insert(0,str(ROOT/'scripts/qwen'))
from qualification_evidence import *
from benchmark_exact import config_args,inspect_admission
from screen_route_selection import configs
from screen_cache import validate_request
from capacity_experiment import validate_probe
out=ROOT/'.cache/benchmarks/decode-gpu-boundary-20260913';out.mkdir(exist_ok=False)
start=time.monotonic();summary=dict(kind='decode_gpu_boundary_v1',complete=False,normal_request_latency_qualified=False,production_promoted=False)
try:
    source=ROOT/'.cache/benchmarks/decode-target-20260913-retry1'
    work=json.loads((source/'workload.json').read_text());save(out/'workload.json',work)
    config=configs()[1];model=ROOT/'.cache/qwen-mixed-reference';prepared=ROOT/'.cache/prepared/q4-records-v1'
    frozen=identity(ROOT,[config],model,prepared,out/'workload.json');frozen['files'][str(Path(__file__).resolve())]=sha(__file__)
    save(out/'identity.json',frozen);guard=EvidenceGuard(frozen,out)
    common=['--model',model,'--artifact','mixed-4_8bit','--prepared',prepared,'--memory-gb','12','--context','8192',*config_args(config)]
    env=dict(os.environ)
    for k in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION'):env.pop(k,None)
    with (ROOT/'.cache/qwen-qualification.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard.check_resources(initial=True)
        with (out/'request.log').open('w') as log:
            inspect_admission(ROOT/'build/qwen/bin/freellm',common,out/'request',12*GiB,512,log,guard,lambda:150-(time.monotonic()-start))
            guard.run([ROOT/'build/qwen/bin/freellm','bench',*common,'--workload-file',out/'workload.json','--repetitions','1','--temperature','0','--seed','0','--gpu-reference','resident-q8-v1','--bench-progress',out/'progress.jsonl','--json',out/'request.json'],stdout=log,timeout=120,env=env)
        raw=json.loads((out/'request.json').read_text());expected={work[0]['name']:json.loads((source/'normal.json').read_text())['runs'][0]['output_token_ids']}
        validate_request(raw,frozen,config,work,expected,gpu_reference='resident-q8-v1',output_tokens=17)
        assert [(r['repetition'],r['boundary']) for r in raw['gpu_references']]==[(0,'before'),(0,'after')]
        probes=[validate_probe(r) for r in raw['gpu_references']]
        summary.update(complete=True,status='captured',warm_ns_per_dispatch=probes,decode_ms_per_token=raw['runs'][0]['decode_wall_ms']/16)
except Exception as e:summary.update(status='failed',error=str(e))
finally:
    summary['elapsed_seconds']=time.monotonic()-start;save(out/'summary.json',summary)
    (out/'collector.py').write_bytes(Path(__file__).read_bytes());seal(out);print(json.dumps(summary))
