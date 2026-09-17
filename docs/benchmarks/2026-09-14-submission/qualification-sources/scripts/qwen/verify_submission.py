#!/usr/bin/env python3
"""Reconstruct submission experiments from sealed raw data and archived sources."""
import argparse
import json
from pathlib import Path

from qualification_evidence import save, sha, verify_seal
from screen_submission import analyze as ownership_decision
from screen_coalesced import configs as coalesced_configs, observe as coalesced_observe, decide
from screen_expert_residency import configs as residency_configs, observe as residency_observe
from stage200 import ORDER, SOURCE, clean_memory
from submission_timeline import analyze as driver_analysis
from verify_stage200 import verify_sources


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def verify(directory):
    directory=Path(directory);results=[]
    source_roots={'profile-01':'diagnostic-sources','ownership-01':'ownership-sources',
                  'coalesced-01':'final-sources','residency-01':'final-sources'}
    for name,source in source_roots.items():
        path=directory/name;verify_seal(path,sha(path/'evidence-files.json'))
        summary=read(path/'summary.json');frozen=read(path/'identity.json')
        provenance=verify_sources(directory,frozen,directory/source)
        result=dict(source=name,recorded_complete=summary['complete'],recorded_status=summary['status'],
            source_seal_sha256=sha(path/'evidence-files.json'),build=frozen['build'],source_provenance=provenance)
        if summary['complete'] is not True:
            require(summary['status'] in ('resource_blocked','failed','interrupted','time_budget_exhausted','incomplete'),
                    'Unfinished run has no terminal status')
            result['qualified']=False;results.append(result);continue
        if name=='profile-01':
            derived=driver_analysis(path)
            require(derived==read(directory/'driver-analysis.json'),'Driver interval reconstruction differs')
            result.update(normal_clean_memory=derived['normal_clean_memory'],traced_clean_memory=derived['traced_clean_memory'])
        elif name=='ownership-01':
            timing,validation=read(path/'timing.json'),read(path/'validation.json')
            require(timing['build']==validation['build']==frozen['build'] and validation['complete'] is True and
                    validation['validation'] is True,'Missing native ownership validation')
            decision=ownership_decision(timing)
            require(all(summary.get(k)==v for k,v in decision.items()),'Changed ownership decision')
            require(timing['fixture_manifest']==validation['fixture_manifest'],'Different operator fixtures')
            require(timing['final_live_bytes']==validation['final_live_bytes']==0,'Retained probe resources')
            result.update(status=decision['status'],clean_memory=decision['clean_memory'],
                          optimistic_ms_per_token=decision['optimistic_ms_per_token'])
        else:
            configs,observe=(coalesced_configs,coalesced_observe) if name=='coalesced-01' else (residency_configs,residency_observe)
            require(summary['configurations']==configs(),'Changed paired configurations')
            require([(r['pair'],r['configuration']) for r in summary['measurements']]==list(ORDER),'Missing alternating request pairs')
            expected={r['name']:r['output_token_ids'] for r in read(SOURCE/'pair-0-control.json')['runs']};hashes=set()
            for m in summary['measurements']:
                raw_path=path/m['source'];digest=sha(raw_path)
                require(digest==m['sha256'] and digest not in hashes,'Changed or reused request');hashes.add(digest)
                raw=read(raw_path);c=next(c for c in configs() if c['name']==m['configuration'])
                require(observe(raw,frozen,c,summary['workload'],expected)==m['requests'],'Changed request measurements')
                require(clean_memory(raw) is True and m['clean_memory'] is True,'Disturbed comparison')
            decision=decide(summary['measurements'])
            require(all(summary.get(k)==v for k,v in decision.items()),'Changed request decision')
            result.update(status=decision['status'],exact_output_tokens=True,exact_dispatch_counts=True,
                clean_memory=True,decode_savings_ms_per_token=decision['decode_savings_ms_per_token'])
        results.append(result)
    return dict(kind='submission_experiment_audit_v1',complete=True,audit_passed=True,experiments=results,
        normal_request_latency_qualified=False,production_promoted=False,
        limitations=['Raw controls, native source identities and measurement arithmetic are reconstructed; inference is not rerun.',
            'The profiled 1460-slot requests compressed and cannot qualify speed. Isolated fixture results do not establish request gains.',
            'Two-pair normal screens cannot replace full-model state, five-pair, long-context or sustained-coding qualification.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();save(args.output,verify(args.directory))
