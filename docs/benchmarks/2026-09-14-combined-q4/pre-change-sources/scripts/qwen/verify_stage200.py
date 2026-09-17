#!/usr/bin/env python3
"""Read-only reconstruction of the bounded stage; never promotes a runtime."""
import argparse
import hashlib
import json
from pathlib import Path

from qualification_evidence import confined, save, sha, verify_seal
from screen_cache import validate_request
from stage200 import ORDER, SOURCE, capacity_decision, clean_memory, profile_row, q3_decision


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def verify_sources(directory,frozen,source_snapshot=None):
    root=Path(frozen['root']);snapshots=[directory/name for name in ('initial-sources','capture-sources','screen-sources')]
    if source_snapshot is not None:snapshots.append(Path(source_snapshot))
    checked=[]
    for name,digest in frozen['files'].items():
        relative=Path(name).relative_to(root)
        if relative.parts[0] not in ('scripts','tests'):continue
        candidates=[root/relative,*[s/relative for s in snapshots]]
        match=next((p for p in candidates if p.is_file() and sha(p)==digest),None)
        require(match is not None,'Missing frozen tooling source: '+str(relative))
        checked.append(dict(original=str(relative),sha256=digest,available_at=str(match)))
    source_root=root if source_snapshot is None else Path(source_snapshot)
    paths=sorted([*(source_root/'src/qwen').glob('*'),*(source_root/'include/qwen').glob('*')])
    paths += [source_root/p for p in ('kernels/metal/qwen.metal','models.lock.json','mixed-models.lock.json','mixed-payload-reuse.lock.json','cmake/qwen.cmake')]
    versions={}
    for mode in ('current','initial'):
        lines=[]
        for p in paths:
            relative=p.relative_to(source_root);saved=directory/'initial-sources'/relative
            source=saved if mode=='initial' and saved.is_file() else p
            lines.append(f'{relative}:{sha(source)}\n')
        versions[mode]=hashlib.sha256(''.join(lines).encode()).hexdigest()
    require(frozen['build'] in versions.values(),'Native source fingerprint cannot be reconstructed')
    return dict(native_source_version=next(k for k,v in versions.items() if v==frozen['build']),tooling_files=len(checked),
                archived_tooling=[s for s in checked if not s['available_at'].startswith(str(root/'scripts')) and not s['available_at'].startswith(str(root/'tests'))])


def verify(directory,source_snapshot=None):
    directory=Path(directory);audited=[]
    for path in sorted(directory.iterdir()):
        if not (path/'evidence-files.json').is_file():continue
        digest=sha(path/'evidence-files.json');verify_seal(path,digest)
        report=read(path/'summary.json');frozen=read(path/'identity.json')
        require(report['identity']=={k:frozen[k] for k in report['identity']},'Changed summary identity')
        row=dict(source=path.name,seal_sha256=digest,recorded_status=report['status'],
                 recorded_complete=report['complete'],build=frozen['build'],source_provenance=verify_sources(directory,frozen,source_snapshot))
        if report['complete'] is not True:
            require(report['status'] in ('failed','resource_blocked','interrupted','time_budget_exhausted'),
                    'Unfinished capture has no terminal disposition')
            row.update(recomputed_status=report['status'],result_qualified=False)
            audited.append(row);continue
        kind=report['kind'];work=report['workload'];configs=report['configurations']
        if kind in ('stage200_capacity_v1','stage200_pressure_v1'):
            expected={r['name']:r['output_token_ids'] for r in read(SOURCE/'pair-0-control.json')['runs']}
            require([(m['pair'],m['configuration']) for m in report['measurements']]==list(ORDER),'Incomplete alternating pairs')
            measurements=[]
            for m in report['measurements']:
                source=confined(path,m['source']);require(sha(source)==m['sha256'],'Changed request source')
                raw=read(source);c=configs[m['configuration']=='candidate']
                observations=validate_request(raw,frozen,c,work,expected,
                    capacity_axis=kind=='stage200_capacity_v1',pressure_axis=kind=='stage200_pressure_v1')
                require(observations==m['requests'] and clean_memory(raw)==m['clean_memory'],'Changed derived request or memory measurements')
                item=dict(m,requests=observations,clean_memory=clean_memory(raw));measurements.append(item)
                if kind=='stage200_pressure_v1':
                    require(m['pressure']==raw['runs'][-1]['after']['memory_pressure'],'Changed pressure events')
            if kind=='stage200_capacity_v1':
                decision=capacity_decision(measurements,configs[0]['expert_slots'],configs[1]['expert_slots'])
                require(all(report[k]==v for k,v in decision.items()),'Changed capacity decision')
                row.update(recomputed_status=decision['status'],decision=decision)
            else:
                exercised=any(e['achieved_slots']<e['before_slots'] for m in measurements if m['configuration']=='candidate' for e in m['pressure']['events'])
                status='observed' if exercised else 'not_exercised'
                require(report['real_pressure_exercised']==exercised and report['status']==status,'Changed pressure outcome')
                row.update(recomputed_status=status,real_pressure_exercised=exercised)
        elif kind=='decode_target_capture_v2':
            expected={}
            for arm in ('normal','traced'):
                ref=report[arm];source=confined(path,ref['source'])
                require(sha(source)==ref['sha256'],'Changed profile request')
                raw=read(source)
                validate_request(raw,frozen,configs[0],work,expected,instrumented=arm=='traced',output_tokens=17)
                require(clean_memory(raw)==ref['clean_memory'],'Changed profile memory summary')
            normal,traced=(read(path/(arm+'.json')) for arm in ('normal','traced'))
            profile=read(path/'commands.json');deps=[json.loads(line) for line in (path/'dependencies.jsonl').read_text().splitlines()]
            require(profile['coverage']=='decode-only','Changed profile coverage')
            timelines=[profile_row(traced,i,profile,deps) for i in range(2)]
            ref=report['timelines'];source=confined(path,ref['source'])
            require(sha(source)==ref['sha256'] and read(source)==timelines,'Changed dependency reconstruction')
            ratios=[b['decode_wall_ms']/a['decode_wall_ms'] for a,b in zip(normal['runs'],traced['runs'])]
            require(report['trace_to_normal_ratios']==ratios,'Changed trace overhead calculation')
            row.update(recomputed_status='captured',captured_tokens=[t['captured_tokens'] for t in timelines],
                       trace_to_normal_ratios=ratios)
        elif kind=='stage200_q3_v1':
            decisions={}
            for split in ('tuning','heldout'):
                if not (path/split/'manifest.json').exists():continue
                manifest=read(path/split/'manifest.json')
                require(manifest.get('complete') is True and manifest['kind']=='q3_probe_capture_v2','Incomplete Q3 capture')
                require(manifest['build']==frozen['build'] and manifest['artifact_revision']==frozen['artifact_revision'],'Changed Q3 model identity')
                require(set(manifest['layers'])=={'0','16','32','47'},'Changed Q3 layer coverage')
                experts=set()
                for layer,data in manifest['layers'].items():
                    require(len(data['experts'])==2 and data['offsets']==list(range(72,80)),'Changed Q3 input coverage')
                    for file in [data['inputs'],*[e['record'] for e in data['experts']]]:
                        source=confined(path/split,file['file'])
                        require(source.stat().st_size==file['bytes'] and sha(source)==file['sha256'],'Changed Q3 fixture bytes')
                    experts.update((int(layer),e['expert']) for e in data['experts'])
                require(len(experts)==8,'Duplicate Q3 experts')
                for mode in ('validate','timing'):
                    result=read(path/(split+'-'+mode+'.json'))
                    require(result['complete'] is True and result['validation'] is (mode=='validate') and
                            result['build']==frozen['build'] and len(result['cases'])==32,'Incomplete Q3 operator coverage')
                    require(result['peak_gpu_bytes']<=512*1024**2 and result['memory']['physical_footprint_peak_bytes']<=2*1024**3,'Q3 budget exceeded')
                    require({(c['layer'],c['expert'],c['rows']) for c in result['cases']}=={(l,e,t) for l,e in experts for t in (1,2,4,8)},'Changed Q3 expert set')
                    for c in result['cases']:
                        require(c['exact_expanded_reference'] is True and
                                (c['q4_record_bytes'],c['q4_aligned_bytes'],c['q3_record_bytes'],c['q3_aligned_bytes'])==
                                (2764800,2768896,2355200,2359296),'Changed Q3 arithmetic or byte accounting')
                decision=q3_decision(result);decisions[split]=decision
                original=report[split]
                for key in ('one_token_geomean_ratio','one_token_upper_95','ratios','clean_memory','advance_to_quality_stage'):
                    require(original[key]==decision[key],'Changed Q3 numerical decision')
            require(decisions,'Missing Q3 decisions')
            if not decisions['tuning']['advance_to_quality_stage']:
                require(set(decisions)=={'tuning'} and report.get('heldout_status')=='not_run','Failed tuning advanced to held-out capture')
            row.update(recomputed_status=list(decisions.values())[-1]['status'],decisions=decisions,
                limitation='The original collector called all failed latency gates regressions. A confidence interval crossing the allowed limit is inconclusive.')
        else:raise ValueError('Unsupported stage report: '+kind)
        row['result_qualified']=False;audited.append(row)
    require(audited,'No stage evidence')
    return dict(kind='stage200_evidence_audit_v1',complete=True,audit_passed=True,attempts=audited,
                production_promoted=False,normal_request_latency_qualified=False,
                limitations=['This reconstructs saved source-bound measurements; it does not rerun inference.',
                    'Failed and timed-out attempts stay incomplete. No historical native binaries are archived.',
                    'The 2K attempt has no completed request or exact TTFT, and no 4K/7K or sustained coding qualification was run.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-snapshot',type=Path,help='Preserved native/tooling source root when the working tree has advanced')
    args=p.parse_args();save(args.output,verify(args.directory,args.source_snapshot))
