"""Bounded selector qualification. Negative results complete; missing evidence does not."""
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import sys

import benchmark_exact as normal
from qualification_evidence import (EvidenceGuard, ResourceBlocked, GiB, identity, save, sha,
                                    seal, verify_seal, import_sealed, confined)
from summarize_decode_profile import summarize as summarize_profile
from qualify_exact_sessions import check_configuration, compare as compare_sessions

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/'docs/benchmarks/2026-09-09-sparse-attention/configs.json'
WORKLOAD = ROOT/'docs/benchmarks/2026-09-08-validation/workload-2k.json'
CASES = ('prompt_2k','prompt_4k','append_128','prompt_7k')
METRICS = ('ttft_ms','decode_ms_per_token','request_ms')


def configurations():
    configs = json.loads(CONFIG.read_text())[:2]
    if [c['name'] for c in configs] != ['cpu-selection','gpu-selection']:
        raise ValueError('Missing selector control/candidate')
    a,b = [dict(c) for c in configs]
    a.pop('name');b.pop('name')
    if a.pop('sparse_selection')!='cpu' or b.pop('sparse_selection')!='gpu' or a!=b or a['attention_score_tiles']!='full':
        raise ValueError('Selector comparison must change only selection')
    return configs


def screen_gate(screen):
    if screen.get('complete') is not True or screen.get('mode')!='screen' or screen.get('comparison_purpose')!='experiment':
        raise ValueError('Requires a complete normal selector screen')
    rows=screen['measurements']
    expected={(p,c,n) for p in range(2) for c in ('cpu-selection','gpu-selection') for n in CASES}
    keys=[(r['pair'],r['configuration'],r['name']) for r in rows]
    if len(keys)!=len(expected) or set(keys)!=expected:
        raise ValueError('Screen requires sixteen unique measurements')
    index=dict(zip(keys,rows));ratios=[]
    for name in CASES:
        tokens=None
        for p,c,n in keys:
            if n!=name:continue
            ids=index[p,c,n].get('output_token_ids')
            if not isinstance(ids,list) or len(ids)!=64 or (tokens is not None and tokens!=ids):
                raise ValueError('Screen generated tokens differ or are missing')
            tokens=ids
        for metric in METRICS:
            pair=[]
            for p in range(2):
                values=[index[p,c,name][metric] for c in ('cpu-selection','gpu-selection')]
                if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in values):
                    raise ValueError('Screen timing must be positive and finite')
                pair.append(values[1]/values[0])
            ratios.append(dict(case=name,metric=metric,ratios=pair,median=statistics.median(pair)))
    promising=any(r['case'] in ('prompt_4k','append_128') and r['metric']=='request_ms' and
                  r['median']<=.99 and all(x<1 for x in r['ratios']) for r in ratios)
    advance=promising and all(r['median']<=1.03 for r in ratios)
    return dict(advance_to_paired=advance,status='promising' if advance else 'improvement_not_demonstrated',
                ratios=ratios,confidence_claim=False,production_promoted=False)


def check_machine(state,evidence):
    machine=state['metal']
    if any(machine[k]!=evidence[v] for k,v in [('build_fingerprint','build'),('device','device'),('physical_bytes','physical_bytes')]):
        raise ValueError('Execution machine or build differs')
    if state['artifact_revision']!=evidence['artifact_revision'] or state['prepared']['manifest_sha256']!=evidence['prepared_manifest_sha256']:
        raise ValueError('Execution artifact differs')
    plan=state['memory_plan']
    if plan['limit_bytes']!=12*GiB or plan['planned_bytes']>12*GiB or state['process']['physical_footprint_bytes']>12*GiB or machine['peak_buffer_bytes']>12*GiB:
        raise ValueError('Execution exceeded or reduced the fixed memory budget')


def check_cached(report,evidence,tokens):
    from benchmark_sparse_attention import check_cached as validate_cached
    result=validate_cached(report,evidence['build'],evidence['artifact_revision'],tokens,'sparse_selection')
    if result['pairs']!=5:raise ValueError('Selector cached comparison requires exactly five pairs')
    for row in report['runs']:
        config=evidence['configurations'][row['variant']=='candidate']
        for state in (row['before'],row['after']):
            check_machine(state,evidence)
            checked=copy.deepcopy(state);checked['execution']['cached_token_replay']=False
            check_configuration(checked,config)
            if state.get('completion_pipeline') is not True:raise ValueError('Missing completion pipeline')
    return result


def check_capture(manifest,report,name,evidence,directory):
    length={'prompt_4k':4097,'prompt_7k':7169,'append_128':4225}[name]
    expected={(layer,'decode',1,length) for layer in (3,47)}
    if name=='append_128':expected|={(layer,'append',128,4224) for layer in (3,47)}
    for data,kind in ((manifest,'sparse_attention_fixture_v1'),(report,'sparse_attention_replay_v1')):
        if data.get('kind')!=kind or data['build_fingerprint']!=evidence['build'] or data['artifact_revision']!=evidence['artifact_revision']:
            raise ValueError('Capture build or artifact differs')
        keys=[(c['layer'],c['phase'],c['tokens'],c['length']) for c in data['cases']]
        if len(keys)!=len(expected) or set(keys)!=expected:raise ValueError('Capture workload geometry differs')
    if report.get('passed') is not True:raise ValueError('Capture replay failed')
    for row in report['cases']:
        arms=row['arms']
        if row.get('exact') is not True or [a['arm'] for a in arms]!=[0,1,2] or any(a['hashes']!=arms[0]['hashes'] for a in arms):
            raise ValueError('Capture did not replay all three arms exactly')
    total=0
    for row in manifest['cases']:
        if row['offset']+row['tokens']!=row['length']:raise ValueError('Capture absolute position differs')
        t,n=row['tokens'],row['length']
        sizes=dict(q=t*6144*4,keys=n*512*4,values=n*512*4,qg=t*12288*4,index_scores=t*(n//4)*4)
        if set(row['tensors'])!=set(sizes):raise ValueError('Missing capture tensors')
        for name,size in sizes.items():
            tensor=row['tensors'][name];path=confined(directory,tensor['file'])
            if tensor['bytes']!=size or path.stat().st_size!=size or sha(path)!=tensor['sha256']:
                raise ValueError('Changed capture tensor')
            total+=size
    if total!=manifest['bytes'] or total>256*1024**2:raise ValueError('Invalid capture byte count')
    return total


def stage_complete(phases):
    needed={'recovery','capture','qualify','cached-4096','cached-2048','cached-7168','screen','profile'}
    if not all(phases.get(p,{}).get('status')=='passed' for p in needed):return False
    gate=phases['screen']['result']
    return not gate['advance_to_paired'] or phases.get('paired',{}).get('status')=='passed'


class Experiment:
    def __init__(self,args):
        self.args=args;self.output=args.output.resolve();self.configs=configurations()
        self.model=ROOT/'.cache/qwen-mixed-reference';self.prepared=ROOT/'.cache/prepared/q4-records-v1'
        self.binary=ROOT/'build/qwen/bin/freellm'
        self.evidence=identity(ROOT,self.configs,self.model,self.prepared,WORKLOAD)
        self.evidence['files'][str(CONFIG.resolve())]=sha(CONFIG)
        self.guard=EvidenceGuard(self.evidence,self.output)
        self.previous=None;self.phases={};self.active=None
        self.seed=json.loads(WORKLOAD.read_text())[0]['tokens']
        self.shared=['--model',str(self.model),'--artifact','mixed-4_8bit','--prepared',str(self.prepared),'--memory-gb','12','--context','8192']
        if args.resume_from:
            self.previous=args.resume_from.resolve()
            old=json.loads((self.previous/'identity.json').read_text())
            if old!=self.evidence:raise ValueError('Resume sources, artifacts, settings or protocol differ')
        save(self.output/'identity.json',self.evidence)

    def publish(self,error=None):
        complete=stage_complete(self.phases)
        decision='unfinished'
        if self.phases.get('screen',{}).get('status')=='passed':
            if not self.phases['screen']['result']['advance_to_paired']:decision='improvement_not_demonstrated'
            elif self.phases.get('paired',{}).get('status')=='passed':
                decision='latency_passed' if self.phases['paired']['result']['stage_latency_passed'] else 'improvement_not_demonstrated'
        report=dict(kind='selector_qualification',identity=self.evidence,phases=self.phases,complete=complete,
                    decision=decision,full_product_acceptance=False,production_promoted=False)
        if error:
            report.update(error=str(error),status='resource_blocked' if isinstance(error,ResourceBlocked) else 'failed')
        else:report['status']='complete' if complete else 'partial'
        save(self.output/'summary.json',report)
        lines=['# GPU selector qualification','',f"Status: **{report['status']}**. Decision: **{decision}**.",'',
               'Native build: `'+self.evidence['build']+'`. Fixed 12GiB budget; CPU/full versus GPU/full.',
               '', '| Phase | Status |','|---|---|']
        lines += [f"| {name} | {row['status']} |" for name,row in self.phases.items()]
        cached=[(name,row['result']) for name,row in self.phases.items() if name.startswith('cached-') and row.get('status')=='passed']
        if cached:
            lines+=['','## Cached-token comparison','','| Context | CPU ms | GPU ms | Paired ratio | 95% interval |',
                    '|---|---:|---:|---:|---|']
            for name,r in cached:
                bounds=r['latency_ratio'];lines.append(f"| {name[7:]} | {r['control_median_ms']:.3f} | {r['candidate_median_ms']:.3f} | {bounds['median']:.4f} | {bounds['low']:.4f}–{bounds['high']:.4f} |")
        for phase in ('screen','paired'):
            if self.phases.get(phase,{}).get('status')!='passed':continue
            rows=self.phases[phase]['result']['measurements']
            lines+=['',f'## Normal requests: {phase}','','| Workload | Arm | Median TTFT s | Median tokens/s | Median request s | Peak Metal GiB |',
                    '|---|---|---:|---:|---:|---:|']
            for case in CASES:
                for arm in ('cpu-selection','gpu-selection'):
                    selected=[r for r in rows if r['name']==case and r['configuration']==arm]
                    if selected:lines.append(f"| {case} | {arm} | {statistics.median(r['ttft_ms'] for r in selected)/1000:.3f} | {statistics.median(r['tokens_per_second'] for r in selected):.3f} | {statistics.median(r['request_ms'] for r in selected)/1000:.3f} | {max(r['peak_metal_bytes'] for r in selected)/GiB:.3f} |")
        if error:lines+=['',str(error)]
        if 'profile' in self.phases and self.phases['profile'].get('status')=='passed':
            lines+=['','## Next optimization',self.phases['profile']['result']['recommendation']]
        lines+=['','Raw reports and hashes are retained in each phase directory. No production promotion or sustained coding-quality qualification is claimed.']
        (self.output/'README.md').write_text('\n'.join(lines)+'\n')

    def phase(self,name,action,validate):
        self.active=name;directory=self.output/name
        if self.previous:
            saved=json.loads((self.previous/'summary.json').read_text()).get('phases',{}).get(name,{})
            if saved.get('status')=='passed':
                import_sealed(self.previous/name,directory,saved['files_sha256'])
                result=validate(directory)
                self.phases[name]=dict(status='passed',result=result,files_sha256=saved['files_sha256'],imported=True)
                self.guard.check_identity();self.guard.check_resources();self.publish();return result
        directory.mkdir()
        self.phases[name]=dict(status='running');self.publish()
        try:
            self.guard.check_identity();self.guard.check_resources()
            action(directory)
            result=validate(directory)
            self.guard.check_identity();self.guard.check_resources()
            self.phases[name]=dict(status='passed',result=result,files_sha256=seal(directory))
            self.publish();return result
        except BaseException as error:
            self.phases[name]=dict(status='resource_blocked' if isinstance(error,ResourceBlocked) else 'failed',error=str(error))
            self.publish(error);raise

    def call(self,command,log,timeout=7200,validation=False):
        env=dict(os.environ)
        for key in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION'):
            env.pop(key,None)
            if validation:env[key]='1'
        print(log.stem,flush=True)
        with log.open('w') as stream:self.guard.run(command,stdout=stream,timeout=timeout,env=env)

    def admit(self,directory,config,cached=False):
        extra=['--cached-token-replay','--expert-slots','480'] if cached else []
        stem=directory/f'admission-{len(list(directory.glob("admission-*.log"))):03d}'
        with stem.with_suffix('.log').open('w') as log:
            return normal.inspect_admission(self.binary,self.shared+normal.config_args(config)+extra,
                                            stem,12*GiB,512,log,self.guard)

    def recovery(self):
        def action(d):
            self.admit(d,self.configs[1],True)
            self.call([ROOT/'build/qwen/test_qwen'],d/'native-tests.log',600,True)
            self.call([sys.executable,'-m','unittest','discover','-s','scripts/qwen','-p','test_*.py'],d/'tooling-tests.log',600)
            self.call([ROOT/'build/qwen/qwen_cached_recovery',self.model,self.prepared,d/'report.json'],d/'recovery.log',600,True)
        def validate(d):
            report=json.loads((d/'report.json').read_text())
            if report.get('passed') is not True or len(report['cases'])!=2:raise ValueError('Missing successful recovery cases')
            for row,target in zip(report['cases'],(1,97)):
                if not all(row.get(k) is True for k in ('cancelled','configuration_restored','gpu_drained','reused_model_succeeded')) or row['window']['target_records']!=target or not row['window']['validated']:
                    raise ValueError('Incomplete recovery evidence')
                if sha(confined(d,row['trace']))!=row['trace_sha256']:raise ValueError('Recovery trace hash differs')
            check_machine(report['before'],self.evidence);check_machine(report['after'],self.evidence)
            return dict(cancelled_and_reused=True)
        return self.phase('recovery',action,validate)

    def capture(self):
        def action(d):
            for name,conversation in normal.workloads(self.seed,2,True).items():
                if name=='prompt_2k':continue
                self.admit(d,self.configs[0]);inp=d/(name+'-workload.json');save(inp,conversation)
                dest=d/(name+'-capture')
                self.call([self.binary,'bench',*self.shared,*normal.config_args(self.configs[0]),'--workload-file',inp,
                           '--repetitions','1','--sparse-capture',dest,'--json',d/(name+'.json')],d/(name+'.log'))
                self.call([ROOT/'build/qwen/qwen_sparse_replay',dest/'manifest.json',d/(name+'-replay.json')],d/(name+'-replay.log'),600,True)
        def validate(d):
            cases=[];total=0
            for name,count in [('prompt_4k',2),('prompt_7k',2),('append_128',4)]:
                report=json.loads((d/(name+'-replay.json')).read_text());manifest=json.loads((d/(name+'-capture/manifest.json')).read_text())
                total+=check_capture(manifest,report,name,self.evidence,d/(name+'-capture'));cases+=report['cases']
                native=json.loads((d/(name+'.json')).read_text())
                if not native.get('complete') or not normal.fixed_sampling(native.get('sampling')):raise ValueError('Incomplete capture request')
                for row in native['runs']:
                    check_machine(row['after'],self.evidence);check_configuration(row['after'],self.configs[0])
            if total>256*1024**2:raise ValueError('Captures exceed 256MiB')
            return dict(exact_cases=len(cases),capture_bytes=total)
        return self.phase('capture',action,validate)

    def qualify(self):
        def action(d):
            for case in ('boundary','append','7k'):
                self.admit(d,self.configs[1])
                config=self.configs[1]
                options={k:v for k,v in config.items() if k not in ('name','kernel_policy','io_workers')}
                self.call([sys.executable,ROOT/'scripts/qwen/qualify_exact_sessions.py','--case',case,'--output',d/case,
                           '--memory-gb','12','--evidence',self.output/'identity.json',*normal.config_args(options)],d/(case+'.log'),43200,True)
        def validate(d):
            for case in ('boundary','append','7k'):
                reference=json.loads((d/case/'reference.json').read_text());candidate=json.loads((d/case/'candidate.json').read_text())
                compare_sessions(reference,candidate,self.evidence['build'])
                n,a={'boundary':(2053,129),'append':(4096,128),'7k':(7168,128)}[case]
                sized=lambda count:(self.seed*((count+len(self.seed)-1)//len(self.seed)))[:count]
                for report in (reference,candidate):
                    config=report['case']
                    for key,value in dict(prefix=sized(n),append=sized(a),continuation=[760,369],context=8192,
                                          layers=48,memory_gib=12,panels=[512],require_exact_panel=True).items():
                        if config.get(key)!=value:raise ValueError('Session workload differs')
                for row in candidate['runs']:
                    for key in ('continued_statistics','after_fresh'):check_configuration(row[key],self.configs[1])
                for report in (reference,candidate):
                    for row in report['runs']:
                        for key in ('continued_statistics','after_fresh'):check_machine(row[key],self.evidence)
            return dict(all_state_exact=True,cases=['boundary','append','7k'])
        return self.phase('qualify',action,validate)

    def cached(self,n):
        tokens=(self.seed*((n+len(self.seed)-1)//len(self.seed)))[:n]+[760]
        def action(d):
            self.admit(d,self.configs[1],True);save(d/'tokens.json',tokens)
            self.call([self.binary,'bench',*self.shared,*normal.config_args(self.configs[1]),'--cached-token-replay',
                       '--cached-compare','--cached-compare-axis','sparse_selection','--tokens-file',d/'tokens.json',
                       '--cached-progress',d/'progress.jsonl',
                       '--repetitions','5','--json',d/'report.json'],d/'cached.log')
        return self.phase('cached-'+str(n),action,lambda d:check_cached(json.loads((d/'report.json').read_text()),self.evidence,tokens))

    def normal(self,mode):
        def action(d):
            save(d/'configs.json',self.configs)
            args=argparse.Namespace(binary=self.binary,artifact='mixed-4_8bit',model=self.model,prepared=self.prepared,
                workload=WORKLOAD,config=d/'configs.json',output=d/'normal',mode=mode,comparison_purpose='experiment',
                pairs=2 if mode=='screen' else 5,memory_gb=12,cases=None,include_7k=mode=='screen',timeout=7200,resume_from=None)
            if self.previous and any((self.previous/mode/'normal'/name).exists() for name in ('summary.json','progress.json')):
                args.resume_from=self.previous/mode/'normal'
            normal.run(args,self.guard)
        def validate(d):
            from benchmark_sparse_attention import acceptance
            report=json.loads((d/'normal/summary.json').read_text())
            # Reconstruct every observation from native reports, even when reusing a sealed phase.
            temporary=d/'validated-pairs'
            if temporary.exists():raise ValueError('Unexpected validation output')
            # import_pairs needs no destination writes when validation_only is selected.
            cases=normal.workloads(self.seed,64 if mode=='screen' else 256,mode=='screen')
            expected_identity=dict(kind='exact_kernel_normal_requests',mode=mode,comparison_purpose='experiment',
                build=self.evidence['build'],artifact='mixed-4_8bit',artifact_revision=self.evidence['artifact_revision'],
                workload_sha256=sha(WORKLOAD),runner_sha256=sha(Path(normal.__file__)),configurations=self.configs,
                budget_bytes=12*GiB,cases=list(cases),pairs=2 if mode=='screen' else 5,
                output_tokens=64 if mode=='screen' else 256,evidence_identity=self.evidence)
            validated=normal.import_pairs(d/'normal',temporary,expected_identity,
                self.configs,normal.workloads(self.seed,64 if mode=='screen' else 256,mode=='screen'),2 if mode=='screen' else 5,
                dict(build=self.evidence['build'],revision=self.evidence['artifact_revision']),64 if mode=='screen' else 256,validation_only=True)
            if len(validated)!=len(report['measurements']):raise ValueError('Incomplete normal pairs')
            if report.get('evidence_identity')!=self.evidence:raise ValueError('Normal evidence identity differs')
            if mode=='screen':result=screen_gate(report)
            else:
                if report.get('complete') is not True:raise ValueError('Incomplete paired comparison')
                recomputed=dict(report,comparisons=normal.summarize(validated,self.configs))
                if recomputed['comparisons']!=report['comparisons']:raise ValueError('Changed confidence summary')
                result=acceptance(recomputed);result['confidence_bounds']=recomputed['comparisons'][0]['confidence_bounds']
            result['measurements']=[{k:r[k] for k in ('name','pair','configuration','ttft_ms','tokens_per_second',
                'decode_ms_per_token','request_ms','p95_ms','peak_metal_bytes','footprint_bytes','expert_slots')} for r in validated]
            return result
        return self.phase(mode,action,validate)

    def profile(self):
        gpu=self.phases.get('paired',{}).get('result',{}).get('stage_latency_passed',False)
        config=self.configs[int(gpu)]
        def action(d):
            self.admit(d,config,True)
            n=4096;tokens=(self.seed*((n+len(self.seed)-1)//len(self.seed)))[:n]+[760];save(d/'tokens.json',tokens)
            self.call([self.binary,'bench',*self.shared,*normal.config_args(config),'--cached-token-replay',
                       '--tokens-file',d/'tokens.json','--repetitions','3','--dispatch-profile',d/'cached-profile.json',
                       '--json',d/'cached.json'],d/'cached.log')
            # Full request diagnostics use existing command groups and complete
            # aggregate counts; bounded detailed samples may truncate at 100k.
            # Eight outputs keep diagnostic overhead and evidence size bounded.
            for name,conversation in normal.workloads(self.seed,8).items():
                if name=='prompt_4k':continue
                self.admit(d,config);save(d/(name+'-workload.json'),conversation)
                self.call([self.binary,'bench',*self.shared,*normal.config_args(config),'--workload-file',d/(name+'-workload.json'),
                           '--repetitions','1','--phase-profile',d/(name+'-profile.json'),'--json',d/(name+'.json')],d/(name+'.log'))
        def validate(d):
            report=json.loads((d/'cached.json').read_text());profile=json.loads((d/'cached-profile.json').read_text())
            attribution=summarize_profile(report,profile)
            if len(report['runs'])!=3 or report['prompt_tokens']!=4096:raise ValueError('Incomplete profile repetitions/context')
            for row in report['runs']:
                check_machine(row['after'],self.evidence)
                checked=copy.deepcopy(row['after']);checked['execution']['cached_token_replay']=False
                check_configuration(checked,config)
            requests=[]
            for name in ('prompt_2k','append_128'):
                native=json.loads((d/(name+'.json')).read_text());detail=json.loads((d/(name+'-profile.json')).read_text())
                if not native.get('complete') or not normal.fixed_sampling(native.get('sampling')):raise ValueError('Incomplete diagnostic request')
                rows=native['runs'];last=rows[-1]
                if last['name']!=name or last['output_tokens']!=8 or last['finish_reason']!='length':raise ValueError('Diagnostic workload differs')
                if name=='append_128' and (len(rows)!=2 or rows[0]['finish_reason']!='primed' or rows[0]['output_tokens'] or
                    last['reused_tokens']!=4096 or last['prefill_tokens']!=128 or last['pending_tokens_ingested']):
                    raise ValueError('Diagnostic append lost computation reuse')
                for row in rows:
                    check_machine(row['after'],self.evidence);check_configuration(row['after'],config)
                    if row['after']['metal']['kernels']['counter_profile'] or not row['profiling_enabled']:
                        raise ValueError('Wrong diagnostic profiling mode')
                # Exact generated output must agree with the uninstrumented screen.
                screen_path=self.output/'screen/normal/summary.json'
                if screen_path.exists():
                    screen=json.loads(screen_path.read_text())
                    expected=next(r['output_token_ids'][:8] for r in screen['measurements'] if r['name']==name and r['configuration']==config['name'])
                    if last['output_token_ids']!=expected:raise ValueError('Profiling changed generated tokens')
                counts=sum(r['count'] for r in detail['operation_summary'].values())
                dispatches=last['after']['metal']['dispatches']-rows[0]['before']['metal']['dispatches']
                if counts!=dispatches:raise ValueError('Incomplete aggregate diagnostic coverage')
                requests.append(dict(case=name,detailed_trace_truncated=detail['truncated'],aggregate_dispatches=counts,
                    ttft_ms_instrumented=last['time_to_first_token_ms'],phases=phase_costs(last),
                    expert_dependencies=detail.get('expert_dependencies',[])))
            result=dict(configuration=config['name'],cached_attribution=attribution,requests=requests,
                        normal_request_latency_qualified=False)
            result['recommendation']=recommend(attribution,requests)
            # validate() is also used for immutable imported evidence; avoid writes.
            return result
        return self.phase('profile',action,validate)


def phase_costs(row):
    result={}
    for name,p in row['phases'].items():
        a,b=p['before'],p['after']
        def reads(s):return s['checkpoint_application_read_bytes']+s['prepared']['application_read_bytes']
        dependencies={key:b.get('phase_dependencies',{}).get(name,{}).get(key,0)-a.get('phase_dependencies',{}).get(name,{}).get(key,0)
                      for key in ('coordinator_wait_ns','read_queue_sum_ns','read_service_sum_ns','ready_to_gpu_sum_ns')}
        result[name]=dict(application_read_bytes=reads(b)-reads(a),dependencies=dependencies,
            cpu_encode_ns=b['metal']['cpu_encode_ns']-a['metal']['cpu_encode_ns'],
            cpu_gpu_wait_ns=b['metal']['cpu_gpu_wait_ns']-a['metal']['cpu_gpu_wait_ns'],
            gpu_command_ns=b['metal']['gpu_command_ns']-a['metal']['gpu_command_ns'],
            cache_before=a['expert_cache'],cache_after=b['expert_cache'],device_before=a['storage'],device_after=b['storage'])
    return result


def recommend(attribution,requests):
    top=attribution['kernels'][0]
    stage=attribution['stages'][0]
    return (f"Investigate the largest measured cached GPU operation, {top['name']} "
            f"({top['mean_ms_per_token']:.2f}ms per token in instrumented passes); "
            f"the largest aggregate stage is {stage['name']}. First reproduce this operation on real captured "
            f"inputs, then design one exact-arithmetic implementation. A 25–50% reduction in that pass "
            f"would remove roughly {top['mean_ms_per_token']*.25:.2f}–{top['mean_ms_per_token']*.5:.2f}ms "
            "of instrumented GPU work, a scenario range rather than a predicted request speedup. "
            "Validate exact operator outputs, routes and all persistent state, then repeat unprofiled paired requests. "
            "Use the separate request dependency measurements to assess whether SSD waits hide that opportunity. "
            "GPU pass times and CPU waits overlap and cannot be added. No new optimization was implemented.")


def run(args):
    if args.screen:raise ValueError('Selector phases use --resume-from for bound screen evidence; --screen belongs to three-arm')
    contexts=args.contexts or [4096,2048,7168]
    if len(set(contexts))!=len(contexts):raise ValueError('Duplicate cached contexts')
    if args.phase=='all' and set(contexts)!={2048,4096,7168}:raise ValueError('All-phase qualification requires all cached contexts')
    if any(os.environ.get(k) not in (None,'','0') for k in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION')):
        raise ValueError('Unset GPU validation variables; correctness phases enable them explicitly')
    args.output.mkdir(parents=True,exist_ok=False)
    experiment=None
    try:
        experiment=Experiment(args)
        lock_path=ROOT/'.cache/qwen-qualification.lock';lock_path.parent.mkdir(exist_ok=True)
        with lock_path.open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ResourceBlocked('Another qualification workload owns the GPU lease')
            experiment.guard.check_identity();experiment.guard.check_resources(initial=True)
            selected=['recovery','capture','qualify','cached','screen','paired','profile'] if args.phase=='all' else [args.phase]
            for phase in selected:
                if phase=='cached':
                    for n in contexts:experiment.cached(n)
                elif phase=='paired':
                    if 'screen' not in experiment.phases:
                        if not experiment.previous:raise ValueError('Paired phase requires --resume-from with a completed screen')
                        # Imports a completed screen; never silently starts another screen.
                        saved=json.loads((experiment.previous/'summary.json').read_text()).get('phases',{}).get('screen',{})
                        if saved.get('status')!='passed':raise ValueError('Paired phase requires a completed screen')
                        experiment.normal('screen')
                    if experiment.phases['screen']['result']['advance_to_paired']:experiment.normal('paired')
                    else:experiment.phases['paired']=dict(status='not_required',reason='screen did not justify longer comparison');experiment.publish()
                elif phase=='profile':
                    if args.phase!='all':
                        if not experiment.previous:raise ValueError('Profile requires --resume-from with a completed decision')
                        prior=json.loads((experiment.previous/'summary.json').read_text()).get('phases',{})
                        if prior.get('screen',{}).get('status')!='passed':raise ValueError('Profile requires a completed screen')
                        experiment.normal('screen')
                        if experiment.phases['screen']['result']['advance_to_paired']:
                            if prior.get('paired',{}).get('status')!='passed':raise ValueError('Profile requires the paired result')
                            experiment.normal('paired')
                    experiment.profile()
                elif phase=='screen':experiment.normal('screen')
                else:getattr(experiment,phase)()
            experiment.publish()
    except BaseException as error:
        if experiment:experiment.publish(error)
        else:save(args.output/'summary.json',dict(complete=False,status='failed',error=str(error),production_promoted=False))
        raise
