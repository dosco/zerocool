#!/usr/bin/env python3
"""Bounded real coding continuations, serial agreement and actual draft costs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import build_mtp_continuation as builder
from benchmark_host import build_probe,preflight
from cache_residency import require
from combined_q4 import freeze
from perfect_draft import configuration,source_input
from prepare_mtp import verify as verify_artifact
from qualification_evidence import ResourceBlocked,save,sha,verify_seal
from screen_mtp_forward import clean,PREPARED,ROOT
from screen_mtp_recovery import equivalent
from stage200 import Experiment

BASE=ROOT/'docs/benchmarks/2026-09-16-mtp-continuation'
PROMPTS=(
    ('merge_intervals','Write a Python function merge_intervals(intervals) that merges overlapping closed intervals. Return sorted intervals without mutating the input. Handle an empty list and intervals that touch at an endpoint. Include the implementation and three small assertions testing empty input, overlapping intervals, and unsorted disjoint intervals. Explain the time complexity.'),
    ('lru_cache','Fix this Python LRU cache and include a complete replacement class plus assertions:\nfrom collections import OrderedDict\nclass LRUCache:\n    def __init__(self, capacity):\n        self.capacity = capacity\n        self.data = OrderedDict()\n    def get(self, key):\n        return self.data.get(key, -1)\n    def put(self, key, value):\n        if len(self.data) >= self.capacity:\n            self.data.popitem(last=False)\n        self.data[key] = value\n\nBoth get and put must update recency. Existing keys must not increase the size or evict another key. Capacity zero must be handled and negative capacity must raise ValueError. A get miss must return -1 without changing recency. Include assertions testing eviction after a hit, overwriting an existing key at full capacity, repeated misses, and zero capacity. Briefly explain the bug and time complexity.'),
    ('retry_backoff','Implement an async TypeScript function retry<T>(operation: () => Promise<T>, maxAttempts: number, delayMs: number): Promise<T>. Retry only rejected operations. Use exponential backoff between attempts and no delay after the final failure. Reject invalid attempt counts or negative delays before invoking operation. Preserve the final thrown error. Include a short deterministic example that fails twice then succeeds and a test of permanent failure.'),
)


def workloads(model,max_tokens):
    from tokenizers import Tokenizer
    require(max_tokens in (16,128,256),'Unsupported continuation screen length')
    tokenizer=Tokenizer.from_file(str(model/'tokenizer.json'))
    generation=json.loads((model/'generation_config.json').read_text());eos=generation['eos_token_id']
    if type(eos) is int:eos=[eos]
    rows=[]
    for name,prompt in PROMPTS:
        text='<|im_start|>user\n'+prompt+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        ids=tokenizer.encode(text,add_special_tokens=False).ids
        require(2<=len(ids)<=512 and all(0<=i<248320 for i in ids),'Invalid coding prompt')
        rows.append(dict(name=name,prompt_text=prompt,prompt_ids=ids,eos_ids=eos,max_tokens=max_tokens,
            draft_slots=32,target_prepared=str(ROOT/'.cache/prepared/q4-records-v1')))
    return rows


def select_cases(rows,names):
    if names is None:return rows
    by_name={r['name']:r for r in rows}
    require(names and len(names)==len(set(names)) and all(n in by_name for n in names),
        'Selected cases must be nonempty, unique and present in the suite')
    return [by_name[n] for n in names]


def setup(exp,directory):
    cfg,proof=builder.verify(directory);require(proof['base_native_fingerprint']==exp.frozen['build'],'Changed native base')
    save(exp.out/'producer.json',proof);save(exp.out/'draft-audit.json',verify_artifact(PREPARED))
    host=build_probe(ROOT/'.cache/benchmark-host'/hashlib.sha256(str(exp.out).encode()).hexdigest())
    freeze(exp,[*builder.inputs(cfg),*builder.generated(cfg['output']),*cfg['objects'],cfg['binary'],*host['files'],
        PREPARED/'manifest.json',PREPARED/'dense.bin',PREPARED/'experts.bin',BASE/'protocol.md',
        exp.model/'tokenizer.json',exp.model/'generation_config.json'])
    exp.env['FREELLM_Q8_EXPANDED']='packed';exp.guard.check_resources(initial=True)
    return cfg,host


def host_check(exp,host,stem,full=True):
    observed=preflight(exp,host,stem)
    if observed['host']['power_source']!='AC Power':raise ResourceBlocked('AC power required')
    raw=json.loads((exp.out/(stem+'-host.json')).read_text())
    if full and raw['process']['reclaimable_bytes']<int(13.5*1024**3):
        raise ResourceBlocked('Fixed 12GiB experiment requires 13.5GiB currently available memory')


def observe(raw,work,input_sha,mode):
    require(raw.get('kind')=='native_mtp_continuation_v1' and raw.get('complete') is True and
        raw['mode']==mode and raw['validation'] is (mode in ('validate','fast-validate')) and
        raw['input_sha256']==input_sha,'Incomplete or incompatible continuation')
    count=8 if raw['validation'] else work['max_tokens'];cycles=raw['cycles'];ids=[]
    offset=len(work['prompt_ids']);proposed=accepted=wall=0
    for c in cycles:
        n=c['committed_tokens'];width=c['width']
        require(c['offset']==offset and width in ((1,) if mode=='serial' else (1,4)) and
            len(c['proposals'])==width and 1<=n<=width and c['accepted_proposals']==n-1 and
            c['wall_ns']>0 and sum(c[k] for k in ('draft_ns','verify_ns','recovery_ns'))<=c['wall_ns'],
            'Invalid continuation cycle coverage')
        require(raw['validation'] or c['forced_rejection'] is False,'Forced timing proposal')
        ids+=c['proposals'][:n];offset+=n;proposed+=width-1;accepted+=n-1;wall+=c['wall_ns']
    require(ids==raw['committed_token_ids'] and 0<len(ids)<=count and raw['generated_tokens']==len(ids) and
        raw['requested_tokens']==count and raw['prompt_tokens']==len(work['prompt_ids']) and raw['eos_ids']==work['eos_ids'] and
        raw['proposed_tokens']==proposed and raw['accepted_proposals']==accepted and
        raw['decode_wall_ns']==wall and raw['tokens_per_second']==len(ids)*1e9/wall and
        raw['decode_including_reporting_ns']>=wall,'Changed continuation totals')
    require(len(raw['row_logits_sha256'])==len(ids) and all(isinstance(h,str) and len(h)==64 for h in raw['row_logits_sha256']),
        'Missing full-vocabulary logit coverage')
    require(not any(i in work['eos_ids'] for i in ids[:-1]) and
        ((raw['stop_reason']=='eos' and ids[-1] in work['eos_ids']) or
         (raw['stop_reason']=='length' and len(ids)==count and ids[-1] not in work['eos_ids'])),'Invalid EOS/length termination')
    require(raw['final_target_state']['valid'] is True and raw['final_target_state']['tokens']==offset and
        all(l['position']==offset for l in raw['final_target_state']['layers']),'Target state position differs')
    if mode!='serial':require(raw['final_draft_state']['valid'] is True and raw['final_draft_state']['position']==offset-1,'Draft state position differs')
    if raw['validation']:require(any(c['forced_rejection'] and c['accepted_proposals']==0 for c in cycles),'Missing forced rejection')
    return dict(**clean(raw),tokens_per_second=raw['tokens_per_second'],generated_tokens=len(ids),
        proposed_tokens=proposed,accepted_proposals=accepted,acceptance_rate=accepted/proposed if proposed else None,
        mean_committed_per_cycle=len(ids)/len(cycles),rejected_cycles=sum(c['committed_tokens']<c['width'] for c in cycles),
        component_ms_per_token={k:sum(c[k+'_ns'] for c in cycles)/len(ids)/1e6 for k in ('draft','verify','recovery','wall')},
        completed_requested_length=len(ids)==count,stop_reason=raw['stop_reason'])


def compare(serial,candidate):
    require(serial['mode']=='serial' and candidate['mode']=='fast-timing' and
        all(serial[k]==candidate[k] for k in ('input_sha256','admission','draft_manifest_sha256','prime_logits_sha256',
            'committed_token_ids','row_logits_sha256','next_id','stop_reason','final_target_state')),
        'Continuation differs from fresh serial decoding')
    require(serial['before']['expert_cache']['diagnostic_cache_state']==candidate['before']['expert_cache']['diagnostic_cache_state'],
        'Different initial target cache')
    return dict(exact_all_logits_tokens_and_state=True,candidate_to_serial_ratio=candidate['decode_wall_ns']/serial['decode_wall_ns'])


def fixture(output,directory):
    source=ROOT/'docs/benchmarks/2026-09-16-mtp-forward/fixture-04';verify_seal(source,sha(source/'evidence-files.json'))
    exp=Experiment(output,'mtp_continuation_fixture_v1',[],dict(kind='same_four_row_fixture'),240)
    with exp:
        cfg,host=setup(exp,directory)
        exp.command([cfg['binary'],'--continuation-self-test'],'bounds',limit=5)
        exp.command([cfg['binary'],'--checkpoint-self-test'],'checkpoint',limit=5)
        data=json.loads((source/'input.json').read_text());original=Path(data['hidden_file'])
        (exp.out/'hidden.f32').write_bytes(original.read_bytes());data['hidden_file']=str(exp.out/'hidden.f32');save(exp.out/'input.json',data)
        freeze(exp,[exp.out/'input.json',exp.out/'hidden.f32']);host_check(exp,host,'fixture',False)
        exp.command([cfg['binary'],exp.model,PREPARED,exp.out/'input.json',exp.out/'native.json','fixture'],'native',limit=90,validation=True)
        native=json.loads((exp.out/'native.json').read_text());require(native['complete'] is True,'Incomplete native fixture')
        exp.report['native']=clean(native)
        exp.command([sys.executable,ROOT/'scripts/qwen/reference_mtp_forward.py','--prepared',PREPARED,'--model',exp.model,
            '--input',exp.out/'input.json','--native',exp.out/'native.json','--output',exp.out/'reference.json'],'reference',limit=100)
        ref=json.loads((exp.out/'reference.json').read_text());require(ref['passed'] is True,'Independent reference failed')
        exp.report['reference']=ref
        if not all(exp.report['native'][k] for k in ('clean_memory','clean_host')):raise ResourceBlocked('Fixture resource disturbance')
        exp.report['status']='forward_fixture_validated'
    return exp.report


def run(output,directory,fixture_source,max_tokens=128,validation=False,validation_source=None,selected_cases=None):
    require(not validation or selected_cases is None,'Case selection is only supported for normal screens')
    verify_seal(fixture_source,sha(fixture_source/'evidence-files.json'))
    fixture_report=json.loads((fixture_source/'summary.json').read_text())
    require(fixture_report['complete'] is True and fixture_report['status']=='forward_fixture_validated','Missing independent fixture')
    if validation:
        work,_=source_input();work.update(eos_ids=[248046,248044],max_tokens=16,draft_slots=32,
            target_prepared=str(ROOT/'.cache/prepared/q4-records-v1'));cases=[work]
    else:cases=select_cases(workloads(ROOT/'.cache/qwen-mixed-reference',max_tokens),selected_cases)
    exp=Experiment(output,'mtp_continuation_validation_v1' if validation else 'mtp_continuation_screen_v1',
        [configuration(4,expert_slots=1460)],cases,480 if validation else 1200)
    with exp:
        cfg,host=setup(exp,directory)
        require(json.loads((fixture_source/'producer.json').read_text())==json.loads((exp.out/'producer.json').read_text()),'Changed fixture producer')
        exp.report.update(fixture_source=str(fixture_source.resolve()),cases=[],paired_confidence_qualified=False,
            screen_scope='rejection_validation' if validation else 'selected_cases' if selected_cases is not None else 'all_cases',
            selected_case_names=[w.get('name','rejection-recovery') for w in cases])
        if not validation:
            require(validation_source is not None,'Missing continuation recovery validation')
            verify_seal(validation_source,sha(validation_source/'evidence-files.json'))
            proof=json.loads((validation_source/'summary.json').read_text())
            require(proof['kind']=='mtp_continuation_validation_v1' and proof['complete'] is True and
                proof['status']=='numerically_validated' and proof['cases']==[dict(case=0,name='rejection-recovery',exact_recovery=True)] and
                json.loads((validation_source/'producer.json').read_text())==json.loads((exp.out/'producer.json').read_text()),
                'Incomplete or incompatible continuation validation')
            freeze(exp,[validation_source/p for p in ('summary.json','producer.json','case-0-validate.json','case-0-fast-validate.json','evidence-files.json')])
            exp.report['validation_source']=str(validation_source.resolve())
        for i,work in enumerate(cases):
            input_path=exp.out/f'case-{i}.json';save(input_path,work);freeze(exp,[input_path]);values={}
            modes=('validate','fast-validate') if validation else (('serial','fast-timing') if i%2==0 else ('fast-timing','serial'))
            for mode in modes:
                stem=f'case-{i}-{mode}';host_check(exp,host,stem)
                exp.command([cfg['binary'],exp.model,PREPARED,input_path,exp.out/(stem+'.json'),mode],stem,
                    limit=180 if validation else 360,validation=validation)
                raw=json.loads((exp.out/(stem+'.json')).read_text());observed=observe(raw,work,sha(input_path),mode)
                exp.report.setdefault('samples',[]).append(dict(case=i,mode=mode,source=stem+'.json',sha256=sha(exp.out/(stem+'.json')),**observed));exp.persist()
                if not observed['clean_memory'] or not observed['clean_host']:raise ResourceBlocked('Continuation memory or host disturbed')
                values[mode]=raw
            if validation:
                equivalent(values['validate'],values['fast-validate']);decision=dict(exact_recovery=True)
            else:decision=compare(values['serial'],values['fast-timing'])
            exp.report['cases'].append(dict(case=i,name=work.get('name','rejection-recovery'),**decision));exp.persist()
        exp.report['status']='numerically_validated' if validation else 'continuation_screen_complete'
        if not validation:
            exp.report['all_candidates_at_least_5_tps']=all(s['tokens_per_second']>=5 for s in exp.report['samples'] if s['mode']=='fast-timing')
            exp.report['all_requested_lengths_complete']=all(s['completed_requested_length'] for s in exp.report['samples'])
    return exp.report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['fixture','validate','screen'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--build',type=Path,required=True)
    p.add_argument('--fixture',type=Path);p.add_argument('--max-tokens',type=int,choices=[16,128,256],default=128)
    p.add_argument('--validation-source',type=Path)
    p.add_argument('--case',dest='selected_cases',action='append',choices=[name for name,_ in PROMPTS],
        help='Run only this named case; repeat for several. Reports explicitly retain the selected scope.')
    a=p.parse_args()
    if a.selected_cases is not None and a.mode!='screen':p.error('--case is only supported with screen')
    if a.mode=='fixture':fixture(a.output,a.build)
    else:
        require(a.fixture is not None,'--fixture is required');run(a.output,a.build,a.fixture,a.max_tokens,a.mode=='validate',a.validation_source,a.selected_cases)
