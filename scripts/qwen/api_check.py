#!/usr/bin/env python3
"""Check a running native server with real inference; preserve partial evidence."""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def admission_error(status):
    if status.get('test_executor'):return 'A test executor cannot establish native correctness'
    if status.get('phase')!='ready':return 'Server must be ready and idle'
    memory=status.get('process') or {}
    if any(type(memory.get(key)) is not int for key in ('physical_footprint_bytes','compressed_bytes','compressed_peak_bytes')):
        return 'Physical memory observations are unavailable'
    if memory['compressed_bytes'] or memory['compressed_peak_bytes']:return 'Engine memory is compressed; restart after freeing memory'
    if memory['physical_footprint_bytes']>22*1024**3:return 'Engine exceeds the 22GiB ceiling'
    return None


def cancellation_observed(status,request_id):
    if not request_id:return False
    return ((status.get('active_request_id')==request_id and status.get('phase') in ('cancelling','draining'))
        or (status.get('last_request_id')==request_id and status.get('last_request_cancelled') is True))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url',default='http://127.0.0.1:8080')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise SystemExit('Output exists; choose a new evidence path')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    records=[];status_latency=[]

    def get(path):
        start=time.monotonic()
        with urllib.request.urlopen(args.url+path,timeout=5) as f:r=json.load(f)
        if path=='/zerocool/status':status_latency.append((time.monotonic()-start)*1000)
        return r

    models=get('/v1/models')['data']
    if len(models)!=1:raise SystemExit('Expected exactly one selected model')
    identity=get('/zerocool/status');model=models[0]['id']
    report=dict(kind='native_api_usability_v1',complete=False,passed=False,model=model,
        initial_status=identity,checks=records,limitations=['Functional checks do not qualify request latency or coding quality.'])

    def save():
        report['status_latency_ms']=status_latency
        args.out.write_text(json.dumps(report,indent=2)+'\n')

    blocked=admission_error(identity)
    if blocked:
        report.update(status='resource_blocked',error=blocked);save();raise SystemExit(blocked)

    def post(body):
        req=urllib.request.Request(args.url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        return urllib.request.urlopen(req,timeout=600)

    def reset():
        req=urllib.request.Request(args.url+'/zerocool/session/reset',data=b'{}',headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=30) as f:assert json.load(f)['status']=='ready'

    def check(name,fn):
        start=time.monotonic();print(json.dumps(dict(phase=name,status='running')),flush=True)
        try:
            detail=fn();row=dict(name=name,passed=detail.get('acknowledgement_within_500ms',True) if isinstance(detail,dict) else True,detail=detail)
        except Exception as e:row=dict(name=name,passed=False,error=str(e))
        row['seconds']=time.monotonic()-start;records.append(row);save()
        print(json.dumps(dict(name=name,passed=row['passed'],seconds=row['seconds'])),flush=True)

    prompt=dict(model=model,messages=[dict(role='user',content='Reply with exactly OK.')],max_tokens=8,temperature=0)

    def plain(body=None):
        with post(body or prompt) as f:r=json.load(f)
        assert r['choices'][0]['message']['content'].strip()=='OK',r
        assert r['created']>0 and r['usage']['completion_tokens']>0
        return r

    def stream():
        text='';finished=False;done=False;usage=None
        with post(dict(prompt,stream=True,stream_options=dict(include_usage=True))) as f:
            for raw in f:
                line=raw.decode().strip()
                if not line.startswith('data: '):continue
                if line=='data: [DONE]':done=True;break
                event=json.loads(line[6:]);assert 'error' not in event,event
                if not event['choices']:usage=event['usage'];continue
                choice=event['choices'][0];text+=choice['delta'].get('content','') or ''
                finished|=choice['finish_reason'] is not None
        assert text.strip()=='OK' and finished and done and usage,(text,finished,done,usage)
        return dict(text=text,usage=usage)

    def invalid():
        for body in [dict(prompt,max_tokens=identity['context_limit']),dict(prompt,model='other-model'),dict(prompt,stop='END')]:
            try:
                with post(body) as f:raise AssertionError(f'Unexpected success: {f.status}')
            except urllib.error.HTTPError as e:
                assert e.code==400
                assert json.load(e)['error']['message']
        return 'Context overflow, unknown model, and unsupported stop rejected'

    def retained():
        reset();first=plain()
        messages=[*prompt['messages'],first['choices'][0]['message'],dict(role='user',content='Reply with exactly OK again.')]
        kept=plain(dict(prompt,messages=messages))
        assert kept['usage']['prompt_tokens_details']['cached_tokens']>0,kept
        reset();fresh=plain(dict(prompt,messages=messages))
        assert kept['zerocool']['output_token_ids']==fresh['zerocool']['output_token_ids']
        changed=plain(dict(prompt,messages=[dict(role='system',content='Follow the user request exactly.'),*messages]))
        assert changed['usage']['prompt_tokens_details']['cached_tokens']==0
        return dict(retained=kept,fresh=fresh,edited=changed)

    def tools():
        body=dict(prompt,max_tokens=96,messages=[dict(role='user',content='Call read_file with path maths.py. Do not answer in prose.')],
            tools=[dict(type='function',function=dict(name='read_file',description='Read a file',parameters=dict(type='object',properties=dict(path=dict(type='string')),required=['path'])))])
        with post(body) as f:r=json.load(f)
        choice=r['choices'][0];assert choice['finish_reason']=='tool_calls',r
        call=choice['message']['tool_calls'][0]
        assert call['function']['name']=='read_file' and json.loads(call['function']['arguments'])=={'path':'maths.py'},call
        messages=[*body['messages'],choice['message'],dict(role='tool',tool_call_id=call['id'],content='def add(a, b): return a + b'),dict(role='user',content='The file has been read. Reply with exactly OK.')]
        continued=plain(dict(prompt,messages=messages,tools=body['tools']))
        return dict(call=r,continued=continued)

    def disconnect(phase):
        reset()
        text=('Read these words and then explain sorting in detail. '+'apple orange banana. '*100) if phase=='prefill' else 'Explain sorting algorithms in considerable detail.'
        with post(dict(prompt,stream=True,max_tokens=256,messages=[dict(role='user',content=text)])) as f:
            request_id=f.headers.get('X-Request-Id');assert request_id,'Missing streaming request identity'
            deadline=time.monotonic()+120
            while time.monotonic()<deadline:
                state=get('/zerocool/status')
                if state['phase']==phase and state.get('active_request_id')==request_id:break
                time.sleep(.05)
            else:raise AssertionError(f'Never observed {phase}')
        started=time.monotonic();ack=None;states=[]
        deadline=started+120
        while time.monotonic()<deadline:
            state=get('/zerocool/status');states.append(state)
            if ack is None and cancellation_observed(state,request_id):ack=(time.monotonic()-started)*1000
            if not state['busy']:break
            time.sleep(.05)
        drain_ms=(time.monotonic()-started)*1000
        detail=dict(request_id=request_id,acknowledgement_ms=ack,acknowledgement_within_500ms=ack is not None and ack<=500,drain_ms=drain_ms,states=states)
        report.setdefault('cancellations',[]).append(detail);save()
        assert not state['busy'] and state.get('last_request_id')==request_id and state.get('last_request_cancelled'),state
        detail['recovery']=plain()
        return detail

    save()
    for name,fn in [('chat',plain),('stream',stream),('invalid_requests',invalid),('retained_and_edited_history',retained),('tool_result_continuation',tools),
                    ('cancel_prefill',lambda:disconnect('prefill')),('cancel_generation',lambda:disconnect('generating'))]:check(name,fn)
    report.update(complete=True,passed=all(r['passed'] for r in records),final_status=get('/zerocool/status'))
    if status_latency:report['status_p95_ms']=sorted(status_latency)[min(len(status_latency)-1,int(len(status_latency)*.95))]
    save();return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
