#!/usr/bin/env python3
"""Exercise a running native server with real inference. No mocked responses."""
import argparse
import json
from pathlib import Path
import socket
import time
import urllib.error
import urllib.request

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--url',default='http://127.0.0.1:8080')
ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args()
records=[]


def post(body):
    req=urllib.request.Request(args.url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    return urllib.request.urlopen(req,timeout=600)


def check(name,fn):
    start=time.monotonic()
    try:
        detail=fn()
        row=dict(name=name,passed=True,seconds=time.monotonic()-start,detail=detail)
    except Exception as e:
        row=dict(name=name,passed=False,seconds=time.monotonic()-start,error=str(e))
    records.append(row)
    args.out.write_text(json.dumps(dict(passed=all(r['passed'] for r in records),checks=records),indent=2)+'\n')
    print(json.dumps(row),flush=True)


prompt=dict(model='qwen3.8-flash-next:4bit',messages=[dict(role='user',content='Reply with exactly OK.')],max_tokens=8,temperature=0)


def plain():
    with post(prompt) as f: response=json.load(f)
    assert response['choices'][0]['message']['content'].strip()=='OK', response
    assert response['usage']['completion_tokens']>0
    return response


def stream():
    text='';finished=False;done=False
    with post(dict(prompt,stream=True)) as f:
        for raw in f:
            line=raw.decode().strip()
            if not line.startswith('data: '): continue
            if line=='data: [DONE]': done=True;break
            event=json.loads(line[6:]);assert 'error' not in event,event
            choice=event['choices'][0];text+=choice['delta'].get('content','')
            finished|=choice['finish_reason'] is not None
    assert text.strip()=='OK' and finished and done,(text,finished,done)
    return dict(text=text,finished=finished,done=done)


def invalid():
    for body in [dict(prompt,max_tokens=8192),dict(prompt,model='other-model')]:
        try:
            with post(body) as f: raise AssertionError(f'Request unexpectedly succeeded: {f.status}')
        except urllib.error.HTTPError as e:
            assert e.code==400
            assert json.load(e)['error']['message']
    return 'Both invalid requests were rejected before inference'


def tools():
    body=dict(prompt,max_tokens=96,messages=[dict(role='user',content='Call read_file with path maths.py. Do not answer in prose.')],
        tools=[dict(type='function',function=dict(name='read_file',description='Read a file',parameters=dict(type='object',properties=dict(path=dict(type='string')),required=['path'])))])
    with post(body) as f: response=json.load(f)
    choice=response['choices'][0];assert choice['finish_reason']=='tool_calls',response
    call=choice['message']['tool_calls'][0]
    assert call['function']['name']=='read_file' and json.loads(call['function']['arguments'])=={'path':'maths.py'},call
    return response


def disconnect():
    from urllib.parse import urlsplit
    url=urlsplit(args.url)
    body=json.dumps(dict(prompt,stream=True,max_tokens=512,messages=[dict(role='user',content='Write a long explanation of sorting algorithms.')])).encode()
    with socket.create_connection((url.hostname,url.port or 80),timeout=30) as s:
        s.sendall(f'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n'.encode()+body)
        assert b'200' in s.recv(4096)
    # The next real request must work after cancellation and buffer draining.
    return plain()


with urllib.request.urlopen(args.url+'/v1/models',timeout=10) as f:
    assert json.load(f)['data'][0]['id']==prompt['model']
for name,fn in [('chat',plain),('stream',stream),('invalid_requests',invalid),('tool_call',tools),('disconnect_recovery',disconnect)]: check(name,fn)
raise SystemExit(0 if all(r['passed'] for r in records) else 1)
