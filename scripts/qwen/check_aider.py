#!/usr/bin/env python3
"""Pinned Aider integration in a disposable folder; real checks never use mocks."""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
from api_check import admission_error

VERSION='0.86.2'


def verify_add(path):
    source=path.read_text();tree=ast.parse(source)
    allowed=(ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.BinOp,ast.Add,ast.Sub,ast.Name,ast.Load,ast.Constant,ast.Expr)
    if any(not isinstance(node,allowed) for node in ast.walk(tree)):
        raise ValueError('Fixture output is outside the bounded arithmetic task')
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef)]
    if len(functions)!=1 or functions[0].name!='add':raise ValueError('Expected one add function')
    namespace={};exec(compile(tree,str(path),'exec'),{'__builtins__':{}},namespace)
    fn=namespace['add'];values=[(2,3,5),(-2,3,1),(0,0,0)]
    return dict(passed=all(fn(a,b)==expected for a,b,expected in values),cases=values,source=source)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url',default='http://127.0.0.1:8080')
    ap.add_argument('--aider',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--transport-only',action='store_true')
    args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    def get(path):
        with urllib.request.urlopen(args.url+path,timeout=5) as f:return json.load(f)
    initial=get('/freellm/status');model='openai/'+get('/v1/models')['data'][0]['id']
    if not args.transport_only:
        blocked=admission_error(initial)
        if blocked:
            (out/'summary.json').write_text(json.dumps(dict(complete=False,passed=False,status='resource_blocked',initial_status=initial,error=blocked),indent=2)+'\n')
            raise SystemExit(blocked)
    binary=args.aider.resolve()
    reported=subprocess.check_output([str(binary),'--version'],text=True).strip()
    if VERSION not in reported:raise SystemExit(f'Expected Aider {VERSION}; got {reported}')
    work=out/'workspace';work.mkdir()
    subprocess.run(['git','init','-q',str(work)],check=True) # Bound Aider's discovered root to this disposable repository.
    (work/'empty.env').write_text('')
    (work/'empty.yml').write_text('{}\n')
    settings=[dict(name=model,edit_format='whole',weak_model_name=model,editor_model_name=model,use_repo_map=False,
        extra_params=dict(max_tokens=256,temperature=0))]
    (work/'model-settings.yml').write_text(json.dumps(settings)) # YAML accepts JSON.
    metadata={model:dict(max_input_tokens=6144,max_output_tokens=256,max_tokens=8192,input_cost_per_token=0,output_cost_per_token=0,litellm_provider='openai',mode='chat')}
    (work/'model-metadata.json').write_text(json.dumps(metadata))
    command=[str(binary),'--model',model,'--weak-model',model,'--editor-model',model,
        '--openai-api-base',args.url+'/v1','--openai-api-key','local-only','--config','empty.yml','--env-file','empty.env',
        '--input-history-file',str(work/'input.history'),'--chat-history-file',str(work/'chat.history.md'),
        '--model-settings-file','model-settings.yml','--model-metadata-file','model-metadata.json','--map-tokens','0',
        '--no-git','--no-auto-commits','--no-analytics','--no-check-update','--no-show-model-warnings',
        '--no-show-release-notes','--no-restore-chat-history','--max-chat-history-tokens','8192',
        '--no-suggest-shell-commands','--no-auto-lint','--no-auto-test','--yes-always','--no-pretty','--timeout','600']
    env=dict(os.environ,OPENAI_API_KEY='local-only',OPENAI_API_BASE=args.url+'/v1',LITELLM_LOCAL_MODEL_COST_MAP='True',AIDER_ANALYTICS='false')
    result=dict(kind='aider_usability_v1',complete=False,passed=False,client_version=reported,
        initial_status=initial,transport_only=args.transport_only,command=command,steps=[])
    def save(): (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    def run(name,message,extra):
        start=time.monotonic()
        with (out/(name+'.log')).open('w') as log:
            completed=subprocess.run([*command,*extra,'--message',message],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        row=dict(name=name,exit_code=completed.returncode,seconds=time.monotonic()-start,log=name+'.log')
        result['steps'].append(row);save()
        if completed.returncode:raise RuntimeError(f'Aider failed; see {name}.log')
    save()
    try:
        if args.transport_only:
            run('transport','Reply with exactly OK.', ['--chat-mode','ask'])
            text=(out/'transport.log').read_text();assert 'OK' in text and 'APIConnectionError' not in text
            result['limitations']=['Checks pinned client protocol only; no real model or coding workflow is qualified.']
        else:
            path=work/'maths.py';path.write_text('def add(a, b):\n    return a - b\n')
            before=verify_add(path);assert not before['passed'];result['initial_failing_tests']=before
            run('edit','Fix add in maths.py to add its two arguments. Change only maths.py. Return the complete file in the requested edit format.', ['--edit-format','whole','maths.py'])
            result['after_edit']=verify_add(path);assert result['after_edit']['passed']
            path.write_text('def add(a, b):\n    return a + b + 1\n')
            result['injected_regression']=verify_add(path);assert not result['injected_regression']['passed']
            run('recovery','The latest tests fail: add(2, 3) returned 6, expected 5, and add(-2, 3) returned 2, expected 1. Re-read maths.py and repair add. Change only maths.py.', ['--edit-format','whole','maths.py'])
            result['after_recovery']=verify_add(path);assert result['after_recovery']['passed']
            result['limitations']=['A small two-invocation coding fixture; not a general coding-quality or sustained-session qualification.']
        result.update(complete=True,passed=True,final_status=get('/freellm/status'))
    except Exception as e:result.update(complete=True,passed=False,error=str(e))
    save();print(json.dumps({k:result[k] for k in ('complete','passed','transport_only')},indent=2));return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
