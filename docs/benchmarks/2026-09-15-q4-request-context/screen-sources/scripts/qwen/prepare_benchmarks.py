#!/usr/bin/env python3
"""Create fixed, inspectable token workloads; results remain unmeasured targets."""
import argparse
import json
from pathlib import Path
from tokenizers import Tokenizer
from verify_checkpoint import ROOT

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--model',type=Path,default=ROOT/'.cache/models/qwen38-flash-next')
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--prompt-file',type=Path,help='Representative coding context to tile into controlled replay fixtures')
args=ap.parse_args()
tokenizer=Tokenizer.from_file(str(args.model/'tokenizer.json'))
text=args.prompt_file.read_text() if args.prompt_file else '''Review this Python code and suggest a correct implementation with tests.
def binary_search(items, value):
    lo, hi = 0, len(items)
    while lo < hi:
        mid = (lo + hi) // 2
        if items[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo < len(items) and items[lo] == value else -1
Consider empty lists, duplicate values, missing values, and boundary cases.
'''
seed=tokenizer.encode(text).ids
if not seed: raise SystemExit('Empty benchmark prompt')
def sized(n): return (seed*((n+len(seed)-1)//len(seed)))[:n]
cases=[dict(name='prompt_2k',tokens=sized(2048),max_tokens=256),
       dict(name='prompt_4k',tokens=sized(4096),max_tokens=256),
       dict(name='append_128',tokens=sized(128),append=True,max_tokens=256),
       dict(name='prompt_7k',tokens=sized(7168),max_tokens=256)]
args.out.parent.mkdir(parents=True,exist_ok=True)
args.out.write_text(json.dumps(cases,indent=2)+'\n')
print(f'Wrote {args.out}; controlled synthetic replay, not a coding-quality score')
