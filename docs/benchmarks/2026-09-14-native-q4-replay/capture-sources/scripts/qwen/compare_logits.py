#!/usr/bin/env python3
"""Compare complete pinned-model logits and write an explicit pass/fail report."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from build_identity import build_fingerprint


def run(args):
    n=json.loads(args.native_report.read_text());r=json.loads(args.reference_report.read_text())
    a=np.fromfile(args.reference,np.float32).astype(np.float64);b=np.fromfile(args.native,np.float32).astype(np.float64)
    revision=n['statistics'].get('artifact_revision')
    if not revision or revision!=r.get('source_revision'):
        raise ValueError('Native and independent reference artifacts differ or are not identified')
    if n['statistics']['metal']['build_fingerprint']!=build_fingerprint(Path(__file__).resolve().parents[2]):
        raise ValueError('Native report does not describe the current build')
    valid=n['tokens']==r['tokens'] and n['layers']==r['layers']==48 and a.size==b.size==248320 and np.isfinite(a).all() and np.isfinite(b).all()
    if not valid: raise SystemExit('Invalid or mismatched full-model fixtures')
    relative=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-9));cosine=float(np.clip(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-9),-1,1))
    result=dict(passed=relative<=0.02 and cosine>=0.9998,layers=48,native_tokens=n['tokens'],reference_tokens=r['tokens'],
        build_fingerprint=n['statistics']['metal']['build_fingerprint'],artifact_revision=revision,reference=r,
        native_logits_sha256=hashlib.sha256(args.native.read_bytes()).hexdigest(),
        reference_logits_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        native_report_sha256=hashlib.sha256(args.native_report.read_bytes()).hexdigest(),
        reference_report_sha256=hashlib.sha256(args.reference_report.read_bytes()).hexdigest(),
        relative_l2=relative,cosine=cosine,reference_greedy_id=int(a.argmax()),native_greedy_id=int(b.argmax()),
        thresholds=dict(relative_l2_max=0.02,cosine_min=0.9998),
        diagnostic_stream_trunk=n['statistics']['diagnostic_stream_trunk'])
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--native',type=Path,required=True)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--native-report',type=Path,required=True)
    ap.add_argument('--reference-report',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    result=run(ap.parse_args())
    raise SystemExit(0 if result['passed'] else 1)
