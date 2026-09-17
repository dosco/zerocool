#!/usr/bin/env python3
"""Choose exact-shape tile rules from alternating, real-activation fixtures."""
import argparse
import json
import statistics
from pathlib import Path
from benchmark_exact import paired_interval


def select(report):
    if report.get('kind')!='captured_operator_screen' or report.get('exact') is not True:
        raise ValueError('Requires exact captured-operator evidence')
    cases={}
    for row in report['measurements']:
        if row.get('exact') is not True:raise ValueError('Changed operator output')
        key=(json.dumps(row['matrix'],sort_keys=True),row['case'])
        values=cases.setdefault(key,{})
        variant=(row['tile'],row.get('output_rows',1),row.get('gate_pair',False))
        identity=(variant,row['repetition'])
        if identity in values or row['wall_ns']<=0:raise ValueError('Duplicate or invalid measurement')
        values[identity]=row['wall_ns']
    shapes={}
    for (shape,_),values in cases.items():
        reference=(1,1,False)
        reps=sorted(r for v,r in values if v==reference)
        if len(reps)<5 or reps!=list(range(len(reps))):raise ValueError('Requires five complete alternating repetitions')
        ratios={}
        for variant in sorted({v for v,r in values}-{reference}):
            if sorted(r for v,r in values if v==variant)!=reps:raise ValueError('Unpaired variant measurements')
            ratios[variant]=[values[variant,r]/values[reference,r] for r in reps]
        if not all((tile,1,False) in ratios for tile in (2,4,8)):raise ValueError('Missing reference tile coverage')
        shapes.setdefault(shape,[]).append(ratios)
    rules=[];evidence=[]
    for shape,cases_for_shape in sorted(shapes.items()):
        matrix=json.loads(shape);candidates=[]
        variants=set.intersection(*(set(case) for case in cases_for_shape))
        for variant in sorted(variants):
            tile,rows,pair=variant
            if matrix['rows']==1 and rows==1 and not pair:continue
            bounds=[paired_interval(case[variant]) for case in cases_for_shape]
            if all(b['high']<1 for b in bounds):
                candidates.append((statistics.mean(b['median'] for b in bounds),variant,bounds))
        if candidates:
            _,(tile,rows,pair),bounds=min(candidates)
            rule=dict(matrix,tile=tile)
            if rows!=1:rule['output_rows']=rows
            if pair:rule['gate_pair']=True
            rules.append(rule);evidence.append(dict(rule=rule,bounds=bounds))
    return dict(kind='exact_shape_policy',build_fingerprint=report['build_fingerprint'],
                artifact_revision=report['artifact_revision'],rules=rules,evidence=evidence,promoted=False)


def merge_reports(reports):
    if not reports:raise ValueError('No capture reports')
    import hashlib
    merged=dict(kind='captured_operator_screen',exact=True,build_fingerprint=reports[0]['build_fingerprint'],
                artifact_revision=reports[0]['artifact_revision'],measurements=[])
    for report in reports:
        if report.get('exact') is not True or any(report[k]!=merged[k] for k in ('kind','build_fingerprint','artifact_revision')):
            raise ValueError('Capture identities differ')
        source=hashlib.sha256(json.dumps(report.get('source',{}),sort_keys=True).encode()).hexdigest()
        merged['measurements'].extend(dict(row,case=source+'/'+row['case']) for row in report['measurements'])
    return merged


def validate_heldout(policy, report):
    if any(policy[k]!=report[k] for k in ('build_fingerprint','artifact_revision')) or report.get('exact') is not True:
        raise ValueError('Held-out identity or arithmetic differs')
    fields=('K','N','rows','group','bits','fused','gathered')
    rules={tuple(rule[k] for k in fields):rule for rule in policy['rules']}
    cases={}
    for row in report['measurements']:
        if row.get('exact') is not True:raise ValueError('Changed held-out output')
        shape=tuple(row['matrix'][k] for k in fields)
        values=cases.setdefault((shape,row['case']),{})
        identity=(row['tile'],row.get('output_rows',1),row.get('gate_pair',False),row['repetition'])
        if identity in values or row['wall_ns']<=0:raise ValueError('Duplicate or invalid held-out measurement')
        values[identity]=row['wall_ns']
    evidence=[];covered=set()
    for (shape,case),values in cases.items():
        if shape not in rules:continue
        rule=rules[shape];variant=(rule['tile'],rule.get('output_rows',1),rule.get('gate_pair',False))
        reps=sorted(r for t,n,p,r in values if (t,n,p)==(1,1,False))
        if len(reps)<5 or reps!=list(range(len(reps))):raise ValueError('Missing held-out repetitions')
        variants={(t,n,p) for t,n,p,r in values}
        # Never drop a candidate sample just because its reference is absent.
        # Every reported series must cover the same complete repetitions.
        if variant not in variants or any(sorted(r for t,n,p,r in values if (t,n,p)==v)!=reps for v in variants):
            raise ValueError('Unpaired held-out measurements')
        bounds=paired_interval([values[(*variant,r)]/values[(1,1,False,r)] for r in reps])
        evidence.append(dict(shape=shape,case=case,bounds=bounds));covered.add(shape)
    missing=[shape for shape in rules if shape not in covered]
    return dict(passed=not missing and bool(evidence) and all(e['bounds']['high']<1 for e in evidence),
                uncovered_shapes=missing,evidence=evidence,normal_request_latency_qualified=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('report',type=Path,nargs='+');ap.add_argument('--validate-policy',type=Path);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();report=merge_reports([json.loads(p.read_text()) for p in args.report]);result=validate_heldout(json.loads(args.validate_policy.read_text()),report) if args.validate_policy else select(report)
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')

    if args.validate_policy and not result['passed']:raise SystemExit(1)
