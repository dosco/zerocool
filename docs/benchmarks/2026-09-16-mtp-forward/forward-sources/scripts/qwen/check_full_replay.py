#!/usr/bin/env python3
"""Requalify the saved five-token oracle and compare complete native state."""
import argparse
import hashlib
import json
from pathlib import Path
from build_identity import build_fingerprint


def run(args):
    root = Path(__file__).resolve().parents[2]
    current = build_fingerprint(root)
    prepared = json.loads(args.prepared_report.read_text())
    source = json.loads(args.source_report.read_text())
    artifact=getattr(args,'artifact','q4-control')
    lock=json.loads((root/('models.lock.json' if artifact=='q4-control' else 'mixed-models.lock.json')).read_text())
    if artifact=='mixed-4_8bit':
        reference_path=args.reference/'reference.json';layers_path=args.reference/'reference-identity.json'
        reference=json.loads(reference_path.read_text());manifest=json.loads(layers_path.read_text())
        if reference['source_revision']!=lock['revision'] or manifest['report_sha256']!=hashlib.sha256(reference_path.read_bytes()).hexdigest():
            raise ValueError('Mixed reference identity differs from the pinned artifact')
        reference_tokens=reference['tokens']
        layer_reference={'checks':[dict(layer=l,output_values=manifest['outputs'][f'layer_{l}.bin']['bytes']//4,
                                      sha256=manifest['outputs'][f'layer_{l}.bin']['sha256']) for l in range(48)]}
    else:
        reference_path=args.reference/'logits.json';layers_path=args.reference/'full-layer-parity.json'
        reference=json.loads(reference_path.read_text());layer_reference=json.loads(layers_path.read_text())
        reference_tokens=reference['reference_tokens']
    checks = []

    def check(name, passed, **detail):
        checks.append(dict(name=name, passed=bool(passed), **detail))

    def sha(data):
        return hashlib.sha256(data).hexdigest()

    for name, report in [('prepared', prepared), ('source', source)]:
        check(name+'_identity', report['statistics']['metal']['build_fingerprint'] == current
              and report['statistics'].get('artifact_revision') == lock['revision']
              and report['tokens'] == reference_tokens and report['layers'] == 48)
    expected = (args.reference/'reference_logits.f32').read_bytes()
    if artifact=='mixed-4_8bit':
        saved=manifest['outputs']['logits.f32']
        if len(expected)!=saved['bytes'] or sha(expected)!=saved['sha256']:
            raise ValueError('Mixed reference logits changed since verification')
    for name, report in [('prepared', prepared), ('source', source)]:
        data = Path(report['logits_file']).read_bytes()
        check(name+'_logits', len(data) == 248320*4 and data == expected,
              sha256=sha(data), reference_sha256=sha(expected), values=len(data)//4)
    if len(layer_reference['checks']) != 48:
        raise ValueError('Incomplete original layer oracle')
    for layer, reference_row in enumerate(layer_reference['checks']):
        data = (args.trace/f'layer_{layer}.bin').read_bytes()
        check(f'layer_{layer}', reference_row['layer'] == layer and
              len(data) == reference_row['output_values']*4 and sha(data) == reference_row['sha256'],
              sha256=sha(data), values=len(data)//4)
    a, b = prepared['state'], source['state']
    check('persistent_state', a == b and a['valid'] and a['tokens'] == len(prepared['tokens']) and len(a['layers']) == 48,
          buffers=sum(sum(isinstance(v, dict) and 'bytes' in v for v in row.values()) for row in a['layers']))
    report = dict(kind='full_pinned_fixture_and_state_requalification', passed=all(c['passed'] for c in checks),
                  build_fingerprint=current, artifact_revision=lock['revision'], tokens=prepared['tokens'], checks=checks,
                  prepared_report_sha256=sha(args.prepared_report.read_bytes()),
                  source_report_sha256=sha(args.source_report.read_bytes()),
                  reference_report_sha256=sha(reference_path.read_bytes()),
                  layer_reference_sha256=sha(layers_path.read_bytes()),
                  longer_context_qualified=False, performance_qualified=False, coding_quality_qualified=False)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(f'{len(checks)} full-model fixture/state checks: {report["passed"]}')
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-report', type=Path, required=True)
    parser.add_argument('--source-report', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=root/'docs/benchmarks/2026-09-07')
    parser.add_argument('--artifact', choices=('q4-control','mixed-4_8bit'), default='q4-control')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
