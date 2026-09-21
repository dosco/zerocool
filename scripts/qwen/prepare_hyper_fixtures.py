#!/usr/bin/env python3
"""Prepare four real complete hyper blocks using selected pinned tensor ranges."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from cache_residency import require
from capture_hyper_inputs import CAPTURE_ENV, instrument
from native_q4_replay import valid_sha256
from qualification_evidence import save, sha
from shared_expert_reference import REVISION, SelectedWeights

ROOT = Path(__file__).resolve().parents[2]
CASES = ((0, 'attention_input', 'attn_hyper_connection', 36),
         (0, 'mlp_input', 'mlp_hyper_connection', 36),
         (3, 'attention_input', 'attn_hyper_connection', 12),
         (3, 'mlp_input', 'mlp_hyper_connection', 12))
TENSORS = dict(norm=('hc_norm.weight', (10240,), 'BF16'),
    down_w=('input_mix_weight_down.weight', (320, 2560), 'U32'),
    down_s=('input_mix_weight_down.scales', (320, 160), 'BF16'),
    down_b=('input_mix_weight_down.biases', (320, 160), 'BF16'),
    up_w=('input_mix_weight_up.weight', (10240, 80), 'U32'),
    up_s=('input_mix_weight_up.scales', (10240, 5), 'BF16'),
    up_b=('input_mix_weight_up.biases', (10240, 5), 'BF16'),
    inject=('block_inject_weight.weight', (4, 10240), 'BF16'))
CAPTURE_BYTES = 4*(10240+2560+4)*4
BYTE_LIMIT = 32*1024**2


def load(path):
    return json.loads(Path(path).read_text())


def frequency_shape_proof(reader):
    """Check all 96 eligible blocks using pinned headers, without weight reads."""
    headers = {}; blocks = []
    for layer in range(48):
        for stage, suffix in (('attention_input','attn_hyper_connection'), ('mlp_input','mlp_hyper_connection')):
            base = f'model.layers.{layer}.{suffix}'
            for projection in ('input_mix_weight_down','input_mix_weight_up'):
                require(reader.config['quantization'].get(base+'.'+projection) == dict(bits=8,group_size=64),
                        'Frequency assignment includes ineligible hyper precision')
            require(base+'.block_inject_weight' not in reader.config['quantization'], 'Quantized hyper injection')
            tensors = {}
            for name, (tail, shape, dtype) in TENSORS.items():
                full = 'language_model.'+base+'.'+tail
                filename = reader.index[full]
                if filename not in headers:
                    reader.verify_file(filename)
                    with (reader.model/filename).open('rb', buffering=0) as stream:
                        prefix = stream.read(8)
                        require(len(prefix) == 8, 'Truncated hyper shape header')
                        size, = struct.unpack('<Q', prefix)
                        require(0 < size <= 16*1024**2, 'Unbounded hyper shape header')
                        raw = stream.read(size)
                        require(len(raw) == size, 'Truncated hyper shape metadata')
                    headers[filename] = (size+8, json.loads(raw))
                start, header = headers[filename]; tensor = header[full]
                require(tensor.get('shape') == list(shape) and tensor.get('dtype') == dtype,
                        'Frequency assignment includes ineligible hyper shape')
                lo, hi = tensor['data_offsets']; size = int(np.prod(shape))*(4 if dtype == 'U32' else 2)
                require(type(lo) is int and type(hi) is int and 0 <= lo < hi and hi-lo == size and
                        start+hi <= reader.states[filename]['size'], 'Invalid hyper shape tensor range')
                tensors[name] = dict(file=filename, offset=start+lo, bytes=size, dtype=dtype, shape=list(shape))
            blocks.append(dict(layer=layer, stage=stage, base=base, tensors=tensors))
    return dict(kind='hyper_frequency_shapes_v1', blocks=blocks, eligible_blocks=96,
                frequencies=[36,36,12,12], source='pinned checkpoint headers and quantization metadata')


def capture_proof(capture, producer_path, report_path):
    capture, producer_path, report_path = map(Path, (capture, producer_path, report_path))
    producer = load(producer_path)
    require(producer.get('kind') == 'hyper_capture_producer_v1' and producer.get('complete') is True and
            producer.get('actual_binary_is_base_native_build') is False and
            producer.get('normal_request_latency_qualified') is False and
            producer.get('capture_environment') == CAPTURE_ENV,
            'Missing truthful source-copy capture producer')
    require(all(valid_sha256(producer.get(key)) for key in ('base_native_fingerprint', 'original_model_sha256',
        'instrumented_model_sha256', 'helper_sha256', 'binary_sha256')), 'Missing capture producer hashes')
    require(sha(Path(producer['binary'])) == producer['binary_sha256'] and
            sha(Path(producer['source_file'])) == producer['instrumented_model_sha256'] and
            sha(ROOT/'scripts/qwen/capture_hyper_inputs.py') == producer['helper_sha256'],
            'Capture binary, source copy or helper changed')
    source = ROOT/'src/engine/model.cpp'
    require(sha(source) == producer['original_model_sha256'] and
            Path(producer['source_file']).read_text() == instrument(source.read_text()),
            'Capture source contains changes beyond the declared raw-input hook')
    require(producer.get('frozen_build_inputs') and all(sha(Path(path)) == expected
            for path, expected in producer['frozen_build_inputs'].items()), 'Capture base build inputs changed')
    report = load(report_path)
    require(report.get('complete') is True and report.get('model_revision') == REVISION and
            isinstance(report.get('runs'), list) and len(report['runs']) == 1,
            'Capture requires one completed pinned-model request')
    row = report['runs'][0]
    require(row.get('prompt_tokens') == 72 and row.get('output_tokens') == 2 and
            len(row.get('output_token_ids', [])) == 2 and len(row.get('token_latency_ms', [])) == 1 and
            row.get('finish_reason') == 'length' and row.get('reused_tokens') == 0 and
            row.get('after', {}).get('metal', {}).get('build_fingerprint') == producer['base_native_fingerprint'],
            'Capture did not complete the declared first decode forward')
    cases = []
    files = {}
    for index, (layer, stage, suffix, frequency) in enumerate(CASES):
        prefix = f'capture_hyper_{index}'
        metadata_path = capture/(prefix+'.json')
        metadata = load(metadata_path)
        require(metadata == dict(id=index, layer=layer, stage=stage,
            base=f'model.layers.{layer}.{suffix}', phase='decode', offset=72, tokens=1,
            input_origin='raw_hyper_input'), 'Changed raw hyper capture identity or origin')
        files[str(metadata_path.resolve())] = sha(metadata_path)
        data = {}
        for name, width in (('input', 10240), ('output', 2560), ('injection', 4)):
            path = capture/f'{prefix}_{name}.bin'
            require(not path.is_symlink() and path.stat().st_size == width*4, 'Invalid raw hyper capture payload size')
            value = path.read_bytes()
            require(np.isfinite(np.frombuffer(value, '<f4')).all(), 'Nonfinite raw hyper capture')
            files[str(path.resolve())] = hashlib.sha256(value).hexdigest()
            data[name] = value
        cases.append((metadata, frequency, data))
    require(sum(len(payload) for _, _, values in cases for payload in values.values()) == CAPTURE_BYTES,
            'Changed capture byte coverage')
    return cases, dict(producer_path=str(producer_path.resolve()), producer_sha256=sha(producer_path),
        producer=producer, native_report_path=str(report_path.resolve()), native_report_sha256=sha(report_path),
        files=files, capture_bytes=CAPTURE_BYTES, raw_input_origin='native_unmodified_hyper_argument')


def prepare(capture, producer_path, report_path, model, output, *, lock_path=ROOT/'mixed-models.lock.json'):
    captured, proof = capture_proof(capture, producer_path, report_path)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    manifest = dict(kind='prepared_hyper_fixtures_v1', complete=False, artifact_revision=REVISION,
        source_capture=proof, artifact_lock_sha256=sha(Path(lock_path)), generator_sha256=sha(Path(__file__)),
        weight_reader_sha256=sha(ROOT/'scripts/qwen/shared_expert_reference.py'),
        raw_input_origin='native_unmodified_hyper_argument', cases=[], files={},
        byte_limit=BYTE_LIMIT, total_bytes=0, includes_complete_hyper_block=True,
        normal_request_latency_qualified=False, production_promoted=False,
        limitations=['Four actual single-token inputs from layers 0 and 3 are representative fixtures, not all-layer correctness evidence.',
            'The source-copy capture adds waits and file writes; its request timings are not performance evidence.',
            'Frequency weights describe 36 GDN and 12 attention layers; weighted operator timings do not predict request latency.'])
    save(output/'manifest.json', manifest)
    reader = SelectedWeights(model, lock_path)
    try:
        manifest['frequency_shape_proof'] = frequency_shape_proof(reader)
        def payload(filename, raw, shape, dtype):
            entry = dict(file=filename, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                         shape=list(shape), dtype=dtype)
            require(manifest['total_bytes']+len(raw) <= BYTE_LIMIT, 'Hyper fixture bytes exceed bound')
            (output/filename).write_bytes(raw)
            manifest['files'][filename] = entry
            manifest['total_bytes'] += len(raw)
            return entry

        for metadata, frequency, data in captured:
            index = metadata['id']; base = metadata['base']
            for suffix in ('input_mix_weight_down', 'input_mix_weight_up'):
                require(reader.config['quantization'].get(base+'.'+suffix) == dict(bits=8, group_size=64),
                        'Hyper projection is not original affine Q8/64')
            require(base+'.block_inject_weight' not in reader.config['quantization'], 'Hyper injection unexpectedly quantized')
            case = dict(id=index, layer=metadata['layer'], stage=metadata['stage'], offset=metadata['offset'],
                base=base, frequency=frequency, tensors={},
                input=payload(f'case{index}.input.f32', data['input'], (1,10240), 'F32'),
                native_output=payload(f'case{index}.output.f32', data['output'], (1,2560), 'F32'),
                native_injection=payload(f'case{index}.injection.f32', data['injection'], (1,4), 'F32'))
            for name, (suffix, shape, dtype) in TENSORS.items():
                source_name = base+'.'+suffix
                value = reader.get(source_name, shape, dtype)
                # SelectedWeights preserves original BF16 bit patterns in the
                # high half of F32. This round trip is checked against raw bytes.
                raw = (value.tobytes() if dtype == 'U32' else
                       (value.view('<u4') >> 16).astype('<u2').tobytes())
                source = reader.tensors[source_name]
                require(hashlib.sha256(raw).hexdigest() == source['sha256'], 'Original hyper weight bytes were not preserved')
                entry = payload(f'case{index}.{name}.bin', raw, shape, dtype)
                entry.update(source_name=source_name, source=source)
                case['tensors'][name] = entry
            manifest['cases'].append(case)
        manifest['source_proof'] = reader.proof()
        require(all(sha(Path(path)) == expected for path, expected in proof['files'].items()) and
                sha(Path(producer_path)) == proof['producer_sha256'] and
                sha(Path(report_path)) == proof['native_report_sha256'], 'Capture source changed during fixture preparation')
        manifest['complete'] = True
        validate_manifest(manifest)
        save(output/'manifest.json', manifest)
        return manifest
    finally:
        reader.close()


def validate_manifest(manifest):
    require(manifest.get('kind') == 'prepared_hyper_fixtures_v1' and manifest.get('complete') is True and
            manifest.get('artifact_revision') == REVISION and manifest.get('includes_complete_hyper_block') is True and
            manifest.get('raw_input_origin') == 'native_unmodified_hyper_argument' and
            manifest.get('normal_request_latency_qualified') is False and manifest.get('production_promoted') is False,
            'Incomplete or incorrectly labeled hyper fixtures')
    frequency = manifest.get('frequency_shape_proof', {})
    require(frequency.get('eligible_blocks') == 96 and frequency.get('frequencies') == [36,36,12,12] and
            [(b.get('layer'),b.get('stage')) for b in frequency.get('blocks',[])] ==
            [(l,s) for l in range(48) for s in ('attention_input','mlp_input')],
            'Missing all-layer metadata support for frequency projection')
    cases = manifest.get('cases', [])
    require(len(cases) == 4 and [(c.get('id'), c.get('layer'), c.get('stage'), c.get('frequency')) for c in cases] ==
            [(i,l,s,f) for i,(l,s,_,f) in enumerate(CASES)], 'Missing or changed complete-block coverage')
    expected_files = {}
    for index, case in enumerate(cases):
        require(case.get('offset') == 72 and case.get('base') == f'model.layers.{CASES[index][0]}.{CASES[index][2]}' and
                set(case.get('tensors', {})) == set(TENSORS), 'Changed hyper block identity or tensor coverage')
        expected = [(case[key], (1,width), 'F32') for key,width in
                    (('input',10240),('native_output',2560),('native_injection',4))]
        expected += [(case['tensors'][key], shape, dtype) for key,(_,shape,dtype) in TENSORS.items()]
        for entry, shape, dtype in expected:
            require(Path(entry.get('file', '')).name == entry.get('file') and entry['file'] not in expected_files and
                    entry.get('shape') == list(shape) and entry.get('dtype') == dtype and
                    entry.get('bytes') == int(np.prod(shape))*(4 if dtype in ('U32','F32') else 2) and
                    valid_sha256(entry.get('sha256')), 'Invalid hyper payload layout or identity')
            expected_files[entry['file']] = entry
        for name, (suffix, shape, dtype) in TENSORS.items():
            entry = case['tensors'][name]; source_name = case['base']+'.'+suffix
            require(entry.get('source_name') == source_name and
                    entry.get('source') == manifest.get('source_proof', {}).get('tensors', {}).get(source_name) and
                    entry['sha256'] == entry['source'].get('sha256') and
                    entry['bytes'] == entry['source'].get('bytes') and
                    entry['shape'] == entry['source'].get('shape') and entry['dtype'] == entry['source'].get('dtype'),
                    'Hyper weight payload differs from selected checkpoint source')
    require(manifest.get('files') == expected_files and manifest.get('byte_limit') == BYTE_LIMIT and
            manifest.get('total_bytes') == sum(entry['bytes'] for entry in expected_files.values()) <= BYTE_LIMIT,
            'Changed hyper fixture byte accounting')
    return manifest


def verify(directory):
    directory = Path(directory); manifest = validate_manifest(load(directory/'manifest.json'))
    for filename, entry in manifest['files'].items():
        path = directory/filename
        require(not path.is_symlink() and path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
                'Changed prepared hyper fixture payload')
    return manifest


def verify_prepared(directory, model, *, lock_path=ROOT/'mixed-models.lock.json'):
    """Reopen selected ranges and the capture proof before submitting GPU work."""
    manifest = verify(directory)
    source = manifest['source_capture']
    require(sha(Path(source['producer_path'])) == source['producer_sha256'] and
            sha(Path(source['native_report_path'])) == source['native_report_sha256'] and
            all(sha(Path(path)) == expected for path,expected in source['files'].items()),
            'Prepared fixture capture provenance changed')
    captures = {Path(path).parent for path in source['files']}
    require(len(captures) == 1, 'Hyper capture paths span different source directories')
    _, current_capture = capture_proof(captures.pop(), source['producer_path'], source['native_report_path'])
    require(current_capture == source, 'Prepared capture producer or request identity changed')
    require(sha(Path(lock_path)) == manifest['artifact_lock_sha256'], 'Prepared fixture checkpoint lock changed')
    reader = SelectedWeights(model, lock_path)
    try:
        require(frequency_shape_proof(reader) == manifest['frequency_shape_proof'],
                'All-layer frequency shape proof changed')
        for case in manifest['cases']:
            for name, (suffix, shape, dtype) in TENSORS.items():
                source_name = case['base']+'.'+suffix
                reader.get(source_name, shape, dtype)
                require(reader.tensors[source_name] == case['tensors'][name]['source'],
                        'Prepared hyper weight differs from currently verified checkpoint')
        proof = reader.proof()
        require(proof == manifest['source_proof'], 'Selected checkpoint source proof changed')
        return dict(manifest_sha256=sha(Path(directory)/'manifest.json'), cases=4,
            original_weight_bytes_verified=True, actual_raw_inputs_verified=True,
            capture_producer_sha256=source['producer_sha256'], total_bytes=manifest['total_bytes'],
            source_proof=proof, producer=source['producer'], normal_request_latency_qualified=False)
    finally:
        reader.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--producer', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=ROOT/'.cache/qwen-mixed-reference')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.capture, args.producer, args.report, args.model, args.output)
    print(json.dumps(dict(complete=result['complete'], cases=len(result['cases']), total_bytes=result['total_bytes'])))
