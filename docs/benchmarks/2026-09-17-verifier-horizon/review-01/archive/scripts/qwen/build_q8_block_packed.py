#!/usr/bin/env python3
"""Compile an isolated packed-Q8 probe with native flags; leave production unchanged."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from build_block_cache_trace import replace
from build_identity import build_fingerprint
from cache_residency import require
from qualification_evidence import save, sha

ROOT = Path(__file__).resolve().parents[2]
SHADER = ROOT/'scripts/qwen/probe_q8_block_packed.metal'
TEMPLATE = ROOT/'scripts/qwen/probe_block_gdn.cpp'
METAL = ROOT/'src/qwen/metal.mm'
KERNEL = 'q8_block_packed_t4_w8'
VARIANT = dict(candidate=KERNEL, control='q8_mm_t4', reference='q8_mm',
    token_tile=4, output_rows=1, lane_width=8, weight_load_bits=32)


def settings(output):
    output = Path(output).resolve(); directory = ROOT/'build/qwen'
    database = directory/'compile_commands.json'; native_link = directory/'CMakeFiles/freellm.dir/link.txt'
    entries = json.loads(database.read_text()); compiler = []; objects = []
    generated = [output/'metal.packed.mm', output/'probe.packed.cpp']
    for source, target in zip((METAL, ROOT/'src/qwen/model.cpp'), generated):
        rows = [e for e in entries if Path(e['file']).resolve() == source]
        require(len(rows) == 1 and Path(rows[0]['directory']).resolve() == directory, 'Missing native compiler command')
        cmd = shlex.split(rows[0]['command'])
        require(cmd.count('-o') == cmd.count('-c') == 1 and Path(cmd[-1]).resolve() == source, 'Changed compile command')
        obj = target.with_suffix('.o'); cmd[-1] = str(target); cmd[cmd.index('-o')+1] = str(obj)
        compiler.append(cmd); objects.append(obj)
    linker = shlex.split(native_link.read_text()); main = 'CMakeFiles/freellm.dir/src/qwen/main.cpp.o'
    require(linker.count(main) == linker.count('-o') == linker.count('libfreellm_lib.a') == 1, 'Changed native linker')
    linker.remove(main); binary = output/'probe-q8-block-packed'; linker[linker.index('-o')+1] = str(binary)
    at = linker.index('libfreellm_lib.a'); linker[at:at] = list(map(str, objects))
    return dict(directory=directory, output=output, database=database, native_link=native_link,
        generated=generated, objects=objects, compiler=compiler, linker=linker, binary=binary)


def sources(cfg):
    shader = SHADER.read_text(); require(')PACKED_Q8"' not in shader, 'Shader delimiter collision')
    metal = replace(METAL.read_text(), '        const auto source=MetalSource;',
        '        const auto source=std::string(MetalSource)+R"PACKED_Q8(\n'+shader+')PACKED_Q8";')
    probe = replace(TEMPLATE.read_text(), 'block_gdn_operator_v1', 'q8_block_packed_operator_v1')
    probe = replace(probe, '// Developer-only four-token Q8 row-pair screen with actual captured inputs.',
        '// Developer-only four-token packed Q8 screen with verified captured inputs.')
    probe = replace(probe, 'cfg.affine_rows=candidate?2:1;', 'cfg.affine_rows=1;')
    probe = replace(probe, 'gpu.configure(cfg);const auto before=process_memory();const auto start=monotonic_ns();',
        'gpu.configure(cfg);const auto before=process_memory();\n'
        '                    const auto dispatches_before=gpu.statistics().at("kernel_dispatches");\n'
        '                    const auto start=monotonic_ns();')
    probe = replace(probe, 'for(int i=0;i<repeats;++i) gpu.linear_into(l,x,4,{out});',
        'for(int i=0;i<repeats;++i) {\n'
        '                        if(candidate) gpu.dispatch("'+KERNEL+'",{l.weight,l.scales,l.biases,{x},{out}},\n'
        '                            {K,N,4,64,0},N*32);\n'
        '                        else gpu.linear_into(l,x,4,{out});\n'
        '                    }')
    probe = replace(probe, 'check(hash(out->data,out->bytes)==expected,"Q8 row pair changed output bits");',
        'check(hash(out->data,out->bytes)==expected,"Packed Q8 changed output bits");\n'
        '                    Json dispatches=Json::object();\n'
        '                    const auto dispatches_after=gpu.statistics().at("kernel_dispatches");\n'
        '                    for(const auto& [name,value]:dispatches_after.items()) {\n'
        '                        const auto count=value.get<uint64_t>()-dispatches_before.value(name,uint64_t(0));\n'
        '                        if(count) dispatches[name]=count;\n'
        '                    }\n'
        '                    check(dispatches==Json{{candidate?"'+KERNEL+'":"q8_mm_t4",repeats}},"Changed kernel selection");')
    probe = replace(probe, '{"output_sha256",expected},{"gpu_ns",', '{"output_sha256",expected},{"kernel_dispatches",dispatches},{"gpu_ns",')
    probe = replace(probe, '        report.update(Json{{"validation",validation}',
        '        report["variant"]=Json::parse(R"VARIANT('+json.dumps(VARIANT, sort_keys=True)+')VARIANT");\n'
        '        report.update(Json{{"validation",validation}')
    return dict(zip(cfg['generated'], (metal, probe)))


def inputs(cfg):
    return [Path(__file__).resolve(), SHADER, TEMPLATE, METAL, ROOT/'scripts/qwen/build_block_cache_trace.py',
        ROOT/'scripts/qwen/build_identity.py', cfg['database'], cfg['native_link'],
        cfg['directory']/'generated/qwen_embedded.hpp', cfg['directory']/'libfreellm_lib.a',
        cfg['directory']/'bin/freellm', *sorted((ROOT/'include/qwen').glob('*.hpp'))]


def proof(cfg):
    return dict(kind='q8_block_packed_producer_v1', base_native_fingerprint=build_fingerprint(ROOT),
        variant=VARIANT, original_inputs={str(p): sha(p) for p in inputs(cfg)},
        generated={str(p): sha(p) for p in sources(cfg)}, compiler=cfg['compiler'], linker=cfg['linker'],
        binary=str(cfg['binary']), production_promoted=False)


def build(output):
    cfg = settings(output); cfg['output'].mkdir(parents=True, exist_ok=False)
    for p, value in sources(cfg).items(): p.write_text(value)
    frozen = proof(cfg); save(cfg['output']/'producer.json', dict(frozen, complete=False))
    with (cfg['output']/'build.log').open('w') as log:
        for cmd in [*cfg['compiler'], cfg['linker']]:
            subprocess.run(cmd, cwd=cfg['directory'], stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True)
    require(frozen == proof(cfg), 'Build inputs changed')
    result = dict(frozen, complete=True, binary_sha256=sha(cfg['binary']), objects={str(p): sha(p) for p in cfg['objects']})
    save(cfg['output']/'producer.json', result); return result


def verify(output, fingerprint):
    cfg = settings(output); saved = json.loads((cfg['output']/'producer.json').read_text())
    expected = dict(proof(cfg), complete=True, binary_sha256=sha(cfg['binary']), objects={str(p): sha(p) for p in cfg['objects']})
    require(saved == expected and saved['base_native_fingerprint'] == fingerprint, 'Changed packed-Q8 producer')
    for p, value in sources(cfg).items(): require(p.read_text() == value, 'Changed generated source')
    return dict(producer=saved, files=[*inputs(cfg), *sources(cfg), *cfg['objects'], cfg['binary'], cfg['output']/'producer.json'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    print(json.dumps(build(p.parse_args().output), indent=2))
