#!/usr/bin/env python3
"""Build a developer-only CLI that captures four raw decode hyper blocks.

Build mode compiles and links a source copy. Run mode performs the declared
bounded capture. Neither modifies production source, objects, library or executable.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from build_identity import build_fingerprint
from cache_residency import require
from qualification_evidence import save, sha

ROOT = Path(__file__).resolve().parents[2]
CAPTURE_ENV = 'ZEROCOOL_HYPER_CAPTURE_DIR'
HOOK = r'''
    // Developer source-copy capture: four complete, raw single-token blocks.
    // It is deliberately distinct from production inference and its timings.
    if(tokens==1 && phase_=="decode") {
        const char* capture_dir=std::getenv("ZEROCOOL_HYPER_CAPTURE_DIR");
        const std::array<std::string,4> capture_bases={
            "model.layers.0.attn_hyper_connection","model.layers.0.mlp_hyper_connection",
            "model.layers.3.attn_hyper_connection","model.layers.3.mlp_hyper_connection"};
        const auto found=std::find(capture_bases.begin(),capture_bases.end(),base);
        if(capture_dir && *capture_dir && found!=capture_bases.end()) {
            const auto index=size_t(found-capture_bases.begin());
            const auto prefix=std::filesystem::path(capture_dir)/("capture_hyper_"+std::to_string(index));
            static std::array<bool,4> captured{};
            if(!captured[index]) {
                std::filesystem::create_directories(capture_dir);
                gpu_.finish();
                auto save_capture=[&](const char* name,const Buf& value,uint64_t bytes) {
                    const auto path=prefix.string()+"_"+name+".bin";
                    if(!value || value->bytes<bytes || std::filesystem::exists(path))
                        throw std::runtime_error("invalid or existing hyper capture");
                    std::ofstream file(path,std::ios::binary);
                    file.write(reinterpret_cast<const char*>(value->data),std::streamsize(bytes));
                    if(!file) throw std::runtime_error("cannot write hyper capture");
                };
                save_capture("input",x,10240*4);
                save_capture("output",out,2560*4);
                save_capture("injection",injection,4*4);
                const auto metadata=prefix.string()+".json";
                if(std::filesystem::exists(metadata)) throw std::runtime_error("existing hyper metadata");
                Json entry={{"id",index},{"layer",index<2?0:3},
                    {"stage",index%2?"mlp_input":"attention_input"},{"base",base},
                    {"phase",phase_},{"offset",trace_offset_},{"tokens",tokens},
                    {"input_origin","raw_hyper_input"}};
                std::ofstream file(metadata);file<<entry.dump(2)<<'\n';
                if(!file) throw std::runtime_error("cannot write hyper metadata");
                captured[index]=true;
            }
        }
    }
'''


def instrument(source):
    start = source.index('std::pair<Buf,Buf> Model::hyper(')
    end = source.index('\nBuf Model::conv(', start)
    section = source[start:end]
    marker = '    return {out,injection};'
    require(section.count(marker) == 1 and CAPTURE_ENV not in source,
            'Changed native hyper capture seam')
    return '#include <cstdlib>\n'+source[:start]+section.replace(marker, HOOK+marker)+source[end:]


def build(output, *, root=ROOT):
    output, root = Path(output).resolve(), Path(root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = root/'src/engine/model.cpp'
    build_dir = root/'build/qwen'
    commands_path = build_dir/'compile_commands.json'
    link_path = build_dir/'CMakeFiles/zerocool.dir/link.txt'
    commands = json.loads(commands_path.read_text())
    entries = [entry for entry in commands if Path(entry['file']).resolve() == source]
    require(len(entries) == 1, 'Missing unique native model compile command')
    original = entries[0]
    require(Path(original['directory']).resolve() == build_dir, 'Unexpected native build directory')
    generated, obj, binary = (output/name for name in ('model.capture.cpp', 'model.capture.o', 'zerocool-hyper-capture'))
    generated.write_text(instrument(source.read_text()))
    compile_command = shlex.split(original['command'])
    require(compile_command.count('-o') == compile_command.count('-c') == 1 and
            compile_command[-1] == str(source), 'Unexpected model compiler command')
    compile_command[compile_command.index('-o')+1] = str(obj)
    compile_command[-1] = str(generated)
    link_command = shlex.split(link_path.read_text())
    require(link_command.count('-o') == 1 and link_command.count('libzerocool_lib.a') == 1,
            'Unexpected CLI linker command')
    link_command[link_command.index('-o')+1] = str(binary)
    link_command.insert(link_command.index('libzerocool_lib.a'), str(obj))
    inputs = [source, commands_path, link_path, build_dir/'libzerocool_lib.a',
              build_dir/'CMakeFiles/zerocool.dir/src/engine/main.cpp.o', build_dir/'bin/zerocool']
    frozen = {str(path): sha(path) for path in inputs}
    producer = dict(kind='hyper_capture_producer_v1', complete=False,
        instrumentation='developer source-copy Model::hyper hook; production build unchanged',
        base_native_fingerprint=build_fingerprint(root), original_model_sha256=sha(source),
        instrumented_model_sha256=sha(generated), source_file=str(generated),
        helper_sha256=sha(Path(__file__)), frozen_build_inputs=frozen,
        compile_command=compile_command, link_command=link_command,
        binary=str(binary), capture_environment=CAPTURE_ENV,
        actual_binary_is_base_native_build=False, normal_request_latency_qualified=False)
    save(output/'producer.json', producer)
    with (output/'build.log').open('w') as log:
        for command in (compile_command, link_command):
            subprocess.run(command, cwd=build_dir, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=180)
    require(frozen == {str(path): sha(path) for path in inputs}, 'Native build changed during developer compilation')
    producer.update(complete=True, binary_sha256=sha(binary), object_sha256=sha(obj))
    save(output/'producer.json', producer)
    return producer


def run_capture(output, producer_path):
    from benchmark_exact import config_args, inspect_admission
    from combined_q4 import configs, freeze
    from q4_request_context import workload
    from prepare_hyper_fixtures import prepare, verify_prepared
    from qualification_evidence import ResourceBlocked
    from stage200 import Experiment

    producer_path = Path(producer_path).resolve()
    producer = json.loads(producer_path.read_text())
    binary = Path(producer['binary'])
    require(producer.get('complete') is True and sha(binary) == producer.get('binary_sha256') and
            producer.get('base_native_fingerprint') == build_fingerprint(ROOT) and
            producer.get('helper_sha256') == sha(Path(__file__)), 'Changed source-copy capture build')
    config = configs()[0]
    work = [dict(workload()[0], max_tokens=2)]
    exp = Experiment(Path(output), 'hyper_raw_capture_v1', [config], work, 240)
    with exp:
        exp.report.update(producer=producer, producer_path=str(producer_path),
            actual_binary_is_base_native_build=False, performance_comparison=False,
            capture_scope='first decode token; complete raw hyper inputs and outputs in layers 0 and 3')
        paths = [producer_path, binary, Path(producer['source_file']),
                 ROOT/'docs/benchmarks/2026-09-15-hyper-fusion/protocol.md',
                 *[Path(path) for path in producer['frozen_build_inputs']]]
        freeze(exp, paths)
        capture = exp.out/'capture'
        capture.mkdir()
        exp.env[CAPTURE_ENV] = str(capture)
        exp.guard.check_resources(initial=True)
        common = ['--model', exp.model, '--artifact', 'mixed-4_8bit', '--prepared', exp.prepared,
                  '--memory-gb', '12', '--context', '8192', *config_args(config)]
        with (exp.out/'capture.admission.log').open('w') as log:
            admission = inspect_admission(binary, common, exp.out/'capture', 12*1024**3, 512,
                                          log, exp.guard, exp.left)
        if json.loads(Path(admission).read_text())['current_admission']['expert_slots'] != 1072:
            raise ResourceBlocked('Requested 1072-slot hyper capture was not admitted')
        report = exp.out/'capture.json'
        exp.command([binary, 'bench', *common, '--workload-file', exp.out/'workload.json',
            '--repetitions', '1', '--temperature', '0', '--seed', '0',
            '--bench-progress', exp.out/'capture.progress.jsonl', '--json', report], 'capture', limit=150)
        exp.report['phase'] = 'prepare_verified_fixtures'; exp.persist()
        manifest = prepare(capture, producer_path, report, exp.model, exp.out/'fixtures')
        exp.report['fixtures'] = verify_prepared(exp.out/'fixtures', exp.model)
        exp.report['fixture_manifest_sha256'] = sha(exp.out/'fixtures/manifest.json')
        exp.report['capture_cases'] = len(manifest['cases'])
        exp.report['status'] = 'captured'
    return exp.report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('build', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--producer', type=Path)
    args = parser.parse_args()
    require(args.mode != 'run' or args.producer is not None, 'Capture run requires a producer receipt')
    result = build(args.output) if args.mode == 'build' else run_capture(args.output, args.producer)
    print(json.dumps(result, indent=2))
