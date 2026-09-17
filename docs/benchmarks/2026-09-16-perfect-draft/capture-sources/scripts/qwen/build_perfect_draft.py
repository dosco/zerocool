#!/usr/bin/env python3
"""Build a source-copy perfect-draft probe without modifying production code.

Only decode blocks of one, two or four tokens may request every row's logits.
Priming keeps the original last-row head. This helper never runs the GPU probe.
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
START = 'std::vector<float> Model::compute_logits(const Buf& h,uint32_t T) {\n'
END = '\nJson Model::decode_counters() const {'
FORWARD = 'std::vector<float> Model::forward(std::span<const int> ids,State& state,bool logits,const std::atomic<bool>* cancel) {\n'
CONFIGURE = r'''    // Developer perfect-draft verifier: existing multi-token kernel selection.
    auto verifier_kernels=options_.kernels;
    if(phase_=="decode") {
        if(ids.size()!=1 && ids.size()!=2 && ids.size()!=4)
            throw std::invalid_argument("perfect-draft decode block must contain 1, 2, or 4 tokens");
        verifier_kernels.token_tile=uint32_t(ids.size());
    }
    gpu_.configure(verifier_kernels);
'''
ALL_ROWS = r'''    // Developer perfect-draft diagnostic; production last-row output is unchanged.
    if(phase_=="decode" && T!=1 && T!=2 && T!=4)
        throw std::invalid_argument("perfect-draft decode block must contain 1, 2, or 4 tokens");
    if(phase_=="decode" && T>1) {
        gpu_.label("logits",-1,T,trace_offset_);
        auto [mixed,unused]=hyper(h,"model.hyper_connection_mixer",T,false); (void)unused;
        auto out=gpu_.linear(resident_->linear("lm_head"),mixed,T);
        gpu_.finish();finish_expert_tail();
        const auto values=out->floats();
        if(values.size()!=uint64_t(T)*Vocab) throw std::runtime_error("invalid perfect-draft logits width");
        for(float value:values) if(!std::isfinite(value))
            throw std::runtime_error("non-finite model logits");
        return {values.begin(),values.end()};
    }
'''
CACHE_START = 'Json ExpertCache::json() const {\n'
CACHE_END = '\nExpertCache::Lease::Lease(std::shared_ptr<Entry> e,int acquisition)'
CACHE_STATE = r'''    // Developer-only deterministic eviction state; no counters or timestamps.
    Json diagnostic_slots=Json::array();
    size_t diagnostic_occupied=0;
    const auto key_or_null=[](const Entry* entry)->Json {
        return entry?Json(entry->key.value()):Json(nullptr);
    };
    for(size_t index=0;index<slots_.size();++index) {
        const auto& entry=slots_[index];
        if(!entry) {diagnostic_slots.push_back(nullptr);continue;}
        ++diagnostic_occupied;
        const auto found=lookup_.find(entry->key.value());
        if(found==lookup_.end() || found->second!=index)
            throw std::logic_error("inconsistent diagnostic cache lookup");
        const bool valid=entry->future.valid();
        const bool ready=valid && entry->future.wait_for(std::chrono::seconds(0))==std::future_status::ready;
        diagnostic_slots.push_back({{"slot",index},{"key",entry->key.value()},
            {"referenced",entry->referenced},{"pins",entry->pins},
            {"future_valid",valid},{"ready",ready},{"queue",entry->queue},
            {"previous",key_or_null(entry->previous)},{"next",key_or_null(entry->next)}});
    }
    if(diagnostic_occupied!=lookup_.size()) throw std::logic_error("inconsistent diagnostic cache occupancy");
    Json diagnostic={{"kind","expert_cache_eviction_state_v1"},{"policy",cache_policy_name(policy_)},
        {"capacity",slots_.size()},{"stride",stride_},{"hand",hand_},{"protected",protected_},
        {"oldest",Json::array({key_or_null(oldest_[0]),key_or_null(oldest_[1])})},
        {"newest",Json::array({key_or_null(newest_[0]),key_or_null(newest_[1])})},
        {"slots",std::move(diagnostic_slots)}};
    const auto serialized=diagnostic.dump();
    if(serialized.size()>std::numeric_limits<CC_LONG>::max())
        throw std::runtime_error("diagnostic cache state exceeds hash input bound");
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(serialized.data(),CC_LONG(serialized.size()),digest);
    std::string hexadecimal;hexadecimal.reserve(2*CC_SHA256_DIGEST_LENGTH);
    constexpr char digits[]="0123456789abcdef";
    for(auto byte:digest) {hexadecimal+=digits[byte>>4];hexadecimal+=digits[byte&15];}
    result["diagnostic_cache_state"]=std::move(hexadecimal);
'''


def instrument(source):
    """Prepend the guarded all-row path; retain the original body byte-for-byte."""
    require(source.count(START) == 1 and source.count(END) == 1 and source.count(FORWARD) == 1 and
            'perfect-draft decode block' not in source, 'Changed or already instrumented logits seam')
    start = source.index(START)+len(START)
    end = source.index(END, start)
    original = source[start:end]
    require(original.count('hyper(h,"model.hyper_connection_mixer",T,false)') == 1 and
            original.count('gpu_.linear(resident_->linear("lm_head"),last,1)') == 1 and
            original.count('const std::array<int,1> row={int(T-1)};') == 1,
            'Native final hyper/head implementation changed')
    return (source[:start]+ALL_ROWS+source[start:]).replace(FORWARD,FORWARD+CONFIGURE,1)


def instrument_storage(source):
    require(source.count(CACHE_START) == 1 and source.count(CACHE_END) == 1 and
            'diagnostic_cache_state' not in source, 'Changed or already instrumented cache state seam')
    start = source.index(CACHE_START)+len(CACHE_START)
    end = source.index(CACHE_END,start)
    section = source[start:end]
    marker = '    return result;'
    require(section.count(marker) == 1 and 'auto result=stats_.json();' in section,
            'Native cache statistics implementation changed')
    return source[:start]+section.replace(marker,CACHE_STATE+marker,1)+source[end:]


def commands(root, output, harness):
    """Use current native compiler/linker flags; do not link the production main."""
    root, output, harness = map(lambda path: Path(path).resolve(), (root, output, harness))
    build_dir = root/'build/qwen'
    source = root/'src/qwen/model.cpp'
    storage_source = root/'src/qwen/storage.cpp'
    compile_path = build_dir/'compile_commands.json'
    link_path = build_dir/'CMakeFiles/freellm.dir/link.txt'
    database = json.loads(compile_path.read_text())
    entries = [entry for entry in database
               if Path(entry['file']).resolve() == source]
    require(len(entries) == 1 and Path(entries[0]['directory']).resolve() == build_dir,
            'Missing unique native model compiler command')
    original = shlex.split(entries[0]['command'])
    require(original.count('-o') == original.count('-c') == 1 and Path(original[-1]).resolve() == source,
            'Unexpected native model compile command')
    generated = output/'model.perfect-draft.cpp'
    storage_generated = output/'storage.perfect-draft.cpp'
    storage_entries = [entry for entry in database if Path(entry['file']).resolve() == storage_source]
    require(len(storage_entries) == 1 and Path(storage_entries[0]['directory']).resolve() == build_dir,
            'Missing unique native storage compiler command')
    storage_command = shlex.split(storage_entries[0]['command'])
    require(storage_command.count('-o') == storage_command.count('-c') == 1 and
            Path(storage_command[-1]).resolve() == storage_source, 'Unexpected native storage compile command')
    object_paths = [output/'model.perfect-draft.o', output/'storage.perfect-draft.o', output/'probe.perfect-draft.o']
    compile_commands = []
    for input_path, object_path, template in zip((generated, storage_generated, harness),
            object_paths, (original, storage_command, original)):
        command = list(template)
        command[command.index('-o')+1] = str(object_path)
        command[-1] = str(input_path)
        compile_commands.append(command)
    link = shlex.split(link_path.read_text())
    main = 'CMakeFiles/freellm.dir/src/qwen/main.cpp.o'
    require(link.count('-o') == link.count('libfreellm_lib.a') == link.count(main) == 1,
            'Unexpected native CLI linker command')
    link.remove(main)
    binary = output/'probe-perfect-draft'
    link[link.index('-o')+1] = str(binary)
    index = link.index('libfreellm_lib.a')
    link[index:index] = [str(object_paths[2]), str(object_paths[0]), str(object_paths[1])]
    return dict(directory=build_dir, source=source, generated=generated, binary=binary,
        storage_source=storage_source, storage_generated=storage_generated,
        objects=object_paths, compiler=compile_commands, linker=link,
        compile_database=compile_path, native_link_command=link_path)


def frozen_inputs(settings, harness):
    build_dir = settings['directory']
    return [settings['source'], settings['storage_source'], harness, Path(__file__).resolve(),
        settings['compile_database'], settings['native_link_command'], build_dir/'libfreellm_lib.a',
        build_dir/'CMakeFiles/freellm.dir/src/qwen/main.cpp.o', build_dir/'bin/freellm']


def contracts():
    return dict(
        cache_state_contract=dict(path='expert_cache.diagnostic_cache_state', hash='SHA256',
            scope='ordered slots, keys, CLOCK hand, references, pins, readiness, policy, stride and SLRU links',
            excludes='addresses, counters, read timing', observes='only on ExpertCache::json calls; no cache mutation'),
        all_row_logits_contract=dict(phase='decode', block_lengths=[1,2,4],
            layout='row-major token then vocabulary', output_values='tokens * Vocab',
            priming='original final hyper and last-row head unchanged',
            final_hyper='existing multi-token final hyper', head='existing vectorized lm_head(mixed,tokens)',
            kernels='decode token_tile equals block length; original configuration outside decode',
            one_token='original compute_logits body unchanged'))


def verify_producer(binary, expected_base_build, *, root=ROOT):
    """Verify the live source-copy build; fail closed on omitted or changed inputs."""
    root, binary = Path(root).resolve(), Path(binary).resolve()
    receipt = binary.parent/'producer.json'
    producer = json.loads(receipt.read_text())
    harness = root/'scripts/qwen/probe_perfect_draft.cpp'
    settings = commands(root, binary.parent, harness)
    require(binary == settings['binary'], 'Unexpected developer binary path')
    for key, value in dict(kind='perfect_draft_producer_v1', complete=True,
            base_native_fingerprint=expected_base_build, actual_binary_is_base_native_build=False,
            production_main_linked=False, normal_request_latency_qualified=False,
            production_promoted=False, **contracts()).items():
        require(key in producer and producer[key] == value and
                (not isinstance(value, bool) or producer[key] is value), f'Invalid producer {key}')
    require(build_fingerprint(root) == expected_base_build, 'Base native fingerprint changed')
    inputs = frozen_inputs(settings, harness)
    require(producer.get('frozen_build_inputs') == {str(path):sha(path) for path in inputs},
            'Changed or incomplete producer inputs')
    for key, path in dict(binary=binary, instrumented_source=settings['generated'],
            instrumented_storage_source=settings['storage_generated'], harness=harness).items():
        require(producer.get(key) == str(path), f'Invalid producer {key} path')
    hashes = dict(binary_sha256=binary, original_model_sha256=settings['source'],
        instrumented_model_sha256=settings['generated'], original_storage_sha256=settings['storage_source'],
        instrumented_storage_sha256=settings['storage_generated'], harness_sha256=harness,
        helper_sha256=Path(__file__).resolve())
    for key, path in hashes.items():
        require(producer.get(key) == sha(path), f'Changed producer {key}')
    require(producer.get('object_sha256') == {str(path):sha(path) for path in settings['objects']},
            'Changed or incomplete producer objects')
    require(settings['generated'].read_text() == instrument(settings['source'].read_text()) and
            settings['storage_generated'].read_text() == instrument_storage(settings['storage_source'].read_text()),
            'Generated source differs from prescribed instrumentation')
    require(producer.get('compiler_commands') == settings['compiler'] and
            producer.get('link_command') == settings['linker'], 'Changed producer compiler/link commands')
    files = list(dict.fromkeys([receipt,*inputs,settings['generated'],settings['storage_generated'],
                               *settings['objects'],binary]))
    return dict(producer=producer, files=files)


def build(output, harness=None, *, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    harness = Path(harness or root/'scripts/qwen/probe_perfect_draft.cpp').resolve()
    require(harness.is_file(), 'Missing standalone perfect-draft harness')
    settings = commands(root, output, harness)
    output.mkdir(parents=True, exist_ok=False)
    settings['generated'].write_text(instrument(settings['source'].read_text()))
    settings['storage_generated'].write_text(instrument_storage(settings['storage_source'].read_text()))
    build_dir = settings['directory']
    inputs = frozen_inputs(settings, harness)
    frozen = {str(path): sha(path) for path in inputs}
    producer = dict(kind='perfect_draft_producer_v1', complete=False,
        base_native_fingerprint=build_fingerprint(root), actual_binary_is_base_native_build=False,
        original_model_sha256=sha(settings['source']), instrumented_model_sha256=sha(settings['generated']),
        original_storage_sha256=sha(settings['storage_source']), instrumented_storage_sha256=sha(settings['storage_generated']),
        instrumented_storage_source=str(settings['storage_generated']),
        instrumented_source=str(settings['generated']), harness=str(harness), harness_sha256=sha(harness),
        helper_sha256=sha(Path(__file__)), frozen_build_inputs=frozen,
        compiler_commands=settings['compiler'], link_command=settings['linker'],
        binary=str(settings['binary']), production_main_linked=False,
        **contracts(),
        normal_request_latency_qualified=False, production_promoted=False)
    save(output/'producer.json', producer)
    try:
        with (output/'build.log').open('w') as log:
            for command in [*settings['compiler'], settings['linker']]:
                subprocess.run(command, cwd=build_dir, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=180)
        require(frozen == {str(path): sha(path) for path in inputs},
                'Native build, harness or helper changed during source-copy compilation')
        require(build_fingerprint(root) == producer['base_native_fingerprint'],
                'Native source fingerprint changed during source-copy compilation')
        producer.update(complete=True, binary_sha256=sha(settings['binary']),
            object_sha256={str(path):sha(path) for path in settings['objects']})
    except Exception as error:
        producer['error'] = str(error)
        raise
    finally:
        save(output/'producer.json', producer)
    return producer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--harness', type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.harness), indent=2))
