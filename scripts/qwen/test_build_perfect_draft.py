import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import build_perfect_draft as builder


class PerfectDraftBuilderTest(unittest.TestCase):
    def test_only_guarded_decode_path_is_added_and_original_body_is_retained(self):
        original = (builder.ROOT/'src/engine/model.cpp').read_text()
        generated = builder.instrument(original)
        self.assertEqual(generated.replace(builder.ALL_ROWS,'',1).replace(builder.CONFIGURE,'',1),original)
        self.assertEqual(generated.count(builder.ALL_ROWS),1)
        self.assertIn('phase_=="decode" && T!=1 && T!=2 && T!=4',builder.ALL_ROWS)
        self.assertIn('phase_=="decode" && T>1',builder.ALL_ROWS)
        self.assertIn('gpu_.linear(resident_->linear("lm_head"),mixed,T)',builder.ALL_ROWS)
        self.assertIn('values.size()!=uint64_t(T)*Vocab',builder.ALL_ROWS)
        self.assertIn('verifier_kernels.token_tile=uint32_t(ids.size())',builder.CONFIGURE)
        self.assertIn('auto verifier_kernels=options_.kernels;',builder.CONFIGURE)
        self.assertIn('if(phase_=="prefill") options_.completion_pipeline=false;',builder.CONFIGURE)
        self.assertIn('~RestoreCompletionPipeline() { target=saved; }',builder.CONFIGURE)
        self.assertIn('restore_completion{options_.completion_pipeline,options_.completion_pipeline}',builder.CONFIGURE)
        with self.assertRaises(ValueError): builder.instrument(generated)
        changed = original.replace('const std::array<int,1> row={int(T-1)};',
                                   'const std::array<int,1> row={0};')
        with self.assertRaises(ValueError): builder.instrument(changed)

    def test_objects_precede_archive_and_production_main_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); build_dir = root/'build/qwen'
            (build_dir/'CMakeFiles/zerocool.dir').mkdir(parents=True)
            (root/'src/engine').mkdir(parents=True)
            model = root/'src/engine/model.cpp'
            compile_entry = dict(directory=str(build_dir),file=str(model),command=
                f'/usr/bin/c++ -I{root}/include -O3 -std=c++23 -fno-fast-math '
                f'-o CMakeFiles/zerocool_lib.dir/model.cpp.o -c {model}')
            storage_entry = dict(compile_entry,file=str(root/'src/engine/storage.cpp'),
                command=compile_entry['command'].replace('model.cpp','storage.cpp'))
            (build_dir/'compile_commands.json').write_text(json.dumps([compile_entry,storage_entry]))
            (build_dir/'CMakeFiles/zerocool.dir/link.txt').write_text(
                '/usr/bin/c++ -O3 CMakeFiles/zerocool.dir/src/engine/main.cpp.o '
                '-o bin/zerocool libzerocool_lib.a -framework Metal -framework Foundation')
            output = root/'developer'
            commands = builder.commands(root,output,root/'probe.cpp')
            self.assertNotIn('CMakeFiles/zerocool.dir/src/engine/main.cpp.o',commands['linker'])
            archive = commands['linker'].index('libzerocool_lib.a')
            self.assertEqual(commands['linker'][archive-3:archive],
                [str(output/'probe.perfect-draft.o'),str(output/'model.perfect-draft.o'),str(output/'storage.perfect-draft.o')])
            for command in commands['compiler']:
                self.assertIn('-fno-fast-math',command)
                self.assertIn('-std=c++23',command)
                self.assertTrue(Path(command[command.index('-o')+1]).is_relative_to(output))
            self.assertEqual(commands['compiler'][0][-1],str(output/'model.perfect-draft.cpp'))
            self.assertEqual(commands['compiler'][1][-1],str(output/'storage.perfect-draft.cpp'))
            self.assertEqual(commands['compiler'][2][-1],str(root/'probe.cpp'))

    def test_cache_hash_hook_retains_native_behavior_and_covers_eviction_state(self):
        original = (builder.ROOT/'src/engine/storage.cpp').read_text()
        generated = builder.instrument_storage(original)
        self.assertEqual(generated.replace(builder.CACHE_STATE,'',1),original)
        for field in ('slots_', 'lookup_', 'hand_', 'policy_', 'stride_', 'protected_', 'oldest_', 'newest_',
                      'referenced', 'pins', 'future.valid()', 'future.wait_for', 'previous', 'next'):
            self.assertIn(field,builder.CACHE_STATE)
        for omitted in ('stats_', 'timing', 'monotonic_ns', 'future.get()', 'touch(', 'acquire('):
            self.assertNotIn(omitted,builder.CACHE_STATE)
        self.assertIn('CC_SHA256(serialized.data()',builder.CACHE_STATE)
        self.assertIn('result["diagnostic_cache_state"]',builder.CACHE_STATE)
        with self.assertRaises(ValueError): builder.instrument_storage(generated)

    def test_ambiguous_compiler_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); build_dir = root/'build/qwen'
            build_dir.mkdir(parents=True)
            (build_dir/'compile_commands.json').write_text('[]')
            with self.assertRaises(ValueError): builder.commands(root,root/'out',root/'probe.cpp')

    def test_producer_verification_rejects_omissions_and_self_consistent_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();build_dir=root/'build/qwen'
            for relative in ('src/engine','scripts/qwen','build/qwen/bin','build/qwen/CMakeFiles/zerocool.dir/src/engine'):
                (root/relative).mkdir(parents=True)
            for name in ('model.cpp','storage.cpp'):
                (root/'src/engine'/name).write_text((builder.ROOT/'src/engine'/name).read_text())
            harness=root/'scripts/qwen/probe_perfect_draft.cpp';harness.write_text('int main() {}')
            entries=[dict(directory=str(build_dir),file=str(root/'src/engine'/name),command=
                f'/usr/bin/c++ -std=c++23 -o {name}.o -c {root}/src/engine/{name}')
                for name in ('model.cpp','storage.cpp')]
            (build_dir/'compile_commands.json').write_text(json.dumps(entries))
            (build_dir/'CMakeFiles/zerocool.dir/link.txt').write_text('/usr/bin/c++ '
                'CMakeFiles/zerocool.dir/src/engine/main.cpp.o -o bin/zerocool libzerocool_lib.a')
            for relative in ('libzerocool_lib.a','CMakeFiles/zerocool.dir/src/engine/main.cpp.o','bin/zerocool'):
                (build_dir/relative).write_bytes(relative.encode())
            output=root/'developer'
            def compile_stub(command, **kwargs):
                Path(command[command.index('-o')+1]).write_bytes(' '.join(command).encode())
            with patch.object(builder,'build_fingerprint',return_value='base'),patch.object(builder.subprocess,'run',side_effect=compile_stub):
                producer=builder.build(output,root=root);binary=output/'probe-perfect-draft'
                proof=builder.verify_producer(binary,'base',root=root)
                self.assertEqual(proof['producer'],producer)
                self.assertIn(output/'producer.json',proof['files'])
                for key in ('frozen_build_inputs','object_sha256'):
                    changed=json.loads(json.dumps(producer));changed[key].pop(next(iter(changed[key])))
                    builder.save(output/'producer.json',changed)
                    with self.assertRaisesRegex(ValueError,'incomplete'): builder.verify_producer(binary,'base',root=root)
                changed=json.loads(json.dumps(producer));changed['link_command'].insert(1,'unexpected.o')
                builder.save(output/'producer.json',changed)
                with self.assertRaisesRegex(ValueError,'compiler/link'): builder.verify_producer(binary,'base',root=root)
                generated=output/'model.perfect-draft.cpp';generated.write_text(generated.read_text()+'\n// unapproved\n')
                changed=json.loads(json.dumps(producer));changed['instrumented_model_sha256']=builder.sha(generated)
                builder.save(output/'producer.json',changed)
                with self.assertRaisesRegex(ValueError,'prescribed instrumentation'): builder.verify_producer(binary,'base',root=root)


if __name__ == '__main__': unittest.main()
