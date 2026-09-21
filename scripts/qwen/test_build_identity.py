from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from build_identity import build_fingerprint, configuration_identity

ROOT = Path(__file__).resolve().parents[2]


class BuildIdentityTests(unittest.TestCase):
    def test_current_configured_native_identity_matches_python(self):
        header = (ROOT/'build/qwen/generated/qwen_embedded.hpp').read_text()
        fingerprint = re.search(r'BuildFingerprint = "([a-f0-9]{64})"', header)[1]
        self.assertEqual(build_fingerprint(ROOT), fingerprint, 'Reconfigure/rebuild stale native sources')

    def test_cmake_agrees_including_case_sensitive_release_flags(self):
        source = (ROOT/'cmake/qwen.cmake').read_text()
        fragment = 'file(GLOB ZEROCOOL_BUILD_INPUTS'+source.split('file(GLOB ZEROCOOL_BUILD_INPUTS', 1)[1]
        fragment = fragment.split('set_property(DIRECTORY', 1)[0].replace(' CONFIGURE_DEPENDS', '')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); build = root/'build/qwen'
            for name in ('src/engine/z.cpp', 'include/engine/a.hpp', 'kernels/metal/qwen.metal',
                         'models.lock.json', 'mixed-models.lock.json', 'mixed-payload-reuse.lock.json',
                         'cmake/qwen.cmake', 'cmake/qwen_embedded.hpp.in', 'CMakeLists.txt'):
                p = root/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(name)
            compiler = build/'CMakeFiles/test/CMakeCXXCompiler.cmake'
            compiler.parent.mkdir(parents=True)
            compiler.write_text('set(CMAKE_CXX_COMPILER_ID "AppleClang")\nset(CMAKE_CXX_COMPILER_VERSION "17.0")\n')
            identities = set()
            for mode, flags, sanitizer in (('Release', '-O3 -DNDEBUG', 'none'),
                                           ('Release', '-O2 -DNDEBUG', 'none'),
                                           ('Debug', '-g', 'address')):
                values = dict(CMAKE_BUILD_TYPE=mode, CMAKE_CXX_FLAGS='-fno-fast-math', ZEROCOOL_SANITIZE=sanitizer,
                              **{'CMAKE_CXX_FLAGS_'+mode.upper(): flags})
                (build/'CMakeCache.txt').write_text(''.join(f'{k}:STRING={v}\n' for k, v in values.items()))
                setup = dict(values, CMAKE_SOURCE_DIR=str(root), CMAKE_CXX_COMPILER_ID='AppleClang',
                             CMAKE_CXX_COMPILER_VERSION='17.0')
                script = root/'compare.cmake'
                script.write_text(''.join(f'set({k} "{v}")\n' for k, v in setup.items())+fragment+
                                  'file(WRITE "'+str(root/'actual')+'" "${ZEROCOOL_BUILD_FINGERPRINT}")\n')
                subprocess.run(['cmake', '-P', str(script)], check=True, capture_output=True, timeout=10)
                current = build_fingerprint(root); identities.add(current)
                self.assertEqual(current, (root/'actual').read_text())
                self.assertIn(flags, configuration_identity(build))
            self.assertEqual(len(identities), 3)
            before = build_fingerprint(root)
            (root/'CMakeLists.txt').write_text('changed')
            self.assertNotEqual(before, build_fingerprint(root))
            (build/'CMakeCache.txt').write_text('')
            with self.assertRaises(ValueError): configuration_identity(build)

if __name__ == '__main__': unittest.main()
