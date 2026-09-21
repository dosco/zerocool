"""Source identity matching cmake/qwen.cmake for qualification evidence."""
import hashlib
from pathlib import Path
import re


def configuration_identity(build: Path) -> str:
    """Read configured compiler settings; do not guess Release or sanitizer flags."""
    cache = {}
    for line in (build/'CMakeCache.txt').read_text().splitlines():
        match = re.fullmatch(r'([^:#/][^:]*):[^=]+=(.*)', line)
        if match: cache[match[1]] = match[2]
    paths = list((build/'CMakeFiles').glob('*/CMakeCXXCompiler.cmake'))
    if len(paths) != 1: raise ValueError('Missing or ambiguous configured C++ compiler identity')
    compiler = paths[0].read_text()
    def setting(name):
        values = re.findall(r'^set\('+re.escape(name)+r' "([^"\n]*)"\)$', compiler, re.MULTILINE)
        if len(values) != 1 or not values[0]: raise ValueError('Missing configured '+name)
        return values[0]
    for key in ('CMAKE_BUILD_TYPE', 'CMAKE_CXX_FLAGS', 'ZEROCOOL_SANITIZE'):
        if key not in cache: raise ValueError('Missing configured '+key)
    mode = cache['CMAKE_BUILD_TYPE']
    flags = cache['CMAKE_CXX_FLAGS']+' '+cache.get('CMAKE_CXX_FLAGS_'+mode.upper(), '')
    return (f'toolchain:{setting("CMAKE_CXX_COMPILER_ID")}-{setting("CMAKE_CXX_COMPILER_VERSION")}\n'
            f'build_type:{mode}\nflags:{flags}\nsanitizer:{cache["ZEROCOOL_SANITIZE"]}\n')


def build_fingerprint(root: Path) -> str:
    root = Path(root)
    paths = sorted(list((root / 'src/engine').glob('*')) + list((root / 'include/engine').glob('*')))
    paths += [root / p for p in ('kernels/metal/qwen.metal', 'models.lock.json', 'mixed-models.lock.json',
        'mixed-payload-reuse.lock.json', 'cmake/qwen.cmake', 'cmake/qwen_embedded.hpp.in', 'CMakeLists.txt')]
    identity = ''.join(f'{p.relative_to(root)}:{hashlib.sha256(p.read_bytes()).hexdigest()}\n' for p in paths)
    identity += configuration_identity(root/'build/qwen')
    return hashlib.sha256(identity.encode()).hexdigest()
