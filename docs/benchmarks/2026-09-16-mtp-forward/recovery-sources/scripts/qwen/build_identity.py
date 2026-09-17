"""Source identity matching cmake/qwen.cmake for qualification evidence."""
import hashlib
from pathlib import Path


def build_fingerprint(root: Path) -> str:
    paths = sorted(list((root / 'src/qwen').glob('*')) + list((root / 'include/qwen').glob('*')))
    paths += [root / 'kernels/metal/qwen.metal', root / 'models.lock.json', root / 'mixed-models.lock.json', root / 'mixed-payload-reuse.lock.json', root / 'cmake/qwen.cmake']
    identity = ''.join(f'{p.relative_to(root)}:{hashlib.sha256(p.read_bytes()).hexdigest()}\n' for p in paths)
    return hashlib.sha256(identity.encode()).hexdigest()
