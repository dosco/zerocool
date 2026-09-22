"""Skip a model-free check when the benchmark evidence it reads is not present.

`docs/benchmarks/**` is deliberately unversioned, so a clean checkout carries
these checks but not the recorded runs they read as fixed inputs. Erroring there
reports an absent file as though it were a defect and buries the rest of the
suite; skipping names what is missing and leaves every other check meaningful.
Evidence a script needs as a fixed input can instead be allowlisted in
.gitignore, which is the right answer when the file is small and stable.
"""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def require(*paths):
    """Skip the calling module unless every path exists under the repository root."""
    missing = sorted({str(p) for p in paths if not (ROOT/p).exists()})
    if missing:
        raise unittest.SkipTest('unversioned benchmark evidence absent: '+', '.join(missing))
