"""Build a PyPI-uploadable distribution.

The dependency on `minicheck` is declared as a PEP 508 direct reference
(`minicheck @ git+https://...`) so that specforge installs today, with no package
index involved. PyPI **rejects** direct references on upload, so the artifact built by a
plain `python -m build` is correct for GitHub installs and unusable for PyPI.

This script swaps the direct reference for an ordinary version constraint, builds, and
puts the file back. Run it only when minicheck is actually on PyPI — until then the
resulting artifact would declare a dependency that cannot be resolved, which is the
defect this whole arrangement exists to avoid.

    python build_pypi.py            # writes dist/ and restores pyproject.toml
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).parent
PYPROJECT = ROOT / "pyproject.toml"
DIRECT = re.compile(r'dependencies = \["minicheck @ git\+[^"]*"\]')
PINNED = 'dependencies = ["minicheck>=0.2.0"]'


def main() -> int:
    original = PYPROJECT.read_text(encoding="utf-8")
    if not DIRECT.search(original):
        print("pyproject.toml has no direct minicheck reference; nothing to swap.", file=sys.stderr)
        return 1

    backup = PYPROJECT.with_suffix(".toml.orig")
    shutil.copy2(PYPROJECT, backup)
    try:
        PYPROJECT.write_text(DIRECT.sub(PINNED, original), encoding="utf-8")
        shutil.rmtree(ROOT / "dist", ignore_errors=True)
        subprocess.run([sys.executable, "-m", "build", "-o", "dist"], cwd=ROOT, check=True)
    finally:
        shutil.copy2(backup, PYPROJECT)
        backup.unlink()

    built = sorted(p.name for p in (ROOT / "dist").glob("*"))
    print(f"\nbuilt for PyPI: {built}")
    print("pyproject.toml restored to the direct reference.")
    print("\nUpload only once minicheck>=0.2.0 exists on PyPI, or the dependency will not resolve:")
    print("    twine upload dist/*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
