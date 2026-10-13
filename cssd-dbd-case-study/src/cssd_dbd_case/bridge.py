"""Read-only imports from the two existing algorithm projects.

This package owns only the DBD experiment orchestration.  It deliberately does
not copy, patch, configure, or write into either sibling algorithm project.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTICLE_ROOT = PROJECT_ROOT.parent
IBSCABC_ROOT = ARTICLE_ROOT / "cssd-kg-edr"
BASELINES_ROOT = ARTICLE_ROOT / "cssd-comparison-baselines"
IBSCABC_SRC = IBSCABC_ROOT / "src"
BASELINES_SRC = BASELINES_ROOT / "src"


def _activate() -> None:
    required = (
        IBSCABC_SRC / "cssd_ibscabc",
        BASELINES_SRC / "cssd_baselines",
    )
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise RuntimeError("missing sibling source package(s): " + ", ".join(missing))
    # Importing a source tree normally creates __pycache__ entries beside it.
    # Disable that side effect so the two sibling experiment projects remain
    # genuinely read-only dependencies of this isolated case-study runner.
    sys.dont_write_bytecode = True
    # The baseline package imports cssd_ibscabc during module initialization.
    # Put both roots on sys.path without importing or mutating sibling configs.
    for source in (IBSCABC_SRC, BASELINES_SRC):
        source_text = str(source)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)


def source_tree_sha256(root: Path) -> str:
    """Hash Python source files for a reproducible read-only dependency record."""

    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py"), key=lambda item: str(item).lower()):
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


_activate()
