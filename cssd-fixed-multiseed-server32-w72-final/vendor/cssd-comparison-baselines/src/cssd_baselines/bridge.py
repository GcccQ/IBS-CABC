"""Load the paper's shared CSSD representation and evaluator from its sibling project.

Only problem-independent infrastructure is reused: instance parsing, direct two-stage
batch representation, deterministic decoding, feasibility checks, metrics and the
common seed portfolio.  All comparison search logic lives in this project.
"""

from __future__ import annotations

import sys
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[2]
ARTICLE_ROOT = BASELINE_ROOT.parent
REFERENCE_ROOT = ARTICLE_ROOT / "cssd-kg-edr"
REFERENCE_SRC = REFERENCE_ROOT / "src"

if not (REFERENCE_SRC / "cssd_ibscabc").is_dir():
    raise RuntimeError(f"missing sibling IBS-CABC source package: {REFERENCE_SRC}")

reference_text = str(REFERENCE_SRC)
if reference_text not in sys.path:
    sys.path.insert(0, reference_text)
