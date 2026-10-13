"""Resolve the locked solver sources in a workstation or server-bundle layout."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTICLE = ROOT.parent
PROJECTS = ("cssd-kg-edr", "cssd-comparison-baselines", "kraul-dbd-reproduction")


def project_source(project: str) -> Path:
    """Prefer self-contained vendored sources, then the local sibling checkout."""
    candidates = (ROOT / "vendor" / project / "src", ARTICLE / project / "src")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    rendered = " or ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"missing solver source for {project}: expected {rendered}")


SOURCE_ROOTS = {project: project_source(project) for project in PROJECTS}
sys.dont_write_bytecode = True

# One outer process equals one independent algorithm run.  Force every common
# numerical backend to one thread before NumPy/OR-Tools or solver code imports.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"
os.environ.setdefault("PYTHONHASHSEED", "0")

for source_root in SOURCE_ROOTS.values():
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)
