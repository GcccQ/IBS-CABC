"""IBS-CABC for two-stage CSSD parallel-batch scheduling."""

import os

# The paper protocols require one internal numerical thread per independent run.
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"

__version__ = "1.2.0"
