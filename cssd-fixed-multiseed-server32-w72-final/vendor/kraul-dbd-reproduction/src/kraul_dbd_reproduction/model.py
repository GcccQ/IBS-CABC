from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class AuthorInstance:
    """A public Kraul instance after the Notebook's Sule processing-time rule.

    ``jobs`` has columns ``id, p1, p2, release, family, load1, load2``.  The
    processing times already include the family setup time, exactly as in the
    released Notebook.
    """

    path: Path
    source_sha256: str
    job_count: int
    family_count: int
    machines_stage1: int
    machines_stage2: int
    capacity_stage1: int
    capacity_stage2: int
    jobs: np.ndarray
    raw_jobs: np.ndarray
    spt_order: np.ndarray
    erd_order: np.ndarray
    lpt_order: np.ndarray

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def release_times(self) -> np.ndarray:
        return self.jobs[:, 3]


def _order_by_column(jobs: np.ndarray, column: int, reverse: bool = False) -> np.ndarray:
    # NumPy's default argsort is intentional: this mirrors ``array.argsort()``
    # in the author's Notebook, including its tie behaviour.
    indices = jobs[:, column].argsort()
    if reverse:
        indices = indices[::-1]
    return jobs[indices, 0].astype(np.int64) - 1


def load_instance(path: str | Path) -> AuthorInstance:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8-sig") as handle:
        header = handle.readline().split()
        if len(header) != 6:
            raise ValueError(f"invalid six-value instance header: {source}")
        job_count, family_count, m1, m2, k1, k2 = map(int, header)
        if min(job_count, family_count, m1, m2, k1, k2) <= 0:
            raise ValueError(f"all six instance header values must be positive: {source}")
        rows: list[list[float]] = []
        for row_number in range(1, job_count + 1):
            line = handle.readline()
            if not line:
                raise ValueError(
                    f"instance ended before job row {row_number} of {job_count}: {source}"
                )
            values = line.split()
            if len(values) != 9:
                raise ValueError(
                    f"job row {row_number} must contain 9 values, got {len(values)}: {source}"
                )
            rows.append(list(map(float, values)))
        trailing = [line for line in handle if line.strip()]
    if trailing:
        raise ValueError(f"instance has unexpected trailing rows: {source}")
    raw = np.asarray(rows, dtype=np.float64)
    if raw.shape != (job_count, 9):
        raise ValueError(f"expected {job_count} rows with 9 values, got {raw.shape}: {source}")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"instance contains NaN or infinite values: {source}")
    if not np.array_equal(raw, np.floor(raw)):
        raise ValueError(f"all public-instance values must be integral: {source}")
    expected_ids = np.arange(1, job_count + 1, dtype=np.float64)
    if not np.array_equal(raw[:, 0], expected_ids):
        raise ValueError(f"job IDs must be consecutive 1..J: {source}")
    families = raw[:, 4]
    if np.any(families < 1) or np.any(families > family_count):
        raise ValueError(f"job family outside 1..F: {source}")
    if np.any(raw[:, [1, 2, 3, 5, 6]] < 0):
        raise ValueError(f"processing, release, and setup times cannot be negative: {source}")
    if np.any(raw[:, 7] <= 0) or np.any(raw[:, 8] <= 0):
        raise ValueError(f"job capacity requirements must be positive: {source}")
    if np.any(raw[:, 7] > k1) or np.any(raw[:, 8] > k2):
        raise ValueError(f"a job exceeds a stage capacity: {source}")
    if np.any(np.mod(raw[:, 8], 10) != 0):
        raise ValueError(f"stage-2 capacity requirements must be multiples of 10: {source}")
    for family in range(1, family_count + 1):
        family_rows = raw[families == family]
        if not len(family_rows):
            continue
        if len(np.unique(family_rows[:, 5])) != 1 or len(np.unique(family_rows[:, 6])) != 1:
            raise ValueError(f"family {family} has inconsistent setup times: {source}")

    # Sule's rule in the released code: add family setup to every job's
    # processing time, then remove the two setup columns.
    jobs = raw.copy()
    jobs[:, 1] += jobs[:, 5]
    jobs[:, 2] += jobs[:, 6]
    if not np.all(np.isfinite(jobs[:, [1, 2]])):
        raise ValueError(f"Sule-adjusted processing times overflowed: {source}")
    jobs = np.delete(jobs, [5, 6], axis=1)

    return AuthorInstance(
        path=source,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        job_count=job_count,
        family_count=family_count,
        machines_stage1=m1,
        machines_stage2=m2,
        capacity_stage1=k1,
        capacity_stage2=k2,
        jobs=jobs,
        raw_jobs=raw,
        spt_order=_order_by_column(jobs, 1),
        erd_order=_order_by_column(jobs, 3),
        lpt_order=_order_by_column(jobs, 1, reverse=True),
    )


def official_time_limit(instance: AuthorInstance, tau: float = 1.0) -> float:
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    value = (5000.0 + tau * instance.job_count**2 * (
        (instance.machines_stage1 + instance.machines_stage2) / 2.0
    )) / 1000.0
    return min(value, 1800.0)
