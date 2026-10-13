from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..core.entities import Job, ProblemInstance


class DataValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    root: Path
    instance_count: int
    job_count: int
    invalid_files: tuple[str, ...]
    family_min: int
    family_max: int
    job_min: int
    job_max: int

    @property
    def valid(self) -> bool:
        return not self.invalid_files


def _integer(value: str, name: str, path: Path, row: int) -> int:
    try:
        number = float(value)
    except ValueError as error:
        raise DataValidationError(f"{path}:{row}: {name} must be numeric") from error
    if not math.isfinite(number):
        raise DataValidationError(f"{path}:{row}: {name} must be finite")
    integer = int(number)
    if number != integer:
        raise DataValidationError(f"{path}:{row}: {name} must be integral")
    return integer


def load_instance(path: str | Path) -> ProblemInstance:
    source = Path(path).resolve()
    payload = source.read_bytes()
    lines = payload.decode("utf-8-sig").splitlines()
    if not lines:
        raise DataValidationError(f"{source}: empty instance")
    header = lines[0].split()
    if len(header) != 6:
        raise DataValidationError(f"{source}: header must have six values")
    j_count, families, m1, m2, capacity1, capacity2 = (
        _integer(value, name, source, 1)
        for value, name in zip(header, ("J", "F", "m1", "m2", "k1", "k2"))
    )
    if len(lines) - 1 != j_count or min(j_count, families, m1, m2, capacity1, capacity2) <= 0:
        raise DataValidationError(f"{source}: inconsistent header")
    jobs: list[Job] = []
    for row_number, line in enumerate(lines[1:], 2):
        values = line.split()
        if len(values) != 9:
            raise DataValidationError(f"{source}:{row_number}: expected nine columns")
        fields = [
            _integer(value, name, source, row_number)
            for value, name in zip(
                values,
                (
                    "job_id",
                    "processing_stage1",
                    "processing_stage2",
                    "release",
                    "family",
                    "setup_stage1",
                    "setup_stage2",
                    "size_stage1",
                    "size_stage2",
                ),
            )
        ]
        job = Job(*fields)
        if job.job_id != row_number - 1 or not 1 <= job.family <= families:
            raise DataValidationError(f"{source}:{row_number}: invalid job id/family")
        if (
            min(
                job.processing_stage1,
                job.processing_stage2,
                job.release,
                job.setup_stage1,
                job.setup_stage2,
            )
            < 0
        ):
            raise DataValidationError(f"{source}:{row_number}: negative time")
        if not 1 <= job.size_stage1 <= capacity1 or not 1 <= job.size_stage2 <= capacity2:
            raise DataValidationError(f"{source}:{row_number}: job exceeds capacity")
        if job.size_stage2 % 10:
            raise DataValidationError(
                f"{source}:{row_number}: stage-2 size is not a multiple of 10"
            )
        jobs.append(job)
    for family in range(1, families + 1):
        family_jobs = [job for job in jobs if job.family == family]
        for stage, setups in (
            (1, {job.setup_stage1 for job in family_jobs}),
            (2, {job.setup_stage2 for job in family_jobs}),
        ):
            if len(setups) > 1:
                raise DataValidationError(
                    f"{source}: family {family} has inconsistent stage-{stage} setup times"
                )
    return ProblemInstance(
        source.stem,
        source,
        hashlib.sha256(payload).hexdigest(),
        families,
        m1,
        m2,
        capacity1,
        capacity2,
        tuple(jobs),
    )


def iter_instance_paths(root: str | Path) -> Iterable[Path]:
    yield from sorted(Path(root).resolve().rglob("*.txt"), key=lambda p: str(p).lower())


def audit_dataset(root: str | Path) -> DatasetAudit:
    paths = list(iter_instance_paths(root))
    instances: list[ProblemInstance] = []
    invalid: list[str] = [] if paths else [f"{Path(root).resolve()}: no .txt instances found"]
    for path in paths:
        try:
            instances.append(load_instance(path))
        except (OSError, UnicodeError, DataValidationError) as error:
            invalid.append(f"{path}: {error}")
    counts = [instance.job_count for instance in instances]
    families = [len(instance.active_families) for instance in instances]
    return DatasetAudit(
        Path(root).resolve(),
        len(paths),
        sum(counts),
        tuple(invalid),
        min(families, default=0),
        max(families, default=0),
        min(counts, default=0),
        max(counts, default=0),
    )
