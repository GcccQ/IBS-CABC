from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from ..core.entities import Job, ProblemInstance
from ..io.loader import load_instance

_SEED_PATTERN = re.compile(r"_seed_(\d+)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class CaseWindowAudit:
    days: int
    total_jobs: int
    minimum_jobs: int
    maximum_jobs: int
    mean_jobs: float
    first_seed: int | None
    last_seed: int | None
    contiguous_seeds: bool


def audit_case_window(paths: Sequence[str | Path]) -> CaseWindowAudit:
    """Audit order and scale of a locked multi-day case-study split."""

    if not paths:
        raise ValueError("case-study window is empty")
    resolved = tuple(Path(path).resolve() for path in paths)
    days = tuple(load_instance(path) for path in resolved)
    job_counts = [day.job_count for day in days]
    seeds: list[int] = []
    for path in resolved:
        match = _SEED_PATTERN.search(path.stem)
        if match is None:
            seeds = []
            break
        seeds.append(int(match.group(1)))
    return CaseWindowAudit(
        days=len(days),
        total_jobs=sum(job_counts),
        minimum_jobs=min(job_counts),
        maximum_jobs=max(job_counts),
        mean_jobs=sum(job_counts) / len(job_counts),
        first_seed=seeds[0] if seeds else None,
        last_seed=seeds[-1] if seeds else None,
        contiguous_seeds=(
            bool(seeds)
            and seeds == list(range(seeds[0], seeds[0] + len(seeds)))
        ),
    )


def combine_daily_instances(
    paths: Sequence[str | Path],
    *,
    day_length_minutes: int = 1440,
    machines_stage1: int | None = None,
    machines_stage2: int | None = None,
) -> ProblemInstance:
    """Build the AIO instance by placing consecutive daily instances on one clock.

    Job identifiers are reassigned consecutively and release times receive the
    corresponding day offset.  All other job fields remain unchanged, so AIO
    may form cross-day batches without changing the CSSD model.
    """

    if not paths:
        raise ValueError("AIO requires at least one daily instance")
    if day_length_minutes <= 0:
        raise ValueError("day_length_minutes must be positive")
    resolved = tuple(Path(path).resolve() for path in paths)
    days = tuple(load_instance(path) for path in resolved)
    first = days[0]
    structural_fields = (
        "family_count",
        "capacity_stage1",
        "capacity_stage2",
    )
    for day in days[1:]:
        for field in structural_fields:
            if getattr(day, field) != getattr(first, field):
                raise ValueError(
                    f"AIO daily instances disagree on {field}: "
                    f"{first.source.name} vs {day.source.name}"
                )
    jobs: list[Job] = []
    next_job_id = 1
    for day_index, day in enumerate(days):
        offset = day_index * day_length_minutes
        for job in day.jobs:
            jobs.append(
                replace(
                    job,
                    job_id=next_job_id,
                    release=job.release + offset,
                )
            )
            next_job_id += 1
    digest = hashlib.sha256()
    for path, day in zip(resolved, days):
        digest.update(path.name.encode("utf-8"))
        digest.update(day.source_sha256.encode("ascii"))
    digest.update(str(day_length_minutes).encode("ascii"))
    m1 = first.machines_stage1 if machines_stage1 is None else int(machines_stage1)
    m2 = first.machines_stage2 if machines_stage2 is None else int(machines_stage2)
    if m1 <= 0 or m2 <= 0:
        raise ValueError("machine counts must be positive")
    virtual_source = resolved[0].parent / (
        f"_AIO_{resolved[0].stem}_to_{resolved[-1].stem}.txt"
    )
    return ProblemInstance(
        name=f"AIO_{resolved[0].stem}_to_{resolved[-1].stem}",
        source=virtual_source,
        source_sha256=digest.hexdigest(),
        family_count=first.family_count,
        machines_stage1=m1,
        machines_stage2=m2,
        capacity_stage1=first.capacity_stage1,
        capacity_stage2=first.capacity_stage2,
        jobs=tuple(jobs),
    )


def override_machine_counts(
    instance: ProblemInstance,
    *,
    machines_stage1: int | None = None,
    machines_stage2: int | None = None,
) -> ProblemInstance:
    """Apply a Table-9 machine scenario without changing jobs or capacities."""

    m1 = instance.machines_stage1 if machines_stage1 is None else int(machines_stage1)
    m2 = instance.machines_stage2 if machines_stage2 is None else int(machines_stage2)
    if m1 <= 0 or m2 <= 0:
        raise ValueError("machine counts must be positive")
    return replace(instance, machines_stage1=m1, machines_stage2=m2)
