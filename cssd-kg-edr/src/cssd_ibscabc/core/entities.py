from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Job:
    job_id: int
    processing_stage1: int
    processing_stage2: int
    release: int
    family: int
    setup_stage1: int
    setup_stage2: int
    size_stage1: int
    size_stage2: int

    @property
    def duration_stage1(self) -> int:
        return self.processing_stage1 + self.setup_stage1

    @property
    def duration_stage2(self) -> int:
        return self.processing_stage2 + self.setup_stage2


@dataclass(frozen=True, slots=True)
class ProblemInstance:
    name: str
    source: Path
    source_sha256: str
    family_count: int
    machines_stage1: int
    machines_stage2: int
    capacity_stage1: int
    capacity_stage2: int
    jobs: tuple[Job, ...]
    _jobs_by_id: Mapping[int, Job] = field(init=False, repr=False, compare=False)
    family_job_ids: Mapping[int, tuple[int, ...]] = field(init=False, repr=False, compare=False)
    active_families: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        by_id = {job.job_id: job for job in self.jobs}
        by_family = {
            family: tuple(job.job_id for job in self.jobs if job.family == family)
            for family in range(1, self.family_count + 1)
        }
        object.__setattr__(self, "_jobs_by_id", MappingProxyType(by_id))
        object.__setattr__(self, "family_job_ids", MappingProxyType(by_family))
        object.__setattr__(self, "active_families", tuple(f for f, ids in by_family.items() if ids))

    @property
    def job_count(self) -> int:
        return len(self.jobs)

    @property
    def jobs_by_id(self) -> Mapping[int, Job]:
        return self._jobs_by_id


@dataclass(frozen=True, slots=True)
class Batch:
    batch_id: int
    stage: int
    family: int
    job_ids: tuple[int, ...]
    load: int
    processing_time: int
    release_time: int
    priority_rank: int
    machine_id: int
    start: int
    end: int
    signature: str


@dataclass(frozen=True, slots=True)
class ScheduleMetrics:
    total_completion_time: int
    total_flow_time: int
    average_time_in_system: float
    p95_time_in_system: float
    makespan: int
    waiting_stage1: float
    waiting_stage2: float
    utilization_stage1: float
    utilization_stage2: float
    mean_fill_stage1: float
    mean_fill_stage2: float
    batch_cycles_stage1: int
    batch_cycles_stage2: int

    @property
    def total_batch_cycles(self) -> int:
        return self.batch_cycles_stage1 + self.batch_cycles_stage2


@dataclass(frozen=True, slots=True)
class Schedule:
    instance_name: str
    algorithm: str
    batches_stage1: tuple[Batch, ...]
    batches_stage2: tuple[Batch, ...]
    completion_stage1: tuple[int, ...]
    completion_stage2: tuple[int, ...]
    metrics: ScheduleMetrics
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def objective(self) -> int:
        """Published primary objective: sum of stage-2 completion times."""

        return self.metrics.total_completion_time


@dataclass(frozen=True, slots=True)
class RunResult:
    schedule: Schedule
    evaluations: int
    runtime_seconds: float
    iterations: int
    history: tuple[int, ...]
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    history_evaluations: tuple[int, ...] = ()
