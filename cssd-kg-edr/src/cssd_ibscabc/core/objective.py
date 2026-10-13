from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Iterable, Sequence

from .entities import Batch, ProblemInstance, ScheduleMetrics


@dataclass(frozen=True, slots=True)
class MetricMoments:
    """Sufficient statistics for exact cross-day mean/std aggregation."""

    count: int
    total: float
    total_squares: float

    @classmethod
    def from_values(cls, values: Sequence[int | float]) -> MetricMoments:
        return cls(
            count=len(values),
            total=float(sum(values)),
            total_squares=float(sum(float(value) ** 2 for value in values)),
        )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "count": self.count,
            "sum": self.total,
            "sum_squares": self.total_squares,
        }


def _utilization(batches: Sequence[Batch], machines: int) -> float:
    """Return stage utilization over that stage's own active horizon."""
    if not batches or machines <= 0:
        return 0.0
    horizon = max(batch.end for batch in batches) - min(batch.start for batch in batches)
    if horizon <= 0:
        return 0.0
    return sum(batch.processing_time for batch in batches) / (machines * horizon)


def _mean_fill(batches: Iterable[Batch], capacity: int) -> float:
    values = [batch.load / capacity for batch in batches]
    return fmean(values) if values else 0.0


def _nearest_rank_p95(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)])


def job_level_metric_moments(
    instance: ProblemInstance,
    batches_stage1: Sequence[Batch],
    batches_stage2: Sequence[Batch],
    completion_stage1: Sequence[int],
    completion_stage2: Sequence[int],
) -> dict[str, MetricMoments]:
    """Return paper Table-8 device-level metric moments.

    Persisting the three moments instead of every device observation keeps the
    JSONL bounded while still allowing an exact pooled sample standard
    deviation across all independently solved DBD days.
    """

    stage1_start = {
        job_id: batch.start for batch in batches_stage1 for job_id in batch.job_ids
    }
    stage2_start = {
        job_id: batch.start for batch in batches_stage2 for job_id in batch.job_ids
    }
    time_in_system = [
        completion_stage2[job.job_id - 1] - job.release for job in instance.jobs
    ]
    waiting_stage1 = [
        stage1_start[job.job_id] - job.release for job in instance.jobs
    ]
    waiting_stage2 = [
        stage2_start[job.job_id] - completion_stage1[job.job_id - 1]
        for job in instance.jobs
    ]
    return {
        "time_in_system": MetricMoments.from_values(time_in_system),
        "waiting_stage1": MetricMoments.from_values(waiting_stage1),
        "waiting_stage2": MetricMoments.from_values(waiting_stage2),
    }


def hourly_completed_devices(
    batches: Sequence[Batch],
    *,
    day_length_minutes: int = 1440,
    bin_minutes: int = 60,
) -> tuple[int, ...]:
    """Count devices completed in each time-of-day bin."""

    if day_length_minutes <= 0 or bin_minutes <= 0:
        raise ValueError("day length and bin size must be positive")
    if day_length_minutes % bin_minutes:
        raise ValueError("day length must be divisible by bin size")
    counts = [0] * (day_length_minutes // bin_minutes)
    for batch in batches:
        minute = batch.end % day_length_minutes
        index = min(len(counts) - 1, minute // bin_minutes)
        counts[index] += len(batch.job_ids)
    return tuple(counts)


def calculate_metrics(
    instance: ProblemInstance,
    batches_stage1: Sequence[Batch],
    batches_stage2: Sequence[Batch],
    completion_stage1: Sequence[int],
    completion_stage2: Sequence[int],
) -> ScheduleMetrics:
    stage1_start = {job_id: batch.start for batch in batches_stage1 for job_id in batch.job_ids}
    stage2_start = {job_id: batch.start for batch in batches_stage2 for job_id in batch.job_ids}
    time_in_system = [completion_stage2[job.job_id - 1] - job.release for job in instance.jobs]
    waiting_stage1 = [stage1_start[job.job_id] - job.release for job in instance.jobs]
    waiting_stage2 = [
        stage2_start[job.job_id] - completion_stage1[job.job_id - 1] for job in instance.jobs
    ]
    total_completion = int(sum(completion_stage2))
    total_flow = int(sum(time_in_system))
    makespan = int(max(completion_stage2, default=0))
    return ScheduleMetrics(
        total_completion_time=total_completion,
        total_flow_time=total_flow,
        average_time_in_system=total_flow / max(1, instance.job_count),
        p95_time_in_system=_nearest_rank_p95(time_in_system),
        makespan=makespan,
        waiting_stage1=fmean(waiting_stage1) if waiting_stage1 else 0.0,
        waiting_stage2=fmean(waiting_stage2) if waiting_stage2 else 0.0,
        utilization_stage1=_utilization(batches_stage1, instance.machines_stage1),
        utilization_stage2=_utilization(batches_stage2, instance.machines_stage2),
        mean_fill_stage1=_mean_fill(batches_stage1, instance.capacity_stage1),
        mean_fill_stage2=_mean_fill(batches_stage2, instance.capacity_stage2),
        batch_cycles_stage1=len(batches_stage1),
        batch_cycles_stage2=len(batches_stage2),
    )
