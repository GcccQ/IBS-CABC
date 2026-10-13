from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from math import isclose
from typing import Sequence

from .entities import Batch, ProblemInstance, Schedule
from .objective import calculate_metrics
from .signatures import batch_signature


class FeasibilityError(RuntimeError):
    pass


def _coverage(batches: Sequence[Batch], expected: set[int], stage: int) -> None:
    counts = Counter(job_id for batch in batches for job_id in batch.job_ids)
    missing = sorted(expected - counts.keys())
    duplicates = sorted(job_id for job_id, count in counts.items() if count != 1)
    extras = sorted(counts.keys() - expected)
    if missing or duplicates or extras:
        raise FeasibilityError(
            f"stage {stage} coverage failed: missing={missing[:8]}, "
            f"duplicates={duplicates[:8]}, extras={extras[:8]}"
        )


def _validate_stage(
    instance: ProblemInstance,
    batches: Sequence[Batch],
    stage: int,
    completion_stage1: Sequence[int] | None = None,
) -> None:
    capacity = instance.capacity_stage1 if stage == 1 else instance.capacity_stage2
    machines = instance.machines_stage1 if stage == 1 else instance.machines_stage2
    if {batch.batch_id for batch in batches} != set(range(len(batches))):
        raise FeasibilityError(f"stage {stage} batch ids must be contiguous and unique")
    if {batch.priority_rank for batch in batches} != set(range(len(batches))):
        raise FeasibilityError(f"stage {stage} priority ranks must be contiguous and unique")
    if len({batch.signature for batch in batches}) != len(batches):
        raise FeasibilityError(f"stage {stage} batch signatures must be unique")
    timelines: dict[int, list[Batch]] = defaultdict(list)
    for batch in batches:
        if not batch.job_ids:
            raise FeasibilityError(f"stage {stage} contains an empty batch")
        if batch.stage != stage or not 0 <= batch.machine_id < machines:
            raise FeasibilityError(f"batch {batch.batch_id} has invalid stage or machine")
        selected = [instance.jobs_by_id[job_id] for job_id in batch.job_ids]
        if {job.family for job in selected} != {batch.family}:
            raise FeasibilityError(f"batch {batch.batch_id} mixes incompatible families")
        if batch.signature != batch_signature(stage, batch.family, batch.job_ids):
            raise FeasibilityError(f"batch {batch.batch_id} has an invalid structural signature")
        sizes = [job.size_stage1 if stage == 1 else job.size_stage2 for job in selected]
        durations = [job.duration_stage1 if stage == 1 else job.duration_stage2 for job in selected]
        releases = (
            [job.release for job in selected]
            if stage == 1
            else [completion_stage1[job.job_id - 1] for job in selected]
        )
        if sum(sizes) != batch.load or not 0 < batch.load <= capacity:
            raise FeasibilityError(f"batch {batch.batch_id} violates capacity")
        if max(durations) != batch.processing_time:
            raise FeasibilityError(f"batch {batch.batch_id} processing time mismatch")
        if max(releases) != batch.release_time or batch.start < batch.release_time:
            raise FeasibilityError(f"batch {batch.batch_id} violates release precedence")
        if batch.end != batch.start + batch.processing_time:
            raise FeasibilityError(f"batch {batch.batch_id} has an inconsistent interval")
        timelines[batch.machine_id].append(batch)
    for machine, timeline in timelines.items():
        timeline.sort(key=lambda item: (item.start, item.batch_id))
        for previous, current in zip(timeline, timeline[1:]):
            if current.start < previous.end:
                raise FeasibilityError(
                    f"stage {stage} machine {machine} overlaps batches "
                    f"{previous.batch_id}/{current.batch_id}"
                )


def validate_schedule(instance: ProblemInstance, schedule: Schedule) -> None:
    if schedule.instance_name != instance.name:
        raise FeasibilityError("schedule belongs to a different instance")
    expected = {job.job_id for job in instance.jobs}
    _coverage(schedule.batches_stage1, expected, 1)
    _coverage(schedule.batches_stage2, expected, 2)
    _validate_stage(instance, schedule.batches_stage1, 1)
    _validate_stage(instance, schedule.batches_stage2, 2, schedule.completion_stage1)
    c1 = [0] * instance.job_count
    c2 = [0] * instance.job_count
    for batch in schedule.batches_stage1:
        for job_id in batch.job_ids:
            c1[job_id - 1] = batch.end
    for batch in schedule.batches_stage2:
        for job_id in batch.job_ids:
            c2[job_id - 1] = batch.end
    if tuple(c1) != schedule.completion_stage1 or tuple(c2) != schedule.completion_stage2:
        raise FeasibilityError("completion arrays disagree with decoded batches")
    recomputed = calculate_metrics(
        instance,
        schedule.batches_stage1,
        schedule.batches_stage2,
        schedule.completion_stage1,
        schedule.completion_stage2,
    )
    for name, expected_value in asdict(recomputed).items():
        actual = getattr(schedule.metrics, name)
        if isinstance(expected_value, float):
            if not isclose(actual, expected_value, rel_tol=1e-12, abs_tol=1e-12):
                raise FeasibilityError(f"metric {name} was not independently recomputed")
        elif actual != expected_value:
            raise FeasibilityError(f"metric {name} was not independently recomputed")
