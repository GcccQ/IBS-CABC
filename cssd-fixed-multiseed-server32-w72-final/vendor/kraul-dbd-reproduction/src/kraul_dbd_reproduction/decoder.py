from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .model import AuthorInstance


@dataclass(frozen=True, slots=True)
class Batch:
    stage: int
    family: int
    family_batch_index: int
    job_indices: tuple[int, ...]
    processing_time: float
    release_time: float
    load: float
    machine_id: int = 0
    start_time: float = 0.0
    completion_time: float = 0.0


@dataclass(frozen=True, slots=True)
class DecodedSchedule:
    permutation: np.ndarray
    stage1_batches: tuple[Batch, ...]
    stage2_batches: tuple[Batch, ...]
    start_stage1: np.ndarray
    completion_stage1: np.ndarray
    machine_stage1: np.ndarray
    batch_stage1: np.ndarray
    start_stage2: np.ndarray
    completion_stage2: np.ndarray
    machine_stage2: np.ndarray
    batch_stage2: np.ndarray
    objective: float


def _validate_permutation(permutation: Iterable[int], job_count: int) -> np.ndarray:
    value = np.asarray(list(permutation), dtype=np.int64)
    if value.shape != (job_count,):
        raise ValueError(f"permutation must contain {job_count} genes")
    if not np.array_equal(np.sort(value), np.arange(job_count, dtype=np.int64)):
        raise ValueError("permutation must contain every zero-based job index exactly once")
    return value


def _family_indices(instance: AuthorInstance, permutation: np.ndarray) -> list[np.ndarray]:
    families = instance.jobs[:, 4].astype(np.int64)
    return [permutation[families[permutation] == family] for family in range(1, instance.family_count + 1)]


def _construct_family_batches(
    instance: AuthorInstance,
    indices: np.ndarray,
    release_times: np.ndarray,
    stage: int,
) -> list[Batch]:
    if stage not in (1, 2):
        raise ValueError("stage must be 1 or 2")
    if stage == 2 and len(indices):
        # Released source sorts ascending by stage-2 release and then consumes
        # from the tail, so the actual packing order is latest-release first.
        indices = indices[release_times[indices].argsort()]
    remaining = list(map(int, indices))
    processing_column = stage
    load_column = 4 + stage
    capacity = instance.capacity_stage1 if stage == 1 else instance.capacity_stage2
    family = int(instance.jobs[remaining[0], 4]) if remaining else 0
    batches: list[Batch] = []
    while remaining:
        members: list[int] = []
        load = 0.0
        while remaining:
            candidate = remaining[-1]
            demand = float(instance.jobs[candidate, load_column])
            if load + demand > capacity:
                break
            remaining.pop()
            members.append(candidate)
            load += demand
        if not members:
            raise ValueError(
                f"job {remaining[-1] + 1} cannot enter a stage-{stage} batch of capacity {capacity}"
            )
        batches.append(
            Batch(
                stage=stage,
                family=family,
                family_batch_index=len(batches),
                job_indices=tuple(members),
                processing_time=float(np.max(instance.jobs[members, processing_column])),
                release_time=float(np.max(release_times[members])),
                load=load,
            )
        )
    return batches


def _construct_batches(
    instance: AuthorInstance,
    permutation: np.ndarray,
    release_times: np.ndarray,
    stage: int,
) -> list[list[Batch]]:
    return [
        _construct_family_batches(instance, indices, release_times, stage)
        for indices in _family_indices(instance, permutation)
    ]


def _schedule_batches_spt(family_batches: list[list[Batch]], machine_count: int) -> tuple[Batch, ...]:
    # Tuple order mirrors the Notebook's (family, batch, release, processing)
    # list.  Python min() therefore keeps family/batch insertion order on SPT
    # ties, just as the released implementation does.
    jobs = [batch for batches in family_batches for batch in batches]
    machines = [(machine_id, 0.0, 0.0) for machine_id in range(1, machine_count + 1)]
    scheduled: list[Batch] = []
    current_time = 0.0
    while jobs:
        available = [batch for batch in jobs if batch.release_time <= current_time]
        if available:
            shortest = min(available, key=lambda batch: batch.processing_time)
            machines.sort(key=lambda item: item[1])
            machine_id, _, last_end = machines[0]
            start = max(last_end, shortest.release_time)
            completion = start + shortest.processing_time
            machines[0] = (machine_id, completion, completion)
            scheduled.append(
                Batch(
                    stage=shortest.stage,
                    family=shortest.family,
                    family_batch_index=shortest.family_batch_index,
                    job_indices=shortest.job_indices,
                    processing_time=shortest.processing_time,
                    release_time=shortest.release_time,
                    load=shortest.load,
                    machine_id=machine_id,
                    start_time=start,
                    completion_time=completion,
                )
            )
            jobs.remove(shortest)
            second_machine_load = machines[1][1] if machine_count > 1 else completion
            current_time = max(current_time, min(completion, second_machine_load))
        else:
            # Public instances use integral minutes.  Keeping the unit step is
            # deliberate because it is part of the author's scheduler.
            current_time += 1.0
    return tuple(scheduled)


def _job_times(
    batches: tuple[Batch, ...], job_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    starts = np.zeros(job_count, dtype=np.float64)
    completions = np.zeros(job_count, dtype=np.float64)
    machines = np.zeros(job_count, dtype=np.int64)
    batch_ids = np.zeros(job_count, dtype=np.int64)
    for scheduled_index, batch in enumerate(batches, start=1):
        members = np.asarray(batch.job_indices, dtype=np.int64)
        starts[members] = batch.start_time
        completions[members] = batch.completion_time
        machines[members] = batch.machine_id
        batch_ids[members] = scheduled_index
    return starts, completions, machines, batch_ids


def decode_permutation(instance: AuthorInstance, permutation: Iterable[int]) -> DecodedSchedule:
    order = _validate_permutation(permutation, instance.job_count)
    stage1_by_family = _construct_batches(instance, order, instance.release_times, stage=1)
    stage1_batches = _schedule_batches_spt(stage1_by_family, instance.machines_stage1)
    s1, c1, m1, b1 = _job_times(stage1_batches, instance.job_count)

    stage2_by_family = _construct_batches(instance, order, c1, stage=2)
    stage2_batches = _schedule_batches_spt(stage2_by_family, instance.machines_stage2)
    s2, c2, m2, b2 = _job_times(stage2_batches, instance.job_count)
    return DecodedSchedule(
        permutation=order.copy(),
        stage1_batches=stage1_batches,
        stage2_batches=stage2_batches,
        start_stage1=s1,
        completion_stage1=c1,
        machine_stage1=m1,
        batch_stage1=b1,
        start_stage2=s2,
        completion_stage2=c2,
        machine_stage2=m2,
        batch_stage2=b2,
        objective=float(np.sum(c2)),
    )


def validate_schedule(instance: AuthorInstance, schedule: DecodedSchedule) -> None:
    tolerance = 1e-9
    _validate_permutation(schedule.permutation, instance.job_count)
    vectors = {
        "start_stage1": schedule.start_stage1,
        "completion_stage1": schedule.completion_stage1,
        "machine_stage1": schedule.machine_stage1,
        "batch_stage1": schedule.batch_stage1,
        "start_stage2": schedule.start_stage2,
        "completion_stage2": schedule.completion_stage2,
        "machine_stage2": schedule.machine_stage2,
        "batch_stage2": schedule.batch_stage2,
    }
    for name, values in vectors.items():
        array = np.asarray(values)
        if array.shape != (instance.job_count,):
            raise ValueError(f"{name} must contain one value per job")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or infinite values")
    for name in ("machine_stage1", "batch_stage1", "machine_stage2", "batch_stage2"):
        array = np.asarray(vectors[name])
        if not np.array_equal(array, np.floor(array)):
            raise ValueError(f"{name} must contain integral identifiers")

    expected = set(range(instance.job_count))
    for (
        stage,
        batches,
        capacity,
        machine_count,
        p_column,
        load_column,
        releases,
        starts,
        completions,
        machines,
        batch_ids,
    ) in (
        (
            1,
            schedule.stage1_batches,
            instance.capacity_stage1,
            instance.machines_stage1,
            1,
            5,
            instance.release_times,
            schedule.start_stage1,
            schedule.completion_stage1,
            schedule.machine_stage1,
            schedule.batch_stage1,
        ),
        (
            2,
            schedule.stage2_batches,
            instance.capacity_stage2,
            instance.machines_stage2,
            2,
            6,
            schedule.completion_stage1,
            schedule.start_stage2,
            schedule.completion_stage2,
            schedule.machine_stage2,
            schedule.batch_stage2,
        ),
    ):
        members = [job for batch in batches for job in batch.job_indices]
        if len(members) != instance.job_count or set(members) != expected:
            raise ValueError(f"stage {stage} does not process every job exactly once")
        by_machine: dict[int, list[Batch]] = {}
        family_batch_indices: dict[int, list[int]] = {}
        for scheduled_index, batch in enumerate(batches, start=1):
            if batch.stage != stage:
                raise ValueError(f"stage {stage} contains a batch tagged for stage {batch.stage}")
            if not 1 <= batch.machine_id <= machine_count:
                raise ValueError(f"stage {stage} batch has invalid machine ID {batch.machine_id}")
            scalar_values = (
                batch.processing_time,
                batch.release_time,
                batch.load,
                batch.start_time,
                batch.completion_time,
            )
            if not np.all(np.isfinite(scalar_values)) or min(scalar_values) < 0:
                raise ValueError(f"stage {stage} batch has invalid numeric values")
            rows = np.asarray(batch.job_indices, dtype=np.int64)
            actual_families = set(instance.jobs[rows, 4].astype(int))
            if len(actual_families) != 1:
                raise ValueError(f"stage {stage} batch mixes incompatible families")
            actual_family = next(iter(actual_families))
            if batch.family != actual_family:
                raise ValueError(f"stage {stage} batch has an inconsistent family identifier")
            if batch.family_batch_index < 0:
                raise ValueError(f"stage {stage} batch has a negative family batch index")
            family_batch_indices.setdefault(batch.family, []).append(batch.family_batch_index)
            load = float(np.sum(instance.jobs[rows, load_column]))
            if load > capacity + tolerance or abs(load - batch.load) > tolerance:
                raise ValueError(f"stage {stage} batch violates capacity")
            if abs(float(np.max(instance.jobs[rows, p_column])) - batch.processing_time) > tolerance:
                raise ValueError(f"stage {stage} batch processing time is inconsistent")
            if abs(float(np.max(releases[rows])) - batch.release_time) > tolerance:
                raise ValueError(f"stage {stage} batch release time is inconsistent")
            if batch.start_time + tolerance < batch.release_time:
                raise ValueError(f"stage {stage} batch starts before release")
            if abs(batch.start_time + batch.processing_time - batch.completion_time) > tolerance:
                raise ValueError(f"stage {stage} batch completion is inconsistent")
            if not np.allclose(starts[rows], batch.start_time, rtol=0.0, atol=tolerance):
                raise ValueError(f"stage {stage} job starts disagree with its batch")
            if not np.allclose(completions[rows], batch.completion_time, rtol=0.0, atol=tolerance):
                raise ValueError(f"stage {stage} job completions disagree with its batch")
            if not np.all(np.asarray(machines)[rows] == batch.machine_id):
                raise ValueError(f"stage {stage} job machine IDs disagree with its batch")
            if not np.all(np.asarray(batch_ids)[rows] == scheduled_index):
                raise ValueError(f"stage {stage} job batch IDs disagree with scheduled order")
            by_machine.setdefault(batch.machine_id, []).append(batch)
        for family, indices in family_batch_indices.items():
            if sorted(indices) != list(range(len(indices))):
                raise ValueError(
                    f"stage {stage} family {family} has inconsistent family batch indices"
                )
        for machine_batches in by_machine.values():
            ordered = sorted(machine_batches, key=lambda batch: batch.start_time)
            for left, right in zip(ordered, ordered[1:]):
                if left.completion_time > right.start_time + tolerance:
                    raise ValueError(f"stage {stage} has overlapping batches on one machine")
    if np.any(schedule.start_stage1 + tolerance < instance.release_times):
        raise ValueError("a stage-1 job starts before its release")
    if np.any(schedule.start_stage2 + tolerance < schedule.completion_stage1):
        raise ValueError("a stage-2 job starts before stage-1 completion")
    if not np.isfinite(schedule.objective) or schedule.objective < 0:
        raise ValueError("objective must be finite and non-negative")
    if abs(schedule.objective - float(np.sum(schedule.completion_stage2))) > tolerance:
        raise ValueError("objective does not equal sum(C_j2)")
