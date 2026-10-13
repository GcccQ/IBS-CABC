from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Callable, Sequence

from . import bridge as _bridge  # noqa: F401
from cssd_ibscabc.core.entities import ProblemInstance, Schedule
from cssd_ibscabc.representation.batch_structure import (
    BatchStructureSolution,
    stage_batches,
    with_stage_batches,
)


@dataclass(frozen=True, slots=True)
class Neighbor:
    solution: BatchStructureSolution
    move_key: str
    operator: str


def _size(instance: ProblemInstance, stage: int, job_id: int) -> int:
    job = instance.jobs_by_id[job_id]
    return job.size_stage1 if stage == 1 else job.size_stage2


def _duration(instance: ProblemInstance, stage: int, job_id: int) -> int:
    job = instance.jobs_by_id[job_id]
    return job.duration_stage1 if stage == 1 else job.duration_stage2


def _capacity(instance: ProblemInstance, stage: int) -> int:
    return instance.capacity_stage1 if stage == 1 else instance.capacity_stage2


def _load(instance: ProblemInstance, stage: int, group: Sequence[int]) -> int:
    return sum(_size(instance, stage, job_id) for job_id in group)


def _mutable_batches(
    solution: BatchStructureSolution, stage: int
) -> dict[int, list[list[int]]]:
    return {
        family: [list(group) for group in groups]
        for family, groups in stage_batches(solution, stage).items()
    }


def _move(
    instance: ProblemInstance,
    solution: BatchStructureSolution,
    stage: int,
    batches: dict[int, list[list[int]]],
    operator: str,
    family: int | str,
    priority: Sequence[str] | None = None,
) -> Neighbor | None:
    try:
        candidate = with_stage_batches(instance, solution, stage, batches, priority)
    except ValueError:
        return None
    if candidate.canonical_signature == solution.canonical_signature:
        return None
    return Neighbor(candidate, f"{operator}:s{stage}:f{family}", operator)


def _split(instance, solution, schedule, rng) -> Neighbor | None:
    stage = rng.choice((1, 2))
    batches = _mutable_batches(solution, stage)
    candidates = [
        (family, index)
        for family, groups in batches.items()
        for index, group in enumerate(groups)
        if len(group) >= 2
    ]
    if not candidates:
        return None
    family, index = rng.choice(candidates)
    group = list(batches[family][index])
    rng.shuffle(group)
    best_cut = min(
        range(1, len(group)),
        key=lambda cut: abs(
            _load(instance, stage, group[:cut]) - _load(instance, stage, group[cut:])
        ),
    )
    batches[family][index : index + 1] = [group[:best_cut], group[best_cut:]]
    return _move(instance, solution, stage, batches, "split", family)


def _merge(instance, solution, schedule, rng) -> Neighbor | None:
    stage = rng.choice((1, 2))
    batches = _mutable_batches(solution, stage)
    capacity = _capacity(instance, stage)
    candidates = []
    for family, groups in batches.items():
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                if _load(instance, stage, groups[left]) + _load(
                    instance, stage, groups[right]
                ) <= capacity:
                    candidates.append((family, left, right))
    if not candidates:
        return None
    family, left, right = rng.choice(candidates)
    batches[family][left].extend(batches[family][right])
    batches[family].pop(right)
    return _move(instance, solution, stage, batches, "merge", family)


def _relocate(instance, solution, schedule, rng) -> Neighbor | None:
    stage = rng.choice((1, 2))
    batches = _mutable_batches(solution, stage)
    capacity = _capacity(instance, stage)
    choices = []
    for family, groups in batches.items():
        if len(groups) < 2:
            continue
        loads = [_load(instance, stage, group) for group in groups]
        for source, group in enumerate(groups):
            for job_id in group:
                for target in range(len(groups)):
                    if source != target and loads[target] + _size(
                        instance, stage, job_id
                    ) <= capacity:
                        choices.append((family, source, target, job_id))
    if not choices:
        return None
    family, source, target, job_id = rng.choice(choices)
    batches[family][source].remove(job_id)
    batches[family][target].append(job_id)
    if not batches[family][source]:
        batches[family].pop(source)
    return _move(instance, solution, stage, batches, "relocate", family)


def _swap_jobs(instance, solution, schedule, rng) -> Neighbor | None:
    stage = rng.choice((1, 2))
    batches = _mutable_batches(solution, stage)
    capacity = _capacity(instance, stage)
    choices = []
    for family, groups in batches.items():
        loads = [_load(instance, stage, group) for group in groups]
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                for left_job in groups[left]:
                    for right_job in groups[right]:
                        left_load = (
                            loads[left]
                            - _size(instance, stage, left_job)
                            + _size(instance, stage, right_job)
                        )
                        right_load = (
                            loads[right]
                            - _size(instance, stage, right_job)
                            + _size(instance, stage, left_job)
                        )
                        if left_load <= capacity and right_load <= capacity:
                            choices.append((family, left, right, left_job, right_job))
    if not choices:
        return None
    family, left, right, left_job, right_job = rng.choice(choices)
    li = batches[family][left].index(left_job)
    ri = batches[family][right].index(right_job)
    batches[family][left][li] = right_job
    batches[family][right][ri] = left_job
    return _move(instance, solution, stage, batches, "swap_jobs", family)


def _ordered_jobs(
    instance: ProblemInstance,
    schedule: Schedule,
    stage: int,
    jobs: list[int],
    mode: str,
    rng: Random,
) -> list[int]:
    if mode == "random":
        rng.shuffle(jobs)
        return jobs

    def ready(job_id: int) -> int:
        return (
            instance.jobs_by_id[job_id].release
            if stage == 1
            else schedule.completion_stage1[job_id - 1]
        )

    reverse = mode == "lpt"
    if mode in {"spt", "lpt"}:
        jobs.sort(
            key=lambda job_id: (_duration(instance, stage, job_id), ready(job_id), job_id),
            reverse=reverse,
        )
    elif mode == "size":
        jobs.sort(key=lambda job_id: (-_size(instance, stage, job_id), ready(job_id), job_id))
    else:
        jobs.sort(key=lambda job_id: (ready(job_id), _duration(instance, stage, job_id), job_id))
    return jobs


def _pack(
    instance: ProblemInstance,
    stage: int,
    jobs: Sequence[int],
    strategy: str,
    close_fraction: float,
) -> list[list[int]]:
    capacity = _capacity(instance, stage)
    groups: list[list[int]] = []
    loads: list[int] = []
    for job_id in jobs:
        size = _size(instance, stage, job_id)
        feasible = [
            (capacity - loads[index] - size, index)
            for index in range(len(groups))
            if loads[index] < close_fraction * capacity and loads[index] + size <= capacity
        ]
        if strategy == "best_fit" and feasible:
            _, index = min(feasible)
            groups[index].append(job_id)
            loads[index] += size
        elif strategy == "next_fit" and groups and loads[-1] + size <= capacity:
            groups[-1].append(job_id)
            loads[-1] += size
        else:
            groups.append([job_id])
            loads.append(size)
    return groups


def _rebuild_family_mode(
    instance,
    solution,
    schedule,
    rng,
    mode: str,
    operator: str,
) -> Neighbor | None:
    stage = rng.choice((1, 2))
    batches = _mutable_batches(solution, stage)
    family = rng.choice(list(instance.active_families))
    jobs = [job_id for group in batches[family] for job_id in group]
    jobs = _ordered_jobs(instance, schedule, stage, jobs, mode, rng)
    batches[family] = _pack(
        instance,
        stage,
        jobs,
        rng.choice(("best_fit", "next_fit")),
        rng.choice((0.70, 0.85, 1.00)),
    )
    return _move(instance, solution, stage, batches, operator, family)


def _rebuild_family(instance, solution, schedule, rng) -> Neighbor | None:
    return _rebuild_family_mode(
        instance,
        solution,
        schedule,
        rng,
        rng.choice(("random", "ready", "spt", "lpt", "size")),
        "rebuild_family",
    )


def _priority_insert(instance, solution, schedule, rng) -> Neighbor | None:
    stage = rng.choice((1, 2))
    priority = list(
        solution.stage1_batch_priority if stage == 1 else solution.stage2_batch_priority
    )
    if len(priority) < 2:
        return None
    old = rng.randrange(len(priority))
    new = rng.randrange(len(priority) - 1)
    if new >= old:
        new += 1
    priority.insert(new, priority.pop(old))
    batches = _mutable_batches(solution, stage)
    return _move(instance, solution, stage, batches, "priority_insert", "all", priority)


def _priority_swap(instance, solution, schedule, rng) -> Neighbor | None:
    stage = rng.choice((1, 2))
    priority = list(
        solution.stage1_batch_priority if stage == 1 else solution.stage2_batch_priority
    )
    if len(priority) < 2:
        return None
    left, right = rng.sample(range(len(priority)), 2)
    priority[left], priority[right] = priority[right], priority[left]
    batches = _mutable_batches(solution, stage)
    return _move(instance, solution, stage, batches, "priority_swap", "all", priority)


SMALL_OPERATORS: tuple[Callable[..., Neighbor | None], ...] = (
    _split,
    _merge,
    _relocate,
    _swap_jobs,
    _rebuild_family,
    _priority_insert,
    _priority_swap,
)


def small_neighbor(
    instance: ProblemInstance,
    solution: BatchStructureSolution,
    schedule: Schedule,
    rng: Random,
) -> Neighbor | None:
    operators = list(SMALL_OPERATORS)
    rng.shuffle(operators)
    for operator in operators:
        candidate = operator(instance, solution, schedule, rng)
        if candidate is not None:
            return candidate
    return None


def partner_guided_neighbor(
    instance: ProblemInstance,
    solution: BatchStructureSolution,
    partner: BatchStructureSolution,
    schedule: Schedule,
    rng: Random,
) -> Neighbor | None:
    for _ in range(12):
        stage = rng.choice((1, 2))
        family = rng.choice(list(instance.active_families))
        current = _mutable_batches(solution, stage)
        partner_groups = stage_batches(partner, stage)[family]
        job_id = rng.choice(list(instance.family_job_ids[family]))
        partner_group = next(group for group in partner_groups if job_id in group)
        mates = [item for item in partner_group if item != job_id]
        if not mates:
            continue
        mate = rng.choice(mates)
        source = next(i for i, group in enumerate(current[family]) if job_id in group)
        target = next(i for i, group in enumerate(current[family]) if mate in group)
        if source == target:
            continue
        if _load(instance, stage, current[family][target]) + _size(
            instance, stage, job_id
        ) > _capacity(instance, stage):
            continue
        current[family][source].remove(job_id)
        current[family][target].append(job_id)
        if not current[family][source]:
            current[family].pop(source)
        candidate = _move(instance, solution, stage, current, "partner_relocate", family)
        if candidate is not None:
            return candidate
    return small_neighbor(instance, solution, schedule, rng)


def _window_rebuild(instance, solution, schedule, rng) -> Neighbor | None:
    stage = rng.choice((1, 2))
    batches = _mutable_batches(solution, stage)
    families = [family for family, groups in batches.items() if len(groups) >= 2]
    if not families:
        return _rebuild_family(instance, solution, schedule, rng)
    family = rng.choice(families)
    groups = batches[family]
    width = rng.randint(2, min(5, len(groups)))
    start = rng.randrange(0, len(groups) - width + 1)
    jobs = [job_id for group in groups[start : start + width] for job_id in group]
    jobs = _ordered_jobs(
        instance,
        schedule,
        stage,
        jobs,
        rng.choice(("random", "ready", "spt", "lpt", "size")),
        rng,
    )
    rebuilt = _pack(
        instance,
        stage,
        jobs,
        rng.choice(("best_fit", "next_fit")),
        rng.choice((0.70, 0.85, 1.00)),
    )
    groups[start : start + width] = rebuilt
    return _move(instance, solution, stage, batches, "window_rebuild", family)


def _priority_segment(instance, solution, schedule, rng) -> Neighbor | None:
    stage = rng.choice((1, 2))
    priority = list(
        solution.stage1_batch_priority if stage == 1 else solution.stage2_batch_priority
    )
    if len(priority) < 3:
        return _priority_insert(instance, solution, schedule, rng)
    width = rng.randint(1, max(1, min(4, len(priority) // 3)))
    start = rng.randrange(0, len(priority) - width + 1)
    segment = priority[start : start + width]
    del priority[start : start + width]
    target = rng.randrange(len(priority) + 1)
    priority[target:target] = segment
    batches = _mutable_batches(solution, stage)
    return _move(instance, solution, stage, batches, "priority_segment", "all", priority)


LARGE_OPERATOR_NAMES = (
    "random_family_rebuild",
    "release_family_rebuild",
    "duration_family_rebuild",
    "window_rebuild",
    "priority_segment_relocate",
)


def large_neighbor(
    name: str,
    instance: ProblemInstance,
    solution: BatchStructureSolution,
    schedule: Schedule,
    rng: Random,
) -> Neighbor | None:
    if name == "random_family_rebuild":
        return _rebuild_family_mode(
            instance, solution, schedule, rng, "random", "random_family_rebuild"
        )
    if name == "release_family_rebuild":
        return _rebuild_family_mode(
            instance, solution, schedule, rng, "ready", "release_family_rebuild"
        )
    if name == "duration_family_rebuild":
        return _rebuild_family_mode(
            instance,
            solution,
            schedule,
            rng,
            rng.choice(("spt", "lpt", "size")),
            "duration_family_rebuild",
        )
    if name == "window_rebuild":
        return _window_rebuild(instance, solution, schedule, rng)
    if name == "priority_segment_relocate":
        return _priority_segment(instance, solution, schedule, rng)
    raise KeyError(f"unknown large-neighborhood operator: {name}")
