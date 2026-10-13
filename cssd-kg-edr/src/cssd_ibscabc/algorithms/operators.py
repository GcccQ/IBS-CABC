from __future__ import annotations

from random import Random
from typing import Callable, Mapping, Sequence

from ..core.entities import ProblemInstance, Schedule
from ..representation.batch_structure import (
    BatchStructureSolution,
    batch_signature,
    stage_batches,
    with_stage_batches,
)
from .bottleneck import BottleneckState

Operator = Callable[
    [ProblemInstance, BatchStructureSolution, Schedule, BottleneckState, Random, int],
    BatchStructureSolution,
]


def _copy_batches(solution: BatchStructureSolution, stage: int) -> dict[int, list[list[int]]]:
    return {
        family: [list(group) for group in groups]
        for family, groups in stage_batches(solution, stage).items()
    }


def _size(instance: ProblemInstance, stage: int, job_id: int) -> int:
    job = instance.jobs_by_id[job_id]
    return job.size_stage1 if stage == 1 else job.size_stage2


def _load(instance: ProblemInstance, stage: int, group: Sequence[int]) -> int:
    return sum(_size(instance, stage, job_id) for job_id in group)


def _capacity(instance: ProblemInstance, stage: int) -> int:
    return instance.capacity_stage1 if stage == 1 else instance.capacity_stage2


def _stage(state: BottleneckState, rng: Random) -> int:
    # Use the pressure gap as confidence.  Near-ties explore both stages;
    # decisive diagnoses focus without completely freezing cross-stage moves.
    focus_probability = 0.50 + 0.45 * state.stage_confidence
    return state.stage if rng.random() < focus_probability else 3 - state.stage


def split_batch(instance, solution, schedule, state, rng, depth):
    stage = _stage(state, rng)
    batches = _copy_batches(solution, stage)
    decoded = {
        batch.signature: batch
        for batch in (schedule.batches_stage1 if stage == 1 else schedule.batches_stage2)
    }
    candidates = []
    for family, groups in batches.items():
        for index, group in enumerate(groups):
            if len(group) <= 1:
                continue
            jobs = [instance.jobs_by_id[job_id] for job_id in group]
            durations = [job.duration_stage1 if stage == 1 else job.duration_stage2 for job in jobs]
            batch = decoded[batch_signature(stage, family, group)]
            score = (batch.start - batch.release_time) + max(durations) - min(durations)
            candidates.append((score, family, index))
    if not candidates:
        return solution
    candidates.sort(reverse=True)
    pool = candidates[: max(1, (len(candidates) + 3) // 4)]
    _, family, index = rng.choice(pool)
    group = batches[family][index]
    # Load-balanced cut is less myopic than a random cardinality split.
    best_cut = min(
        range(1, len(group)),
        key=lambda cut: abs(
            _load(instance, stage, group[:cut]) - _load(instance, stage, group[cut:])
        ),
    )
    batches[family][index : index + 1] = [group[:best_cut], group[best_cut:]]
    return with_stage_batches(instance, solution, stage, batches)


def merge_batch(instance, solution, schedule, state, rng, depth):
    stage = _stage(state, rng)
    batches = _copy_batches(solution, stage)
    capacity = _capacity(instance, stage)
    candidates = []
    for family, groups in batches.items():
        loads = [_load(instance, stage, group) for group in groups]
        candidates.extend(
            (family, left, left + 1)
            for left in range(len(groups))
            if left + 1 < len(groups) and loads[left] + loads[left + 1] <= capacity
        )
    if not candidates:
        return solution
    family, left, right = rng.choice(candidates)
    batches[family][left].extend(batches[family][right])
    batches[family].pop(right)
    return with_stage_batches(instance, solution, stage, batches)


def move_job(instance, solution, schedule, state, rng, depth):
    stage = _stage(state, rng)
    batches = _copy_batches(solution, stage)
    capacity = _capacity(instance, stage)
    families = [family for family, groups in batches.items() if len(groups) >= 2]
    for _ in range(32):
        if not families:
            break
        family = rng.choice(families)
        groups = batches[family]
        target = rng.randrange(len(groups))
        slack = capacity - _load(instance, stage, groups[target])
        sources = [
            index
            for index, group in enumerate(groups)
            if index != target and any(_size(instance, stage, job_id) <= slack for job_id in group)
        ]
        if not sources:
            continue
        source = rng.choice(sources)
        movable = [job_id for job_id in groups[source] if _size(instance, stage, job_id) <= slack]
        job_id = rng.choice(movable)
        groups[source].remove(job_id)
        groups[target].append(job_id)
        if not groups[source]:
            groups.pop(source)
        return with_stage_batches(instance, solution, stage, batches)
    return solution


def ejection_chain(instance, solution, schedule, state, rng, depth):
    stage = _stage(state, rng)
    batches = _copy_batches(solution, stage)
    capacity = _capacity(instance, stage)
    families = [family for family, groups in batches.items() if len(groups) >= 2]
    if not families:
        return solution
    family = rng.choice(families)
    groups = batches[family]
    source = rng.randrange(len(groups))
    if not groups[source]:
        return solution
    carried = rng.choice(groups[source])
    current = source
    changed = False
    for _ in range(max(2, depth)):
        targets = []
        for index, target_group in enumerate(groups):
            if index == current:
                continue
            projected = _load(instance, stage, target_group) + _size(instance, stage, carried)
            if projected <= capacity:
                targets.append(index)
                continue
            overflow = projected - capacity
            if any(
                job_id != carried and _size(instance, stage, job_id) >= overflow
                for job_id in target_group
            ):
                targets.append(index)
        if not targets:
            break
        target = rng.choice(targets)
        groups[current].remove(carried)
        groups[target].append(carried)
        changed = True
        if _load(instance, stage, groups[target]) <= capacity:
            carried = -1
            break
        overflow = _load(instance, stage, groups[target]) - capacity
        ejectable = [
            item
            for item in groups[target]
            if item != carried and _size(instance, stage, item) >= overflow
        ]
        # Target filtering above guarantees this set is non-empty.
        carried = min(ejectable, key=lambda item: _size(instance, stage, item))
        current = target
    if carried != -1:
        groups[current].remove(carried)
        groups.append([carried])
    groups[:] = [group for group in groups if group]
    if not changed:
        return solution
    return with_stage_batches(instance, solution, stage, batches)


def _priority_move(instance, solution, state, rng, mode, critical_signature=None):
    # A critical-batch move must operate on the diagnosed stage.  The previous
    # hard-coded stage 2 silently contradicted a stage-1 diagnosis.
    stage = state.stage if mode == "critical" else _stage(state, rng)
    priority = list(
        solution.stage1_batch_priority if stage == 1 else solution.stage2_batch_priority
    )
    if len(priority) < 2:
        return solution
    if mode == "critical":
        try:
            old = priority.index(critical_signature or state.critical_signature)
        except ValueError:
            old = rng.randrange(1, len(priority))
        new = rng.randrange(0, old) if old else 0
        priority.insert(new, priority.pop(old))
    elif mode == "swap":
        left = rng.randrange(len(priority) - 1)
        priority[left], priority[left + 1] = priority[left + 1], priority[left]
    else:
        old = rng.randrange(max(1, len(priority) // 2), len(priority))
        new = rng.randrange(0, old)
        priority.insert(new, priority.pop(old))
    return with_stage_batches(instance, solution, stage, stage_batches(solution, stage), priority)


def critical_batch_insert(instance, solution, schedule, state, rng, depth):
    batches = schedule.batches_stage1 if state.stage == 1 else schedule.batches_stage2
    movable = [batch for batch in batches if batch.priority_rank > 0]
    if not movable:
        return solution
    critical = next(
        (batch for batch in movable if batch.signature == state.critical_signature),
        None,
    )
    if critical is None:
        critical = max(
            movable,
            key=lambda batch: (
                len(batch.job_ids) * max(1, batch.end - batch.release_time),
                batch.processing_time,
            ),
        )
    return _priority_move(
        instance,
        solution,
        state,
        rng,
        "critical",
        critical_signature=critical.signature,
    )


def batch_swap_tail_relocate(instance, solution, schedule, state, rng, depth):
    mode = "swap" if rng.random() < 0.5 else "tail"
    return _priority_move(instance, solution, state, rng, mode)


def partial_reconstruction(instance, solution, schedule, state, rng, depth):
    stage = _stage(state, rng)
    batches = _copy_batches(solution, stage)
    family = rng.choice(list(instance.active_families))
    jobs = [job_id for group in batches[family] for job_id in group]
    rule = rng.choice(("ERD", "SPT", "DOWNSTREAM", "RANDOM"))
    if rule != "RANDOM":

        def ready_time(job_id: int) -> int:
            return (
                instance.jobs_by_id[job_id].release
                if stage == 1
                else schedule.completion_stage1[job_id - 1]
            )

        def stage_duration(job_id: int) -> int:
            job = instance.jobs_by_id[job_id]
            return job.duration_stage1 if stage == 1 else job.duration_stage2

        def reconstruction_key(job_id: int) -> tuple[int, int, int]:
            if rule == "SPT":
                return (stage_duration(job_id), ready_time(job_id), job_id)
            if rule == "DOWNSTREAM":
                return (instance.jobs_by_id[job_id].duration_stage2, ready_time(job_id), job_id)
            return (ready_time(job_id), stage_duration(job_id), job_id)

        jobs.sort(key=reconstruction_key)
    else:
        rng.shuffle(jobs)
    capacity = _capacity(instance, stage)
    threshold = capacity * rng.choice((0.70, 0.85, 1.00))
    rebuilt: list[list[int]] = []
    current: list[int] = []
    load = 0
    for job_id in jobs:
        size = _size(instance, stage, job_id)
        if current and (load >= threshold or load + size > capacity):
            rebuilt.append(current)
            current, load = [], 0
        current.append(job_id)
        load += size
    if current:
        rebuilt.append(current)
    batches[family] = rebuilt
    return with_stage_batches(instance, solution, stage, batches)


OPERATORS: Mapping[str, Operator] = {
    "split": split_batch,
    "merge": merge_batch,
    "move": move_job,
    "ejection_chain": ejection_chain,
    "critical_batch_insert": critical_batch_insert,
    "batch_swap_tail_relocate": batch_swap_tail_relocate,
    "partial_reconstruction": partial_reconstruction,
}


def apply_operator(
    name: str,
    instance: ProblemInstance,
    solution: BatchStructureSolution,
    schedule: Schedule,
    state: BottleneckState,
    rng: Random,
    depth: int,
) -> BatchStructureSolution:
    return OPERATORS[name](instance, solution, schedule, state, rng, depth)
