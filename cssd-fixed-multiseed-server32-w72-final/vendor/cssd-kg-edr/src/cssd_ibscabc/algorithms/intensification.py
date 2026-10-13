from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

from ..core.entities import ProblemInstance
from ..representation.batch_structure import (
    BatchStructureSolution,
    batch_signature,
    stage_batches,
    with_stage_batches,
)
from .common import EvaluationBudget, FoodSource


@dataclass(frozen=True, slots=True)
class CriticalDescentResult:
    source: FoodSource
    evaluations: int
    passes: int
    accepted_moves: int
    stage1_evaluations: int
    stage2_evaluations: int
    objective_gain: int


def _priority_insertion_candidates(
    instance: ProblemInstance,
    source: FoodSource,
    critical_batch_count: int,
) -> Iterator[tuple[int, BatchStructureSolution]]:
    """Enumerate promising earlier insertions of high-contribution batches.

    A stage-2 batch contributes ``job_count * completion_time`` exactly to the
    primary objective. Stage 2 is therefore searched before stage 1.  Every
    candidate preserves both batch partitions and changes one priority list
    only, so feasibility is retained by construction.
    """

    for stage, decoded_batches in (
        (2, source.schedule.batches_stage2),
        (1, source.schedule.batches_stage1),
    ):
        priority = (
            source.solution.stage1_batch_priority
            if stage == 1
            else source.solution.stage2_batch_priority
        )
        rank = {signature: index for index, signature in enumerate(priority)}
        critical = sorted(
            decoded_batches,
            key=lambda batch: (
                len(batch.job_ids) * batch.end,
                len(batch.job_ids),
                batch.processing_time,
                batch.signature,
            ),
            reverse=True,
        )[:critical_batch_count]
        for batch in critical:
            old_position = rank[batch.signature]
            # Only earlier insertion is considered. Moving a late, high-
            # contribution batch later cannot directly reduce its completion
            # contribution and greatly expands the neighborhood on J1000.
            for target_position in range(old_position):
                candidate_priority = list(priority)
                signature = candidate_priority.pop(old_position)
                candidate_priority.insert(target_position, signature)
                yield (
                    stage,
                    with_stage_batches(
                        instance,
                        source.solution,
                        stage,
                        stage_batches(source.solution, stage),
                        candidate_priority,
                    ),
                )


def _analytical_priority_candidates(
    instance: ProblemInstance,
    source: FoodSource,
    critical_batch_count: int,
) -> Iterator[tuple[int, BatchStructureSolution]]:
    """Enumerate objective-aligned complete priority repairs.

    A completed stage-2 batch contributes ``|batch| * C_batch`` to the primary
    objective.  WSPT-style ranks explicitly expose that multiplicity, while
    release-aware variants remain useful when only a subset is ready.  These
    full-list repairs cost only a few evaluations and precede the much larger
    insertion neighborhood.
    """

    del critical_batch_count
    for stage, decoded_batches in (
        (2, source.schedule.batches_stage2),
        (1, source.schedule.batches_stage1),
    ):
        rules = (
            lambda batch: (
                batch.processing_time / len(batch.job_ids),
                batch.release_time,
                batch.processing_time,
                batch.family,
                batch.signature,
            ),
            lambda batch: (
                batch.release_time,
                batch.processing_time / len(batch.job_ids),
                batch.family,
                batch.signature,
            ),
            lambda batch: (
                batch.release_time + batch.processing_time / len(batch.job_ids),
                batch.processing_time / len(batch.job_ids),
                batch.family,
                batch.signature,
            ),
            lambda batch: (
                -len(batch.job_ids),
                batch.processing_time,
                batch.release_time,
                batch.family,
                batch.signature,
            ),
            lambda batch: (
                batch.processing_time,
                batch.release_time,
                -len(batch.job_ids),
                batch.family,
                batch.signature,
            ),
        )
        for rule in rules:
            priority = tuple(
                batch.signature for batch in sorted(decoded_batches, key=rule)
            )
            yield (
                stage,
                with_stage_batches(
                    instance,
                    source.solution,
                    stage,
                    stage_batches(source.solution, stage),
                    priority,
                ),
            )


def _size(instance: ProblemInstance, stage: int, job_id: int) -> int:
    job = instance.jobs_by_id[job_id]
    return job.size_stage1 if stage == 1 else job.size_stage2


def _critical_formation_candidates(
    instance: ProblemInstance,
    source: FoodSource,
    critical_batch_count: int,
) -> Iterator[tuple[int, BatchStructureSolution]]:
    """Enumerate split, merge and adjacent relocation moves around critical batches."""

    for stage, decoded_batches in (
        (2, source.schedule.batches_stage2),
        (1, source.schedule.batches_stage1),
    ):
        base = stage_batches(source.solution, stage)
        capacity = instance.capacity_stage1 if stage == 1 else instance.capacity_stage2
        locations = {
            batch_signature(stage, family, group): (family, index)
            for family, groups in base.items()
            for index, group in enumerate(groups)
        }
        critical = sorted(
            decoded_batches,
            key=lambda batch: (
                len(batch.job_ids) * batch.end,
                len(batch.job_ids),
                batch.processing_time,
                batch.signature,
            ),
            reverse=True,
        )[:critical_batch_count]
        for batch in critical:
            family, index = locations[batch.signature]
            groups = [list(group) for group in base[family]]
            critical_group = groups[index]

            for cut in range(1, len(critical_group)):
                candidate_batches = {
                    key: [list(group) for group in value]
                    for key, value in base.items()
                }
                candidate_batches[family][index : index + 1] = [
                    critical_group[:cut],
                    critical_group[cut:],
                ]
                yield (
                    stage,
                    with_stage_batches(
                        instance,
                        source.solution,
                        stage,
                        candidate_batches,
                    ),
                )

            for neighbor in (index - 1, index + 1):
                if not 0 <= neighbor < len(groups):
                    continue
                merged = critical_group + groups[neighbor]
                if sum(_size(instance, stage, job_id) for job_id in merged) <= capacity:
                    candidate_batches = {
                        key: [list(group) for group in value]
                        for key, value in base.items()
                    }
                    left, right = sorted((index, neighbor))
                    combined = (
                        candidate_batches[family][left]
                        + candidate_batches[family][right]
                    )
                    candidate_batches[family][left : right + 1] = [combined]
                    yield (
                        stage,
                        with_stage_batches(
                            instance,
                            source.solution,
                            stage,
                            candidate_batches,
                        ),
                    )

            # Keep relocation local to the two adjacent batches.  The wider
            # all-pairs version consumed the fixed intensification budget on
            # low-yield candidates and regressed the paired J500/J1000 test.
            for other in (index - 1, index + 1):
                if not 0 <= other < len(groups):
                    continue
                for source_index, target_index in ((index, other), (other, index)):
                    target_load = sum(
                        _size(instance, stage, job_id)
                        for job_id in groups[target_index]
                    )
                    slack = capacity - target_load
                    for job_id in groups[source_index]:
                        if _size(instance, stage, job_id) > slack:
                            continue
                        candidate_batches = {
                            key: [list(group) for group in value]
                            for key, value in base.items()
                        }
                        candidate_batches[family][source_index].remove(job_id)
                        candidate_batches[family][target_index].append(job_id)
                        if not candidate_batches[family][source_index]:
                            candidate_batches[family].pop(source_index)
                        yield (
                            stage,
                            with_stage_batches(
                                instance,
                                source.solution,
                                stage,
                                candidate_batches,
                            ),
                        )


def _pack_window(
    instance: ProblemInstance,
    stage: int,
    order: list[int],
    close_fraction: float,
    method: str,
) -> list[list[int]]:
    capacity = instance.capacity_stage1 if stage == 1 else instance.capacity_stage2
    close_load = close_fraction * capacity
    if method == "next":
        result: list[list[int]] = []
        current: list[int] = []
        load = 0
        for job_id in order:
            size = _size(instance, stage, job_id)
            if current and (load >= close_load or load + size > capacity):
                result.append(current)
                current = []
                load = 0
            current.append(job_id)
            load += size
        if current:
            result.append(current)
        return result

    batches: list[list[int]] = []
    loads: list[int] = []
    for job_id in order:
        size = _size(instance, stage, job_id)
        feasible = [
            (capacity - loads[index] - size, index)
            for index in range(len(batches))
            if loads[index] < close_load and loads[index] + size <= capacity
        ]
        if feasible:
            _, index = min(feasible)
            batches[index].append(job_id)
            loads[index] += size
        else:
            batches.append([job_id])
            loads.append(size)
    return batches


def _critical_window_reconstruction_candidates(
    instance: ProblemInstance,
    source: FoodSource,
    critical_batch_count: int,
) -> Iterator[tuple[int, BatchStructureSolution]]:
    """Destroy and rebuild a small same-family window around critical batches."""

    for stage, decoded_batches in (
        (2, source.schedule.batches_stage2),
        (1, source.schedule.batches_stage1),
    ):
        base = stage_batches(source.solution, stage)
        locations = {
            batch_signature(stage, family, group): (family, index)
            for family, groups in base.items()
            for index, group in enumerate(groups)
        }
        critical = sorted(
            decoded_batches,
            key=lambda batch: (
                len(batch.job_ids) * batch.end,
                len(batch.job_ids),
                batch.end,
                batch.signature,
            ),
            reverse=True,
        )[:critical_batch_count]
        for batch in critical:
            family, center = locations[batch.signature]
            groups = base[family]
            radii = (1, 2, 3, 4) if instance.job_count >= 800 else (1, 2)
            windows: list[tuple[int, int]] = []
            for radius in radii:
                window = (
                    max(0, center - radius),
                    min(len(groups), center + radius + 1),
                )
                if window not in windows:
                    windows.append(window)
            for left, right in windows:
                if right - left < 2:
                    continue
                pool = [job_id for group in groups[left:right] for job_id in group]
                jobs = instance.jobs_by_id
                ready = (
                    (lambda job_id: jobs[job_id].release)
                    if stage == 1
                    else (
                        lambda job_id: source.schedule.completion_stage1[job_id - 1]
                    )
                )
                orderings = (
                    sorted(
                        pool,
                        key=lambda job_id: (
                            ready(job_id),
                            _size(instance, stage, job_id),
                            job_id,
                        ),
                    ),
                    sorted(
                        pool,
                        key=lambda job_id: (
                            _size(instance, stage, job_id),
                            ready(job_id),
                            job_id,
                        ),
                    ),
                    sorted(
                        pool,
                        key=lambda job_id: (
                            -_size(instance, stage, job_id),
                            ready(job_id),
                            job_id,
                        ),
                    ),
                )
                for order in orderings:
                    for close_fraction in (0.70, 0.85, 1.00):
                        for method in ("next", "best"):
                            rebuilt = _pack_window(
                                instance,
                                stage,
                                order,
                                close_fraction,
                                method,
                            )
                            candidate_batches = {
                                key: [list(group) for group in value]
                                for key, value in base.items()
                            }
                            candidate_batches[family][left:right] = rebuilt
                            yield (
                                stage,
                                with_stage_batches(
                                    instance,
                                    source.solution,
                                    stage,
                                    candidate_batches,
                                ),
                            )


def _batch_elimination_candidates(
    instance: ProblemInstance,
    source: FoodSource,
    critical_batch_count: int,
) -> Iterator[tuple[int, BatchStructureSolution]]:
    """Try to remove under-filled cycles by redistributing all contained jobs."""

    for stage, decoded_batches in (
        (2, source.schedule.batches_stage2),
        (1, source.schedule.batches_stage1),
    ):
        base = stage_batches(source.solution, stage)
        capacity = instance.capacity_stage1 if stage == 1 else instance.capacity_stage2
        locations = {
            batch_signature(stage, family, group): (family, index)
            for family, groups in base.items()
            for index, group in enumerate(groups)
        }
        candidates = sorted(
            decoded_batches,
            key=lambda batch: (
                batch.load / capacity,
                -batch.end,
                len(batch.job_ids),
                batch.signature,
            ),
        )[: max(critical_batch_count * 2, critical_batch_count)]
        for batch in candidates:
            family, remove_index = locations[batch.signature]
            groups = [list(group) for group in base[family]]
            if len(groups) <= 1:
                continue
            for descending in (True, False):
                moving = sorted(
                    groups[remove_index],
                    key=lambda job_id: (
                        _size(instance, stage, job_id),
                        job_id,
                    ),
                    reverse=descending,
                )
                for tightest in (True, False):
                    rebuilt = [list(group) for group in groups]
                    rebuilt.pop(remove_index)
                    loads = [
                        sum(_size(instance, stage, job_id) for job_id in group)
                        for group in rebuilt
                    ]
                    feasible = True
                    for job_id in moving:
                        size = _size(instance, stage, job_id)
                        destinations = [
                            (capacity - load - size, index)
                            for index, load in enumerate(loads)
                            if load + size <= capacity
                        ]
                        if not destinations:
                            feasible = False
                            break
                        remaining, target = (
                            min(destinations) if tightest else max(destinations)
                        )
                        rebuilt[target].append(job_id)
                        loads[target] = capacity - remaining
                    if not feasible:
                        continue
                    candidate_batches = {
                        key: [list(group) for group in value]
                        for key, value in base.items()
                    }
                    candidate_batches[family] = rebuilt
                    yield (
                        stage,
                        with_stage_batches(
                            instance,
                            source.solution,
                            stage,
                            candidate_batches,
                        ),
                    )


def _cross_stage_alignment_candidates(
    instance: ProblemInstance,
    source: FoodSource,
    critical_batch_count: int,
) -> Iterator[tuple[int, BatchStructureSolution]]:
    """Use critical stage-2 batches to repair their upstream synchronization."""

    stage1 = stage_batches(source.solution, 1)
    priority = source.solution.stage1_batch_priority
    rank = {signature: index for index, signature in enumerate(priority)}
    job_signature = {
        job_id: batch_signature(1, family, group)
        for family, groups in stage1.items()
        for group in groups
        for job_id in group
    }
    critical = sorted(
        source.schedule.batches_stage2,
        key=lambda batch: (
            len(batch.job_ids)
            * (
                batch.end
                + max(source.schedule.completion_stage1[job_id - 1] for job_id in batch.job_ids)
                - min(source.schedule.completion_stage1[job_id - 1] for job_id in batch.job_ids)
            ),
            len(batch.job_ids),
            batch.end,
            batch.signature,
        ),
        reverse=True,
    )[:critical_batch_count]
    for downstream in critical:
        affected = sorted(
            {job_signature[job_id] for job_id in downstream.job_ids},
            key=rank.__getitem__,
        )
        affected_set = set(affected)
        retained = [signature for signature in priority if signature not in affected_set]
        earliest = min(rank[signature] for signature in affected)
        targets = sorted(
            {
                0,
                max(0, earliest // 3),
                max(0, earliest // 2),
                max(0, earliest - len(affected)),
            }
        )
        for target in targets:
            insertion = min(target, len(retained))
            candidate_priority = tuple(
                retained[:insertion] + affected + retained[insertion:]
            )
            yield (
                1,
                with_stage_batches(
                    instance,
                    source.solution,
                    1,
                    stage1,
                    candidate_priority,
                ),
            )

        family = downstream.family
        groups = stage1[family]
        target_jobs = set(downstream.job_ids)
        affected_indices = [
            index
            for index, group in enumerate(groups)
            if target_jobs.intersection(group)
        ]
        if not affected_indices:
            continue
        pool = [
            job_id
            for index in affected_indices
            for job_id in groups[index]
        ]
        if len(pool) > max(250, instance.job_count // 2):
            continue
        first_index = min(affected_indices)
        remaining_groups = [
            list(group)
            for index, group in enumerate(groups)
            if index not in set(affected_indices)
        ]
        orders = (
            sorted(
                pool,
                key=lambda job_id: (
                    job_id not in target_jobs,
                    instance.jobs_by_id[job_id].release,
                    _size(instance, 1, job_id),
                    job_id,
                ),
            ),
            sorted(
                pool,
                key=lambda job_id: (
                    instance.jobs_by_id[job_id].release,
                    job_id not in target_jobs,
                    _size(instance, 1, job_id),
                    job_id,
                ),
            ),
        )
        for order in orders:
            for close_fraction in (0.75, 0.90, 1.00):
                for method in ("next", "best"):
                    rebuilt = _pack_window(
                        instance,
                        1,
                        order,
                        close_fraction,
                        method,
                    )
                    candidate_batches = {
                        key: [list(group) for group in value]
                        for key, value in stage1.items()
                    }
                    insertion = min(first_index, len(remaining_groups))
                    candidate_batches[family] = (
                        remaining_groups[:insertion]
                        + rebuilt
                        + remaining_groups[insertion:]
                    )
                    yield (
                        1,
                        with_stage_batches(
                            instance,
                            source.solution,
                            1,
                            candidate_batches,
                        ),
                    )


CandidateFactory = Callable[
    [ProblemInstance, FoodSource, int],
    Iterator[tuple[int, BatchStructureSolution]],
]


def _best_improvement_descent(
    instance: ProblemInstance,
    source: FoodSource,
    budget: EvaluationBudget,
    max_evaluations: int,
    candidate_factory: CandidateFactory,
    operator_label: str,
    *,
    critical_batch_count: int = 5,
) -> CriticalDescentResult:
    if max_evaluations <= 0 or critical_batch_count <= 0:
        return CriticalDescentResult(source, 0, 0, 0, 0, 0, 0)

    initial_objective = source.objective
    current = source
    evaluations = 0
    passes = 0
    accepted_moves = 0
    stage_evaluations = {1: 0, 2: 0}

    while evaluations < max_evaluations and not budget.exhausted:
        passes += 1
        pass_best = current
        seen: set[str] = set()
        for stage, candidate in candidate_factory(
            instance,
            current,
            critical_batch_count,
        ):
            if evaluations >= max_evaluations or budget.exhausted:
                break
            signature = candidate.canonical_signature
            if signature in seen:
                continue
            seen.add(signature)
            schedule = budget.evaluate(candidate)
            if schedule is None:
                break
            evaluations += 1
            stage_evaluations[stage] += 1
            if schedule.objective < pass_best.objective:
                pass_best = FoodSource(
                    candidate,
                    schedule,
                    0,
                    f"{operator_label}_stage{stage}",
                )
        if pass_best.objective >= current.objective:
            break
        current = pass_best
        accepted_moves += 1

    return CriticalDescentResult(
        current,
        evaluations,
        passes,
        accepted_moves,
        stage_evaluations[1],
        stage_evaluations[2],
        initial_objective - current.objective,
    )


def critical_batch_descent(
    instance: ProblemInstance,
    source: FoodSource,
    budget: EvaluationBudget,
    max_evaluations: int,
    *,
    critical_batch_count: int = 5,
) -> CriticalDescentResult:
    """Optimize critical-batch priority by budget-charged best improvement."""

    return _best_improvement_descent(
        instance,
        source,
        budget,
        max_evaluations,
        _priority_insertion_candidates,
        "critical_priority_descent",
        critical_batch_count=critical_batch_count,
    )


def analytical_priority_descent(
    instance: ProblemInstance,
    source: FoodSource,
    budget: EvaluationBudget,
    max_evaluations: int,
) -> CriticalDescentResult:
    """Try a small set of full-list priority rules before local insertion."""

    return _best_improvement_descent(
        instance,
        source,
        budget,
        max_evaluations,
        _analytical_priority_candidates,
        "analytical_priority_descent",
        critical_batch_count=1,
    )


def critical_formation_descent(
    instance: ProblemInstance,
    source: FoodSource,
    budget: EvaluationBudget,
    max_evaluations: int,
    *,
    critical_batch_count: int = 5,
) -> CriticalDescentResult:
    """Optimize critical-batch partitions by split/merge/relocation descent."""

    return _best_improvement_descent(
        instance,
        source,
        budget,
        max_evaluations,
        _critical_formation_candidates,
        "critical_formation_descent",
        critical_batch_count=critical_batch_count,
    )


def critical_window_reconstruction_descent(
    instance: ProblemInstance,
    source: FoodSource,
    budget: EvaluationBudget,
    max_evaluations: int,
    *,
    critical_batch_count: int = 5,
) -> CriticalDescentResult:
    """Apply destroy/reconstruct to same-family windows around critical batches."""

    return _best_improvement_descent(
        instance,
        source,
        budget,
        max_evaluations,
        _critical_window_reconstruction_candidates,
        "critical_window_reconstruction",
        critical_batch_count=critical_batch_count,
    )


def batch_elimination_descent(
    instance: ProblemInstance,
    source: FoodSource,
    budget: EvaluationBudget,
    max_evaluations: int,
    *,
    critical_batch_count: int = 5,
) -> CriticalDescentResult:
    """Eliminate avoidable processing cycles before window reconstruction."""

    return _best_improvement_descent(
        instance,
        source,
        budget,
        max_evaluations,
        _batch_elimination_candidates,
        "batch_elimination",
        critical_batch_count=critical_batch_count,
    )


def cross_stage_alignment_descent(
    instance: ProblemInstance,
    source: FoodSource,
    budget: EvaluationBudget,
    max_evaluations: int,
    *,
    critical_batch_count: int = 8,
) -> CriticalDescentResult:
    """Repair stage-1 structure according to critical stage-2 synchronization."""

    return _best_improvement_descent(
        instance,
        source,
        budget,
        max_evaluations,
        _cross_stage_alignment_candidates,
        "cross_stage_alignment",
        critical_batch_count=critical_batch_count,
    )

