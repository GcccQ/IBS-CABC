from __future__ import annotations

from collections import defaultdict
from typing import Literal, Sequence

from ..core.entities import ProblemInstance
from ..representation.batch_structure import (
    BatchStructureSolution,
    batch_signature,
    make_solution,
)
from .dispatch import BatchPlan, dispatch_parallel

DispatchRule = Literal["FCFS", "SPT", "LPT", "WSPT"]


def _threshold(delta: float, capacity: int) -> float:
    if delta <= 0:
        raise ValueError("delta must be positive")
    return delta * capacity if delta <= 1.0 else delta


def _consume_from_tail(
    instance: ProblemInstance,
    order: Sequence[int],
    stage: int,
    delta: float,
) -> list[tuple[int, ...]]:
    """Convert one family order into feasible batches for seed construction."""

    capacity = instance.capacity_stage1 if stage == 1 else instance.capacity_stage2
    threshold = min(float(capacity), _threshold(delta, capacity))
    pending = list(order)
    result: list[tuple[int, ...]] = []
    while pending:
        batch: list[int] = []
        load = 0
        while pending and load < threshold:
            job_id = pending[-1]
            job = instance.jobs_by_id[job_id]
            size = job.size_stage1 if stage == 1 else job.size_stage2
            if load + size > capacity:
                break
            batch.append(pending.pop())
            load += size
        result.append(tuple(batch))
    return result


def batch_priority(
    instance: ProblemInstance,
    stage: int,
    batches: dict[int, list[tuple[int, ...]]],
    rule: DispatchRule,
    upstream_completion: tuple[int, ...] | None = None,
) -> list[str]:
    if stage == 2 and upstream_completion is None:
        raise ValueError("stage-2 priority requires stage-1 completion times")
    records: list[tuple[object, ...]] = []
    for family, groups in batches.items():
        for ordinal, group in enumerate(groups):
            jobs = [instance.jobs_by_id[job_id] for job_id in group]
            duration = max(
                job.duration_stage1 if stage == 1 else job.duration_stage2 for job in jobs
            )
            release = max(
                job.release if stage == 1 else upstream_completion[job.job_id - 1]  # type: ignore[index]
                for job in jobs
            )
            if rule == "SPT":
                rule_key = duration
            elif rule == "WSPT":
                rule_key = duration / len(group)
            elif rule == "LPT":
                rule_key = -duration
            else:
                rule_key = release
            records.append((rule_key, family, ordinal, batch_signature(stage, family, group)))
    return [record[-1] for record in sorted(records)]


def stage1_completion(
    instance: ProblemInstance,
    batches: dict[int, list[tuple[int, ...]]],
    priority: Sequence[str],
) -> tuple[int, ...]:
    ranks = {signature: rank for rank, signature in enumerate(priority)}
    plans = []
    for family, groups in batches.items():
        for group in groups:
            jobs = [instance.jobs_by_id[job_id] for job_id in group]
            signature = batch_signature(1, family, group)
            plans.append(
                BatchPlan(
                    1,
                    family,
                    tuple(group),
                    sum(job.size_stage1 for job in jobs),
                    max(job.duration_stage1 for job in jobs),
                    max(job.release for job in jobs),
                    ranks[signature],
                    signature,
                )
            )
    scheduled = dispatch_parallel(plans, instance.machines_stage1)
    completion = [0] * instance.job_count
    for batch in scheduled:
        for job_id in batch.job_ids:
            completion[job_id - 1] = batch.end
    return tuple(completion)


def permutation_to_structure(
    instance: ProblemInstance,
    permutation: Sequence[int],
    delta1: float = 1.0,
    delta2: float = 1.0,
    dispatch_rule: DispatchRule = "SPT",
) -> BatchStructureSolution:
    """Create a complete direct two-stage structure from an ordered seed."""

    expected = {job.job_id for job in instance.jobs}
    if len(permutation) != instance.job_count or set(permutation) != expected:
        raise ValueError("permutation must contain every job exactly once")
    by_family: dict[int, list[int]] = defaultdict(list)
    for job_id in permutation:
        by_family[instance.jobs_by_id[job_id].family].append(job_id)
    stage1 = {
        family: _consume_from_tail(instance, order, 1, delta1)
        for family, order in by_family.items()
    }
    priority1 = batch_priority(instance, 1, stage1, dispatch_rule)
    completion1 = stage1_completion(instance, stage1, priority1)
    stage2_order = {
        family: sorted(
            order,
            key=lambda job_id: (completion1[job_id - 1], job_id),
        )
        for family, order in by_family.items()
    }
    stage2 = {
        family: _consume_from_tail(instance, order, 2, delta2)
        for family, order in stage2_order.items()
    }
    return make_solution(
        instance,
        stage1,
        stage2,
        priority1,
        batch_priority(instance, 2, stage2, dispatch_rule, completion1),
    )
