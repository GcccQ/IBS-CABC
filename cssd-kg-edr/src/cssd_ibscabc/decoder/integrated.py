from __future__ import annotations

from typing import Mapping

from ..core.entities import ProblemInstance, Schedule
from ..core.feasibility import validate_schedule
from ..core.objective import calculate_metrics
from ..representation.batch_structure import (
    BatchStructureSolution,
    batch_signature,
    stage_batches,
    validate_structure,
)
from .dispatch import BatchPlan, dispatch_parallel


def _plans(
    instance: ProblemInstance,
    solution: BatchStructureSolution,
    stage: int,
    upstream_completion: tuple[int, ...] | None = None,
) -> tuple[BatchPlan, ...]:
    groups = stage_batches(solution, stage)
    priority = solution.stage1_batch_priority if stage == 1 else solution.stage2_batch_priority
    ranks = {signature: rank for rank, signature in enumerate(priority)}
    plans: list[BatchPlan] = []
    for family in sorted(groups):
        for group in groups[family]:
            jobs = [instance.jobs_by_id[job_id] for job_id in group]
            signature = batch_signature(stage, family, group)
            if stage == 1:
                sizes = [job.size_stage1 for job in jobs]
                durations = [job.duration_stage1 for job in jobs]
                releases = [job.release for job in jobs]
            else:
                if upstream_completion is None:
                    raise ValueError("stage 2 requires upstream completion times")
                sizes = [job.size_stage2 for job in jobs]
                durations = [job.duration_stage2 for job in jobs]
                releases = [upstream_completion[job.job_id - 1] for job in jobs]
            plans.append(
                BatchPlan(
                    stage,
                    family,
                    tuple(group),
                    sum(sizes),
                    max(durations),
                    max(releases),
                    ranks[signature],
                    signature,
                )
            )
    return tuple(plans)


def _completion(job_count: int, batches: tuple) -> tuple[int, ...]:
    values = [0] * job_count
    for batch in batches:
        for job_id in batch.job_ids:
            values[job_id - 1] = batch.end
    return tuple(values)


def _weighted_completion(batches: tuple) -> int:
    return sum(len(batch.job_ids) * batch.end for batch in batches)


def decode_structure(
    instance: ProblemInstance,
    solution: BatchStructureSolution,
    *,
    algorithm: str = "IBS-CABC",
    validate: bool = True,
    strategic_idle_best_response: bool = False,
    metadata: Mapping[str, object] | None = None,
) -> Schedule:
    """Decode one complete direct batch structure across both coupled stages."""

    validate_structure(instance, solution)
    stage1 = dispatch_parallel(_plans(instance, solution, 1), instance.machines_stage1)
    completion1 = _completion(instance.job_count, stage1)
    stage2_plans = _plans(instance, solution, 2, completion1)
    stage2_non_delay = dispatch_parallel(stage2_plans, instance.machines_stage2)
    stage2 = stage2_non_delay
    strategic_idle_selected = False
    if strategic_idle_best_response:
        stage2_idle = dispatch_parallel(
            stage2_plans,
            instance.machines_stage2,
            strategic_idle=True,
        )
        if _weighted_completion(stage2_idle) < _weighted_completion(stage2_non_delay):
            stage2 = stage2_idle
            strategic_idle_selected = True
    completion2 = _completion(instance.job_count, stage2)
    schedule = Schedule(
        instance.name,
        algorithm,
        stage1,
        stage2,
        completion1,
        completion2,
        calculate_metrics(instance, stage1, stage2, completion1, completion2),
        dict(metadata or {})
        | {
            "structure_sha256": solution.canonical_signature,
            "stage2_dispatch_best_response": (
                "strategic_idle" if strategic_idle_selected else "non_delay"
            ),
        },
    )
    if validate:
        validate_schedule(instance, schedule)
    return schedule
