from __future__ import annotations

from math import inf
from random import Random
from typing import Callable

from ..core.entities import Job, ProblemInstance
from ..decoder.construction import batch_priority, permutation_to_structure, stage1_completion
from ..representation.batch_structure import (
    BatchStructureSolution,
    make_solution,
    stage_batches,
)


def _rule_order(instance: ProblemInstance, rule: str, rng: Random) -> tuple[int, ...]:
    rules: dict[str, Callable[[Job], tuple[object, ...]]] = {
        "ERD": lambda job: (job.release, job.duration_stage1, job.job_id),
        "SPT": lambda job: (job.duration_stage1, job.release, job.job_id),
        "LPT": lambda job: (-job.duration_stage1, job.release, job.job_id),
        "SIZE": lambda job: (-(job.size_stage1 + job.size_stage2), job.release, job.job_id),
        "DOWNSTREAM": lambda job: (job.duration_stage2, job.release, job.job_id),
    }
    if rule == "RANDOM":
        jobs = list(instance.jobs)
        rng.shuffle(jobs)
        return tuple(job.job_id for job in jobs)
    return tuple(job.job_id for job in sorted(instance.jobs, key=rules[rule]))


def _stage_size(instance: ProblemInstance, stage: int, job_id: int) -> int:
    job = instance.jobs_by_id[job_id]
    return job.size_stage1 if stage == 1 else job.size_stage2


def _restricted_best_fit(
    instance: ProblemInstance,
    stage: int,
    order: tuple[int, ...],
    close_fraction: float,
) -> list[tuple[int, ...]]:
    """Pack an ordered family while protecting already well-filled batches.

    The CSSD instances have discrete cassette sizes and family-constant batch
    durations.  Filling an early batch with more jobs therefore lowers more
    completion-time terms at once.  Once a bin reaches ``close_fraction`` it is
    closed, preventing later large jobs from destroying that cardinality bias.
    """

    capacity = instance.capacity_stage1 if stage == 1 else instance.capacity_stage2
    close_load = close_fraction * capacity
    batches: list[list[int]] = []
    loads: list[int] = []
    for job_id in order:
        size = _stage_size(instance, stage, job_id)
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
    return [tuple(batch) for batch in batches]


def _next_fit(
    instance: ProblemInstance,
    stage: int,
    order: tuple[int, ...],
) -> list[tuple[int, ...]]:
    capacity = instance.capacity_stage1 if stage == 1 else instance.capacity_stage2
    batches: list[tuple[int, ...]] = []
    current: list[int] = []
    load = 0
    for job_id in order:
        size = _stage_size(instance, stage, job_id)
        if current and load + size > capacity:
            batches.append(tuple(current))
            current = []
            load = 0
        current.append(job_id)
        load += size
    if current:
        batches.append(tuple(current))
    return batches


def cardinality_aware_seeds(
    instance: ProblemInstance,
) -> list[BatchStructureSolution]:
    """Build deterministic two-stage seeds aligned with the primary objective."""

    seeds: list[BatchStructureSolution] = []
    seen: set[str] = set()
    for stage1_rule in ("ERD", "SIZE_ASC"):
        for close_fraction in (0.70, 0.85, 1.00):
            stage1: dict[int, list[tuple[int, ...]]] = {}
            for family in instance.active_families:
                jobs = [instance.jobs_by_id[job_id] for job_id in instance.family_job_ids[family]]
                if stage1_rule == "ERD":
                    jobs.sort(
                        key=lambda job: (
                            job.release,
                            job.size_stage1,
                            job.job_id,
                        )
                    )
                else:
                    jobs.sort(
                        key=lambda job: (
                            job.size_stage1,
                            job.release,
                            job.job_id,
                        )
                    )
                stage1[family] = _restricted_best_fit(
                    instance,
                    1,
                    tuple(job.job_id for job in jobs),
                    close_fraction,
                )
            priority1 = batch_priority(instance, 1, stage1, "WSPT")
            completion1 = stage1_completion(instance, stage1, priority1)
            stage2 = {
                family: _next_fit(
                    instance,
                    2,
                    tuple(
                        sorted(
                            instance.family_job_ids[family],
                            key=lambda job_id: (
                                completion1[job_id - 1],
                                instance.jobs_by_id[job_id].size_stage2,
                                job_id,
                            ),
                        )
                    ),
                )
                for family in instance.active_families
            }
            seed = make_solution(
                instance,
                stage1,
                stage2,
                priority1,
                batch_priority(instance, 2, stage2, "WSPT", completion1),
            )
            if seed.canonical_signature not in seen:
                seeds.append(seed)
                seen.add(seed.canonical_signature)
    return seeds


def _release_congestion_partition(
    instance: ProblemInstance,
    order: tuple[int, ...],
    upstream_completion: tuple[int, ...],
    release_weight: float,
) -> list[tuple[int, ...]]:
    """Optimize consecutive stage-2 cuts under release/congestion pressure.

    A fixed fill threshold cannot decide whether waiting for one more job is
    worth delaying every job already in a batch. The dynamic program balances
    that release exposure against the processing time imposed on the jobs that
    remain downstream:

    ``alpha * |B| * r(B) + p(B) * remaining_jobs / m2``.

    This is a construction surrogate only. Every returned partition is still
    decoded and compared under the unchanged full two-stage objective.
    """

    if release_weight < 0:
        raise ValueError("release_weight must be non-negative")
    job_count = len(order)
    if job_count == 0:
        return []
    capacity = instance.capacity_stage2
    machine_count = instance.machines_stage2
    costs = [inf] * (job_count + 1)
    predecessors = [-1] * (job_count + 1)
    costs[0] = 0.0
    for right in range(1, job_count + 1):
        load = 0
        processing = 0
        release = 0
        for left in range(right - 1, -1, -1):
            job_id = order[left]
            job = instance.jobs_by_id[job_id]
            load += job.size_stage2
            if load > capacity:
                break
            processing = max(processing, job.duration_stage2)
            release = max(release, upstream_completion[job_id - 1])
            group_size = right - left
            candidate = (
                costs[left]
                + release_weight * group_size * release
                + processing * (job_count - left) / machine_count
            )
            if (
                candidate < costs[right] - 1e-12
                or (
                    abs(candidate - costs[right]) <= 1e-12
                    and (predecessors[right] < 0 or left < predecessors[right])
                )
            ):
                costs[right] = candidate
                predecessors[right] = left
    groups: list[tuple[int, ...]] = []
    right = job_count
    while right:
        left = predecessors[right]
        if left < 0:
            raise RuntimeError("stage-2 dynamic partition has no feasible predecessor")
        groups.append(order[left:right])
        right = left
    groups.reverse()
    return groups


def adaptive_stage2_seeds(
    instance: ProblemInstance,
    base: BatchStructureSolution,
) -> list[BatchStructureSolution]:
    """Generate load-adaptive stage-2 descendants of one complete solution."""

    stage1 = {
        family: list(groups)
        for family, groups in stage_batches(base, 1).items()
    }
    completion1 = stage1_completion(
        instance,
        stage1,
        base.stage1_batch_priority,
    )
    # The grid represents release/congestion regimes. It is evaluated as a
    # portfolio and never selected from an instance name or published answer.
    release_weights = (0.0, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0)
    candidates: list[BatchStructureSolution] = []
    seen: set[str] = set()
    for mode in ("ready", "ready_duration"):
        for release_weight in release_weights:
            stage2: dict[int, list[tuple[int, ...]]] = {}
            for family in instance.active_families:
                order = list(instance.family_job_ids[family])
                if mode == "ready":
                    order.sort(
                        key=lambda job_id: (
                            completion1[job_id - 1],
                            job_id,
                        )
                    )
                else:
                    order.sort(
                        key=lambda job_id: (
                            completion1[job_id - 1],
                            instance.jobs_by_id[job_id].duration_stage2,
                            instance.jobs_by_id[job_id].size_stage2,
                            job_id,
                        )
                    )
                stage2[family] = _release_congestion_partition(
                    instance,
                    tuple(order),
                    completion1,
                    release_weight,
                )
            candidate = make_solution(
                instance,
                stage1,
                stage2,
                base.stage1_batch_priority,
                batch_priority(instance, 2, stage2, "WSPT", completion1),
            )
            if (
                candidate.canonical_signature != base.canonical_signature
                and candidate.canonical_signature not in seen
            ):
                candidates.append(candidate)
                seen.add(candidate.canonical_signature)
    return candidates


def seed_portfolio(
    instance: ProblemInstance,
    count: int,
    delta_pairs: tuple[tuple[float, float], ...],
    rng: Random,
) -> list[BatchStructureSolution]:
    rules = ("ERD", "SPT", "LPT", "SIZE", "DOWNSTREAM", "RANDOM")
    seeds: list[BatchStructureSolution] = []
    seen: set[str] = set()
    for seed in cardinality_aware_seeds(instance):
        if len(seeds) >= count:
            return seeds
        seeds.append(seed)
        seen.add(seed.canonical_signature)
    attempts = 0
    while len(seeds) < count and attempts < count * 20:
        rule = rules[attempts % len(rules)]
        delta1, delta2 = delta_pairs[(attempts // len(rules)) % len(delta_pairs)]
        order = _rule_order(instance, rule, rng)
        seed = permutation_to_structure(instance, order, delta1, delta2, "SPT")
        if seed.canonical_signature not in seen:
            seeds.append(seed)
            seen.add(seed.canonical_signature)
        attempts += 1
    while len(seeds) < count:
        order = _rule_order(instance, "RANDOM", rng)
        delta1, delta2 = rng.choice(delta_pairs)
        seed = permutation_to_structure(instance, order, delta1, delta2, "SPT")
        if seed.canonical_signature not in seen:
            seeds.append(seed)
            seen.add(seed.canonical_signature)
    return seeds


def random_seed(
    instance: ProblemInstance,
    delta_pairs: tuple[tuple[float, float], ...],
    rng: Random,
) -> BatchStructureSolution:
    """Build one independent complete batch-structure scout solution."""

    order = _rule_order(instance, "RANDOM", rng)
    delta1, delta2 = rng.choice(delta_pairs)
    return permutation_to_structure(instance, order, delta1, delta2, "SPT")
