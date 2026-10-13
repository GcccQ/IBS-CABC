from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

from ortools.sat.python import cp_model

from ..core.entities import Batch, ProblemInstance, Schedule
from ..core.feasibility import validate_schedule
from ..core.objective import calculate_metrics
from ..core.signatures import batch_signature
from ..decoder.integrated import decode_structure
from ..representation.batch_structure import (
    BatchStructureSolution,
    make_solution,
    with_stage_batches,
)
from .common import FoodSource


@dataclass(frozen=True, slots=True)
class AssignmentRepairResult:
    source: FoodSource
    model_calls: int
    feasible_models: int
    optimal_models: int
    accepted_iterations: int
    objective_gain: int
    runtime_seconds: float
    solver_runtime_seconds: float
    max_solver_call_runtime_seconds: float
    method_sequence: tuple[str, ...]


@dataclass(slots=True)
class _RepairCounters:
    model_calls: int = 0
    feasible_models: int = 0
    optimal_models: int = 0
    solver_runtime_seconds: float = 0.0
    max_solver_call_runtime_seconds: float = 0.0


def _new_solver(
    seconds: float,
    workers: int,
    seed: int,
) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.01, seconds)
    solver.parameters.num_search_workers = max(1, workers)
    solver.parameters.random_seed = max(0, seed % 2_147_483_647)
    solver.parameters.log_search_progress = False
    return solver


def _model_seconds(
    seconds_per_family: float,
    deadline: float | None,
    remaining_models: int,
) -> float:
    if deadline is None:
        return seconds_per_family
    remaining = max(0.0, deadline - time.perf_counter())
    return min(seconds_per_family, remaining / max(1, remaining_models))


def _deadline_expired(deadline: float | None) -> bool:
    return deadline is not None and time.perf_counter() >= deadline


def _stage2_groups(
    instance: ProblemInstance,
    schedule: Schedule,
    *,
    seconds_per_family: float,
    workers: int,
    seed: int,
    iteration: int,
    deadline: float | None,
    remaining_iterations: int,
    counters: _RepairCounters,
    allow_empty_batches: bool,
) -> list[dict[int, list[int]]]:
    groups = {batch.batch_id: list(batch.job_ids) for batch in schedule.batches_stage2}
    candidates: list[dict[int, list[int]]] = []
    families = instance.active_families
    for family_index, family in enumerate(families):
        if _deadline_expired(deadline):
            break
        remaining_models = (
            len(families) - family_index + max(0, remaining_iterations - 1) * len(families)
        )
        seconds = _model_seconds(
            seconds_per_family,
            deadline,
            remaining_models,
        )
        if seconds < 0.01:
            break
        jobs = [job for job in instance.jobs if job.family == family]
        batches = [batch for batch in schedule.batches_stage2 if batch.family == family]
        current = {(job_id, batch.batch_id) for batch in batches for job_id in batch.job_ids}
        model = cp_model.CpModel()
        variables: dict[tuple[int, int], cp_model.IntVar] = {}
        by_job: dict[int, list[cp_model.IntVar]] = defaultdict(list)
        by_batch: dict[int, list[cp_model.IntVar]] = defaultdict(list)
        for job in jobs:
            if _deadline_expired(deadline):
                return candidates
            for batch in batches:
                if schedule.completion_stage1[job.job_id - 1] > batch.start:
                    continue
                variable = model.new_bool_var(f"x_{job.job_id}_{batch.batch_id}")
                model.add_hint(
                    variable,
                    int((job.job_id, batch.batch_id) in current),
                )
                variables[job.job_id, batch.batch_id] = variable
                by_job[job.job_id].append(variable)
                by_batch[batch.batch_id].append(variable)
        if any(not by_job[job.job_id] for job in jobs):
            continue
        for job in jobs:
            if _deadline_expired(deadline):
                return candidates
            model.add_exactly_one(by_job[job.job_id])
        for batch in batches:
            if not allow_empty_batches:
                model.add(sum(by_batch[batch.batch_id]) >= 1)
            model.add(
                sum(
                    instance.jobs_by_id[job_id].size_stage2 * variable
                    for (job_id, batch_id), variable in variables.items()
                    if batch_id == batch.batch_id
                )
                <= instance.capacity_stage2
            )
        model.minimize(
            sum(
                batch.end * variables[job.job_id, batch.batch_id]
                for job in jobs
                for batch in batches
                if (job.job_id, batch.batch_id) in variables
            )
        )
        solve_seconds = _model_seconds(
            seconds_per_family,
            deadline,
            remaining_models,
        )
        if solve_seconds < 0.01:
            break
        solver = _new_solver(
            solve_seconds,
            workers,
            seed + 1009 * iteration + 97 * family,
        )
        counters.model_calls += 1
        solve_started = time.perf_counter()
        status = solver.solve(model)
        solve_runtime = time.perf_counter() - solve_started
        counters.solver_runtime_seconds += solve_runtime
        counters.max_solver_call_runtime_seconds = max(
            counters.max_solver_call_runtime_seconds,
            solve_runtime,
        )
        if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            continue
        counters.feasible_models += 1
        counters.optimal_models += int(status == cp_model.OPTIMAL)
        for batch in batches:
            groups[batch.batch_id] = []
        for (job_id, batch_id), variable in variables.items():
            if solver.value(variable):
                groups[batch_id].append(job_id)
        candidates.append({batch_id: list(job_ids) for batch_id, job_ids in groups.items()})
    return candidates


def _joint_groups(
    instance: ProblemInstance,
    schedule: Schedule,
    *,
    seconds_per_family: float,
    workers: int,
    seed: int,
    iteration: int,
    deadline: float | None,
    remaining_iterations: int,
    counters: _RepairCounters,
    allow_empty_batches: bool,
) -> list[tuple[dict[int, list[int]], dict[int, list[int]]]]:
    groups1 = {batch.batch_id: list(batch.job_ids) for batch in schedule.batches_stage1}
    groups2 = {batch.batch_id: list(batch.job_ids) for batch in schedule.batches_stage2}
    candidates: list[tuple[dict[int, list[int]], dict[int, list[int]]]] = []
    families = instance.active_families
    for family_index, family in enumerate(families):
        if _deadline_expired(deadline):
            break
        remaining_models = (
            len(families) - family_index + max(0, remaining_iterations - 1) * len(families)
        )
        seconds = _model_seconds(
            seconds_per_family,
            deadline,
            remaining_models,
        )
        if seconds < 0.01:
            break
        jobs = [job for job in instance.jobs if job.family == family]
        batches1 = [batch for batch in schedule.batches_stage1 if batch.family == family]
        batches2 = [batch for batch in schedule.batches_stage2 if batch.family == family]
        current1 = {(job_id, batch.batch_id) for batch in batches1 for job_id in batch.job_ids}
        current2 = {(job_id, batch.batch_id) for batch in batches2 for job_id in batch.job_ids}
        model = cp_model.CpModel()
        variables1: dict[tuple[int, int], cp_model.IntVar] = {}
        variables2: dict[tuple[int, int], cp_model.IntVar] = {}
        by_job1: dict[int, list[cp_model.IntVar]] = defaultdict(list)
        by_job2: dict[int, list[cp_model.IntVar]] = defaultdict(list)
        by_batch1: dict[int, list[cp_model.IntVar]] = defaultdict(list)
        by_batch2: dict[int, list[cp_model.IntVar]] = defaultdict(list)
        for job in jobs:
            if _deadline_expired(deadline):
                return candidates
            for batch in batches1:
                if job.release > batch.start:
                    continue
                variable = model.new_bool_var(f"x1_{job.job_id}_{batch.batch_id}")
                model.add_hint(
                    variable,
                    int((job.job_id, batch.batch_id) in current1),
                )
                variables1[job.job_id, batch.batch_id] = variable
                by_job1[job.job_id].append(variable)
                by_batch1[batch.batch_id].append(variable)
            for batch in batches2:
                variable = model.new_bool_var(f"x2_{job.job_id}_{batch.batch_id}")
                model.add_hint(
                    variable,
                    int((job.job_id, batch.batch_id) in current2),
                )
                variables2[job.job_id, batch.batch_id] = variable
                by_job2[job.job_id].append(variable)
                by_batch2[batch.batch_id].append(variable)
        if any(not by_job1[job.job_id] or not by_job2[job.job_id] for job in jobs):
            continue
        horizon = max(batch.end for batch in batches1)
        for job in jobs:
            if _deadline_expired(deadline):
                return candidates
            model.add_exactly_one(by_job1[job.job_id])
            model.add_exactly_one(by_job2[job.job_id])
            completion = model.new_int_var(0, horizon, f"c1_{job.job_id}")
            model.add(
                completion
                == sum(
                    batch.end * variables1[job.job_id, batch.batch_id]
                    for batch in batches1
                    if (job.job_id, batch.batch_id) in variables1
                )
            )
            for batch in batches2:
                variable = variables2.get((job.job_id, batch.batch_id))
                if variable is not None:
                    model.add(completion <= batch.start).only_enforce_if(variable)
        for batch in batches1:
            if not allow_empty_batches:
                model.add(sum(by_batch1[batch.batch_id]) >= 1)
            model.add(
                sum(
                    instance.jobs_by_id[job_id].size_stage1 * variable
                    for (job_id, batch_id), variable in variables1.items()
                    if batch_id == batch.batch_id
                )
                <= instance.capacity_stage1
            )
        for batch in batches2:
            if not allow_empty_batches:
                model.add(sum(by_batch2[batch.batch_id]) >= 1)
            model.add(
                sum(
                    instance.jobs_by_id[job_id].size_stage2 * variable
                    for (job_id, batch_id), variable in variables2.items()
                    if batch_id == batch.batch_id
                )
                <= instance.capacity_stage2
            )
        model.minimize(
            sum(
                batch.end * variables2[job.job_id, batch.batch_id]
                for job in jobs
                for batch in batches2
                if (job.job_id, batch.batch_id) in variables2
            )
        )
        solve_seconds = _model_seconds(
            seconds_per_family,
            deadline,
            remaining_models,
        )
        if solve_seconds < 0.01:
            break
        solver = _new_solver(
            solve_seconds,
            workers,
            seed + 1009 * iteration + 97 * family,
        )
        counters.model_calls += 1
        solve_started = time.perf_counter()
        status = solver.solve(model)
        solve_runtime = time.perf_counter() - solve_started
        counters.solver_runtime_seconds += solve_runtime
        counters.max_solver_call_runtime_seconds = max(
            counters.max_solver_call_runtime_seconds,
            solve_runtime,
        )
        if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            continue
        counters.feasible_models += 1
        counters.optimal_models += int(status == cp_model.OPTIMAL)
        for batch in batches1:
            groups1[batch.batch_id] = []
        for batch in batches2:
            groups2[batch.batch_id] = []
        for (job_id, batch_id), variable in variables1.items():
            if solver.value(variable):
                groups1[batch_id].append(job_id)
        for (job_id, batch_id), variable in variables2.items():
            if solver.value(variable):
                groups2[batch_id].append(job_id)
        candidates.append(
            (
                {batch_id: list(job_ids) for batch_id, job_ids in groups1.items()},
                {batch_id: list(job_ids) for batch_id, job_ids in groups2.items()},
            )
        )
    return candidates


def _rebuild_stage2(
    instance: ProblemInstance,
    solution: BatchStructureSolution,
    schedule: Schedule,
    groups: dict[int, list[int]],
) -> BatchStructureSolution:
    batches: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    priorities: list[tuple[int, str]] = []
    for batch in sorted(schedule.batches_stage2, key=lambda item: item.batch_id):
        group = tuple(sorted(groups[batch.batch_id]))
        if not group:
            continue
        batches[batch.family].append(group)
        priorities.append(
            (
                batch.priority_rank,
                batch_signature(2, batch.family, group),
            )
        )
    return with_stage_batches(
        instance,
        solution,
        2,
        batches,
        [signature for _, signature in sorted(priorities)],
    )


def _rebuild_joint(
    instance: ProblemInstance,
    schedule: Schedule,
    groups1: dict[int, list[int]],
    groups2: dict[int, list[int]],
) -> BatchStructureSolution:
    batches1: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    batches2: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    priorities1: list[tuple[int, str]] = []
    priorities2: list[tuple[int, str]] = []
    for batch in sorted(schedule.batches_stage1, key=lambda item: item.batch_id):
        group = tuple(sorted(groups1[batch.batch_id]))
        if not group:
            continue
        batches1[batch.family].append(group)
        priorities1.append(
            (
                batch.priority_rank,
                batch_signature(1, batch.family, group),
            )
        )
    for batch in sorted(schedule.batches_stage2, key=lambda item: item.batch_id):
        group = tuple(sorted(groups2[batch.batch_id]))
        if not group:
            continue
        batches2[batch.family].append(group)
        priorities2.append(
            (
                batch.priority_rank,
                batch_signature(2, batch.family, group),
            )
        )
    return make_solution(
        instance,
        batches1,
        batches2,
        [signature for _, signature in sorted(priorities1)],
        [signature for _, signature in sorted(priorities2)],
    )


def _explicit_schedule(
    instance: ProblemInstance,
    solution: BatchStructureSolution,
    baseline: Schedule,
    *,
    groups1: dict[int, list[int]] | None,
    groups2: dict[int, list[int]],
) -> Schedule:
    stage1_records = []
    for batch in baseline.batches_stage1:
        group = tuple(sorted(batch.job_ids if groups1 is None else groups1[batch.batch_id]))
        if group:
            stage1_records.append((batch, group))
    priority1 = {
        original.batch_id: rank
        for rank, (original, _) in enumerate(
            sorted(stage1_records, key=lambda item: item[0].priority_rank)
        )
    }
    times1: dict[int, tuple[int, int]] = {}
    machine_end1: dict[int, int] = defaultdict(int)
    for original, group in sorted(
        stage1_records,
        key=lambda item: (
            item[0].machine_id,
            item[0].start,
            item[0].batch_id,
        ),
    ):
        jobs = [instance.jobs_by_id[job_id] for job_id in group]
        processing = max(job.duration_stage1 for job in jobs)
        release = max(job.release for job in jobs)
        start = max(
            release,
            machine_end1[original.machine_id],
        )
        times1[original.batch_id] = (start, start + processing)
        machine_end1[original.machine_id] = start + processing
    batches1 = []
    for batch_id, (original, group) in enumerate(stage1_records):
        jobs = [instance.jobs_by_id[job_id] for job_id in group]
        processing = max(job.duration_stage1 for job in jobs)
        release = max(job.release for job in jobs)
        start, end = times1[original.batch_id]
        batches1.append(
            Batch(
                batch_id,
                1,
                original.family,
                group,
                sum(job.size_stage1 for job in jobs),
                processing,
                release,
                priority1[original.batch_id],
                original.machine_id,
                start,
                end,
                batch_signature(1, original.family, group),
            )
        )
    completion1 = [0] * instance.job_count
    for batch in batches1:
        for job_id in batch.job_ids:
            completion1[job_id - 1] = batch.end

    stage2_records = []
    for batch in baseline.batches_stage2:
        group = tuple(sorted(groups2[batch.batch_id]))
        if group:
            stage2_records.append((batch, group))
    priority2 = {
        original.batch_id: rank
        for rank, (original, _) in enumerate(
            sorted(stage2_records, key=lambda item: item[0].priority_rank)
        )
    }
    times2: dict[int, tuple[int, int]] = {}
    machine_end2: dict[int, int] = defaultdict(int)
    for original, group in sorted(
        stage2_records,
        key=lambda item: (
            item[0].machine_id,
            item[0].start,
            item[0].batch_id,
        ),
    ):
        jobs = [instance.jobs_by_id[job_id] for job_id in group]
        processing = max(job.duration_stage2 for job in jobs)
        release = max(completion1[job.job_id - 1] for job in jobs)
        start = max(
            release,
            machine_end2[original.machine_id],
        )
        times2[original.batch_id] = (start, start + processing)
        machine_end2[original.machine_id] = start + processing
    batches2 = []
    for batch_id, (original, group) in enumerate(stage2_records):
        jobs = [instance.jobs_by_id[job_id] for job_id in group]
        processing = max(job.duration_stage2 for job in jobs)
        release = max(completion1[job.job_id - 1] for job in jobs)
        start, end = times2[original.batch_id]
        batches2.append(
            Batch(
                batch_id,
                2,
                original.family,
                group,
                sum(job.size_stage2 for job in jobs),
                processing,
                release,
                priority2[original.batch_id],
                original.machine_id,
                start,
                end,
                batch_signature(2, original.family, group),
            )
        )
    completion2 = [0] * instance.job_count
    for batch in batches2:
        for job_id in batch.job_ids:
            completion2[job_id - 1] = batch.end
    metrics = calculate_metrics(
        instance,
        batches1,
        batches2,
        completion1,
        completion2,
    )
    schedule = Schedule(
        instance.name,
        baseline.algorithm,
        tuple(batches1),
        tuple(batches2),
        tuple(completion1),
        tuple(completion2),
        metrics,
        {
            "structure_sha256": solution.canonical_signature,
            "assignment_repair_explicit": True,
        },
    )
    validate_schedule(instance, schedule)
    return schedule


def family_decomposed_assignment_repair(
    instance: ProblemInstance,
    source: FoodSource,
    *,
    seconds_per_family: float,
    workers: int,
    iterations: int,
    joint_max_jobs: int,
    seed: int,
    allow_empty_batches: bool = True,
    max_total_seconds: float | None = None,
) -> AssignmentRepairResult:
    """Reassign jobs inside incumbent batch slots, then decode the full structure.

    The assignment model is separated by family.  On small/medium instances,
    the first iteration jointly reassigns both stages; subsequent iterations
    reassign stage 2 using the newly decoded stage-1 completion times.  Large
    instances use the lower-dimensional stage-2 model throughout.
    """

    started = time.perf_counter()
    deadline = None if max_total_seconds is None else started + max(0.0, max_total_seconds)
    counters = _RepairCounters()
    initial_objective = source.objective
    current = source
    accepted = 0
    methods: list[str] = []
    for iteration in range(max(0, iterations)):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        remaining_iterations = iterations - iteration
        joint = instance.job_count <= joint_max_jobs and iteration == 0
        if joint:
            methods.append("joint_two_stage")
            group_candidates = _joint_groups(
                instance,
                current.schedule,
                seconds_per_family=seconds_per_family,
                workers=workers,
                seed=seed,
                iteration=iteration,
                deadline=deadline,
                remaining_iterations=remaining_iterations,
                counters=counters,
                allow_empty_batches=allow_empty_batches,
            )
            candidate_pairs = [
                (
                    _rebuild_joint(
                        instance,
                        current.schedule,
                        groups1,
                        groups2,
                    ),
                    groups1,
                    groups2,
                )
                for groups1, groups2 in group_candidates
                if not _deadline_expired(deadline)
            ]
            candidate_sources = [
                FoodSource(
                    candidate_solution,
                    _explicit_schedule(
                        instance,
                        candidate_solution,
                        current.schedule,
                        groups1=groups1,
                        groups2=groups2,
                    ),
                    0,
                    "family_decomposed_assignment_repair",
                )
                for candidate_solution, groups1, groups2 in candidate_pairs
                if not _deadline_expired(deadline)
            ]
            candidate_sources.extend(
                FoodSource(
                    candidate_solution,
                    decode_structure(
                        instance,
                        candidate_solution,
                        algorithm=current.schedule.algorithm,
                    ),
                    0,
                    "family_decomposed_assignment_repair",
                )
                for candidate_solution, _, _ in candidate_pairs
                if not _deadline_expired(deadline)
            )
        else:
            methods.append("stage2")
            group_candidates = _stage2_groups(
                instance,
                current.schedule,
                seconds_per_family=seconds_per_family,
                workers=workers,
                seed=seed,
                iteration=iteration,
                deadline=deadline,
                remaining_iterations=remaining_iterations,
                counters=counters,
                allow_empty_batches=allow_empty_batches,
            )
            candidate_pairs = [
                (
                    _rebuild_stage2(
                        instance,
                        current.solution,
                        current.schedule,
                        groups,
                    ),
                    groups,
                )
                for groups in group_candidates
                if not _deadline_expired(deadline)
            ]
            candidate_sources = [
                FoodSource(
                    candidate_solution,
                    _explicit_schedule(
                        instance,
                        candidate_solution,
                        current.schedule,
                        groups1=None,
                        groups2=groups,
                    ),
                    0,
                    "family_decomposed_assignment_repair",
                )
                for candidate_solution, groups in candidate_pairs
                if not _deadline_expired(deadline)
            ]
            candidate_sources.extend(
                FoodSource(
                    candidate_solution,
                    decode_structure(
                        instance,
                        candidate_solution,
                        algorithm=current.schedule.algorithm,
                    ),
                    0,
                    "family_decomposed_assignment_repair",
                )
                for candidate_solution, _ in candidate_pairs
                if not _deadline_expired(deadline)
            )
        iteration_best = current
        best_by_signature: dict[str, FoodSource] = {}
        for candidate_source in candidate_sources:
            signature = candidate_source.solution.canonical_signature
            if signature == current.solution.canonical_signature:
                continue
            present = best_by_signature.get(signature)
            if present is None or candidate_source.objective < present.objective:
                best_by_signature[signature] = candidate_source
        for candidate_source in best_by_signature.values():
            if candidate_source.objective < iteration_best.objective:
                iteration_best = candidate_source
        if iteration_best.objective < current.objective:
            current = iteration_best
            accepted += 1
    return AssignmentRepairResult(
        current,
        counters.model_calls,
        counters.feasible_models,
        counters.optimal_models,
        accepted,
        initial_objective - current.objective,
        time.perf_counter() - started,
        counters.solver_runtime_seconds,
        counters.max_solver_call_runtime_seconds,
        tuple(methods),
    )
