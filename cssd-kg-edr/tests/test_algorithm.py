from dataclasses import replace
from random import Random

from cssd_ibscabc.algorithms import solve_ibscabc
from cssd_ibscabc.algorithms.assignment_repair import (
    _explicit_schedule,
    family_decomposed_assignment_repair,
)
from cssd_ibscabc.algorithms.common import EvaluationBudget, FoodSource, SearchConfig
from cssd_ibscabc.algorithms.ibscabc import IBSCABC
from cssd_ibscabc.algorithms.seeding import (
    adaptive_stage2_seeds,
    cardinality_aware_seeds,
    seed_portfolio,
)
from cssd_ibscabc.core.feasibility import validate_schedule
from cssd_ibscabc.core.entities import Schedule
from cssd_ibscabc.core.objective import calculate_metrics
from cssd_ibscabc.decoder import decode_structure, permutation_to_structure
from cssd_ibscabc.io import load_instance
from cssd_ibscabc.representation import stage_batches, validate_structure
from cssd_ibscabc.representation.batch_structure import batch_signature


def test_search_respects_evaluation_budget_and_seed(small_instance):
    config = SearchConfig(colony_size=4, archive_size=8, max_evaluations=32, seed=77)
    first = solve_ibscabc(small_instance, config)
    second = solve_ibscabc(small_instance, config)
    assert first.evaluations == second.evaluations == 32
    assert first.history == second.history
    assert first.schedule.objective == second.schedule.objective
    assert first.schedule.algorithm == "IBS-CABC"
    assert all(value > 0 for value in first.diagnostics["role_calls"].values())
    assert (
        first.diagnostics["decode_calls"] + first.diagnostics["decode_cache_hits"]
        == first.evaluations
    )
    assert first.diagnostics["archive_descriptor_cache"] <= first.diagnostics["archive_size"]
    assert first.diagnostics["archive_distance_cache"] <= (
        first.diagnostics["archive_size"] * (first.diagnostics["archive_size"] - 1) // 2
    )
    assert 0 <= first.diagnostics["final_formation_role_diversity"] <= 1
    assert 0 <= first.diagnostics["final_sequencing_role_diversity"] <= 1
    assert first.diagnostics["final_unique_sources"] > 0


def test_single_joint_colony_supports_tiny_budgets(small_instance):
    result = solve_ibscabc(
        small_instance,
        SearchConfig(colony_size=4, archive_size=4, max_evaluations=2, seed=77),
    )
    assert result.evaluations == 2
    assert result.schedule.objective > 0


def test_adaptive_stage2_seeds_are_deterministic_and_feasible(small_instance):
    base = seed_portfolio(
        small_instance,
        4,
        SearchConfig().delta_seed_pairs,
        Random(11),
    )[0]
    first = adaptive_stage2_seeds(small_instance, base)
    second = adaptive_stage2_seeds(small_instance, base)
    assert first
    assert [item.canonical_signature for item in first] == [
        item.canonical_signature for item in second
    ]
    assert len({item.canonical_signature for item in first}) == len(first)
    for candidate in first:
        schedule = decode_structure(small_instance, candidate)
        validate_schedule(small_instance, schedule)


def test_oversized_assignment_model_is_disabled_before_construction(data_root):
    instance = load_instance(sorted((data_root / "Instances100-1000" / "J_1000").glob("*.txt"))[0])
    solver = IBSCABC(
        instance,
        SearchConfig(max_evaluations=1, seed=3),
    )
    assert solver.assignment_repair_applicable is False


def test_assignment_repair_honors_expired_total_deadline(small_instance):
    solution = cardinality_aware_seeds(small_instance)[0]
    schedule = decode_structure(small_instance, solution)
    result = family_decomposed_assignment_repair(
        small_instance,
        FoodSource(solution, schedule),
        seconds_per_family=10.0,
        workers=1,
        iterations=3,
        joint_max_jobs=600,
        seed=5,
        max_total_seconds=0.001,
    )
    assert result.model_calls == 0
    assert result.runtime_seconds < 0.1


def test_evaluation_cache_preserves_budget_count(small_instance):
    solution = permutation_to_structure(
        small_instance,
        tuple(job.job_id for job in small_instance.jobs),
        0.70,
        0.70,
    )
    budget = EvaluationBudget(small_instance, 2, None, "IBS-CABC")
    first = budget.evaluate(solution)
    second = budget.evaluate(solution)
    assert first is second
    assert budget.evaluations == 2
    assert budget.decode_calls == 1
    assert budget.cache_hits == 1
    assert budget.cache_size == 1

    other = permutation_to_structure(
        small_instance,
        tuple(reversed(tuple(job.job_id for job in small_instance.jobs))),
        0.70,
        0.70,
    )
    bounded = EvaluationBudget(small_instance, 3, None, "IBS-CABC", cache_capacity=1)
    assert bounded.evaluate(solution) is not None
    assert bounded.evaluate(other) is not None
    assert bounded.evaluate(solution) is not None
    assert bounded.evaluations == bounded.decode_calls == 3
    assert bounded.cache_hits == 0
    assert bounded.cache_evictions == 2
    assert bounded.cache_size == 1


def test_slot_anchored_schedule_is_independently_feasible(small_instance):
    solution = cardinality_aware_seeds(small_instance)[0]
    schedule = EvaluationBudget(
        small_instance,
        1,
        None,
        "IBS-CABC",
    ).evaluate(solution)
    assert schedule is not None
    groups1 = {batch.batch_id: list(batch.job_ids) for batch in schedule.batches_stage1}
    groups2 = {batch.batch_id: list(batch.job_ids) for batch in schedule.batches_stage2}
    explicit = _explicit_schedule(
        small_instance,
        solution,
        schedule,
        groups1=groups1,
        groups2=groups2,
    )
    validate_schedule(small_instance, explicit)
    assert explicit.objective == schedule.objective
    assert explicit.metadata["assignment_repair_explicit"] is True


def test_assignment_repair_removes_obsolete_incumbent_idle_time(small_instance):
    solution = cardinality_aware_seeds(small_instance)[0]
    schedule = decode_structure(small_instance, solution)
    batches1 = tuple(
        replace(batch, start=batch.start + 100, end=batch.end + 100)
        for batch in schedule.batches_stage1
    )
    batches2 = tuple(
        replace(
            batch,
            release_time=batch.release_time + 100,
            start=batch.start + 200,
            end=batch.end + 200,
        )
        for batch in schedule.batches_stage2
    )
    completion1 = tuple(value + 100 for value in schedule.completion_stage1)
    completion2 = tuple(value + 200 for value in schedule.completion_stage2)
    delayed = Schedule(
        schedule.instance_name,
        schedule.algorithm,
        batches1,
        batches2,
        completion1,
        completion2,
        calculate_metrics(
            small_instance,
            batches1,
            batches2,
            completion1,
            completion2,
        ),
    )
    validate_schedule(small_instance, delayed)
    repaired = _explicit_schedule(
        small_instance,
        solution,
        delayed,
        groups1={batch.batch_id: list(batch.job_ids) for batch in batches1},
        groups2={batch.batch_id: list(batch.job_ids) for batch in batches2},
    )
    validate_schedule(small_instance, repaired)
    assert repaired.objective == schedule.objective


def test_bsc_neighbors_preserve_batch_formations_and_scout_is_complete(small_instance):
    solver = IBSCABC(
        small_instance,
        SearchConfig(
            colony_size=4,
            archive_size=8,
            max_evaluations=100,
            abandonment_limit=1,
            seed=17,
        ),
    )
    population = solver._initial_colony()
    source = population[0]
    for _ in range(10):
        _, candidate = solver._neighbor(source, "sequencing")
        assert candidate.stage1_family_orders == source.solution.stage1_family_orders
        assert candidate.stage1_cuts == source.solution.stage1_cuts
        assert candidate.stage2_family_orders == source.solution.stage2_family_orders
        assert candidate.stage2_cuts == source.solution.stage2_cuts
        validate_structure(small_instance, candidate)
    scout = solver._scout_candidate(source)
    assert scout.canonical_signature != source.solution.canonical_signature
    validate_structure(small_instance, scout)


def test_single_joint_population_and_both_neighborhoods_are_active(small_instance):
    result = solve_ibscabc(
        small_instance,
        SearchConfig(
            colony_size=4,
            archive_size=8,
            max_evaluations=160,
            seed=23,
        ),
    )
    diagnostics = result.diagnostics
    assert diagnostics["architecture"] == {
        "algorithm": "IBS-CABC",
        "representation": "direct_two_stage_batch_structure",
        "population": "single_joint_colony",
        "construction": ("release_congestion_dynamic_stage2_partition_with_delayed_dominance_gate"),
        "formation_neighborhood": "BFC",
        "sequencing_neighborhood": "BSC",
        "role_schedule": "strict_alternation",
        "cross_colony_cooperation": "removed",
        "archive": "diverse_elite_archive",
        "decoder": "complete_integrated_two_stage_decoder",
        "primary_objective": "sum_stage2_completion_times",
        "bottleneck_aware": True,
        "intensification": "critical_reconstruction_and_cross_stage_alignment",
        "assignment_repair": ("family_decomposed_joint_and_stage2_slot_reassignment"),
        "stage2_dispatch": ("dominance_screened_non_delay_or_strategic_idle"),
    }
    assert diagnostics["archive_size"] > 0
    assert diagnostics["role_calls"]["formation"] > 0
    assert diagnostics["role_calls"]["sequencing"] > 0
    assert diagnostics["initial_population_size"] == 4
    assert diagnostics["initial_unique_sources"] == 4
    assert diagnostics["final_unique_sources"] > 0
    assert not any(key.startswith("cooperation_") for key in diagnostics)
    assert 0 < diagnostics["intensification_evaluations"] <= 144
    assert diagnostics["intensification_evaluations"] == (
        diagnostics["intensification_stage1_evaluations"]
        + diagnostics["intensification_stage2_evaluations"]
    )
    assert diagnostics["intensification_evaluations"] == (
        diagnostics["intensification_analytical_evaluations"]
        + diagnostics["intensification_elimination_evaluations"]
        + diagnostics["intensification_priority_evaluations"]
        + diagnostics["intensification_reconstruction_evaluations"]
        + diagnostics["intensification_cross_stage_evaluations"]
        + diagnostics["intensification_formation_evaluations"]
    )
    assert diagnostics["intensification_objective_gain"] >= 0
    assert diagnostics["intensification_analytical_gain"] >= 0
    assert diagnostics["intensification_elimination_gain"] >= 0
    assert diagnostics["intensification_priority_gain"] >= 0
    assert diagnostics["intensification_reconstruction_gain"] >= 0
    assert diagnostics["intensification_cross_stage_gain"] >= 0
    assert diagnostics["intensification_cross_stage_triggered"] is False
    assert diagnostics["intensification_formation_gain"] >= 0
    assert diagnostics["assignment_repair_model_calls"] > 0
    assert diagnostics["assignment_repair_feasible_models"] > 0
    assert diagnostics["assignment_repair_objective_gain"] >= 0
    assert diagnostics["assignment_repair_solver_runtime_seconds"] > 0
    assert (
        diagnostics["assignment_repair_solver_runtime_seconds"]
        <= diagnostics["assignment_repair_runtime_seconds"]
    )
    assert diagnostics["assignment_repair_max_solver_call_runtime_seconds"] > 0
    assert (
        diagnostics["assignment_repair_max_solver_call_runtime_seconds"]
        <= diagnostics["assignment_repair_solver_runtime_seconds"]
    )


def test_bfc_and_bsc_update_the_same_population(small_instance):
    solver = IBSCABC(
        small_instance,
        SearchConfig(
            colony_size=4,
            archive_size=8,
            max_evaluations=80,
            seed=29,
        ),
    )
    population = solver._initial_colony()
    evaluations_before = solver.budget.evaluations
    solver._phase(population, "formation")
    formation_end = solver.budget.evaluations
    solver._phase(population, "sequencing")

    assert formation_end > evaluations_before
    assert solver.budget.evaluations > formation_end
    assert solver.role_calls["formation"] > 0
    assert solver.role_calls["sequencing"] > 0
    assert len(population) == solver.initial_population_size
    assert all(source.schedule.objective > 0 for source in population)


def test_cardinality_aware_seeds_are_deterministic_and_wspt_ranked(small_instance):
    first = cardinality_aware_seeds(small_instance)
    second = cardinality_aware_seeds(small_instance)
    assert first
    assert [seed.canonical_signature for seed in first] == [
        seed.canonical_signature for seed in second
    ]
    assert len({seed.canonical_signature for seed in first}) == len(first)
    for seed in first:
        validate_structure(small_instance, seed)
        for stage, priority in (
            (1, seed.stage1_batch_priority),
            (2, seed.stage2_batch_priority),
        ):
            ratio = {
                batch_signature(stage, family, group): max(
                    (
                        small_instance.jobs_by_id[job_id].duration_stage1
                        if stage == 1
                        else small_instance.jobs_by_id[job_id].duration_stage2
                    )
                    for job_id in group
                )
                / len(group)
                for family, groups in stage_batches(seed, stage).items()
                for group in groups
            }
            observed = [ratio[signature] for signature in priority]
            assert observed == sorted(observed)
