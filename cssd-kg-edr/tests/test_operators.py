from dataclasses import replace
from random import Random

from cssd_ibscabc.io import load_instance
from cssd_ibscabc.algorithms.bottleneck import (
    compatible_operators,
    diagnose,
    operator_weights,
)
from cssd_ibscabc.algorithms.operators import OPERATORS, apply_operator
from cssd_ibscabc.decoder import decode_structure, permutation_to_structure
from cssd_ibscabc.representation import validate_structure


def test_exactly_seven_problem_specific_neighborhood_families():
    assert set(OPERATORS) == {
        "split",
        "merge",
        "move",
        "ejection_chain",
        "critical_batch_insert",
        "batch_swap_tail_relocate",
        "partial_reconstruction",
    }


def test_every_operator_preserves_feasibility(small_instance):
    solution = permutation_to_structure(
        small_instance, tuple(job.job_id for job in small_instance.jobs), 0.70, 0.70
    )
    schedule = decode_structure(small_instance, solution)
    for seed, name in enumerate(OPERATORS, 1):
        candidate = apply_operator(
            name,
            small_instance,
            solution,
            schedule,
            diagnose(small_instance, schedule),
            Random(seed),
            3,
        )
        validate_structure(small_instance, candidate)
        decoded = decode_structure(small_instance, candidate)
        assert decoded.objective > 0


def test_stagnation_routes_formation_search_to_reconstruction(small_instance):
    solution = permutation_to_structure(
        small_instance, tuple(job.job_id for job in small_instance.jobs), 0.70, 0.70
    )
    schedule = decode_structure(small_instance, solution)
    state = diagnose(small_instance, schedule, stagnation=5)
    assert compatible_operators(state, "formation")[0] == "partial_reconstruction"


def test_sequencing_route_has_nonuniform_context_weights(small_instance):
    solution = permutation_to_structure(
        small_instance, tuple(job.job_id for job in small_instance.jobs), 0.70, 0.70
    )
    schedule = decode_structure(small_instance, solution)
    state = diagnose(small_instance, schedule)
    weights = operator_weights(state, "sequencing")
    assert set(weights) == {"critical_batch_insert", "batch_swap_tail_relocate"}
    assert len(set(weights.values())) > 1
    assert all(weight > 0 for weight in weights.values())


def test_critical_insert_respects_diagnosed_stage(small_instance):
    solution = permutation_to_structure(
        small_instance, tuple(job.job_id for job in small_instance.jobs), 0.70, 0.70
    )
    schedule = decode_structure(small_instance, solution)
    movable = [batch for batch in schedule.batches_stage1 if batch.priority_rank > 0]
    assert movable
    critical = max(movable, key=lambda batch: len(batch.job_ids) * batch.end)
    state = replace(
        diagnose(small_instance, schedule),
        stage=1,
        critical_signature=critical.signature,
    )
    candidate = apply_operator(
        "critical_batch_insert",
        small_instance,
        solution,
        schedule,
        state,
        Random(1),
        3,
    )
    assert candidate.stage1_batch_priority != solution.stage1_batch_priority
    assert candidate.stage2_batch_priority == solution.stage2_batch_priority


def test_bottleneck_pressure_is_translation_invariant(small_instance):
    solution = permutation_to_structure(
        small_instance, tuple(job.job_id for job in small_instance.jobs), 0.70, 0.70
    )
    schedule = decode_structure(small_instance, solution)
    shift = 10_000
    shifted = replace(
        schedule,
        batches_stage1=tuple(
            replace(
                batch,
                release_time=batch.release_time + shift,
                start=batch.start + shift,
                end=batch.end + shift,
            )
            for batch in schedule.batches_stage1
        ),
        batches_stage2=tuple(
            replace(
                batch,
                release_time=batch.release_time + shift,
                start=batch.start + shift,
                end=batch.end + shift,
            )
            for batch in schedule.batches_stage2
        ),
        completion_stage1=tuple(value + shift for value in schedule.completion_stage1),
        completion_stage2=tuple(value + shift for value in schedule.completion_stage2),
        metrics=replace(
            schedule.metrics,
            total_completion_time=schedule.metrics.total_completion_time
            + shift * small_instance.job_count,
            makespan=schedule.metrics.makespan + shift,
        ),
    )
    state = diagnose(small_instance, schedule)
    shifted_state = diagnose(small_instance, shifted)
    assert shifted_state == state
    assert 0 <= state.stage_confidence <= 1
    assert 0 <= state.waiting_ratio
    assert 0 <= state.critical_contribution <= 1
    assert 0 <= state.critical_fill <= 1


def test_randomized_operator_stress_has_zero_infeasible_candidates(small_instance):
    solution = permutation_to_structure(
        small_instance, tuple(job.job_id for job in small_instance.jobs), 0.70, 0.70
    )
    schedule = decode_structure(small_instance, solution)
    rng = Random(20260722)
    for _ in range(200):
        name = rng.choice(tuple(OPERATORS))
        candidate = apply_operator(
            name,
            small_instance,
            solution,
            schedule,
            diagnose(small_instance, schedule),
            rng,
            3,
        )
        validate_structure(small_instance, candidate)
        schedule = decode_structure(small_instance, candidate)
        solution = candidate


def test_multiscale_published_instance_operator_stress(data_root):
    project = data_root.parent / "cssd-kg-edr"
    relative_paths = (
        project / "data" / "splits" / "paper_fixed.txt"
    ).read_text().splitlines()
    instances = []
    selected_sizes = set()
    for relative in relative_paths:
        instance = load_instance(data_root / relative)
        if instance.job_count in (500, 1000) and instance.job_count not in selected_sizes:
            instances.append(instance)
            selected_sizes.add(instance.job_count)
    for instance in instances:
        solution = permutation_to_structure(
            instance, tuple(job.job_id for job in instance.jobs), 1.0, 1.0
        )
        schedule = decode_structure(instance, solution)
        rng = Random(instance.job_count)
        for _ in range(60):
            candidate = apply_operator(
                rng.choice(tuple(OPERATORS)),
                instance,
                solution,
                schedule,
                diagnose(instance, schedule),
                rng,
                3,
            )
            validate_structure(instance, candidate)
            schedule = decode_structure(instance, candidate)
            solution = candidate
