from dataclasses import replace

import pytest

from cssd_ibscabc.core.feasibility import FeasibilityError, validate_schedule
from cssd_ibscabc.decoder import decode_structure, permutation_to_structure
from cssd_ibscabc.decoder.dispatch import BatchPlan, dispatch_parallel
from cssd_ibscabc.representation import (
    BatchStructureSolution,
    role_structural_distance,
    stage_batches,
    structural_distance,
    with_stage_batches,
)


def test_structure_serialization_and_integrated_decoder(small_instance):
    order = tuple(job.job_id for job in small_instance.jobs)
    structure = permutation_to_structure(small_instance, order, 0.85, 0.85)
    restored = BatchStructureSolution.from_dict(structure.to_dict())
    assert structure.canonical_signature == restored.canonical_signature
    assert structural_distance(small_instance, structure, restored) == 0
    schedule = decode_structure(small_instance, restored)
    validate_schedule(small_instance, schedule)
    assert schedule.objective == sum(schedule.completion_stage2)
    for job in small_instance.jobs:
        assert (
            schedule.completion_stage2[job.job_id - 1] >= schedule.completion_stage1[job.job_id - 1]
        )
    for batches, machines, observed in (
        (
            schedule.batches_stage1,
            small_instance.machines_stage1,
            schedule.metrics.utilization_stage1,
        ),
        (
            schedule.batches_stage2,
            small_instance.machines_stage2,
            schedule.metrics.utilization_stage2,
        ),
    ):
        horizon = max(batch.end for batch in batches) - min(batch.start for batch in batches)
        expected = sum(batch.processing_time for batch in batches) / (machines * horizon)
        assert observed == pytest.approx(expected)


def test_integrated_decoder_is_deterministic(small_instance):
    order = tuple(job.job_id for job in small_instance.jobs)
    structure = permutation_to_structure(small_instance, order)
    first = decode_structure(small_instance, structure)
    second = decode_structure(small_instance, structure)
    assert first == second


def test_strategic_idle_improves_weighted_batch_completion_when_dominant():
    ready = BatchPlan(2, 1, (1,), 1, 10, 0, 0, "ready")
    urgent = BatchPlan(
        2,
        1,
        tuple(range(2, 12)),
        10,
        1,
        1,
        1,
        "urgent",
    )
    non_delay = dispatch_parallel((ready, urgent), 1)
    strategic = dispatch_parallel((ready, urgent), 1, strategic_idle=True)

    non_delay_objective = sum(
        len(batch.job_ids) * batch.end for batch in non_delay
    )
    strategic_objective = sum(
        len(batch.job_ids) * batch.end for batch in strategic
    )
    assert strategic[0].signature == "urgent"
    assert strategic[0].start == 1
    assert strategic_objective < non_delay_objective


def test_integrated_strategic_idle_best_response_never_worsens_fixed_structure(
    small_instance,
):
    structure = permutation_to_structure(
        small_instance,
        tuple(job.job_id for job in small_instance.jobs),
    )
    non_delay = decode_structure(
        small_instance,
        structure,
        strategic_idle_best_response=False,
    )
    best_response = decode_structure(
        small_instance,
        structure,
        strategic_idle_best_response=True,
    )
    assert best_response.objective <= non_delay.objective
    assert best_response.metadata["stage2_dispatch_best_response"] in {
        "non_delay",
        "strategic_idle",
    }


def test_internal_order_is_latent_genotype_but_same_current_phenotype(small_instance):
    order = tuple(job.job_id for job in small_instance.jobs)
    structure = permutation_to_structure(small_instance, order, 0.70, 0.70)
    batches = stage_batches(structure, 1)
    family = next(
        family for family, groups in batches.items() if any(len(group) > 1 for group in groups)
    )
    index = next(index for index, group in enumerate(batches[family]) if len(group) > 1)
    changed = {key: list(groups) for key, groups in batches.items()}
    changed[family][index] = tuple(reversed(changed[family][index]))
    equivalent = with_stage_batches(small_instance, structure, 1, changed)
    assert equivalent.canonical_signature != structure.canonical_signature
    assert structural_distance(small_instance, equivalent, structure) == 0
    assert (
        decode_structure(small_instance, equivalent).objective
        == decode_structure(small_instance, structure).objective
    )


def test_role_distances_isolate_formation_and_sequencing_components(small_instance):
    order = tuple(job.job_id for job in small_instance.jobs)
    structure = permutation_to_structure(small_instance, order, 0.70, 0.70)
    reordered = with_stage_batches(
        small_instance,
        structure,
        1,
        stage_batches(structure, 1),
        tuple(reversed(structure.stage1_batch_priority)),
    )
    assert (
        role_structural_distance(
            small_instance, structure, reordered, "formation"
        )
        == 0
    )
    assert (
        role_structural_distance(
            small_instance, structure, reordered, "sequencing"
        )
        > 0
    )


def test_formation_change_inherits_semantic_priority(small_instance):
    order = tuple(job.job_id for job in small_instance.jobs)
    structure = permutation_to_structure(small_instance, order, 0.70, 0.70)
    batches = stage_batches(structure, 1)
    rank = {signature: index for index, signature in enumerate(structure.stage1_batch_priority)}
    candidates = []
    from cssd_ibscabc.core.signatures import batch_signature

    for family, groups in batches.items():
        for index, group in enumerate(groups):
            if len(group) > 1:
                candidates.append((rank[batch_signature(1, family, group)], family, index))
    old_rank, family, index = min(candidates)
    group = batches[family][index]
    changed = {key: list(groups) for key, groups in batches.items()}
    changed[family][index : index + 1] = [group[:1], group[1:]]
    updated = with_stage_batches(small_instance, structure, 1, changed)
    new_signatures = {
        batch_signature(1, family, group[:1]),
        batch_signature(1, family, group[1:]),
    }
    new_ranks = [
        position
        for position, signature in enumerate(updated.stage1_batch_priority)
        if signature in new_signatures
    ]
    assert min(new_ranks) <= old_rank + 1


def test_validator_rejects_corrupted_batch_signature(small_instance):
    structure = permutation_to_structure(
        small_instance,
        tuple(job.job_id for job in small_instance.jobs),
    )
    schedule = decode_structure(small_instance, structure)
    corrupted = replace(schedule.batches_stage1[0], signature="corrupted")
    invalid = replace(schedule, batches_stage1=(corrupted, *schedule.batches_stage1[1:]))
    with pytest.raises(FeasibilityError, match="signature"):
        validate_schedule(small_instance, invalid)
