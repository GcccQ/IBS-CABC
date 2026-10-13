from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

from ..core.entities import ProblemInstance
from ..core.signatures import batch_signature


def _freeze(mapping: Mapping[int, Sequence[int]]) -> Mapping[int, tuple[int, ...]]:
    return MappingProxyType(
        {int(family): tuple(int(value) for value in values) for family, values in mapping.items()}
    )


@dataclass(frozen=True, slots=True)
class BatchStructureSolution:
    stage1_family_orders: Mapping[int, tuple[int, ...]]
    stage1_cuts: Mapping[int, tuple[int, ...]]
    stage1_batch_priority: tuple[str, ...]
    stage2_family_orders: Mapping[int, tuple[int, ...]]
    stage2_cuts: Mapping[int, tuple[int, ...]]
    stage2_batch_priority: tuple[str, ...]
    _canonical_signature: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage1_family_orders", _freeze(self.stage1_family_orders))
        object.__setattr__(self, "stage1_cuts", _freeze(self.stage1_cuts))
        object.__setattr__(self, "stage2_family_orders", _freeze(self.stage2_family_orders))
        object.__setattr__(self, "stage2_cuts", _freeze(self.stage2_cuts))
        object.__setattr__(
            self, "stage1_batch_priority", tuple(map(str, self.stage1_batch_priority))
        )
        object.__setattr__(
            self, "stage2_batch_priority", tuple(map(str, self.stage2_batch_priority))
        )
        # Πk is part of the genotype even when two within-batch orders currently
        # decode to the same schedule: later split/move neighborhoods use that
        # latent order and therefore reach different descendants.
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        object.__setattr__(
            self,
            "_canonical_signature",
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage1_family_orders": {
                str(key): list(value) for key, value in self.stage1_family_orders.items()
            },
            "stage1_cuts": {str(key): list(value) for key, value in self.stage1_cuts.items()},
            "stage1_batch_priority": list(self.stage1_batch_priority),
            "stage2_family_orders": {
                str(key): list(value) for key, value in self.stage2_family_orders.items()
            },
            "stage2_cuts": {str(key): list(value) for key, value in self.stage2_cuts.items()},
            "stage2_batch_priority": list(self.stage2_batch_priority),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> BatchStructureSolution:
        def mapping(name: str) -> dict[int, tuple[int, ...]]:
            raw = payload[name]
            if not isinstance(raw, Mapping):
                raise ValueError(f"{name} must be a mapping")
            return {int(key): tuple(map(int, value)) for key, value in raw.items()}

        return cls(
            mapping("stage1_family_orders"),
            mapping("stage1_cuts"),
            tuple(map(str, payload["stage1_batch_priority"])),
            mapping("stage2_family_orders"),
            mapping("stage2_cuts"),
            tuple(map(str, payload["stage2_batch_priority"])),
        )

    @property
    def canonical_signature(self) -> str:
        return self._canonical_signature


def _stage_fields(
    solution: BatchStructureSolution, stage: int
) -> tuple[Mapping[int, tuple[int, ...]], Mapping[int, tuple[int, ...]], tuple[str, ...]]:
    if stage == 1:
        return (
            solution.stage1_family_orders,
            solution.stage1_cuts,
            solution.stage1_batch_priority,
        )
    if stage == 2:
        return (
            solution.stage2_family_orders,
            solution.stage2_cuts,
            solution.stage2_batch_priority,
        )
    raise ValueError("stage must be 1 or 2")


def stage_batches(solution: BatchStructureSolution, stage: int) -> dict[int, list[tuple[int, ...]]]:
    orders, cuts, _ = _stage_fields(solution, stage)
    result: dict[int, list[tuple[int, ...]]] = {}
    for family, order in orders.items():
        previous = 0
        result[family] = []
        for cut in cuts[family]:
            result[family].append(tuple(order[previous:cut]))
            previous = cut
    return result


def _fields_from_batches(
    batches: Mapping[int, Sequence[Sequence[int]]],
) -> tuple[dict[int, tuple[int, ...]], dict[int, tuple[int, ...]]]:
    orders: dict[int, tuple[int, ...]] = {}
    cuts: dict[int, tuple[int, ...]] = {}
    for family in sorted(batches):
        groups = [tuple(map(int, group)) for group in batches[family]]
        orders[family] = tuple(job_id for group in groups for job_id in group)
        running = 0
        endpoints = []
        for group in groups:
            running += len(group)
            endpoints.append(running)
        cuts[family] = tuple(endpoints)
    return orders, cuts


def _signatures(stage: int, batches: Mapping[int, Sequence[Sequence[int]]]) -> tuple[str, ...]:
    return tuple(
        batch_signature(stage, family, group)
        for family in sorted(batches)
        for group in batches[family]
    )


def _reconcile_priority(
    stage: int,
    batches: Mapping[int, Sequence[Sequence[int]]],
    priority: Sequence[str] | None,
) -> tuple[str, ...]:
    required = _signatures(stage, batches)
    required_set = set(required)
    kept = [signature for signature in (priority or ()) if signature in required_set]
    kept_set = set(kept)
    return tuple(kept + [signature for signature in required if signature not in kept_set])


def _transfer_priority(
    solution: BatchStructureSolution,
    stage: int,
    batches: Mapping[int, Sequence[Sequence[int]]],
) -> tuple[str, ...]:
    """Transfer old batch ranks to a changed partition by shared job membership.

    Formation operators create new signatures.  Appending every new signature
    to the tail would silently turn a local formation move into a disruptive
    sequencing move.  Earliest inherited job rank keeps the new batches near the
    batches they replace, while family/ordinal/signature provide deterministic
    tie-breaking.
    """

    old_batches = stage_batches(solution, stage)
    old_priority = solution.stage1_batch_priority if stage == 1 else solution.stage2_batch_priority
    signature_rank = {signature: rank for rank, signature in enumerate(old_priority)}
    job_rank = {
        job_id: signature_rank[batch_signature(stage, family, group)]
        for family, groups in old_batches.items()
        for group in groups
        for job_id in group
    }
    records: list[tuple[float, int, int, str]] = []
    for family in sorted(batches):
        for ordinal, group in enumerate(batches[family]):
            signature = batch_signature(stage, family, group)
            # Earliest inherited rank protects a newly formed batch from being
            # pushed behind every batch it partly replaces.  This is aligned
            # with the regular objective, where delaying a multi-job batch has
            # a multiplied completion-time cost.
            inherited = float(min(job_rank[job_id] for job_id in group))
            records.append((inherited, family, ordinal, signature))
    return tuple(record[-1] for record in sorted(records))


def make_solution(
    instance: ProblemInstance,
    stage1: Mapping[int, Sequence[Sequence[int]]],
    stage2: Mapping[int, Sequence[Sequence[int]]],
    priority1: Sequence[str] | None = None,
    priority2: Sequence[str] | None = None,
) -> BatchStructureSolution:
    order1, cuts1 = _fields_from_batches(stage1)
    order2, cuts2 = _fields_from_batches(stage2)
    solution = BatchStructureSolution(
        order1,
        cuts1,
        _reconcile_priority(1, stage1, priority1),
        order2,
        cuts2,
        _reconcile_priority(2, stage2, priority2),
    )
    validate_structure(instance, solution)
    return solution


def with_stage_batches(
    instance: ProblemInstance,
    solution: BatchStructureSolution,
    stage: int,
    batches: Mapping[int, Sequence[Sequence[int]]],
    priority: Sequence[str] | None = None,
) -> BatchStructureSolution:
    transferred = priority if priority is not None else _transfer_priority(solution, stage, batches)
    if stage == 1:
        return make_solution(
            instance,
            batches,
            stage_batches(solution, 2),
            transferred,
            solution.stage2_batch_priority,
        )
    if stage == 2:
        return make_solution(
            instance,
            stage_batches(solution, 1),
            batches,
            solution.stage1_batch_priority,
            transferred,
        )
    raise ValueError("stage must be 1 or 2")


def validate_structure(instance: ProblemInstance, solution: BatchStructureSolution) -> None:
    expected_families = set(instance.active_families)
    for stage in (1, 2):
        orders, cuts, priority = _stage_fields(solution, stage)
        if set(orders) != expected_families or set(cuts) != expected_families:
            raise ValueError(f"stage {stage} does not contain every active family")
        batches = stage_batches(solution, stage)
        capacity = instance.capacity_stage1 if stage == 1 else instance.capacity_stage2
        for family in instance.active_families:
            order = orders[family]
            endpoints = cuts[family]
            if set(order) != set(instance.family_job_ids[family]) or len(order) != len(
                instance.family_job_ids[family]
            ):
                raise ValueError(f"stage {stage} family {family} coverage is invalid")
            if (
                not endpoints
                or endpoints[-1] != len(order)
                or any(left >= right for left, right in zip((0, *endpoints[:-1]), endpoints))
            ):
                raise ValueError(f"stage {stage} family {family} cuts are invalid")
            for group in batches[family]:
                load = sum(
                    instance.jobs_by_id[job_id].size_stage1
                    if stage == 1
                    else instance.jobs_by_id[job_id].size_stage2
                    for job_id in group
                )
                if not 0 < load <= capacity:
                    raise ValueError(f"stage {stage} family {family} batch exceeds capacity")
        required = _signatures(stage, batches)
        if len(priority) != len(required) or set(priority) != set(required):
            raise ValueError(f"stage {stage} priority must cover every batch exactly once")


def structural_distance(
    instance: ProblemInstance,
    left: BatchStructureSolution,
    right: BatchStructureSolution,
) -> float:
    return descriptor_distance(
        structural_descriptor(instance, left), structural_descriptor(instance, right)
    )


StructureDescriptor = tuple[tuple[str, ...], tuple[float, ...], tuple[str, ...], tuple[float, ...]]


def structural_descriptor(
    instance: ProblemInstance, solution: BatchStructureSolution
) -> StructureDescriptor:
    descriptors: list[tuple[str, ...] | tuple[float, ...]] = []
    for stage in (1, 2):
        batches = stage_batches(solution, stage)
        membership = {
            job_id: batch_signature(stage, family, group)
            for family, groups in batches.items()
            for group in groups
            for job_id in group
        }
        _, _, priority = _stage_fields(solution, stage)
        rank = {
            signature: position / max(1, len(priority) - 1)
            for position, signature in enumerate(priority)
        }
        descriptors.append(tuple(membership[job.job_id] for job in instance.jobs))
        descriptors.append(tuple(rank[membership[job.job_id]] for job in instance.jobs))
    return (descriptors[0], descriptors[1], descriptors[2], descriptors[3])  # type: ignore[return-value]


def descriptor_distance(left: StructureDescriptor, right: StructureDescriptor) -> float:
    comparisons = sum(len(values) for values in left)
    differences = sum(a != b for a, b in zip(left[0], right[0]))
    differences += sum(abs(a - b) for a, b in zip(left[1], right[1]))
    differences += sum(a != b for a, b in zip(left[2], right[2]))
    differences += sum(abs(a - b) for a, b in zip(left[3], right[3]))
    return differences / max(1, comparisons)


def role_descriptor_distance(
    left: StructureDescriptor,
    right: StructureDescriptor,
    role: Literal["formation", "sequencing"],
) -> float:
    """Measure only the structural component owned by one bee colony."""

    if role == "formation":
        comparisons = len(left[0]) + len(left[2])
        differences = sum(a != b for a, b in zip(left[0], right[0]))
        differences += sum(a != b for a, b in zip(left[2], right[2]))
    elif role == "sequencing":
        comparisons = len(left[1]) + len(left[3])
        differences = sum(abs(a - b) for a, b in zip(left[1], right[1]))
        differences += sum(abs(a - b) for a, b in zip(left[3], right[3]))
    else:
        raise ValueError("role must be 'formation' or 'sequencing'")
    return differences / max(1, comparisons)


def role_structural_distance(
    instance: ProblemInstance,
    left: BatchStructureSolution,
    right: BatchStructureSolution,
    role: Literal["formation", "sequencing"],
) -> float:
    return role_descriptor_distance(
        structural_descriptor(instance, left),
        structural_descriptor(instance, right),
        role,
    )
