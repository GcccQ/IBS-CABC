from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ..core.entities import Batch


@dataclass(frozen=True, slots=True)
class BatchPlan:
    stage: int
    family: int
    job_ids: tuple[int, ...]
    load: int
    processing_time: int
    release_time: int
    priority_rank: int
    signature: str


def dispatch_parallel(
    plans: Sequence[BatchPlan],
    machine_count: int,
    selection_key: Callable[[BatchPlan], tuple[object, ...]] | None = None,
    *,
    strategic_idle: bool = False,
) -> tuple[Batch, ...]:
    """Event-driven dispatch on identical parallel machines.

    At each machine-release event, the best currently ready batch is selected.
    If no batch is ready, only the earliest-free machine advances to the next
    release.  ``strategic_idle`` optionally admits a strictly screened wait for
    one future stage-2 batch. Deterministic tie-breaking makes paired-seed
    experiments exact.
    """

    if machine_count <= 0:
        raise ValueError("machine_count must be positive")
    key = selection_key or (
        lambda plan: (
            plan.priority_rank,
            plan.release_time,
            plan.processing_time,
            plan.family,
            plan.signature,
        )
    )
    remaining = list(plans)
    machine_ready = [0] * machine_count
    scheduled: list[Batch] = []
    while remaining:
        machine = min(range(machine_count), key=lambda item: (machine_ready[item], item))
        event_time = machine_ready[machine]
        ready = [plan for plan in remaining if plan.release_time <= event_time]
        if not ready:
            event_time = min(plan.release_time for plan in remaining)
            machine = min(
                range(machine_count),
                key=lambda item: (max(machine_ready[item], event_time), item),
            )
            event_time = max(machine_ready[machine], event_time)
            ready = [plan for plan in remaining if plan.release_time <= event_time]
        chosen = min(ready, key=key)
        if strategic_idle and chosen.stage == 2:
            # A non-delay schedule is not dominant for 1|r_j|sum(w_j C_j).
            # Here a batch weight is exactly its number of jobs because every
            # member completes with the batch. Reserve the machine for a
            # not-yet-ready batch only when a two-batch completion surrogate,
            # including the earliest alternative machine, strictly improves.
            # The caller still compares this complete schedule with the
            # ordinary non-delay schedule, so this look-ahead can never worsen
            # the decoded objective.
            future = sorted(
                (
                    plan
                    for plan in remaining
                    if plan.release_time > event_time
                ),
                key=key,
            )[:8]
            other_ready = min(
                (
                    ready_time
                    for index, ready_time in enumerate(machine_ready)
                    if index != machine
                ),
                default=float("inf"),
            )
            weight_ready = len(chosen.job_ids)
            finish_ready = event_time + chosen.processing_time
            best_future: BatchPlan | None = None
            best_saving = 0.0
            for candidate in future:
                weight_future = len(candidate.job_ids)
                future_start_without_idle = min(
                    max(finish_ready, candidate.release_time),
                    max(other_ready, candidate.release_time),
                )
                cost_without_idle = (
                    weight_ready * finish_ready
                    + weight_future
                    * (future_start_without_idle + candidate.processing_time)
                )
                future_finish_with_idle = (
                    candidate.release_time + candidate.processing_time
                )
                ready_start_after_idle = min(
                    future_finish_with_idle,
                    max(other_ready, event_time),
                )
                cost_with_idle = (
                    weight_future * future_finish_with_idle
                    + weight_ready
                    * (ready_start_after_idle + chosen.processing_time)
                )
                saving = cost_without_idle - cost_with_idle
                if saving > best_saving:
                    best_saving = saving
                    best_future = candidate
            if best_future is not None:
                chosen = best_future
                event_time = chosen.release_time
        start = max(event_time, chosen.release_time)
        end = start + chosen.processing_time
        scheduled.append(
            Batch(
                batch_id=len(scheduled),
                stage=chosen.stage,
                family=chosen.family,
                job_ids=chosen.job_ids,
                load=chosen.load,
                processing_time=chosen.processing_time,
                release_time=chosen.release_time,
                priority_rank=chosen.priority_rank,
                machine_id=machine,
                start=start,
                end=end,
                signature=chosen.signature,
            )
        )
        machine_ready[machine] = end
        remaining.remove(chosen)
    return tuple(scheduled)
