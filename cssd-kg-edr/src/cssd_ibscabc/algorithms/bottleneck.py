from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..core.entities import ProblemInstance, Schedule

FORMATION_OPERATORS = (
    "split",
    "merge",
    "move",
    "ejection_chain",
    "partial_reconstruction",
)
SEQUENCING_OPERATORS = ("critical_batch_insert", "batch_swap_tail_relocate")


@dataclass(frozen=True, slots=True)
class BottleneckState:
    stage: int
    stage_confidence: float
    utilization: float
    waiting: float
    waiting_ratio: float
    mean_fill: float
    avoidable_idle: float
    critical_signature: str
    critical_contribution: float
    critical_job_count: int
    critical_fill: float
    stagnation: int
    pressure: float


def diagnose(instance: ProblemInstance, schedule: Schedule, stagnation: int = 0) -> BottleneckState:
    stage_records = []
    for stage, batches, utilization, waiting, fill, machines, capacity in (
        (
            1,
            schedule.batches_stage1,
            schedule.metrics.utilization_stage1,
            schedule.metrics.waiting_stage1,
            schedule.metrics.mean_fill_stage1,
            instance.machines_stage1,
            instance.capacity_stage1,
        ),
        (
            2,
            schedule.batches_stage2,
            schedule.metrics.utilization_stage2,
            schedule.metrics.waiting_stage2,
            schedule.metrics.mean_fill_stage2,
            instance.machines_stage2,
            instance.capacity_stage2,
        ),
    ):
        busy = sum(batch.processing_time for batch in batches)
        first_release = min((batch.release_time for batch in batches), default=0)
        horizon = max((batch.end for batch in batches), default=0) - first_release
        # Idle before a stage can possibly start is not caused by its batch
        # structure.  Idle inside the active release-to-completion horizon is
        # potentially reducible by regrouping or resequencing.
        avoidable_idle = max(
            0.0,
            (machines * horizon - busy) / max(1, machines * horizon),
        )
        # A bottleneck diagnosis must be invariant to a common shift in all
        # release/completion times.  Normalizing by the absolute makespan made
        # otherwise identical instances receive different routes.
        waiting_ratio = waiting / max(1, horizon)
        critical = max(
            batches,
            key=lambda batch: (
                len(batch.job_ids) * max(1, batch.end - batch.release_time),
                batch.processing_time,
            ),
        )
        contributions = [
            len(batch.job_ids) * max(1, batch.end - batch.release_time) for batch in batches
        ]
        # Normalize within each stage.  The old denominator was the stage-2
        # objective for both stages and structurally suppressed stage 1.
        critical_contribution = max(contributions) / max(1, sum(contributions))
        wait_signal = min(1.0, 4.0 * waiting_ratio)
        pressure = 0.50 * utilization + 0.30 * wait_signal + 0.20 * critical_contribution
        stage_records.append(
            (
                pressure,
                stage,
                utilization,
                waiting,
                fill,
                avoidable_idle,
                critical,
                critical_contribution,
                waiting_ratio,
                len(critical.job_ids),
                critical.load / max(1, capacity),
            )
        )
    stage_records.sort(reverse=True)
    (
        pressure,
        stage,
        utilization,
        waiting,
        fill,
        idle,
        critical,
        contribution,
        waiting_ratio,
        critical_job_count,
        critical_fill,
    ) = stage_records[0]
    runner_up = stage_records[1][0] if len(stage_records) > 1 else 0.0
    confidence = max(0.0, min(1.0, (pressure - runner_up) / max(pressure, 1e-12)))
    return BottleneckState(
        stage,
        confidence,
        utilization,
        waiting,
        waiting_ratio,
        fill,
        idle,
        critical.signature,
        contribution,
        critical_job_count,
        critical_fill,
        stagnation,
        pressure,
    )


def operator_weights(state: BottleneckState, colony: str) -> Mapping[str, float]:
    """Return a soft, exhaustive routing distribution.

    Continuous signals are used instead of a fixed fill threshold.  Every
    valid neighborhood retains positive mass, so a noisy diagnosis cannot
    remove exploration.  The weights encode search opportunities rather than
    equating low fill with a compulsory merge.
    """

    wait = min(1.0, 4.0 * state.waiting_ratio)
    idle = max(0.0, min(1.0, state.avoidable_idle))
    critical = min(1.0, 4.0 * state.critical_contribution)
    fill_deficit = max(0.0, min(1.0, 1.0 - state.mean_fill))
    stagnation = min(1.0, state.stagnation / 10.0)

    if colony == "formation":
        weights = {name: 1.0 for name in FORMATION_OPERATORS}
        if state.critical_job_count > 1:
            # Splitting a high-contribution multi-job batch can advance several
            # completion times and had the strongest L1 acceptance rate.
            weights["split"] += 0.80 * critical + 0.55 * wait + 0.20 * state.critical_fill
        # A low mean fill alone is not evidence that merging helps sum(Cj):
        # merging can wait for a late job and lengthen the batch duration.
        weights["merge"] += 0.30 * fill_deficit * idle
        weights["move"] += 0.45 * wait + 0.35 * idle
        weights["ejection_chain"] += 0.35 * wait + 0.35 * idle
        weights["partial_reconstruction"] += 0.45 * idle + 3.00 * stagnation
        return weights
    if colony == "sequencing":
        return {
            "critical_batch_insert": 1.0 + 0.90 * critical + 0.65 * wait,
            "batch_swap_tail_relocate": 1.0 + 0.75 * idle + 0.75 * stagnation,
        }
    raise ValueError("colony must be 'formation' or 'sequencing'")


def compatible_operators(state: BottleneckState, colony: str) -> tuple[str, ...]:
    """Rank all compatible operators for diagnostics and reproducibility."""

    weights = operator_weights(state, colony)
    return tuple(sorted(weights, key=lambda name: (-weights[name], name)))
