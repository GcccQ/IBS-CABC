from __future__ import annotations

from dataclasses import dataclass

from ..core.entities import ProblemInstance


@dataclass(frozen=True, slots=True)
class BudgetProtocol:
    name: str
    max_evaluations: int | None
    time_limit_seconds: float | None


def published_time_limit(instance: ProblemInstance, tau: float = 1.0) -> float:
    """CPU-time budget used in the reference paper, converted to seconds."""

    if tau <= 0:
        raise ValueError("tau must be positive")
    return min(
        (
            5000.0
            + tau
            * instance.job_count**2
            * ((instance.machines_stage1 + instance.machines_stage2) / 2.0)
        )
        / 1000.0,
        1800.0,
    )


def resolve_protocol(spec: dict[str, object], instance: ProblemInstance) -> BudgetProtocol:
    mode = str(spec.get("mode", "evaluations"))
    name = str(spec.get("name", mode))
    if mode == "evaluations":
        return BudgetProtocol(name, int(spec["max_evaluations"]), None)
    if mode == "official_time":
        return BudgetProtocol(
            name, None, published_time_limit(instance, float(spec.get("tau", 1.0)))
        )
    if mode == "wall_clock":
        return BudgetProtocol(name, None, float(spec["time_limit_seconds"]))
    raise ValueError(f"unknown protocol mode: {mode}")
