from __future__ import annotations

import os
from typing import Mapping

from . import bridge as _bridge  # noqa: F401

# One DBD task must remain one compute task.  The outer runner owns parallelism.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

from cssd_baselines.alns import solve_alns  # noqa: E402
from cssd_baselines.common import BaselineConfig  # noqa: E402
from cssd_baselines.tabu_search import solve_tabu_search  # noqa: E402
from cssd_ibscabc.algorithms import solve_ibscabc  # noqa: E402
from cssd_ibscabc.algorithms.common import SearchConfig  # noqa: E402
from cssd_ibscabc.core.entities import ProblemInstance, RunResult  # noqa: E402


def _normalized_parameters(parameters: Mapping[str, object]) -> dict[str, object]:
    values = dict(parameters)
    pairs = values.get("delta_seed_pairs")
    if pairs is not None:
        if not isinstance(pairs, (list, tuple)):
            raise ValueError("delta_seed_pairs must be a sequence")
        normalized_pairs: list[tuple[float, float]] = []
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("delta_seed_pairs must contain two-number pairs")
            normalized_pairs.append((float(pair[0]), float(pair[1])))
        values["delta_seed_pairs"] = tuple(normalized_pairs)
    return values


def validate_parameters(algorithm: str, parameters: Mapping[str, object]) -> None:
    """Fail before a long run if a solver configuration is misspelled or invalid."""

    values = _normalized_parameters(parameters)
    if algorithm == "ibscabc":
        config = SearchConfig(
            **values,
            seed=0,
            max_evaluations=1,
            time_limit_seconds=None,
        )
        if not config.bottleneck_aware or not config.intensification_enabled:
            raise ValueError("IBS-CABC requires bottleneck-aware intensification")
        return
    if algorithm in {"alns", "ts"}:
        BaselineConfig(
            **values,
            seed=0,
            max_evaluations=1,
            time_limit_seconds=None,
        )
        return
    raise KeyError(f"unsupported DBD solver: {algorithm!r}")


def solve(
    algorithm: str,
    instance: ProblemInstance,
    *,
    seed: int,
    max_evaluations: int | None,
    time_limit_seconds: float | None,
    parameters: Mapping[str, object],
) -> RunResult:
    """Call one existing solver without modifying its source project."""

    values = _normalized_parameters(parameters)
    if algorithm == "ibscabc":
        config = SearchConfig(
            **values,
            seed=seed,
            max_evaluations=max_evaluations,
            time_limit_seconds=time_limit_seconds,
        )
        if not config.bottleneck_aware or not config.intensification_enabled:
            raise ValueError("formal IBS-CABC requires bottleneck-aware intensification")
        return solve_ibscabc(instance, config)

    if algorithm not in {"alns", "ts"}:
        raise KeyError(f"unsupported DBD solver: {algorithm!r}")
    baseline = BaselineConfig(
        **values,
        seed=seed,
        max_evaluations=max_evaluations,
        time_limit_seconds=time_limit_seconds,
    )
    if algorithm == "alns":
        return solve_alns(instance, baseline)
    return solve_tabu_search(instance, baseline)
