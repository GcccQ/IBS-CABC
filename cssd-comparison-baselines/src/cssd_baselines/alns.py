from __future__ import annotations

import math
from collections import Counter

from .common import BaselineConfig, SearchRuntime, Source, roulette_index
from .neighborhoods import LARGE_OPERATOR_NAMES, large_neighbor
from . import bridge as _bridge  # noqa: F401
from cssd_ibscabc.core.entities import ProblemInstance, RunResult


def _temperature(runtime: SearchRuntime, reference: int) -> float:
    config = runtime.config
    if config.time_limit_seconds is not None:
        progress = min(1.0, runtime.budget.elapsed / config.time_limit_seconds)
    elif config.max_evaluations is not None:
        progress = min(1.0, runtime.budget.evaluations / config.max_evaluations)
    else:
        progress = 1.0
    ratio = config.initial_temperature_ratio * (
        config.final_temperature_ratio / config.initial_temperature_ratio
    ) ** progress
    return max(1e-9, reference * ratio)


def solve_alns(
    instance: ProblemInstance,
    config: BaselineConfig | None = None,
) -> RunResult:
    """Adaptive large-neighborhood search with SA acceptance."""

    runtime = SearchRuntime(instance, config or BaselineConfig(), "ALNS")
    initial = runtime.initial_population()
    current = min(initial, key=lambda source: source.objective)
    best = current
    weights = {name: 1.0 for name in LARGE_OPERATOR_NAMES}
    scores: Counter[str] = Counter()
    uses: Counter[str] = Counter()
    selections: Counter[str] = Counter()
    improvements: Counter[str] = Counter()
    accepted_worse = 0
    accepted_total = 0
    restarts = 0
    segment_position = 0
    stagnant = 0
    iterations = 0

    while not runtime.budget.exhausted:
        iterations += 1
        name = LARGE_OPERATOR_NAMES[
            roulette_index(runtime.rng, [weights[item] for item in LARGE_OPERATOR_NAMES])
        ]
        selections[name] += 1
        candidate: Source | None = None
        for _ in range(runtime.config.candidate_attempts):
            if runtime.budget.exhausted:
                break
            move = large_neighbor(
                name,
                runtime.instance,
                current.solution,
                current.schedule,
                runtime.rng,
            )
            if move is None:
                continue
            evaluated = runtime.evaluate(move.solution)
            if evaluated is not None and (
                candidate is None or evaluated.objective < candidate.objective
            ):
                candidate = evaluated

        reward = 0.0
        if candidate is not None:
            delta = candidate.objective - current.objective
            temperature = _temperature(runtime, max(1, best.objective))
            accepted = delta <= 0 or runtime.rng.random() < math.exp(-delta / temperature)
            if accepted:
                accepted_total += 1
                if delta > 0:
                    accepted_worse += 1
                current = candidate
                reward = 1.0
                if candidate.objective < best.objective:
                    best = candidate
                    runtime.record(best)
                    improvements[name] += 1
                    stagnant = 0
                    reward = 8.0
                elif delta < 0:
                    reward = 4.0
            if candidate.objective >= best.objective:
                stagnant += 1
        else:
            stagnant += 1

        scores[name] += reward
        uses[name] += 1
        segment_position += 1
        if segment_position >= runtime.config.stagnation_limit:
            reaction = runtime.config.reaction_factor
            for operator in LARGE_OPERATOR_NAMES:
                observed = scores[operator] / max(1, uses[operator])
                weights[operator] = max(
                    0.10,
                    (1.0 - reaction) * weights[operator] + reaction * observed,
                )
            scores.clear()
            uses.clear()
            segment_position = 0

        # The common stagnation limit is used as ALNS's diversification trigger.
        if stagnant >= runtime.config.stagnation_limit and not runtime.budget.exhausted:
            current = runtime.rng.choice(initial)
            stagnant = 0
            restarts += 1

    return runtime.finish(
        best,
        iterations,
        {
            "variant": "adaptive_large_neighborhood_search",
            "acceptance": "simulated_annealing",
            "operator_weights": weights,
            "operator_selections": dict(selections),
            "operator_improvements": dict(improvements),
            "accepted_moves": accepted_total,
            "accepted_worsening_moves": accepted_worse,
            "diversification_restarts": restarts,
        },
    )
