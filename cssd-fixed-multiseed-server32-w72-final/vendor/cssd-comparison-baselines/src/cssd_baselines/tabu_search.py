from __future__ import annotations

from collections import Counter

from .common import BaselineConfig, SearchRuntime, Source
from .neighborhoods import small_neighbor
from . import bridge as _bridge  # noqa: F401
from cssd_ibscabc.core.entities import ProblemInstance, RunResult


def solve_tabu_search(
    instance: ProblemInstance,
    config: BaselineConfig | None = None,
) -> RunResult:
    """Best-admissible tabu search with aspiration and restart diversification."""

    runtime = SearchRuntime(instance, config or BaselineConfig(), "TS")
    initial = runtime.initial_population()
    current = min(initial, key=lambda source: source.objective)
    best = current
    tabu_until: dict[str, int] = {}
    selections: Counter[str] = Counter()
    improvements: Counter[str] = Counter()
    aspiration_uses = 0
    diversifications = 0
    stagnant = 0
    iterations = 0

    while not runtime.budget.exhausted:
        iterations += 1
        neighborhood: list[tuple[Source, str, str, bool]] = []
        seen: set[str] = set()
        attempts = 0
        maximum_attempts = max(
            runtime.config.neighborhood_size * runtime.config.candidate_attempts * 3,
            runtime.config.neighborhood_size,
        )
        while (
            len(neighborhood) < runtime.config.neighborhood_size
            and attempts < maximum_attempts
            and not runtime.budget.exhausted
        ):
            attempts += 1
            move = small_neighbor(
                runtime.instance,
                current.solution,
                current.schedule,
                runtime.rng,
            )
            if move is None or move.solution.canonical_signature in seen:
                continue
            seen.add(move.solution.canonical_signature)
            candidate = runtime.evaluate(move.solution)
            if candidate is None:
                break
            is_tabu = tabu_until.get(move.move_key, 0) > iterations
            aspiration = is_tabu and candidate.objective < best.objective
            neighborhood.append((candidate, move.move_key, move.operator, aspiration))

        if not neighborhood:
            break
        admissible = [
            item
            for item in neighborhood
            if tabu_until.get(item[1], 0) <= iterations or item[3]
        ]
        selected = min(admissible or neighborhood, key=lambda item: item[0].objective)
        current, move_key, operator, aspiration = selected
        selections[operator] += 1
        if aspiration:
            aspiration_uses += 1
        tabu_until[move_key] = iterations + runtime.config.tabu_tenure
        tabu_until = {
            key: expiry for key, expiry in tabu_until.items() if expiry > iterations
        }

        if current.objective < best.objective:
            best = current
            runtime.record(best)
            improvements[operator] += 1
            stagnant = 0
        else:
            stagnant += 1

        if stagnant >= runtime.config.stagnation_limit and not runtime.budget.exhausted:
            diversified = runtime.scout()
            if diversified is not None:
                current = diversified
                if current.objective < best.objective:
                    best = current
                    runtime.record(best)
                tabu_until.clear()
                diversifications += 1
            stagnant = 0

    return runtime.finish(
        best,
        iterations,
        {
            "variant": "best_admissible_tabu_search",
            "acceptance": "best_admissible_with_aspiration",
            "tabu_attribute": "operator_stage_family",
            "operator_selections": dict(selections),
            "operator_improvements": dict(improvements),
            "aspiration_uses": aspiration_uses,
            "diversification_restarts": diversifications,
        },
    )
