from __future__ import annotations

from collections import Counter

from .common import BaselineConfig, SearchRuntime, Source, roulette_index
from .neighborhoods import partner_guided_neighbor, small_neighbor
from . import bridge as _bridge  # noqa: F401
from cssd_ibscabc.core.entities import ProblemInstance, RunResult


def _partner(index: int, count: int, runtime: SearchRuntime) -> int:
    if count <= 1:
        return index
    selected = runtime.rng.randrange(count - 1)
    return selected + 1 if selected >= index else selected


def _candidate(
    runtime: SearchRuntime,
    source: Source,
    partner: Source,
    operator_counts: Counter[str],
) -> Source | None:
    best: Source | None = None
    for _ in range(runtime.config.candidate_attempts):
        if runtime.budget.exhausted:
            break
        if runtime.rng.random() < 0.5:
            move = partner_guided_neighbor(
                runtime.instance,
                source.solution,
                partner.solution,
                source.schedule,
                runtime.rng,
            )
        else:
            move = small_neighbor(
                runtime.instance,
                source.solution,
                source.schedule,
                runtime.rng,
            )
        if move is None:
            continue
        operator_counts[move.operator] += 1
        evaluated = runtime.evaluate(move.solution)
        if evaluated is not None and (best is None or evaluated.objective < best.objective):
            best = evaluated
    return best


def solve_standard_abc(
    instance: ProblemInstance,
    config: BaselineConfig | None = None,
) -> RunResult:
    """Discrete standard ABC without bottleneck routing or diversity archives."""

    runtime = SearchRuntime(instance, config or BaselineConfig(), "Standard ABC")
    population = runtime.initial_population()
    best = min(population, key=lambda source: source.objective)
    operator_counts: Counter[str] = Counter()
    operator_improvements: Counter[str] = Counter()
    scouts = 0
    iterations = 0

    def attempt(index: int) -> None:
        nonlocal best
        source = population[index]
        partner = population[_partner(index, len(population), runtime)]
        candidate = _candidate(runtime, source, partner, operator_counts)
        if candidate is not None and candidate.objective < source.objective:
            candidate.trials = 0
            population[index] = candidate
            operator_improvements["greedy_acceptance"] += 1
            if candidate.objective < best.objective:
                best = candidate
                runtime.record(best)
        else:
            source.trials += 1

    while not runtime.budget.exhausted:
        iterations += 1
        # Employed-bee phase: one source-local greedy trial per food source.
        for index in range(len(population)):
            if runtime.budget.exhausted:
                break
            attempt(index)

        # Onlooker phase: objective-only roulette, with no diversity or
        # bottleneck terms.  This is the key control against IBS-CABC.
        for _ in range(runtime.config.onlooker_count):
            if runtime.budget.exhausted:
                break
            minimum = min(source.objective for source in population)
            weights = [minimum / max(1, source.objective) for source in population]
            attempt(roulette_index(runtime.rng, weights))

        # Scout phase follows the same limit=10 used in the formal IBS-CABC run.
        for index, source in enumerate(population):
            if runtime.budget.exhausted:
                break
            if source.trials < runtime.config.stagnation_limit:
                continue
            replacement = runtime.scout()
            if replacement is not None:
                population[index] = replacement
                scouts += 1
                if replacement.objective < best.objective:
                    best = replacement
                    runtime.record(best)

    return runtime.finish(
        best,
        iterations,
        {
            "variant": "discrete_standard_abc",
            "selection": "objective_only_inverse_fitness",
            "acceptance": "strict_greedy",
            "archive": "none",
            "operator_counts": dict(operator_counts),
            "operator_improvements": dict(operator_improvements),
            "scout_replacements": scouts,
        },
    )
