from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from random import Random
from typing import Mapping

# Keep one baseline task on one compute thread.  External task parallelism is
# controlled by the experiment runner and matches the IBS-CABC formal run.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

from . import bridge as _bridge  # noqa: E402,F401
from cssd_ibscabc.algorithms.common import EvaluationBudget  # noqa: E402
from cssd_ibscabc.algorithms.seeding import random_seed, seed_portfolio  # noqa: E402
from cssd_ibscabc.core.entities import ProblemInstance, RunResult, Schedule  # noqa: E402
from cssd_ibscabc.decoder.integrated import decode_structure  # noqa: E402
from cssd_ibscabc.representation.batch_structure import BatchStructureSolution  # noqa: E402


DEFAULT_DELTA_PAIRS: tuple[tuple[float, float], ...] = (
    (1.00, 1.00),
    (0.93, 1.00),
    (1.00, 0.95),
    (0.67, 0.78),
    (0.40, 0.52),
    (0.85, 0.85),
    (0.70, 0.70),
)


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    population_size: int = 20
    stagnation_limit: int = 10
    candidate_attempts: int = 2
    max_evaluations: int | None = 10_000
    time_limit_seconds: float | None = None
    cache_capacity: int = 256
    delta_seed_pairs: tuple[tuple[float, float], ...] = DEFAULT_DELTA_PAIRS
    seed: int = 101
    onlooker_count: int = 20
    reaction_factor: float = 0.20
    initial_temperature_ratio: float = 0.01
    final_temperature_ratio: float = 0.00001
    neighborhood_size: int = 20
    tabu_tenure: int = 10

    def __post_init__(self) -> None:
        if self.population_size < 4:
            raise ValueError("population_size must be at least four")
        if self.stagnation_limit <= 0 or self.candidate_attempts <= 0:
            raise ValueError("stagnation_limit and candidate_attempts must be positive")
        if self.max_evaluations is None and self.time_limit_seconds is None:
            raise ValueError("an evaluation or wall-clock budget is required")
        if self.max_evaluations is not None and self.max_evaluations <= 0:
            raise ValueError("max_evaluations must be positive")
        if self.time_limit_seconds is not None and self.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")
        if self.cache_capacity <= 0 or self.onlooker_count <= 0:
            raise ValueError("cache_capacity and onlooker_count must be positive")
        if not 0 < self.reaction_factor <= 1:
            raise ValueError("reaction_factor must lie in (0, 1]")
        if not 0 < self.final_temperature_ratio < self.initial_temperature_ratio:
            raise ValueError("temperature ratios must be positive and decreasing")
        if self.neighborhood_size <= 0 or self.tabu_tenure <= 0:
            raise ValueError("neighborhood_size and tabu_tenure must be positive")


@dataclass(slots=True)
class Source:
    solution: BatchStructureSolution
    schedule: Schedule
    trials: int = 0

    @property
    def objective(self) -> int:
        return self.schedule.objective


class SearchRuntime:
    def __init__(
        self,
        instance: ProblemInstance,
        config: BaselineConfig,
        label: str,
    ) -> None:
        self.instance = instance
        self.config = config
        self.label = label
        self.rng = Random(config.seed)
        self.budget = EvaluationBudget(
            instance,
            config.max_evaluations,
            config.time_limit_seconds,
            label,
            cache_capacity=max(config.cache_capacity, 4 * config.population_size),
        )
        self.history: list[int] = []
        self.history_evaluations: list[int] = []

    def evaluate(self, solution: BatchStructureSolution) -> Source | None:
        schedule = self.budget.evaluate(solution)
        return None if schedule is None else Source(solution, schedule)

    def initial_population(self) -> list[Source]:
        population: list[Source] = []
        for solution in seed_portfolio(
            self.instance,
            self.config.population_size,
            self.config.delta_seed_pairs,
            self.rng,
        ):
            source = self.evaluate(solution)
            if source is None:
                break
            population.append(source)
        if not population:
            raise RuntimeError("budget ended before one initial solution was evaluated")
        best = min(population, key=lambda item: item.objective)
        self.record(best)
        return population

    def scout(self) -> Source | None:
        return self.evaluate(random_seed(self.instance, self.config.delta_seed_pairs, self.rng))

    def record(self, source: Source) -> None:
        if self.history and source.objective >= self.history[-1]:
            return
        self.history.append(source.objective)
        self.history_evaluations.append(self.budget.evaluations)

    def finish(
        self,
        best: Source,
        iterations: int,
        diagnostics: Mapping[str, object],
    ) -> RunResult:
        final_dispatch = decode_structure(
            self.instance,
            best.solution,
            algorithm=self.label,
            strategic_idle_best_response=True,
        )
        final_schedule = (
            final_dispatch if final_dispatch.objective < best.schedule.objective else best.schedule
        )
        if final_schedule.objective < best.schedule.objective:
            best = Source(best.solution, final_schedule)
            self.record(best)
        payload = {
            "architecture": {
                "algorithm": self.label,
                "representation": "direct_two_stage_batch_structure",
                "decoder": "shared_complete_integrated_two_stage_decoder",
                "stage2_dispatch": "non_delay_search_plus_final_strategic_idle_best_response",
                "primary_objective": "sum_stage2_completion_times",
                "bottleneck_aware": False,
                "assignment_repair": "disabled",
            },
            "config": asdict(self.config),
            "evaluation_requests": self.budget.evaluations,
            "decode_calls": self.budget.decode_calls,
            "decode_cache_hits": self.budget.cache_hits,
            "decode_cache_evictions": self.budget.cache_evictions,
            "process_cpu_runtime_seconds": self.budget.process_elapsed,
            "wall_runtime_seconds": self.budget.wall_elapsed,
            "solution": best.solution.to_dict(),
        }
        payload.update(diagnostics)
        return RunResult(
            best.schedule,
            self.budget.evaluations,
            self.budget.elapsed,
            iterations,
            tuple(self.history),
            payload,
            tuple(self.history_evaluations),
        )


def roulette_index(rng: Random, weights: list[float]) -> int:
    total = sum(max(0.0, weight) for weight in weights)
    if total <= 0:
        return rng.randrange(len(weights))
    target = rng.random() * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += max(0.0, weight)
        if cumulative >= target:
            return index
    return len(weights) - 1
