from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from math import isfinite
from numbers import Integral, Real
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .decoder import DecodedSchedule, decode_permutation, validate_schedule
from .model import AuthorInstance


@dataclass(frozen=True, slots=True)
class GAConfig:
    profile_name: str
    population_size: int = 100
    mutation_rate: float = 0.40
    crossover_rate: float = 0.95
    tournament_size: int = 3
    max_elitism_percentage: float = 0.25
    min_elitism_percentage: float = 0.05
    adjustment_interval: int = 10
    local_search_moves: int = 2500
    restart_interval: int = 50
    max_restart_attempts: int = 10
    mutation_type: str = "insert"
    crossover_type: str = "TPX"
    selection_type: str = "roulette"
    selection_semantics: str = "released_objective_proportional"

    def validate(self) -> None:
        if not self.profile_name.strip():
            raise ValueError("profile_name must be non-empty")
        integer_fields = {
            "population_size": self.population_size,
            "tournament_size": self.tournament_size,
            "adjustment_interval": self.adjustment_interval,
            "local_search_moves": self.local_search_moves,
            "restart_interval": self.restart_interval,
            "max_restart_attempts": self.max_restart_attempts,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        if self.population_size < 4 or self.population_size % 2:
            raise ValueError("population_size must be even and at least 4")
        if not 2 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size must be in [2, population_size]")
        rate_fields = {
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "max_elitism_percentage": self.max_elitism_percentage,
            "min_elitism_percentage": self.min_elitism_percentage,
        }
        for name, value in rate_fields.items():
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
                raise ValueError(f"{name} must be a finite real number")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")
        if not 0.0 < self.min_elitism_percentage <= self.max_elitism_percentage < 1.0:
            raise ValueError("elitism percentages must satisfy 0 < min <= max < 1")
        elite_size = int(self.max_elitism_percentage * self.population_size)
        if elite_size < 1:
            raise ValueError("max_elitism_percentage produces an empty elite set")
        if self.max_restart_attempts > 0 and 3 * elite_size + 3 > self.population_size:
            raise ValueError("restart regeneration would exceed population_size")
        if self.adjustment_interval <= 0:
            raise ValueError("adjustment_interval must be positive")
        if self.local_search_moves < 0:
            raise ValueError("local_search_moves cannot be negative")
        if self.restart_interval <= 0 or self.max_restart_attempts < 0:
            raise ValueError("restart settings must be non-negative and interval positive")
        if self.crossover_type != "TPX":
            raise ValueError("the released implementation only defines the selected TPX crossover")
        if self.mutation_type not in {"insert", "flip", "inversion", "scramble"}:
            raise ValueError(f"unsupported mutation type: {self.mutation_type}")
        if self.selection_type not in {"roulette", "tournament"}:
            raise ValueError(f"unsupported selection type: {self.selection_type}")
        if self.selection_semantics not in {
            "released_objective_proportional",
            "corrected_inverse_objective",
        }:
            raise ValueError(f"unsupported selection semantics: {self.selection_semantics}")


GA_PROFILES: Mapping[str, GAConfig] = MappingProxyType(
    {
        # Final calibrated settings reported by the paper.  Search control flow
        # and selection direction remain those of the released source code.
        "paper_final": GAConfig(profile_name="paper_final"),
        # Constants literally present in Genetic_Algorithm_Example.ipynb.
        "notebook_example": GAConfig(
            profile_name="notebook_example",
            crossover_rate=0.80,
            mutation_type="flip",
        ),
        # An explicitly named diagnostic variant; never used by the formal
        # reproduction because it changes the released roulette semantics.
        "corrected_selection": GAConfig(
            profile_name="corrected_selection",
            selection_semantics="corrected_inverse_objective",
        ),
    }
)


def config_from_profile(name: str, overrides: Mapping[str, object] | None = None) -> GAConfig:
    try:
        config = GA_PROFILES[name]
    except KeyError as error:
        raise ValueError(f"unknown GA profile: {name!r}") from error
    if overrides:
        allowed = set(asdict(config)) - {"profile_name"}
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(f"unknown GA overrides: {', '.join(sorted(unknown))}")
        config = replace(config, **dict(overrides))
    config.validate()
    return config


@dataclass(frozen=True, slots=True)
class GARunResult:
    schedule: DecodedSchedule
    runtime_seconds: float
    process_cpu_seconds: float
    evaluations: int
    generations: int
    restart_count: int
    last_generation_best: float
    final_population_best: float
    best_seen_during_search: float
    history: tuple[float, ...]
    incumbent_history: tuple[float, ...]
    history_elapsed_seconds: tuple[float, ...]
    history_evaluations: tuple[int, ...]
    timing: Mapping[str, float]
    config: GAConfig
    termination_reason: str


class _EvaluationLimitReached(RuntimeError):
    """Internal signal used to make development evaluation budgets exact."""


class _GeneticAlgorithm:
    def __init__(
        self,
        instance: AuthorInstance,
        config: GAConfig,
        seed: int,
        time_limit_seconds: float,
        max_evaluations: int | None,
    ) -> None:
        config.validate()
        if not isfinite(float(time_limit_seconds)) or time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be finite and positive")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be an integer in [0, 2^32 - 1]")
        if max_evaluations is not None and (
            isinstance(max_evaluations, bool)
            or not isinstance(max_evaluations, Integral)
            or max_evaluations <= 0
        ):
            raise ValueError("max_evaluations must be a positive integer when supplied")
        if instance.job_count < 3:
            raise ValueError("the released TPX implementation requires at least three jobs")
        self.instance = instance
        self.config = config
        self.rng = np.random.RandomState(seed)
        self.time_limit_seconds = float(time_limit_seconds)
        self.max_evaluations = max_evaluations
        self.evaluations = 0
        self.population = self._generate_population()
        self.fitness_scores: list[float] = []
        self.elitism_percentage = config.max_elitism_percentage
        self.elite_size = int(self.elitism_percentage * config.population_size)
        self.current_restart_attempt = 0
        self.history: list[float] = []
        self.incumbent_history: list[float] = []
        self.history_elapsed_seconds: list[float] = []
        self.history_evaluations: list[int] = []
        self.best_seen_score = float("inf")
        self.best_seen_order = self.instance.erd_order.copy()
        self.timing = {
            "fitness_seconds": 0.0,
            "selection_seconds": 0.0,
            "crossover_seconds": 0.0,
            "mutation_seconds": 0.0,
            "local_search_seconds": 0.0,
            "restart_seconds": 0.0,
        }

    def _generate_population(self) -> list[np.ndarray]:
        population = [
            self.rng.permutation(np.arange(self.instance.job_count, dtype=np.int64))
            for _ in range(self.config.population_size - 3)
        ]
        population.extend(
            [
                self.instance.spt_order.copy(),
                self.instance.erd_order.copy(),
                self.instance.lpt_order.copy(),
            ]
        )
        return population

    def _fitness(self, individual: np.ndarray) -> float:
        if self.max_evaluations is not None and self.evaluations >= self.max_evaluations:
            raise _EvaluationLimitReached
        value = decode_permutation(self.instance, individual).objective
        self.evaluations += 1
        if value < self.best_seen_score:
            self.best_seen_score = value
            self.best_seen_order = individual.copy()
        return value

    def _evaluate_population(self) -> list[float]:
        return [self._fitness(individual) for individual in self.population]

    def _tpx_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        length = self.instance.job_count
        segment = length // 3
        point1 = self.rng.randint(length - segment)
        point2 = self.rng.randint(point1 + segment, length)
        middle = parent1[point1:point2].copy()
        used = set(map(int, middle))
        start: list[int] = []
        index = 0
        while len(start) < point1:
            value = int(parent2[index])
            if value not in used:
                start.append(value)
                used.add(value)
            index += 1
        tail: list[int] = []
        while len(used) < length:
            value = int(parent2[index])
            if value not in used:
                tail.append(value)
                used.add(value)
            index += 1
        return np.concatenate(
            (np.asarray(start, dtype=np.int64), middle, np.asarray(tail, dtype=np.int64))
        )

    def _insert_mutation(self, individual: np.ndarray) -> np.ndarray:
        mutation_point = self.rng.randint(self.instance.job_count)
        insertion_point = self.rng.randint(self.instance.job_count)
        value = individual[mutation_point]
        return np.insert(np.delete(individual, mutation_point), insertion_point, value)

    def _flip_mutation(self, individual: np.ndarray) -> np.ndarray:
        left, right = self.rng.choice(self.instance.job_count, size=2, replace=False)
        individual[left], individual[right] = individual[right], individual[left]
        return individual

    def _inversion_mutation(self, individual: np.ndarray) -> np.ndarray:
        left, right = sorted(self.rng.choice(self.instance.job_count, size=2, replace=False))
        individual[left : right + 1] = individual[left : right + 1][::-1]
        return individual

    def _scramble_mutation(self, individual: np.ndarray) -> np.ndarray:
        length = self.instance.job_count
        segment = length // 3
        point1 = self.rng.randint(length - segment)
        point2 = self.rng.randint(point1 + segment, length)
        subsequence = individual[point1:point2]
        self.rng.shuffle(subsequence)
        individual[point1:point2] = subsequence
        return individual

    def _mutate_unconditionally(self, individual: np.ndarray, mutation_type: str) -> np.ndarray:
        if mutation_type == "insert":
            return self._insert_mutation(individual)
        if mutation_type == "flip":
            return self._flip_mutation(individual)
        if mutation_type == "inversion":
            return self._inversion_mutation(individual)
        if mutation_type == "scramble":
            return self._scramble_mutation(individual)
        raise AssertionError(mutation_type)

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        if self.rng.rand() < self.config.mutation_rate:
            return self._mutate_unconditionally(individual.copy(), self.config.mutation_type)
        return individual

    def _roulette_selection(self) -> np.ndarray:
        scores = np.asarray(self.fitness_scores, dtype=np.float64)
        if self.config.selection_semantics == "released_objective_proportional":
            probabilities = scores / np.sum(scores)
        else:
            inverse = 1.0 / np.maximum(scores, np.finfo(np.float64).tiny)
            probabilities = inverse / np.sum(inverse)
        return self.rng.choice(
            self.config.population_size,
            size=self.config.population_size,
            p=probabilities,
        )

    def _tournament_selection(self) -> np.ndarray:
        parents: list[int] = []
        while len(parents) < self.config.population_size:
            candidates = self.rng.choice(
                self.config.population_size,
                size=self.config.tournament_size,
                replace=False,
            )
            candidate_scores = [self.fitness_scores[index] for index in candidates]
            # The author's code uses argmax here.  Preserve it under released
            # semantics and use argmin only in the explicitly corrected mode.
            chooser = np.argmax if self.config.selection_semantics.startswith("released") else np.argmin
            parents.append(int(candidates[chooser(candidate_scores)]))
        return np.asarray(parents, dtype=np.int64)

    def _select(self) -> np.ndarray:
        if self.config.selection_type == "roulette":
            return self._roulette_selection()
        return self._tournament_selection()

    def _local_search(self, individual: np.ndarray) -> np.ndarray:
        current_fitness = self._fitness(individual)
        first_indices = np.arange(len(individual))
        second_indices = np.arange(len(individual))
        self.rng.shuffle(first_indices)
        self.rng.shuffle(second_indices)
        attempts = 0
        for gene_index in first_indices:
            for new_position in second_indices:
                if new_position == gene_index:
                    continue
                attempts += 1
                candidate = individual.copy()
                value = candidate[gene_index]
                candidate = np.insert(np.delete(candidate, gene_index), new_position, value)
                candidate_fitness = self._fitness(candidate)
                if candidate_fitness < current_fitness:
                    individual = candidate
                    current_fitness = candidate_fitness
                # This strict comparison reproduces the source's 2501 tested
                # moves when LOCAL_SEARCH_MOVES is 2500.
                if attempts > self.config.local_search_moves:
                    return individual
        return individual

    def _regenerate_population(self, elite_population: list[np.ndarray]) -> list[np.ndarray]:
        # Preserve the released aliasing behaviour: scramble mutates ``i`` in
        # place, so the corresponding elite entry in ``regenerated`` changes
        # as well before the same array is appended again.
        regenerated = elite_population.copy()
        for individual in elite_population:
            regenerated.append(self._insert_mutation(individual.copy()))
            regenerated.append(self._scramble_mutation(individual))
        regenerated.extend(
            [
                self.instance.spt_order.copy(),
                self.instance.erd_order.copy(),
                self.instance.lpt_order.copy(),
            ]
        )
        regenerated.extend(
            self.rng.permutation(np.arange(self.instance.job_count, dtype=np.int64))
            for _ in range(self.config.population_size - len(regenerated))
        )
        return regenerated

    def run(self) -> GARunResult:
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        generation = 0
        restart_counter = 0
        restart_count = 0
        previous = float("inf")
        trigger_local_search = False
        last_generation_best = float("inf")
        evaluation_limit_reached = False

        def within_budget() -> bool:
            if time.perf_counter() - wall_start >= self.time_limit_seconds:
                return False
            return self.max_evaluations is None or self.evaluations < self.max_evaluations

        while within_budget():
            generation += 1
            restart_counter += 1
            started = time.perf_counter()
            try:
                self.fitness_scores = self._evaluate_population()
            except _EvaluationLimitReached:
                evaluation_limit_reached = True
            finally:
                self.timing["fitness_seconds"] += time.perf_counter() - started
            if evaluation_limit_reached:
                break

            last_generation_best = float(np.min(self.fitness_scores))
            if last_generation_best < previous:
                trigger_local_search = True
                restart_counter = 1
            self.history.append(last_generation_best)
            self.incumbent_history.append(self.best_seen_score)
            self.history_elapsed_seconds.append(time.perf_counter() - wall_start)
            self.history_evaluations.append(self.evaluations)
            previous = last_generation_best

            # The released condition is ``generation % interval == -1`` and
            # is therefore unreachable for positive generation numbers.  It
            # remains intentionally absent here; elite size stays at 25%.
            elite_indices = np.argsort(self.fitness_scores)[: self.elite_size]
            elite_population = [self.population[index].copy() for index in elite_indices]

            started = time.perf_counter()
            try:
                if trigger_local_search:
                    elite_population[0] = self._local_search(elite_population[0])
                    trigger_local_search = False
            except _EvaluationLimitReached:
                evaluation_limit_reached = True
            finally:
                self.timing["local_search_seconds"] += time.perf_counter() - started
            if evaluation_limit_reached:
                break

            started = time.perf_counter()
            try:
                low_diversity = len(set(self.fitness_scores)) < self.config.population_size // 2
                fixed_restart = restart_counter % self.config.restart_interval == 0
                if (fixed_restart or low_diversity) and (
                    self.current_restart_attempt < self.config.max_restart_attempts
                ):
                    self.population = self._regenerate_population(elite_population)
                    self.current_restart_attempt += 1
                    restart_count += 1
                    self.fitness_scores = self._evaluate_population()
                    elite_indices = np.argsort(self.fitness_scores)[: self.elite_size]
                    elite_population = [self.population[index].copy() for index in elite_indices]
                    restart_counter = 0
            except _EvaluationLimitReached:
                evaluation_limit_reached = True
            finally:
                self.timing["restart_seconds"] += time.perf_counter() - started
            if evaluation_limit_reached:
                break

            started = time.perf_counter()
            selected = self._select()
            self.timing["selection_seconds"] += time.perf_counter() - started

            started = time.perf_counter()
            children: list[np.ndarray] = []
            for index in range(0, len(selected), 2):
                parent1 = self.population[int(selected[index])]
                parent2 = self.population[int(selected[index + 1])]
                if self.rng.rand() < self.config.crossover_rate:
                    children.append(self._tpx_crossover(parent1, parent2))
                    children.append(self._tpx_crossover(parent2, parent1))
                else:
                    children.extend([parent1, parent2])
            children = children[: self.config.population_size - self.elite_size]
            self.timing["crossover_seconds"] += time.perf_counter() - started

            started = time.perf_counter()
            children = [self._mutate(child) for child in children]
            self.timing["mutation_seconds"] += time.perf_counter() - started
            self.population = elite_population + children

        if self.max_evaluations is None:
            # The source evaluates the final population after the time-controlled
            # loop and writes that population's best individual.  This may overrun
            # the nominal wall-clock limit, so runtime and overrun are both exposed.
            final_scores = self._evaluate_population()
            best_index = int(np.argmin(final_scores))
            final_best = float(final_scores[best_index])
            final_order = self.population[best_index]
            termination_reason = "time_limit"
        else:
            # ``max_evaluations`` is a development/testing control absent from
            # the paper.  Returning the best already evaluated individual keeps
            # that diagnostic budget exact, even when it ends mid-generation.
            final_best = self.best_seen_score
            final_order = self.best_seen_order
            termination_reason = (
                "evaluation_limit"
                if evaluation_limit_reached or self.evaluations >= self.max_evaluations
                else "time_limit"
            )
        runtime_seconds = time.perf_counter() - wall_start
        process_cpu_seconds = time.process_time() - cpu_start
        schedule = decode_permutation(self.instance, final_order)
        validate_schedule(self.instance, schedule)
        if not np.isfinite(final_best):
            final_best = schedule.objective
            self.best_seen_score = final_best
        if not np.isfinite(last_generation_best):
            last_generation_best = final_best
        return GARunResult(
            schedule=schedule,
            runtime_seconds=runtime_seconds,
            process_cpu_seconds=process_cpu_seconds,
            evaluations=self.evaluations,
            generations=generation,
            restart_count=restart_count,
            last_generation_best=last_generation_best,
            final_population_best=final_best,
            best_seen_during_search=self.best_seen_score,
            history=tuple(self.history),
            incumbent_history=tuple(self.incumbent_history),
            history_elapsed_seconds=tuple(self.history_elapsed_seconds),
            history_evaluations=tuple(self.history_evaluations),
            timing=MappingProxyType(dict(self.timing)),
            config=self.config,
            termination_reason=termination_reason,
        )


def run_genetic_algorithm(
    instance: AuthorInstance,
    *,
    seed: int,
    time_limit_seconds: float,
    config: GAConfig | None = None,
    max_evaluations: int | None = None,
) -> GARunResult:
    selected = config or GA_PROFILES["paper_final"]
    return _GeneticAlgorithm(
        instance=instance,
        config=selected,
        seed=seed,
        time_limit_seconds=time_limit_seconds,
        max_evaluations=max_evaluations,
    ).run()
