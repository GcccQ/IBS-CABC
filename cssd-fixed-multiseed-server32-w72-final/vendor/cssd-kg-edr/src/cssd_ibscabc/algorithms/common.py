from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from random import Random
from typing import Callable

from ..core.entities import ProblemInstance, Schedule
from ..decoder.integrated import decode_structure
from ..representation.batch_structure import (
    BatchStructureSolution,
    StructureDescriptor,
    descriptor_distance,
    structural_descriptor,
)


@dataclass(frozen=True, slots=True)
class SearchConfig:
    colony_size: int = 20
    abandonment_limit: int = 10
    archive_size: int = 20
    max_evaluations: int | None = 10_000
    time_limit_seconds: float | None = None
    onlooker_quality_weight: float = 0.60
    onlooker_diversity_weight: float = 0.15
    onlooker_bottleneck_weight: float = 0.25
    operator_attempts: int = 2
    ejection_depth: int = 3
    intensification_enabled: bool = True
    intensification_fraction: float = 0.90
    intensification_max_evaluations: int = 9000
    intensification_critical_batches: int = 8
    assignment_repair_enabled: bool = True
    assignment_repair_seconds_per_family: float = 10.0
    assignment_repair_workers: int = 2
    assignment_repair_iterations: int = 3
    assignment_repair_joint_max_jobs: int = 600
    # Above J=600, constructing the family CP-SAT slot model can itself exceed
    # the paper's per-instance deadline before the solver time limit starts.
    assignment_repair_max_jobs: int = 600
    assignment_repair_allow_empty_batches: bool = True
    assignment_repair_time_fraction: float = 0.30
    mathematical_repair_seconds: float = 45.0
    delta_seed_pairs: tuple[tuple[float, float], ...] = (
        (1.00, 1.00),
        (0.93, 1.00),
        (1.00, 0.95),
        (0.67, 0.78),
        (0.40, 0.52),
        (0.85, 0.85),
        (0.70, 0.70),
    )
    bottleneck_aware: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if self.colony_size < 4 or self.abandonment_limit <= 0:
            raise ValueError("invalid colony size/abandonment limit")
        if self.archive_size <= 0:
            raise ValueError("archive size must be positive")
        if self.max_evaluations is None and self.time_limit_seconds is None:
            raise ValueError("an evaluation or time budget is required")
        if self.max_evaluations is not None and self.max_evaluations <= 0:
            raise ValueError("max_evaluations must be positive")
        if self.time_limit_seconds is not None and self.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")
        if self.operator_attempts <= 0 or self.ejection_depth < 2:
            raise ValueError("operator_attempts must be positive and ejection_depth >= 2")
        if not 0 <= self.intensification_fraction < 1:
            raise ValueError("intensification_fraction must lie in [0, 1)")
        if (
            self.intensification_max_evaluations <= 0
            or self.intensification_critical_batches <= 0
        ):
            raise ValueError("intensification limits must be positive")
        if self.assignment_repair_seconds_per_family <= 0:
            raise ValueError("assignment repair time must be positive")
        if (
            self.assignment_repair_workers <= 0
            or self.assignment_repair_iterations <= 0
            or self.assignment_repair_joint_max_jobs <= 0
            or self.assignment_repair_max_jobs <= 0
        ):
            raise ValueError("assignment repair limits must be positive")
        if not 0 <= self.assignment_repair_time_fraction < 1:
            raise ValueError("assignment repair time fraction must lie in [0, 1)")
        if self.mathematical_repair_seconds <= 0:
            raise ValueError("mathematical repair time must be positive")
        if not self.delta_seed_pairs or any(
            not (0 < left <= 1 and 0 < right <= 1) for left, right in self.delta_seed_pairs
        ):
            raise ValueError("delta seed pairs must lie in (0, 1]")
        weights = (
            self.onlooker_quality_weight
            + self.onlooker_diversity_weight
            + self.onlooker_bottleneck_weight
        )
        if (
            min(
                self.onlooker_quality_weight,
                self.onlooker_diversity_weight,
                self.onlooker_bottleneck_weight,
            )
            < 0
        ):
            raise ValueError("onlooker weights must be non-negative")
        if abs(weights - 1.0) > 1e-12:
            raise ValueError("onlooker weights must sum to one")


@dataclass(slots=True)
class FoodSource:
    solution: BatchStructureSolution
    schedule: Schedule
    trials: int = 0
    last_operator: str = "seed"

    @property
    def objective(self) -> int:
        return self.schedule.objective


class EvaluationBudget:
    def __init__(
        self,
        instance: ProblemInstance,
        max_evaluations: int | None,
        time_limit_seconds: float | None,
        algorithm: str,
        cache_capacity: int = 256,
    ) -> None:
        if cache_capacity <= 0:
            raise ValueError("cache_capacity must be positive")
        self.instance = instance
        self.max_evaluations = max_evaluations
        self.time_limit_seconds = time_limit_seconds
        self.algorithm = algorithm
        # The paper calls this a CPU-time budget, but the author's released
        # implementation stops on ``time.time() - start``. perf_counter() is
        # the monotonic equivalent of that elapsed/wall-clock definition. Keep
        # process CPU time separately as a diagnostic; using it for stopping
        # would grant a different budget under external process parallelism.
        self.started = time.perf_counter()
        self.started_process = time.process_time()
        self.evaluations = 0
        self.decode_calls = 0
        self.cache_hits = 0
        self.cache_evictions = 0
        self.cache_capacity = cache_capacity
        self._cache: OrderedDict[str, Schedule] = OrderedDict()

    @property
    def elapsed(self) -> float:
        """Elapsed seconds used by the author's stopping-rule implementation."""

        return time.perf_counter() - self.started

    @property
    def wall_elapsed(self) -> float:
        """Alias retained for backward-compatible raw result fields."""

        return self.elapsed

    @property
    def process_elapsed(self) -> float:
        """Process CPU seconds retained as an engineering diagnostic."""

        return time.process_time() - self.started_process

    @property
    def exhausted(self) -> bool:
        by_eval = self.max_evaluations is not None and self.evaluations >= self.max_evaluations
        by_time = self.time_limit_seconds is not None and self.elapsed >= self.time_limit_seconds
        return by_eval or by_time

    def evaluate(self, solution: BatchStructureSolution) -> Schedule | None:
        if self.exhausted:
            return None
        # One candidate assessment consumes one objective evaluation even when
        # its deterministic two-stage decode is already cached.  This preserves
        # the fixed-evaluation trajectory while avoiding repeated decoding.
        self.evaluations += 1
        signature = solution.canonical_signature
        cached = self._cache.get(signature)
        if cached is not None:
            self.cache_hits += 1
            self._cache.move_to_end(signature)
            return cached
        schedule = decode_structure(self.instance, solution, algorithm=self.algorithm)
        self.decode_calls += 1
        self._cache[signature] = schedule
        if len(self._cache) > self.cache_capacity:
            self._cache.popitem(last=False)
            self.cache_evictions += 1
        return schedule

    @property
    def cache_size(self) -> int:
        return len(self._cache)


@dataclass(slots=True)
class DiverseArchive:
    instance: ProblemInstance
    capacity: int
    entries: list[FoodSource] = field(default_factory=list)
    descriptors: dict[str, StructureDescriptor] = field(default_factory=dict, repr=False)
    distances: dict[tuple[str, str], float] = field(default_factory=dict, repr=False)

    def _descriptor(self, solution: BatchStructureSolution) -> StructureDescriptor:
        signature = solution.canonical_signature
        if signature not in self.descriptors:
            self.descriptors[signature] = structural_descriptor(self.instance, solution)
        return self.descriptors[signature]

    def _distance(self, left: BatchStructureSolution, right: BatchStructureSolution) -> float:
        left_signature = left.canonical_signature
        right_signature = right.canonical_signature
        if left_signature == right_signature:
            return 0.0
        key = (
            (left_signature, right_signature)
            if left_signature < right_signature
            else (right_signature, left_signature)
        )
        if key not in self.distances:
            self.distances[key] = descriptor_distance(
                self._descriptor(left), self._descriptor(right)
            )
        return self.distances[key]

    def _retain_active_cache(self) -> None:
        """Keep only descriptors and distances needed by current elites.

        Accepted sources can change thousands of times in a long run.  Retaining
        every historical descriptor and pairwise distance makes the nominally
        bounded archive an unbounded memory cache.
        """

        active = {entry.solution.canonical_signature for entry in self.entries}
        self.descriptors = {
            signature: descriptor
            for signature, descriptor in self.descriptors.items()
            if signature in active
        }
        self.distances = {
            pair: distance
            for pair, distance in self.distances.items()
            if pair[0] in active and pair[1] in active
        }

    def offer(self, source: FoodSource) -> bool:
        signature = source.solution.canonical_signature
        for index, present in enumerate(self.entries):
            if present.solution.canonical_signature == signature:
                if source.objective < present.objective:
                    self.entries[index] = FoodSource(source.solution, source.schedule)
                    self.entries.sort(key=lambda item: item.objective)
                    self._retain_active_cache()
                    return True
                return False
        candidate = FoodSource(source.solution, source.schedule)
        self.entries.append(candidate)
        self.entries.sort(key=lambda item: item.objective)
        if len(self.entries) > self.capacity:
            count = len(self.entries)
            nearest = [1.0] * count
            for left in range(count):
                for right in range(left + 1, count):
                    distance = self._distance(
                        self.entries[left].solution, self.entries[right].solution
                    )
                    nearest[left] = min(nearest[left], distance)
                    nearest[right] = min(nearest[right], distance)
            # Always retain the best; remove the weakest quality/diversity trade-off.
            remove = max(
                range(1, count),
                key=lambda index: index / max(1, count - 1) - 0.25 * nearest[index],
            )
            self.entries.pop(remove)
        retained = any(entry.solution.canonical_signature == signature for entry in self.entries)
        self._retain_active_cache()
        return retained

    @property
    def best(self) -> FoodSource:
        if not self.entries:
            raise RuntimeError("archive is empty")
        return self.entries[0]

    def diversity(self, solution: BatchStructureSolution) -> float:
        if not self.entries:
            return 1.0
        signature = solution.canonical_signature
        others = [
            entry for entry in self.entries if entry.solution.canonical_signature != signature
        ]
        if not others:
            return 1.0
        distance = min(self._distance(solution, entry.solution) for entry in others)
        # A colony source that is not retained by the archive is an ephemeral
        # query.  Do not let its descriptor or distance pairs accumulate.
        self._retain_active_cache()
        return distance

    def sample(self, rng: Random) -> FoodSource:
        if not self.entries:
            raise RuntimeError("archive is empty")
        # Bias to quality without collapsing to the current best.
        return self.entries[int((rng.random() ** 2) * len(self.entries))]


def weighted_choice(rng: Random, weights: list[float]) -> int:
    total = sum(max(0.0, weight) for weight in weights)
    if total <= 0:
        return rng.randrange(len(weights))
    target = rng.random() * total
    running = 0.0
    for index, weight in enumerate(weights):
        running += max(0.0, weight)
        if running >= target:
            return index
    return len(weights) - 1


ProgressCallback = Callable[[int, int | None, int, float], None]
