from __future__ import annotations

from dataclasses import asdict
from random import Random
from statistics import fmean
from typing import Literal

from ..core.entities import ProblemInstance, RunResult
from ..decoder.integrated import decode_structure
from ..representation.batch_structure import BatchStructureSolution, role_structural_distance
from .assignment_repair import family_decomposed_assignment_repair
from .bottleneck import (
    diagnose,
    operator_weights,
)
from .common import (
    DiverseArchive,
    EvaluationBudget,
    FoodSource,
    ProgressCallback,
    SearchConfig,
    weighted_choice,
)
from .intensification import (
    analytical_priority_descent,
    batch_elimination_descent,
    critical_batch_descent,
    critical_formation_descent,
    critical_window_reconstruction_descent,
    cross_stage_alignment_descent,
)
from .operators import apply_operator
from .seeding import adaptive_stage2_seeds, random_seed, seed_portfolio


def _role_diversity(
    instance: ProblemInstance,
    population: list[FoodSource],
    role: Literal["formation", "sequencing"],
) -> float:
    distances = [
        role_structural_distance(
            instance,
            population[left].solution,
            population[right].solution,
            role,
        )
        for left in range(len(population))
        for right in range(left + 1, len(population))
    ]
    return fmean(distances) if distances else 0.0


class IBSCABC:
    """IBS-CABC with one population of complete two-stage batch structures.

    Batch formation and batch sequencing remain distinct neighborhoods, but
    they update the same food sources.  This preserves their complementary
    search semantics without the information loss and duplicate evaluations
    caused by two independent colonies and periodic cross-colony mapping.
    """

    def __init__(
        self,
        instance: ProblemInstance,
        config: SearchConfig | None = None,
        *,
        schedule_algorithm: str | None = None,
    ):
        self.instance = instance
        self.config = config or SearchConfig()
        if not self.config.bottleneck_aware:
            raise ValueError("IBS-CABC requires bottleneck-aware routing")
        if not self.config.intensification_enabled:
            raise ValueError("IBS-CABC requires critical-batch intensification")
        self.schedule_algorithm = schedule_algorithm or "IBS-CABC"
        self.rng = Random(self.config.seed)
        self.effective_colony_size = self.config.colony_size
        self.effective_archive_size = self.config.archive_size
        self.assignment_repair_applicable = (
            self.config.assignment_repair_enabled
            and self.instance.job_count <= self.config.assignment_repair_max_jobs
        )
        search_time_limit = self.config.time_limit_seconds
        if (
            search_time_limit is not None
            and self.assignment_repair_applicable
        ):
            search_time_limit *= 1.0 - self.config.assignment_repair_time_fraction
        normal_cache_capacity = max(
            64,
            4 * self.effective_colony_size + 2 * self.effective_archive_size,
        )
        self.budget = EvaluationBudget(
            instance,
            self.config.max_evaluations,
            search_time_limit,
            self.schedule_algorithm,
            cache_capacity=normal_cache_capacity,
        )
        self.archive = DiverseArchive(instance, self.effective_archive_size)
        self.operator_calls: dict[str, int] = {}
        self.operator_improvements: dict[str, int] = {}
        self.operator_selections: dict[str, int] = {}
        self.operator_noops: dict[str, int] = {}
        self.role_calls: dict[str, int] = {"formation": 0, "sequencing": 0}
        self.role_improvements: dict[str, int] = {"formation": 0, "sequencing": 0}
        self.routing_stage_counts: dict[str, int] = {"stage1": 0, "stage2": 0}
        self.scout_attempts: dict[str, int] = {"formation": 0, "sequencing": 0}
        self.scout_changes: dict[str, int] = {"formation": 0, "sequencing": 0}
        self.scout_noops: dict[str, int] = {"formation": 0, "sequencing": 0}
        self.initial_population_size = 0
        self.initial_unique_sources = 0
        self.intensification_evaluations = 0
        self.intensification_passes = 0
        self.intensification_accepted_moves = 0
        self.intensification_stage1_evaluations = 0
        self.intensification_stage2_evaluations = 0
        self.intensification_objective_gain = 0
        self.intensification_analytical_evaluations = 0
        self.intensification_analytical_gain = 0
        self.intensification_elimination_evaluations = 0
        self.intensification_elimination_gain = 0
        self.intensification_priority_evaluations = 0
        self.intensification_priority_gain = 0
        self.intensification_reconstruction_evaluations = 0
        self.intensification_reconstruction_gain = 0
        self.intensification_formation_evaluations = 0
        self.intensification_formation_gain = 0
        self.intensification_cross_stage_evaluations = 0
        self.intensification_cross_stage_gain = 0
        self.intensification_cross_stage_triggered = False
        self.assignment_repair_model_calls = 0
        self.assignment_repair_feasible_models = 0
        self.assignment_repair_optimal_models = 0
        self.assignment_repair_accepted_iterations = 0
        self.assignment_repair_objective_gain = 0
        self.assignment_repair_runtime_seconds = 0.0
        self.assignment_repair_solver_runtime_seconds = 0.0
        self.assignment_repair_max_solver_call_runtime_seconds = 0.0
        self.assignment_repair_method_sequence: tuple[str, ...] = ()
        self.assignment_repair_rounds = 0
        # Wall-clock accounting for the optional final CP-SAT round.  Any
        # reserve that CP-SAT does not consume is returned to the stochastic
        # search so that enabling repair cannot silently shorten the common
        # per-instance time budget.
        self.final_repair_reserved_seconds = 0.0
        self.final_repair_runtime_seconds = 0.0
        self.final_repair_objective_gain = 0
        self.post_final_repair_search_seconds = 0.0
        self.post_final_repair_search_evaluations = 0
        self.post_final_repair_search_objective_gain = 0
        self.adaptive_seed_evaluations = 0
        self.adaptive_seed_improvements = 0
        self.adaptive_seed_best_gain = 0
        self.adaptive_seed_injected = False
        self.best_adaptive_seed: FoodSource | None = None

    def _initial_colony(self) -> list[FoodSource]:
        base_sources: list[FoodSource] = []
        candidate_sources: list[FoodSource] = []
        seen: set[str] = set()
        for solution in seed_portfolio(
            self.instance,
            self.effective_colony_size,
            self.config.delta_seed_pairs,
            self.rng,
        ):
            schedule = self.budget.evaluate(solution)
            if schedule is None:
                break
            source = FoodSource(solution, schedule)
            base_sources.append(source)
            candidate_sources.append(source)
            seen.add(solution.canonical_signature)
        if not base_sources:
            raise RuntimeError("budget ended before one initial solution was evaluated")
        base_best = min(source.objective for source in base_sources)
        adaptive_limit = 3 * 16
        if (
            self.config.max_evaluations is not None
            and self.config.max_evaluations
            < max(100, 10 * self.effective_colony_size)
        ):
            # Tiny test/evaluation budgets must still reach both ABC roles.
            adaptive_limit = 0
        # Expand only a few strong complete solutions. The descendants alter
        # stage-2 batch boundaries while retaining each parent's upstream
        # structure, so they complement the rule/random portfolio.
        for parent in sorted(base_sources, key=lambda source: source.objective)[:3]:
            for solution in adaptive_stage2_seeds(
                self.instance,
                parent.solution,
            ):
                if (
                    self.budget.exhausted
                    or self.adaptive_seed_evaluations >= adaptive_limit
                ):
                    break
                signature = solution.canonical_signature
                if signature in seen:
                    continue
                seen.add(signature)
                schedule = self.budget.evaluate(solution)
                if schedule is None:
                    break
                self.adaptive_seed_evaluations += 1
                if schedule.objective < base_best:
                    self.adaptive_seed_improvements += 1
                candidate_sources.append(
                    FoodSource(
                        solution,
                        schedule,
                        0,
                        "adaptive_stage2_seed",
                    )
                )
        for source in base_sources:
            self.archive.offer(source)
        population = [
            FoodSource(
                source.solution,
                source.schedule,
                source.trials,
                source.last_operator,
            )
            for source in base_sources
        ]
        adaptive_sources = [
            source
            for source in candidate_sources
            if source.last_operator == "adaptive_stage2_seed"
        ]
        if adaptive_sources:
            self.best_adaptive_seed = min(
                adaptive_sources,
                key=lambda source: source.objective,
            )
        self.adaptive_seed_best_gain = (
            0
            if self.best_adaptive_seed is None
            else max(0, base_best - self.best_adaptive_seed.objective)
        )
        self.initial_population_size = len(population)
        self.initial_unique_sources = len(
            {source.solution.canonical_signature for source in population}
        )
        return population

    def _routing_state(self, source: FoodSource):
        state = diagnose(self.instance, source.schedule, source.trials)
        self.routing_stage_counts[f"stage{state.stage}"] += 1
        return state

    def _neighbor(
        self,
        source: FoodSource,
        role: Literal["formation", "sequencing"],
    ) -> tuple[str, BatchStructureSolution]:
        state = self._routing_state(source)
        contextual = dict(operator_weights(state, role))
        names = tuple(contextual)
        # Bounded success smoothing only refines the deterministic bottleneck
        # weights; every compatible operator stays selectable.
        success = {
            name: (self.operator_improvements.get(name, 0) + 1.0)
            / (self.operator_selections.get(name, 0) + 10.0)
            for name in names
        }
        strongest = max(success.values(), default=1.0)
        weights = [
            contextual[name]
            * (0.80 + 0.40 * success[name] / max(strongest, 1e-12))
            for name in names
        ]

        available_names = list(names)
        available_weights = list(weights)
        last_name = names[0]
        for _ in range(min(self.config.operator_attempts, len(available_names))):
            selected = weighted_choice(self.rng, available_weights)
            name = available_names.pop(selected)
            available_weights.pop(selected)
            last_name = name
            self.operator_selections[name] = self.operator_selections.get(name, 0) + 1
            candidate = apply_operator(
                name,
                self.instance,
                source.solution,
                source.schedule,
                state,
                self.rng,
                self.config.ejection_depth,
            )
            if candidate.canonical_signature != source.solution.canonical_signature:
                return name, candidate
            self.operator_noops[name] = self.operator_noops.get(name, 0) + 1
        return last_name, source.solution

    def _try_update(
        self,
        population: list[FoodSource],
        index: int,
        role: Literal["formation", "sequencing"],
    ) -> None:
        source = population[index]
        name, candidate = self._neighbor(source, role)
        if candidate.canonical_signature == source.solution.canonical_signature:
            source.trials += 1
            return
        self.operator_calls[name] = self.operator_calls.get(name, 0) + 1
        self.role_calls[role] += 1
        schedule = self.budget.evaluate(candidate)
        if schedule is None:
            return
        if schedule.objective < source.objective:
            population[index] = FoodSource(candidate, schedule, 0, name)
            self.operator_improvements[name] = self.operator_improvements.get(name, 0) + 1
            self.role_improvements[role] += 1
            self.archive.offer(population[index])
        else:
            source.trials += 1

    def _onlooker_scores(self, population: list[FoodSource]) -> list[float]:
        objectives = [source.objective for source in population]
        low, high = min(objectives), max(objectives)
        scores = []
        for source in population:
            quality = 1.0 if high == low else (high - source.objective) / (high - low)
            diversity = self.archive.diversity(source.solution)
            pressure = diagnose(self.instance, source.schedule, source.trials).pressure
            score = (
                self.config.onlooker_quality_weight * quality
                + self.config.onlooker_diversity_weight * diversity
                + self.config.onlooker_bottleneck_weight * pressure
            )
            scores.append(score + 1e-12)
        return scores

    def _scout_candidate(self, source: FoodSource) -> BatchStructureSolution:
        """Create an independent complete solution for an abandoned source."""

        for _ in range(self.config.operator_attempts + 2):
            candidate = random_seed(
                self.instance,
                self.config.delta_seed_pairs,
                self.rng,
            )
            if candidate.canonical_signature != source.solution.canonical_signature:
                return candidate
        return source.solution

    def _phase(
        self,
        population: list[FoodSource],
        role: Literal["formation", "sequencing"],
    ) -> None:
        for index in range(len(population)):
            if self.budget.exhausted:
                return
            self._try_update(population, index, role)

        scores = self._onlooker_scores(population)
        for _ in range(len(population)):
            if self.budget.exhausted:
                return
            self._try_update(population, weighted_choice(self.rng, scores), role)

        for index, source in enumerate(list(population)):
            if self.budget.exhausted:
                return
            if source.trials < self.config.abandonment_limit:
                continue
            self.scout_attempts[role] += 1
            candidate = self._scout_candidate(source)
            if candidate.canonical_signature == source.solution.canonical_signature:
                self.scout_noops[role] += 1
                source.trials = 0
                continue
            schedule = self.budget.evaluate(candidate)
            if schedule is not None:
                population[index] = FoodSource(candidate, schedule, 0, "scout_random")
                self.archive.offer(population[index])
                self.scout_changes[role] += 1

    def _intensification_reserve(self) -> int:
        if not self.config.intensification_enabled:
            return 0
        if self.config.max_evaluations is None:
            return self.config.intensification_max_evaluations
        return min(
            self.config.intensification_max_evaluations,
            int(self.config.max_evaluations * self.config.intensification_fraction),
        )

    def _main_search_available(self, reserve: int) -> bool:
        if self.budget.exhausted:
            return False
        if (
            self.config.max_evaluations is not None
            and self.budget.evaluations >= self.config.max_evaluations - reserve
        ):
            return False
        if (
            self.config.time_limit_seconds is not None
            and self.config.intensification_enabled
            and self.budget.elapsed
            >= self.config.time_limit_seconds
            * (1.0 - self.config.intensification_fraction)
        ):
            return False
        return True

    def _intensify(
        self,
        population: list[FoodSource],
        allowance: int,
        *,
        critical_batch_count: int | None = None,
        force_cross_stage: bool = False,
        cross_stage_limit: int = 900,
    ) -> None:
        if not self.config.intensification_enabled or allowance <= 0:
            return
        critical_batches = (
            self.config.intensification_critical_batches
            if critical_batch_count is None
            else critical_batch_count
        )
        incumbent = self.archive.best
        analytical = analytical_priority_descent(
            self.instance,
            incumbent,
            self.budget,
            min(48, allowance),
        )
        elimination = batch_elimination_descent(
            self.instance,
            analytical.source,
            self.budget,
            (
                min(300, max(0, allowance - analytical.evaluations))
                if self.instance.job_count >= 800
                else 0
            ),
            critical_batch_count=critical_batches,
        )
        # Full-list analytical ranking and final re-ranking dominate local
        # single-batch insertion on the audited 30-run development set.
        priority_allowance = 0
        priority = critical_batch_descent(
            self.instance,
            elimination.source,
            self.budget,
            priority_allowance,
            critical_batch_count=critical_batches,
        )
        remaining_after_priority = max(
            0,
            allowance
            - analytical.evaluations
            - elimination.evaluations
            - priority.evaluations,
        )
        metrics = priority.source.schedule.metrics
        cross_stage_triggered = (
            self.instance.job_count >= 800
            and (
                force_cross_stage
                or (
                    metrics.mean_fill_stage1 >= 0.98
                    and metrics.waiting_stage2 > metrics.waiting_stage1
                )
            )
        )
        self.intensification_cross_stage_triggered |= cross_stage_triggered
        cross_stage_reserve = (
            min(
                max(0, cross_stage_limit),
                max(0, remaining_after_priority - 32),
            )
            if cross_stage_triggered
            else 0
        )
        reconstruction_allowance = max(
            0,
            remaining_after_priority - cross_stage_reserve - 32,
        )
        reconstruction = critical_window_reconstruction_descent(
            self.instance,
            priority.source,
            self.budget,
            reconstruction_allowance,
            critical_batch_count=critical_batches,
        )
        cross_stage_allowance = min(
            cross_stage_reserve,
            max(
                0,
                allowance
                - analytical.evaluations
                - elimination.evaluations
                - priority.evaluations
                - reconstruction.evaluations
                - 32,
            ),
        )
        cross_stage = cross_stage_alignment_descent(
            self.instance,
            reconstruction.source,
            self.budget,
            cross_stage_allowance,
            critical_batch_count=critical_batches,
        )
        formation_allowance = max(
            0,
            allowance
            - analytical.evaluations
            - elimination.evaluations
            - priority.evaluations
            - reconstruction.evaluations
            - cross_stage.evaluations,
        )
        # Preserve a tiny tail for a final full-list priority repair because
        # formation/reconstruction creates new batch signatures.
        formation_allowance = max(0, formation_allowance - 32)
        formation = critical_formation_descent(
            self.instance,
            cross_stage.source,
            self.budget,
            formation_allowance,
            critical_batch_count=critical_batches,
        )
        final_analytical = analytical_priority_descent(
            self.instance,
            formation.source,
            self.budget,
            max(
                0,
                allowance
                - analytical.evaluations
                - elimination.evaluations
                - priority.evaluations
                - reconstruction.evaluations
                - cross_stage.evaluations
                - formation.evaluations,
            ),
        )
        results = (
            analytical,
            elimination,
            priority,
            reconstruction,
            cross_stage,
            formation,
            final_analytical,
        )
        self.intensification_evaluations += sum(
            result.evaluations for result in results
        )
        self.intensification_passes += sum(result.passes for result in results)
        self.intensification_accepted_moves += sum(
            result.accepted_moves for result in results
        )
        self.intensification_stage1_evaluations += sum(
            result.stage1_evaluations for result in results
        )
        self.intensification_stage2_evaluations += sum(
            result.stage2_evaluations for result in results
        )
        self.intensification_analytical_evaluations += (
            analytical.evaluations + final_analytical.evaluations
        )
        self.intensification_analytical_gain += (
            analytical.objective_gain + final_analytical.objective_gain
        )
        self.intensification_elimination_evaluations += elimination.evaluations
        self.intensification_elimination_gain += elimination.objective_gain
        self.intensification_priority_evaluations += priority.evaluations
        self.intensification_priority_gain += priority.objective_gain
        self.intensification_reconstruction_evaluations += (
            reconstruction.evaluations
        )
        self.intensification_reconstruction_gain += reconstruction.objective_gain
        self.intensification_cross_stage_evaluations += cross_stage.evaluations
        self.intensification_cross_stage_gain += cross_stage.objective_gain
        self.intensification_formation_evaluations += formation.evaluations
        self.intensification_formation_gain += formation.objective_gain
        final_source = final_analytical.source
        self.intensification_objective_gain += (
            incumbent.objective - final_source.objective
        )
        if final_source.objective >= incumbent.objective:
            return
        self.archive.offer(final_source)
        # Keep the intensified complete solution in the live population.  If
        # its original source is still present, update it; otherwise replace
        # only the current worst source and retain population cardinality.
        incumbent_index = next(
            (
                index
                for index, source in enumerate(population)
                if source.solution.canonical_signature
                == incumbent.solution.canonical_signature
            ),
            None,
        )
        target = (
            incumbent_index
            if incumbent_index is not None
            else max(range(len(population)), key=lambda index: population[index].objective)
        )
        population[target] = final_source

    def _mathematical_repair(
        self,
        population: list[FoodSource],
    ) -> bool:
        """Apply the configured exact neighborhoods once and update the archive."""

        objective_before = self.archive.best.objective
        repair_time_limit = self.config.mathematical_repair_seconds
        if self.config.time_limit_seconds is not None:
            repair_time_limit = min(
                repair_time_limit,
                max(
                    0.0,
                    self.config.time_limit_seconds - self.budget.elapsed,
                ),
            )

        if self.assignment_repair_applicable and repair_time_limit >= 0.01:
            repair = family_decomposed_assignment_repair(
                self.instance,
                self.archive.best,
                seconds_per_family=(
                    self.config.assignment_repair_seconds_per_family
                ),
                workers=self.config.assignment_repair_workers,
                iterations=self.config.assignment_repair_iterations,
                joint_max_jobs=self.config.assignment_repair_joint_max_jobs,
                seed=self.config.seed,
                allow_empty_batches=(
                    self.config.assignment_repair_allow_empty_batches
                ),
                max_total_seconds=repair_time_limit,
            )
            self.assignment_repair_rounds += 1
            self.assignment_repair_model_calls += repair.model_calls
            self.assignment_repair_feasible_models += repair.feasible_models
            self.assignment_repair_optimal_models += repair.optimal_models
            self.assignment_repair_accepted_iterations += (
                repair.accepted_iterations
            )
            self.assignment_repair_objective_gain += repair.objective_gain
            self.assignment_repair_runtime_seconds += repair.runtime_seconds
            self.assignment_repair_solver_runtime_seconds += (
                repair.solver_runtime_seconds
            )
            self.assignment_repair_max_solver_call_runtime_seconds = max(
                self.assignment_repair_max_solver_call_runtime_seconds,
                repair.max_solver_call_runtime_seconds,
            )
            self.assignment_repair_method_sequence += repair.method_sequence
            if repair.source.objective < self.archive.best.objective:
                self.archive.offer(repair.source)
                target = max(
                    range(len(population)),
                    key=lambda index: population[index].objective,
                )
                population[target] = repair.source

        return self.archive.best.objective < objective_before

    def _inject_adaptive_seed(self, population: list[FoodSource]) -> bool:
        """Inject the delayed construction seed only if it still dominates."""

        source = self.best_adaptive_seed
        if source is None or source.objective >= self.archive.best.objective:
            return False
        self.archive.offer(source)
        target = max(
            range(len(population)),
            key=lambda index: population[index].objective,
        )
        population[target] = FoodSource(
            source.solution,
            source.schedule,
            0,
            source.last_operator,
        )
        self.adaptive_seed_injected = True
        return True

    def run(self, progress: ProgressCallback | None = None) -> RunResult:
        population = self._initial_colony()
        history = [self.archive.best.objective]
        history_evaluations = [self.budget.evaluations]
        iterations = 0
        no_evaluation_iterations = 0
        termination_reason = "budget_exhausted"
        reserve = self._intensification_reserve()
        def search_step() -> bool:
            nonlocal iterations, no_evaluation_iterations, termination_reason
            iterations += 1
            evaluations_before = self.budget.evaluations
            # Both neighborhoods act on the same complete solutions.  Strict
            # alternation gives them equal opportunities under a fixed budget.
            role: Literal["formation", "sequencing"] = (
                "formation" if iterations % 2 else "sequencing"
            )
            self._phase(population, role)
            if self.budget.evaluations == evaluations_before:
                no_evaluation_iterations += 1
            else:
                no_evaluation_iterations = 0
                history.append(self.archive.best.objective)
                history_evaluations.append(self.budget.evaluations)
            if progress is not None:
                progress(
                    self.budget.evaluations,
                    self.config.max_evaluations,
                    self.archive.best.objective,
                    self.budget.elapsed,
                )
            if no_evaluation_iterations >= max(10, self.config.abandonment_limit):
                termination_reason = "no_evaluable_candidate"
                return False
            return True

        # Preserve a fixed share of either the evaluation or wall-clock budget
        # for critical-batch descent, then return unused reserve to the main
        # search if the descent neighborhood is exhausted early.
        while self._main_search_available(reserve):
            if not search_step():
                break

        if not self.budget.exhausted and termination_reason != "no_evaluable_candidate":
            allowance = reserve
            if self.config.max_evaluations is not None:
                allowance = min(
                    allowance,
                    max(
                        0,
                        self.config.max_evaluations
                        - self.budget.evaluations
                    ),
                )
            evaluations_before = self.budget.evaluations
            objective_before = self.archive.best.objective
            self._intensify(population, allowance)
            if self.budget.evaluations > evaluations_before:
                history.append(self.archive.best.objective)
                history_evaluations.append(self.budget.evaluations)
                if progress is not None:
                    progress(
                        self.budget.evaluations,
                        self.config.max_evaluations,
                        self.archive.best.objective,
                        self.budget.elapsed,
                    )
            if self.archive.best.objective < objective_before:
                no_evaluation_iterations = 0

        while (
            not self.budget.exhausted
            and termination_reason != "no_evaluable_candidate"
            and (
                self.config.max_evaluations is None
                or self.budget.evaluations < self.config.max_evaluations
            )
        ):
            if not search_step():
                break

        self._inject_adaptive_seed(population)
        objective_before_repair = self.archive.best.objective
        self._mathematical_repair(population)
        # The time protocol reserves a fraction for mathematical repair by
        # temporarily shortening the search budget.  If repair finishes early,
        # return every unused second to the metaheuristic instead of terminating
        # at the reservation boundary.
        if self.config.time_limit_seconds is not None:
            self.budget.time_limit_seconds = self.config.time_limit_seconds
        if self.archive.best.objective < objective_before_repair:
            history.append(self.archive.best.objective)
            history_evaluations.append(self.budget.evaluations)
            if progress is not None:
                progress(
                    self.budget.evaluations,
                    self.config.max_evaluations,
                    self.archive.best.objective,
                    self.budget.elapsed,
                )

        final_repair_reserve = 0.0
        if (
            self.assignment_repair_applicable
            and self.config.time_limit_seconds is not None
        ):
            final_repair_reserve = min(
                self.config.mathematical_repair_seconds,
                0.10 * self.config.time_limit_seconds,
            )
        self.final_repair_reserved_seconds = final_repair_reserve
        no_evaluation_iterations = 0
        termination_reason = "budget_exhausted"
        while (
            not self.budget.exhausted
            and (
                self.config.time_limit_seconds is None
                or self.budget.elapsed
                < self.config.time_limit_seconds - final_repair_reserve
            )
        ):
            if not search_step():
                break

        objective_before_final_repair = self.archive.best.objective
        if final_repair_reserve >= 0.01 and not self.budget.exhausted:
            final_repair_started = self.budget.elapsed
            self._mathematical_repair(population)
            self.final_repair_runtime_seconds = max(
                0.0,
                self.budget.elapsed - final_repair_started,
            )
            self.final_repair_objective_gain = max(
                0,
                objective_before_final_repair - self.archive.best.objective,
            )
            if self.archive.best.objective < objective_before_final_repair:
                history.append(self.archive.best.objective)
                history_evaluations.append(self.budget.evaluations)
                if progress is not None:
                    progress(
                        self.budget.evaluations,
                        self.config.max_evaluations,
                        self.archive.best.objective,
                        self.budget.elapsed,
                    )

            # CP-SAT commonly finishes well before its conservative reserve,
            # especially on J100/J250.  Continue the same predeclared BFC/BSC
            # search with the remaining wall-clock budget.  The elite archive
            # retains the repaired incumbent, so this continuation cannot
            # degrade the returned solution.
            no_evaluation_iterations = 0
            termination_reason = "budget_exhausted"
            post_search_started = self.budget.elapsed
            post_search_evaluations = self.budget.evaluations
            post_search_objective = self.archive.best.objective
            while not self.budget.exhausted:
                if not search_step():
                    break
            self.post_final_repair_search_seconds = max(
                0.0,
                self.budget.elapsed - post_search_started,
            )
            self.post_final_repair_search_evaluations = max(
                0,
                self.budget.evaluations - post_search_evaluations,
            )
            self.post_final_repair_search_objective_gain = max(
                0,
                post_search_objective - self.archive.best.objective,
            )

        best = self.archive.best
        final_dispatch = decode_structure(
            self.instance,
            best.solution,
            algorithm=self.schedule_algorithm,
            strategic_idle_best_response=True,
        )
        final_schedule = (
            final_dispatch
            if final_dispatch.objective < best.schedule.objective
            else best.schedule
        )
        explicit_realization = bool(
            final_schedule.metadata.get("assignment_repair_explicit", False)
        )
        final_signatures = {
            source.solution.canonical_signature for source in population
        }
        return RunResult(
            final_schedule,
            self.budget.evaluations,
            self.budget.elapsed,
            iterations,
            tuple(history),
            {
                "architecture": {
                    "algorithm": self.schedule_algorithm,
                    "representation": "direct_two_stage_batch_structure",
                    "population": "single_joint_colony",
                    "construction": (
                        "release_congestion_dynamic_stage2_partition_"
                        "with_delayed_dominance_gate"
                    ),
                    "formation_neighborhood": "BFC",
                    "sequencing_neighborhood": "BSC",
                    "role_schedule": "strict_alternation",
                    "cross_colony_cooperation": "removed",
                    "archive": "diverse_elite_archive",
                    "decoder": "complete_integrated_two_stage_decoder",
                    "stage2_dispatch": (
                        "dominance_screened_non_delay_or_strategic_idle"
                    ),
                    "primary_objective": "sum_stage2_completion_times",
                    "bottleneck_aware": self.config.bottleneck_aware,
                    "intensification": (
                        "critical_reconstruction_and_cross_stage_alignment"
                        if self.config.intensification_enabled
                        else "disabled"
                    ),
                    "assignment_repair": (
                        "family_decomposed_joint_and_stage2_slot_reassignment"
                        if self.config.assignment_repair_enabled
                        else "disabled"
                    ),
                },
                "config": asdict(self.config),
                "operator_selections": self.operator_selections,
                "operator_calls": self.operator_calls,
                "operator_improvements": self.operator_improvements,
                "operator_noops": self.operator_noops,
                "role_calls": self.role_calls,
                "role_improvements": self.role_improvements,
                "routing_stage_counts": self.routing_stage_counts,
                "scout_attempts": self.scout_attempts,
                "scout_changes": self.scout_changes,
                "scout_noops": self.scout_noops,
                "intensification_evaluations": self.intensification_evaluations,
                "intensification_passes": self.intensification_passes,
                "intensification_accepted_moves": (
                    self.intensification_accepted_moves
                ),
                "intensification_stage1_evaluations": (
                    self.intensification_stage1_evaluations
                ),
                "intensification_stage2_evaluations": (
                    self.intensification_stage2_evaluations
                ),
                "intensification_objective_gain": (
                    self.intensification_objective_gain
                ),
                "intensification_analytical_evaluations": (
                    self.intensification_analytical_evaluations
                ),
                "intensification_analytical_gain": (
                    self.intensification_analytical_gain
                ),
                "intensification_elimination_evaluations": (
                    self.intensification_elimination_evaluations
                ),
                "intensification_elimination_gain": (
                    self.intensification_elimination_gain
                ),
                "intensification_priority_evaluations": (
                    self.intensification_priority_evaluations
                ),
                "intensification_priority_gain": (
                    self.intensification_priority_gain
                ),
                "intensification_reconstruction_evaluations": (
                    self.intensification_reconstruction_evaluations
                ),
                "intensification_reconstruction_gain": (
                    self.intensification_reconstruction_gain
                ),
                "intensification_cross_stage_evaluations": (
                    self.intensification_cross_stage_evaluations
                ),
                "intensification_cross_stage_gain": (
                    self.intensification_cross_stage_gain
                ),
                "intensification_cross_stage_triggered": (
                    self.intensification_cross_stage_triggered
                ),
                "intensification_formation_evaluations": (
                    self.intensification_formation_evaluations
                ),
                "intensification_formation_gain": (
                    self.intensification_formation_gain
                ),
                "assignment_repair_model_calls": (
                    self.assignment_repair_model_calls
                ),
                "assignment_repair_feasible_models": (
                    self.assignment_repair_feasible_models
                ),
                "assignment_repair_optimal_models": (
                    self.assignment_repair_optimal_models
                ),
                "assignment_repair_accepted_iterations": (
                    self.assignment_repair_accepted_iterations
                ),
                "assignment_repair_objective_gain": (
                    self.assignment_repair_objective_gain
                ),
                "assignment_repair_runtime_seconds": (
                    self.assignment_repair_runtime_seconds
                ),
                "assignment_repair_solver_runtime_seconds": (
                    self.assignment_repair_solver_runtime_seconds
                ),
                "assignment_repair_max_solver_call_runtime_seconds": (
                    self.assignment_repair_max_solver_call_runtime_seconds
                ),
                "assignment_repair_method_sequence": list(
                    self.assignment_repair_method_sequence
                ),
                "assignment_repair_rounds": self.assignment_repair_rounds,
                "final_repair_reserved_seconds": (
                    self.final_repair_reserved_seconds
                ),
                "final_repair_runtime_seconds": (
                    self.final_repair_runtime_seconds
                ),
                "final_repair_objective_gain": (
                    self.final_repair_objective_gain
                ),
                "post_final_repair_search_seconds": (
                    self.post_final_repair_search_seconds
                ),
                "post_final_repair_search_evaluations": (
                    self.post_final_repair_search_evaluations
                ),
                "post_final_repair_search_objective_gain": (
                    self.post_final_repair_search_objective_gain
                ),
                "adaptive_seed_evaluations": self.adaptive_seed_evaluations,
                "adaptive_seed_improvements": self.adaptive_seed_improvements,
                "adaptive_seed_best_gain": self.adaptive_seed_best_gain,
                "adaptive_seed_injected": self.adaptive_seed_injected,
                "initial_population_size": self.initial_population_size,
                "effective_colony_size": self.effective_colony_size,
                "effective_archive_capacity": self.effective_archive_size,
                "assignment_repair_applicable": self.assignment_repair_applicable,
                "initial_unique_sources": self.initial_unique_sources,
                "final_unique_sources": len(final_signatures),
                "final_formation_role_diversity": _role_diversity(
                    self.instance,
                    population,
                    "formation",
                ),
                "final_sequencing_role_diversity": _role_diversity(
                    self.instance,
                    population,
                    "sequencing",
                ),
                "archive_size": len(self.archive.entries),
                "archive_descriptor_cache": len(self.archive.descriptors),
                "archive_distance_cache": len(self.archive.distances),
                "evaluation_requests": self.budget.evaluations,
                "decode_calls": self.budget.decode_calls,
                "decode_cache_hits": self.budget.cache_hits,
                "decode_cache_evictions": self.budget.cache_evictions,
                "decode_cache_size": self.budget.cache_size,
                "decode_cache_capacity": self.budget.cache_capacity,
                "cpu_runtime_seconds": self.budget.process_elapsed,
                "wall_runtime_seconds": self.budget.wall_elapsed,
                "termination_reason": termination_reason,
                "solution": best.solution.to_dict(),
                "final_schedule_realization": (
                    "matheuristic_explicit"
                    if explicit_realization
                    else "integrated_decoder"
                ),
                "stage2_dispatch_best_response": final_schedule.metadata.get(
                    "stage2_dispatch_best_response",
                    "explicit_matheuristic",
                ),
                "explicit_batches_stage1": (
                    [asdict(batch) for batch in best.schedule.batches_stage1]
                    if explicit_realization
                    else []
                ),
                "explicit_batches_stage2": (
                    [asdict(batch) for batch in best.schedule.batches_stage2]
                    if explicit_realization
                    else []
                ),
            },
            tuple(history_evaluations),
        )


def solve_ibscabc(
    instance: ProblemInstance,
    config: SearchConfig | None = None,
    progress: ProgressCallback | None = None,
) -> RunResult:
    return IBSCABC(instance, config).run(progress)
