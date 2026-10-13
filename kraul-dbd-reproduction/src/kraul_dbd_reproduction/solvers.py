from __future__ import annotations

import time
from dataclasses import asdict
from typing import Mapping

from .decoder import decode_permutation, validate_schedule
from .ga import config_from_profile, run_genetic_algorithm
from .metrics import schedule_metrics, schedule_payload
from .model import AuthorInstance


def solve_ffcbspt(instance: AuthorInstance, day_length_minutes: int = 1440) -> dict[str, object]:
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    # FCFS is the Notebook's ERD chromosome; its shared decoder performs
    # family-compatible first-fit batching followed by dynamic SPT dispatch.
    schedule = decode_permutation(instance, instance.erd_order)
    validate_schedule(instance, schedule)
    return {
        **schedule_metrics(instance, schedule, day_length_minutes=day_length_minutes),
        "schedule": schedule_payload(instance, schedule),
        "runtime_seconds": time.perf_counter() - wall_start,
        "process_cpu_seconds": time.process_time() - cpu_start,
        "evaluations": 1,
        "generations": 0,
        "restart_count": 0,
        "algorithm_diagnostics": {
            "decoder": "released_notebook",
            "initial_order": "ERD_FCFS",
            "batch_dispatch": "dynamic_SPT",
        },
    }


def solve_kraul_ga(
    instance: AuthorInstance,
    *,
    seed: int,
    time_limit_seconds: float,
    profile: str,
    overrides: Mapping[str, object] | None,
    day_length_minutes: int = 1440,
    max_evaluations: int | None = None,
) -> dict[str, object]:
    config = config_from_profile(profile, overrides)
    result = run_genetic_algorithm(
        instance,
        seed=seed,
        time_limit_seconds=time_limit_seconds,
        config=config,
        max_evaluations=max_evaluations,
    )
    return {
        **schedule_metrics(instance, result.schedule, day_length_minutes=day_length_minutes),
        "schedule": schedule_payload(instance, result.schedule),
        "runtime_seconds": result.runtime_seconds,
        "process_cpu_seconds": result.process_cpu_seconds,
        "evaluations": result.evaluations,
        "generations": result.generations,
        "restart_count": result.restart_count,
        "algorithm_diagnostics": {
            "profile": result.config.profile_name,
            "parameters": asdict(result.config),
            "last_generation_best": result.last_generation_best,
            "final_population_best": result.final_population_best,
            "best_seen_during_search": result.best_seen_during_search,
            "history": list(result.history),
            "incumbent_history": list(result.incumbent_history),
            "history_elapsed_seconds": list(result.history_elapsed_seconds),
            "history_evaluations": list(result.history_evaluations),
            "timing": dict(result.timing),
            "termination_reason": result.termination_reason,
            "nominal_time_limit_seconds": time_limit_seconds,
            "time_limit_overrun_seconds": max(0.0, result.runtime_seconds - time_limit_seconds),
        },
    }
