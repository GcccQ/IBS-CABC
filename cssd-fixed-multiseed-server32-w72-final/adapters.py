"""Call existing solvers unchanged and normalize their final metrics."""
from __future__ import annotations

import bootstrap
import time
from dataclasses import asdict
from pathlib import Path

from contracts import IBSCABC_DIAGNOSTIC_FIELDS, validate_result


def run_task(task):
    from cssd_ibscabc.io.loader import load_instance
    from cssd_ibscabc.bounds import published_jobwise_lower_bound
    instance = load_instance(task["instance_path"])
    if instance.source_sha256 != task["instance_sha256"]:
        raise ValueError("instance changed after task creation")
    algorithm, parameters = task["algorithm"], dict(task["parameters"])
    if "delta_seed_pairs" in parameters:
        parameters["delta_seed_pairs"] = tuple(tuple(pair) for pair in parameters["delta_seed_pairs"])
    if algorithm == "kraul_ga":
        from kraul_dbd_reproduction.model import load_instance as load_author
        from kraul_dbd_reproduction.solvers import solve_kraul_ga
        author = load_author(task["instance_path"])
        started, cpu_started = time.perf_counter(), time.process_time()
        raw = solve_kraul_ga(author, seed=task["seed"], time_limit_seconds=task["time_limit_seconds"], max_evaluations=task["max_evaluations"], profile=parameters["profile"], overrides=parameters["overrides"])
        wall, cpu = time.perf_counter() - started, time.process_time() - cpu_started
        metrics = {"objective": raw["objective"], "average_time_in_system": raw["average_time_in_system"], "makespan": max(raw["schedule"]["completion_stage2"]), "utilization_stage1_percent": 100 * raw["utilization_stage1"], "utilization_stage2_percent": 100 * raw["utilization_stage2"], "waiting_stage1": raw["average_waiting_stage1"], "waiting_stage2": raw["average_waiting_stage2"]}
        evidence = raw
        solver_reported_runtime = float(raw["runtime_seconds"])
        diagnostics = raw.get("algorithm_diagnostics", {})
    else:
        from cssd_ibscabc.core.feasibility import validate_schedule
        if algorithm == "ibscabc":
            from cssd_ibscabc.algorithms import solve_ibscabc
            from cssd_ibscabc.algorithms.common import SearchConfig
            cfg = SearchConfig(**parameters, seed=task["seed"], max_evaluations=task["max_evaluations"], time_limit_seconds=task["time_limit_seconds"])
            solver = solve_ibscabc
        else:
            from cssd_baselines.common import BaselineConfig
            from cssd_baselines.alns import solve_alns
            from cssd_baselines.tabu_search import solve_tabu_search
            cfg = BaselineConfig(**parameters, seed=task["seed"], max_evaluations=task["max_evaluations"], time_limit_seconds=task["time_limit_seconds"])
            solver = {"alns": solve_alns, "ts": solve_tabu_search}[algorithm]
        started, cpu_started = time.perf_counter(), time.process_time()
        raw = solver(instance, cfg)
        wall, cpu = time.perf_counter() - started, time.process_time() - cpu_started
        validate_schedule(instance, raw.schedule)
        m = raw.schedule.metrics
        metrics = {"objective": m.total_completion_time, "average_time_in_system": m.average_time_in_system, "makespan": m.makespan, "utilization_stage1_percent": 100 * m.utilization_stage1, "utilization_stage2_percent": 100 * m.utilization_stage2, "waiting_stage1": m.waiting_stage1, "waiting_stage2": m.waiting_stage2}
        evidence = {"evaluations": raw.evaluations, "iterations": raw.iterations, "budget_elapsed_seconds": raw.runtime_seconds, "history": list(raw.history), "history_evaluations": list(raw.history_evaluations), "diagnostics": raw.diagnostics, "schedule": {"batches_stage1": [asdict(b) for b in raw.schedule.batches_stage1], "batches_stage2": [asdict(b) for b in raw.schedule.batches_stage2], "completion_stage1": list(raw.schedule.completion_stage1), "completion_stage2": list(raw.schedule.completion_stage2)}}
        solver_reported_runtime = float(raw.runtime_seconds)
        diagnostics = raw.diagnostics
    lb = published_jobwise_lower_bound(instance)
    nominal = task["time_limit_seconds"]
    result = {
        **task,
        **metrics,
        "status": "success",
        "feasible": True,
        "runtime_seconds": wall,
        "solver_reported_runtime_seconds": solver_reported_runtime,
        "process_cpu_seconds": cpu,
        "runtime_overrun_seconds": max(0.0, wall - nominal) if nominal is not None else 0.0,
        "runtime_underrun_seconds": max(0.0, nominal - wall) if nominal is not None else 0.0,
        "termination_reason": diagnostics.get("termination_reason"),
        "lower_bound": lb,
        "gap_lb_percent": 100 * (metrics["objective"] - lb) / metrics["objective"],
        "evidence": evidence,
    }
    if algorithm == "ibscabc":
        result.update(
            {
                field: diagnostics[field]
                for field in IBSCABC_DIAGNOSTIC_FIELDS
            }
        )
        result["assignment_repair_time_share_percent"] = (
            100.0 * diagnostics["assignment_repair_runtime_seconds"] / wall
            if wall > 0
            else 0.0
        )
    validate_result(result, task)
    return result
