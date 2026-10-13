from __future__ import annotations

import bootstrap
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from pathlib import Path

import yaml
from cssd_ibscabc.io.loader import load_instance
from cssd_ibscabc.experiments.protocols import resolve_protocol
from kraul_dbd_reproduction.model import load_instance as load_author_instance

METHODS = {"alns": "ALNS", "ts": "TS", "kraul_ga": "Kraul-GA", "ibscabc": "IBS-CABC"}
SCALES = (100, 250, 500, 1000)
METRICS = ("objective", "average_time_in_system", "makespan", "utilization_stage1_percent", "utilization_stage2_percent", "waiting_stage1", "waiting_stage2", "runtime_seconds", "gap_lb_percent")
IBSCABC_DIAGNOSTIC_FIELDS = (
    "assignment_repair_applicable",
    "assignment_repair_model_calls",
    "assignment_repair_feasible_models",
    "assignment_repair_optimal_models",
    "assignment_repair_accepted_iterations",
    "assignment_repair_objective_gain",
    "assignment_repair_runtime_seconds",
    "assignment_repair_solver_runtime_seconds",
    "assignment_repair_max_solver_call_runtime_seconds",
    "assignment_repair_rounds",
    "final_repair_reserved_seconds",
    "final_repair_runtime_seconds",
    "final_repair_objective_gain",
    "post_final_repair_search_seconds",
    "post_final_repair_search_evaluations",
    "post_final_repair_search_objective_gain",
)


def digest(value):
    # JSON object keys are strings on disk. Normalize before sorting so a YAML
    # scale map keyed by integers has the same checksum after a JSON round trip.
    normalized = json.loads(json.dumps(value, allow_nan=False))
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def code_hashes():
    hashes = {}
    for project, root in bootstrap.SOURCE_ROOTS.items():
        h = hashlib.sha256()
        for path in sorted(root.rglob("*.py")):
            h.update(path.relative_to(root).as_posix().encode())
            h.update(path.read_bytes())
        hashes[project] = h.hexdigest()
    # Statistical presentation changes do not invalidate an interrupted solver run.
    for name in ("bootstrap.py", "contracts.py", "adapters.py", "run.py"):
        hashes[name] = hashlib.sha256((bootstrap.ROOT / name).read_bytes()).hexdigest()
    return hashes


def environment_contract():
    versions = {}
    for package in ("numpy", "scipy", "ortools", "PyYAML"):
        versions[package] = importlib.metadata.version(package)
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "packages": versions,
        "internal_threads": 1,
    }


def observed_environment():
    """Record host details without making cloud-host renames break resume."""
    return {
        "executable": sys.executable,
        "platform": platform.platform(),
        "host": platform.node(),
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
    }


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if "extends" in config:
        base = yaml.safe_load((Path(path).resolve().parent / config.pop("extends")).read_text(encoding="utf-8-sig"))
        config = base | config
    if config.get("mode") not in {"formal", "smoke"}:
        raise ValueError("mode must be formal or smoke")
    if set(config["algorithms"]) != set(METHODS):
        raise ValueError("configuration must define exactly the four stochastic algorithms")
    if not 1 <= int(config["workers"]) <= 128 or not 0 < float(config["heartbeat_seconds"]) <= 60:
        raise ValueError("invalid workers/heartbeat")
    if not 1 <= int(config.get("worker_recycle_tasks", 8)) <= 1000:
        raise ValueError("worker_recycle_tasks must be in [1, 1000]")
    config["worker_recycle_tasks"] = int(config.get("worker_recycle_tasks", 8))
    for scale, spec in config["scales"].items():
        if int(scale) not in SCALES:
            raise ValueError(f"unexpected scale: {scale}")
        seeds = spec["seeds"]
        if not seeds or any(type(s) is not int or s < 1 for s in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique positive integers")
        if len(seeds) < 2:
            raise ValueError("within-instance sample SD requires at least two runs")
        if config["mode"] == "formal":
            expected = list(range(1, 31 if int(scale) in (100, 250) else 21))
            if seeds != expected or spec["expected_instances"] != 20:
                raise ValueError("formal seeds/counts must follow the 30/30/20/20 protocol")
    if config["mode"] == "formal":
        if set(config["scales"]) != set(SCALES):
            raise ValueError("formal config must cover all four scales")
        if config["protocol"] != {"name": "paper_time_tau1", "mode": "official_time", "tau": 1.0}:
            raise ValueError("formal config must retain the original wall-clock protocol")
    if config["algorithms"]["ibscabc"].get("assignment_repair_workers") != 1:
        raise ValueError("CP-SAT must be single-threaded")
    if set(config["statistics"]["rpr_scope"]) != set(METHODS):
        raise ValueError("RPR denominator must cover the four named algorithms")
    for params in config["algorithms"].values():
        if set(params) & {"seed", "time_limit_seconds", "max_evaluations"}:
            raise ValueError("algorithm parameters cannot override the run protocol")
    return config


def prepare(config, scales=None, algorithms=None):
    selected_scales = sorted(scales or config["scales"])
    selected_algorithms = [a for a in METHODS if algorithms is None or a in algorithms]
    if not set(selected_scales) <= set(config["scales"]) or not selected_algorithms:
        raise ValueError("invalid task selection")
    root = (bootstrap.ROOT / config["data_root"]).resolve()
    instances, tasks = [], []
    seen = set()
    for scale in sorted(config["scales"]):
        spec = config["scales"][scale]
        split = (bootstrap.ROOT / spec["split"]).resolve()
        names = [line.strip() for line in split.read_text(encoding="utf-8-sig").splitlines() if line.strip() and not line.lstrip().startswith("#")]
        if len(names) != spec["expected_instances"]:
            raise ValueError(f"J{scale}: unexpected number of instances in {split}")
        for name in names:
            path = (root / name).resolve()
            if not path.is_relative_to(root):
                raise ValueError("instance path is outside data_root")
            instance, author = load_instance(path), load_author_instance(path)
            if instance.job_count != scale or instance.source_sha256 in seen:
                raise ValueError("duplicate content or wrong-scale instance")
            seen.add(instance.source_sha256)
            # Independent parsers must agree on every processed job field.
            core_rows = [[j.job_id, j.duration_stage1, j.duration_stage2, j.release, j.family, j.size_stage1, j.size_stage2] for j in instance.jobs]
            if core_rows != author.jobs.tolist():
                raise ValueError(f"parser disagreement: {name}")
            for field in ("machines_stage1", "machines_stage2", "capacity_stage1", "capacity_stage2"):
                if getattr(instance, field) != getattr(author, field):
                    raise ValueError(f"parser disagreement: {name}/{field}")
            budget = resolve_protocol(config["protocol"], instance)
            row = {"instance": instance.name, "instance_path": str(path), "relative_instance_path": path.relative_to(root).as_posix(), "instance_sha256": instance.source_sha256, "jobs": scale, "release_sum": sum(j.release for j in instance.jobs), "machines_stage1": instance.machines_stage1, "machines_stage2": instance.machines_stage2, "time_limit_seconds": budget.time_limit_seconds, "max_evaluations": budget.max_evaluations, "expected_seeds": spec["seeds"]}
            instances.append(row)
            if scale not in selected_scales:
                continue
            # Rotate methods at each seed to avoid running one method only at one time of day.
            for seed in spec["seeds"]:
                order = selected_algorithms[seed % len(selected_algorithms):] + selected_algorithms[:seed % len(selected_algorithms)]
                for algorithm in order:
                    task = {**row, "algorithm": algorithm, "algorithm_label": METHODS[algorithm], "seed": seed, "parameters": config["algorithms"][algorithm], "mode": config["mode"]}
                    task.pop("expected_seeds")
                    task["task_id"] = f"J{scale}_{instance.source_sha256[:16]}_{algorithm}_s{seed:02d}"
                    task["task_hash"] = digest(task)
                    tasks.append(task)
    core = {
        "schema": 2,
        "config": config,
        "instances": instances,
        "code_hashes": code_hashes(),
        "environment": environment_contract(),
    }
    experiment = {"core": core, "study_hash": digest(core), "selected_scales": selected_scales, "selected_algorithms": selected_algorithms, "task_count": len(tasks), "task_manifest_hash": digest(tasks)}
    experiment["run_contract"] = digest(experiment)
    experiment["observed_environment"] = observed_environment()
    return experiment, tasks


def validate_result(result, task):
    for key in ("task_id", "task_hash", "instance_sha256", "jobs", "algorithm", "seed", "time_limit_seconds", "mode"):
        if result.get(key) != task[key]:
            raise ValueError(f"checkpoint disagrees with task: {task['task_id']}/{key}")
    if result.get("status") != "success" or result.get("feasible") is not True:
        raise ValueError("checkpoint is not a feasible successful task")
    for field in METRICS:
        value = result.get(field)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid numeric metric: {field}")
    if result["objective"] <= 0:
        raise ValueError("objective must be positive")
    if not math.isclose(result["average_time_in_system"] * task["jobs"], result["objective"] - task["release_sum"], abs_tol=1e-6):
        raise ValueError("TIS/objective identity failed")
    if any(result[f"utilization_stage{k}_percent"] > 100.000001 for k in (1, 2)):
        raise ValueError("invalid utilization")
    for field in (
        "solver_reported_runtime_seconds",
        "process_cpu_seconds",
        "runtime_overrun_seconds",
        "runtime_underrun_seconds",
    ):
        value = result.get(field)
        if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid runtime diagnostic: {field}")
    if task["algorithm"] == "ibscabc":
        for field in IBSCABC_DIAGNOSTIC_FIELDS:
            if field not in result:
                raise ValueError(f"missing IBS-CABC diagnostic: {field}")
            value = result[field]
            if field == "assignment_repair_applicable":
                if not isinstance(value, bool):
                    raise ValueError(f"invalid IBS-CABC diagnostic: {field}")
            elif not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid IBS-CABC diagnostic: {field}")
        share = result.get("assignment_repair_time_share_percent")
        if not isinstance(share, (float, int)) or isinstance(share, bool) or not math.isfinite(share) or share < 0:
            raise ValueError("invalid IBS-CABC repair time share")


def save_checkpoint(root, result, task):
    validate_result(result, task)
    path = Path(root) / "checkpoints" / (task["task_id"] + ".json")
    if path.exists():
        raise ValueError(f"refusing to overwrite completed task: {path}")
    atomic_json(path, {"result": result, "sha256": digest(result)})


def load_checkpoints(root, tasks, include_evidence=False):
    expected = {t["task_id"]: t for t in tasks}
    rows = {}
    for path in sorted((Path(root) / "checkpoints").glob("*.json")):
        envelope = read_json(path)
        result = envelope["result"]
        task_id = result["task_id"]
        if task_id in rows or task_id not in expected or path.stem != task_id:
            raise ValueError(f"duplicate, unexpected, or misnamed checkpoint: {path}")
        if envelope["sha256"] != digest(result):
            raise ValueError(f"checkpoint checksum failed: {path}")
        validate_result(result, expected[task_id])
        # Schedules/history remain on disk; an 8,000-task resume/report must not
        # retain every detailed search trace in RAM at once.
        rows[task_id] = result if include_evidence else {k: v for k, v in result.items() if k != "evidence"}
    return rows
