from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import queue
import sys
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import yaml

from . import bridge
from .alns import solve_alns
from .common import BaselineConfig, DEFAULT_DELTA_PAIRS
from .convergence import write_convergence_reports
from .reporting import print_summary, write_four_algorithm_reports, write_reports
from .standard_abc import solve_standard_abc
from .tabu_search import solve_tabu_search

from cssd_ibscabc.core.feasibility import validate_schedule
from cssd_ibscabc.experiments.protocols import published_time_limit
from cssd_ibscabc.io.loader import load_instance


LABELS = {
    "standard_abc": "Standard ABC",
    "alns": "ALNS",
    "ts": "TS",
}


class _AsyncProgressPrinter:
    """Keep console selection/QuickEdit from blocking result collection."""

    def __init__(self) -> None:
        self._messages: queue.Queue[str | None] = queue.Queue(maxsize=1024)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            message = self._messages.get()
            if message is None:
                return
            print(message, flush=True)

    def emit(self, message: str) -> None:
        try:
            self._messages.put_nowait(message)
        except queue.Full:
            # The JSON progress file remains authoritative if a paused console
            # prevents the background printer from draining its queue.
            pass

    def close(self) -> None:
        try:
            self._messages.put_nowait(None)
        except queue.Full:
            return
        self._thread.join(timeout=2.0)


def _stop_process_pool(executor: ProcessPoolExecutor, futures: object) -> None:
    """Cancel queued work and terminate active workers after Ctrl+C."""
    for future in futures:
        future.cancel()

    terminate_workers = getattr(executor, "terminate_workers", None)
    if terminate_workers is not None:  # Python 3.14+
        terminate_workers()
        return

    # Python <=3.13 has no public immediate-stop API. Capture this executor's
    # workers before shutdown clears the internal mapping, then stop them. The
    # parent alone writes JSONL, so completed rows remain safe for --resume.
    processes = tuple(getattr(executor, "_processes", {}).values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    executor.shutdown(wait=True, cancel_futures=True)


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    algorithm: str
    instance_path: str
    jobs: int
    seed: int
    time_limit_seconds: float | None
    max_evaluations: int | None
    config: Mapping[str, object]


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_file(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _hash_python_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py"), key=lambda item: str(item).lower()):
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _strict_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"required formal result file is missing: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _split_dataset_sha256(data_root: Path, split_path: Path) -> tuple[str, int]:
    paths = sorted(
        (
            (data_root / relative.strip()).resolve()
            for relative in split_path.read_text(encoding="utf-8-sig").splitlines()
            if relative.strip()
        ),
        key=str,
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(data_root)).replace("\\", "/").encode())
        digest.update(path.read_bytes())
    return digest.hexdigest(), len(paths)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_results(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = path.read_text(encoding="utf-8")
    lines = payload.splitlines(keepends=True)
    rows: list[dict[str, object]] = []
    valid_character_count = 0
    for index, line in enumerate(lines):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise
                # A forcibly closed terminal can interrupt the final JSONL
                # append. Preserve every complete row and discard only that
                # incomplete tail so --resume can append cleanly.
                path.write_text(
                    payload[:valid_character_count], encoding="utf-8"
                )
                break
        valid_character_count += len(line)
    return rows


def _run_task(task: Task) -> dict[str, object]:
    # These variables are also set in common.py; repeat in spawned workers so
    # the one-thread contract is explicit before optional numeric libraries load.
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"

    instance = load_instance(task.instance_path)
    parameters = dict(task.config)
    parameters.update(
        seed=task.seed,
        time_limit_seconds=task.time_limit_seconds,
        max_evaluations=task.max_evaluations,
    )
    config = BaselineConfig(**parameters)
    if task.algorithm == "standard_abc":
        result = solve_standard_abc(instance, config)
    elif task.algorithm == "alns":
        result = solve_alns(instance, config)
    elif task.algorithm == "ts":
        result = solve_tabu_search(instance, config)
    else:
        raise KeyError(f"unknown algorithm: {task.algorithm}")
    validate_schedule(instance, result.schedule)
    metrics = result.schedule.metrics
    return {
        "task_id": task.task_id,
        "algorithm": task.algorithm,
        "algorithm_label": LABELS[task.algorithm],
        "instance": instance.name,
        "instance_path": str(instance.source),
        "instance_sha256": instance.source_sha256,
        "jobs": instance.job_count,
        "families": len(instance.active_families),
        "machines_stage1": instance.machines_stage1,
        "machines_stage2": instance.machines_stage2,
        "seed": task.seed,
        "time_limit_seconds": task.time_limit_seconds,
        "max_evaluations": task.max_evaluations,
        "evaluations": result.evaluations,
        "runtime_seconds": result.runtime_seconds,
        "process_cpu_runtime_seconds": result.diagnostics.get(
            "process_cpu_runtime_seconds", ""
        ),
        "iterations": result.iterations,
        "feasible": True,
        "total_completion_time": metrics.total_completion_time,
        "average_time_in_system": metrics.average_time_in_system,
        "c_max": metrics.makespan,
        "utilization_stage1": metrics.utilization_stage1,
        "utilization_stage2": metrics.utilization_stage2,
        "utilization_stage1_percent": 100.0 * metrics.utilization_stage1,
        "utilization_stage2_percent": 100.0 * metrics.utilization_stage2,
        "history": list(result.history),
        "history_evaluations": list(result.history_evaluations),
        "diagnostics": result.diagnostics,
    }


def _alignment(
    reference_manifest: Mapping[str, object],
    config: Mapping[str, object],
) -> tuple[dict[str, object], tuple[tuple[float, float], ...]]:
    experiment = reference_manifest["config"]["experiment"]  # type: ignore[index]
    algorithm = experiment["algorithms"][0]  # type: ignore[index]
    parameters = algorithm["parameters"]  # type: ignore[index]
    common = config["common"]  # type: ignore[index]
    baseline_protocol = config["experiment"]["protocol"]  # type: ignore[index]
    reference_protocols = experiment["protocols"]  # type: ignore[index]
    if len(reference_protocols) != 1:
        raise ValueError("IBS-CABC formal manifest must contain exactly one protocol")
    reference_protocol = reference_protocols[0]
    expected_cache_capacity = max(
        64,
        4 * int(parameters["colony_size"])
        + 2 * int(parameters["archive_size"]),
    )
    checks = {
        "seed_set": list(config["experiment"]["seeds"]) == list(experiment["seeds"]),  # type: ignore[index]
        "external_workers": int(config["experiment"]["workers"])  # type: ignore[index]
        == int(reference_manifest.get("workers", 8)),
        "population_size_N": int(common["population_size"]) == int(parameters["colony_size"]),
        "stagnation_limit": int(common["stagnation_limit"])
        == int(parameters["abandonment_limit"]),
        "candidate_attempts": int(common["candidate_attempts"])
        == int(parameters["operator_attempts"]),
        "decoder_cache_capacity": int(common["cache_capacity"])
        == expected_cache_capacity,
        "time_protocol_name": baseline_protocol["name"]
        == reference_protocol["name"],
        "time_protocol_mode": baseline_protocol["mode"]
        == reference_protocol["mode"],
        "time_protocol_tau": float(baseline_protocol.get("tau", 1.0))
        == float(reference_protocol.get("tau", 1.0)),
        "objective": "sum_stage2_completion_times",
        "decoder": "shared_complete_integrated_two_stage_decoder",
        "single_thread_per_task": True,
    }
    boolean_checks = [value for value in checks.values() if isinstance(value, bool)]
    if not all(boolean_checks):
        failed = [name for name, value in checks.items() if value is False]
        raise ValueError(f"IBS-CABC alignment check failed: {', '.join(failed)}")
    delta_pairs = tuple(tuple(map(float, pair)) for pair in parameters["delta_seed_pairs"])
    if not delta_pairs:
        delta_pairs = DEFAULT_DELTA_PAIRS
    return checks, delta_pairs


def _tasks(
    config: Mapping[str, object],
    *,
    algorithms_override: tuple[str, ...] | None,
    max_instances_per_size: int | None,
    max_evaluations: int | None,
    time_scale: float,
    delta_pairs: tuple[tuple[float, float], ...],
) -> list[Task]:
    root = bridge.BASELINE_ROOT
    experiment = config["experiment"]  # type: ignore[index]
    data_root = _resolve(root, str(experiment["data_root"]))
    split_values = experiment.get("splits") or [experiment["split"]]
    split_paths = [_resolve(root, str(value)) for value in split_values]
    algorithms = algorithms_override or tuple(map(str, experiment["algorithms"]))
    unknown = sorted(set(algorithms) - set(LABELS))
    if unknown:
        raise ValueError(f"unknown algorithms: {', '.join(unknown)}")
    seeds = tuple(map(int, experiment["seeds"]))
    tau = float(experiment["protocol"].get("tau", 1.0))
    common = dict(config["common"])  # type: ignore[arg-type]
    common["delta_seed_pairs"] = delta_pairs

    instances = []
    counts: dict[int, int] = {}
    seen_paths: set[Path] = set()
    for split_path in split_paths:
        for relative in split_path.read_text(encoding="utf-8-sig").splitlines():
            if not relative.strip():
                continue
            path = (data_root / relative.strip()).resolve()
            if path in seen_paths:
                raise ValueError(f"duplicate instance across split files: {path}")
            seen_paths.add(path)
            instance = load_instance(path)
            if max_instances_per_size is not None:
                used = counts.get(instance.job_count, 0)
                if used >= max_instances_per_size:
                    continue
                counts[instance.job_count] = used + 1
            instances.append(instance)

    expected_sizes = {25, 100, 250, 500, 1000}
    if {instance.job_count for instance in instances} != expected_sizes:
        raise ValueError("fixed-scale splits must contain J25/J100/J250/J500/J1000")
    if max_instances_per_size is None:
        actual = {size: sum(item.job_count == size for item in instances) for size in expected_sizes}
        if actual != {size: 20 for size in expected_sizes}:
            raise ValueError(f"formal split must contain 20 instances per size, got {actual}")

    tasks: list[Task] = []
    for instance in sorted(instances, key=lambda item: (-item.job_count, item.name)):
        for algorithm in algorithms:
            algorithm_config = dict(common)
            algorithm_config.update(config.get(algorithm, {}))  # type: ignore[arg-type]
            for seed in seeds:
                time_limit = (
                    None
                    if max_evaluations is not None
                    else published_time_limit(instance, tau) * time_scale
                )
                identity = (
                    f"{algorithm}|{instance.source_sha256}|{seed}|"
                    f"{time_limit}|{max_evaluations}|{json.dumps(algorithm_config, sort_keys=True)}"
                )
                task_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
                tasks.append(
                    Task(
                        task_id,
                        algorithm,
                        str(instance.source),
                        instance.job_count,
                        seed,
                        time_limit,
                        max_evaluations,
                        algorithm_config,
                    )
                )
    return tasks


def _audit_formal_evidence(
    config: Mapping[str, object],
    tasks: list[Task],
    reference_paths: list[Path],
    reference_manifests: list[Mapping[str, object]],
) -> dict[str, object]:
    experiment = config["experiment"]  # type: ignore[index]
    data_root = _resolve(bridge.BASELINE_ROOT, str(experiment["data_root"]))
    split_values = experiment.get("splits") or [experiment["split"]]
    split_paths = [
        _resolve(bridge.BASELINE_ROOT, str(value)) for value in split_values
    ]
    if len(split_paths) != len(reference_manifests):
        raise ValueError("each formal split must have one IBS-CABC reference manifest")

    dataset_checks = []
    for split_path, manifest in zip(split_paths, reference_manifests):
        digest, count = _split_dataset_sha256(data_root, split_path)
        dataset_checks.append(
            {
                "split": str(split_path),
                "instance_count": count,
                "sha256": digest,
                "matches_reference": digest == manifest.get("dataset_sha256"),
            }
        )

    reference_rows: list[dict[str, object]] = []
    for path, manifest in zip(reference_paths, reference_manifests):
        rows = _strict_jsonl(path.parent / "raw" / "results.jsonl")
        if int(manifest.get("task_count_successful", -1)) != len(rows):
            raise ValueError(f"incomplete IBS-CABC reference results: {path.parent}")
        if int(manifest.get("task_count_failed", -1)) != 0:
            raise ValueError(f"failed IBS-CABC reference tasks: {path.parent}")
        reference_rows.extend(rows)

    reference_pairs = {
        (str(row["instance_sha256"]), int(row["seed"])) for row in reference_rows
    }
    instance_hashes = {
        path: _hash_file(Path(path)) for path in {task.instance_path for task in tasks}
    }
    task_pairs = {
        (instance_hashes[task.instance_path], task.seed) for task in tasks
    }
    task_triples = {
        (instance_hashes[task.instance_path], task.seed, task.algorithm)
        for task in tasks
    }
    protocol_name = str(experiment["protocol"]["name"])
    checks: dict[str, object] = {
        "dataset_splits": dataset_checks,
        "dataset_hashes_match": all(
            bool(item["matches_reference"]) for item in dataset_checks
        ),
        "reference_result_count": len(reference_rows),
        "reference_results_complete": len(reference_rows) == 100,
        "reference_algorithm": {
            str(row.get("algorithm")) for row in reference_rows
        }
        == {"ibscabc"},
        "reference_protocol": {
            str(row.get("protocol")) for row in reference_rows
        }
        == {protocol_name},
        "instance_seed_pairs_match": task_pairs == reference_pairs,
        "baseline_task_count": len(tasks),
        "baseline_tasks_unique": len(task_triples) == len(tasks),
        "tasks_per_algorithm": {
            name: sum(task.algorithm == name for task in tasks) for name in LABELS
        },
    }
    boolean_checks = [value for value in checks.values() if isinstance(value, bool)]
    if not all(boolean_checks) or checks["tasks_per_algorithm"] != {
        name: 100 for name in LABELS
    }:
        raise ValueError(
            "formal IBS-CABC/baseline evidence alignment failed: "
            + json.dumps(checks, ensure_ascii=False, sort_keys=True)
        )
    return checks


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=bridge.BASELINE_ROOT / "configs" / "fixed_scale.yaml",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--algorithms", help="comma-separated subset")
    parser.add_argument("--max-instances-per-size", type=int)
    parser.add_argument("--max-evaluations", type=int)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    reference_values = experiment.get("ibscabc_manifests") or [
        experiment["ibscabc_manifest"]
    ]
    reference_paths = [
        _resolve(bridge.BASELINE_ROOT, str(value)) for value in reference_values
    ]
    reference_manifests = [
        json.loads(path.read_text(encoding="utf-8")) for path in reference_paths
    ]
    alignment_results = [
        _alignment(reference_manifest, config)
        for reference_manifest in reference_manifests
    ]
    alignment_checks = dict(alignment_results[0][0])
    alignment_checks["reference_manifest_count"] = len(reference_paths)
    delta_pairs = alignment_results[0][1]
    if any(pairs != delta_pairs for _, pairs in alignment_results[1:]):
        raise ValueError("IBS-CABC reference manifests use different delta_seed_pairs")
    algorithms_override = (
        tuple(item.strip() for item in args.algorithms.split(",") if item.strip())
        if args.algorithms
        else None
    )
    tasks = _tasks(
        config,
        algorithms_override=algorithms_override,
        max_instances_per_size=args.max_instances_per_size,
        max_evaluations=args.max_evaluations,
        time_scale=args.time_scale,
        delta_pairs=delta_pairs,
    )
    formal_four_algorithm_run = (
        len(tasks) == 300
        and {task.algorithm for task in tasks} == set(LABELS)
        and args.max_instances_per_size is None
        and args.max_evaluations is None
        and args.time_scale == 1.0
    )
    workers = args.workers or int(experiment["workers"])
    if workers <= 0:
        raise ValueError("workers must be positive")
    if args.time_scale <= 0:
        raise ValueError("time-scale must be positive")
    reference_worker_counts = {
        int(manifest.get("workers", 8)) for manifest in reference_manifests
    }
    alignment_checks["actual_external_workers"] = (
        len(reference_worker_counts) == 1 and workers in reference_worker_counts
    )
    current_shared_code_sha256 = _hash_python_tree(
        bridge.REFERENCE_SRC / "cssd_ibscabc"
    )
    reference_code_hashes = {
        str(manifest.get("code_sha256", "")) for manifest in reference_manifests
    }
    alignment_checks["shared_decoder_code"] = reference_code_hashes == {
        current_shared_code_sha256
    }
    if formal_four_algorithm_run and not alignment_checks["actual_external_workers"]:
        raise ValueError(
            "formal comparison must use the same external worker count as "
            f"IBS-CABC: expected {sorted(reference_worker_counts)}, got {workers}"
        )
    if formal_four_algorithm_run and not alignment_checks["shared_decoder_code"]:
        raise ValueError(
            "shared IBS-CABC representation/decoder code changed after the reference "
            "runs; start from the archived reference code before a formal comparison"
        )
    formal_evidence_checks = (
        _audit_formal_evidence(
            config,
            tasks,
            reference_paths,
            reference_manifests,
        )
        if formal_four_algorithm_run
        else None
    )

    print(f"tasks={len(tasks)} workers={workers} algorithms={sorted({t.algorithm for t in tasks})}")
    for jobs in (25, 100, 250, 500, 1000):
        limits = sorted(
            {task.time_limit_seconds for task in tasks if task.jobs == jobs and task.time_limit_seconds}
        )
        if limits:
            print(f"J{jobs} wall-clock budget={limits[0]:.2f}s per task")
    if args.dry_run:
        return 0

    config_sha256 = _hash_file(config_path)
    reference_manifest_sha256s = [_hash_file(path) for path in reference_paths]
    baseline_code_sha256 = _hash_python_tree(
        bridge.BASELINE_ROOT / "src" / "cssd_baselines"
    )
    task_manifest_sha256 = _hash_bytes(
        "\n".join(task.task_id for task in tasks).encode("ascii")
    )
    existing_manifest: dict[str, object] | None = None
    if args.resume:
        output_dir = args.resume.resolve()
        if not output_dir.is_dir():
            raise FileNotFoundError(f"resume directory does not exist: {output_dir}")
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"resume manifest does not exist: {manifest_path}")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        resume_checks = {
            "config_sha256": config_sha256,
            "task_manifest_sha256": task_manifest_sha256,
            "baseline_code_sha256": baseline_code_sha256,
            "shared_code_sha256": current_shared_code_sha256,
            "reference_manifest_sha256s": reference_manifest_sha256s,
            "workers": workers,
        }
        mismatched = [
            key
            for key, expected in resume_checks.items()
            if existing_manifest.get(key) != expected
        ]
        if mismatched:
            raise ValueError(
                "resume refused because the formal experiment configuration or "
                "code changed: "
                + ", ".join(mismatched)
                + ". Start a new output directory to avoid mixing evidence."
            )
    elif args.output:
        output_dir = args.output.resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"output directory is not empty; use --resume only for an exact run: {output_dir}"
            )
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = bridge.BASELINE_ROOT / "outputs" / f"fixed_scale_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    result_path = raw_dir / "results.jsonl"
    failure_path = raw_dir / "failures.jsonl"
    existing = _load_results(result_path)
    completed = {str(row["task_id"]) for row in existing}
    pending = [task for task in tasks if task.task_id not in completed]

    manifest = {
        "created_utc": (existing_manifest or {}).get(
            "created_utc", datetime.now(timezone.utc).isoformat()
        ),
        "resumed_utc": (
            datetime.now(timezone.utc).isoformat()
            if existing_manifest is not None
            else None
        ),
        "experiment": experiment["name"],
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "task_manifest_sha256": task_manifest_sha256,
        "baseline_code_sha256": baseline_code_sha256,
        "shared_code_sha256": current_shared_code_sha256,
        "reference_manifest": str(reference_paths[0]),
        "reference_manifests": [str(path) for path in reference_paths],
        "reference_manifest_sha256": _hash_file(reference_paths[0]),
        "reference_manifest_sha256s": reference_manifest_sha256s,
        "reference_dataset_sha256": reference_manifests[0].get("dataset_sha256"),
        "reference_dataset_sha256s": [
            item.get("dataset_sha256") for item in reference_manifests
        ],
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "workers": workers,
        "single_thread_per_task": True,
        "task_count": len(tasks),
        "alignment_checks": alignment_checks,
        "formal_evidence_checks": formal_evidence_checks,
        "algorithms": sorted({task.algorithm for task in tasks}),
        "seeds": sorted({task.seed for task in tasks}),
        "time_scale": args.time_scale,
        "max_evaluations_override": args.max_evaluations,
        "max_instances_per_size": args.max_instances_per_size,
        "common_parameters": config["common"],
        "method_parameters": {name: config.get(name, {}) for name in LABELS},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    started = time.monotonic()

    def write_progress(status: str, failed_count: int) -> None:
        successful = len(completed)
        progress = {
            "status": status,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_dir),
            "total_tasks": len(tasks),
            "successful_tasks": successful,
            "failed_tasks": failed_count,
            "pending_tasks": len(tasks) - successful - failed_count,
            "workers": workers,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        (output_dir / "progress.json").write_text(
            json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    write_progress("running", 0)
    print(f"output={output_dir}", flush=True)

    if not pending:
        baseline_summaries = write_reports(output_dir, existing)
        if formal_four_algorithm_run:
            reference_result_paths = [
                path.parent / "raw" / "results.jsonl" for path in reference_paths
            ]
            summaries = write_four_algorithm_reports(
                output_dir,
                existing,
                reference_result_paths,
            )
            convergence_audit = write_convergence_reports(
                output_dir, existing, reference_result_paths
            )
            print_summary(summaries, title="四算法固定规模正式实验汇总")
            print(f"convergence_status={convergence_audit['status']}")
        else:
            print_summary(baseline_summaries)
        print(f"output={output_dir}")
        return 0

    failures: list[dict[str, object]] = []
    total = len(tasks)
    printer = _AsyncProgressPrinter()
    with result_path.open("a", encoding="utf-8", buffering=1) as result_handle:
        executor = ProcessPoolExecutor(max_workers=workers)
        future_tasks = {executor.submit(_run_task, task): task for task in pending}
        unfinished = set(future_tasks)
        try:
            while unfinished:
                done, _ = wait(
                    unfinished,
                    timeout=30.0,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    write_progress("running", len(failures))
                    printer.emit(
                        f"[progress] {len(completed)}/{total} completed; "
                        f"{min(workers, len(unfinished))} workers active; "
                        f"elapsed={time.monotonic() - started:.0f}s"
                    )
                    continue
                for future in done:
                    unfinished.remove(future)
                    task = future_tasks[future]
                    try:
                        row = future.result()
                        result_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        result_handle.flush()
                        existing.append(row)
                        completed.add(task.task_id)
                        # Persist progress before touching the console.
                        write_progress("running", len(failures))
                        printer.emit(
                            f"[{len(completed)}/{total}] {row['algorithm_label']} "
                            f"J{row['jobs']} {row['instance']} "
                            f"ΣCj2={row['total_completion_time']} "
                            f"TIS={float(row['average_time_in_system']):.2f} "
                            f"Cmax={row['c_max']} "
                            f"U1={float(row['utilization_stage1_percent']):.2f}% "
                            f"U2={float(row['utilization_stage2_percent']):.2f}% "
                            f"runtime={float(row['runtime_seconds']):.2f}s"
                        )
                    except Exception as error:  # pragma: no cover - production failure path
                        failure = {
                            "task_id": task.task_id,
                            "algorithm": task.algorithm,
                            "instance_path": task.instance_path,
                            "seed": task.seed,
                            "error": repr(error),
                            "traceback": traceback.format_exc(),
                        }
                        failures.append(failure)
                        with failure_path.open("a", encoding="utf-8") as failure_handle:
                            failure_handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                        write_progress("running", len(failures))
                        printer.emit(f"FAILED {task.task_id}: {error!r}")
        except KeyboardInterrupt:
            write_progress("interrupted", len(failures))
            _stop_process_pool(executor, unfinished)
            print(
                f"Interrupted safely: {len(completed)}/{total} completed. "
                f"Resume with --resume \"{output_dir}\"",
                flush=True,
            )
            return 130
        else:
            executor.shutdown(wait=True)
        finally:
            printer.close()

    write_progress("failed" if failures else "finished", len(failures))

    baseline_summaries = write_reports(output_dir, existing)
    if failures:
        print_summary(baseline_summaries, title="未完成的三算法部分结果")
        print(f"output={output_dir}")
        raise RuntimeError(f"{len(failures)} task(s) failed; rerun with --resume {output_dir}")
    if formal_four_algorithm_run:
        reference_result_paths = [
            path.parent / "raw" / "results.jsonl" for path in reference_paths
        ]
        summaries = write_four_algorithm_reports(
            output_dir,
            existing,
            reference_result_paths,
        )
        convergence_audit = write_convergence_reports(
            output_dir,
            existing,
            reference_result_paths,
        )
        print_summary(summaries, title="四算法固定规模正式实验汇总")
        print(f"convergence_status={convergence_audit['status']}")
    else:
        print_summary(baseline_summaries)
    print(f"output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
