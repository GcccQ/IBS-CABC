from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import queue
import sys
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from . import bridge as _bridge  # noqa: F401
from .bridge import BASELINES_SRC, IBSCABC_SRC, PROJECT_ROOT, source_tree_sha256
from .config import FORMAL_DATASET_SHA256, ExperimentSpec, load_experiment_spec
from .solvers import solve, validate_parameters

from cssd_ibscabc.bounds import published_jobwise_lower_bound
from cssd_ibscabc.core.feasibility import validate_schedule
from cssd_ibscabc.core.entities import Batch, ProblemInstance
from cssd_ibscabc.core.objective import hourly_completed_devices, job_level_metric_moments
from cssd_ibscabc.experiments.protocols import published_time_limit
from cssd_ibscabc.io.loader import load_instance


HEARTBEAT_SECONDS = 15.0
_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class _AsyncProgressPrinter:
    """Prevent a paused Windows console from blocking result persistence."""

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
            # progress.json remains authoritative when console output is paused.
            pass

    def close(self) -> None:
        try:
            self._messages.put_nowait(None)
        except queue.Full:
            return
        self._thread.join(timeout=2.0)


@contextmanager
def _exclusive_run_lock(run: Path) -> Iterator[None]:
    """Reject concurrent writers targeting the same experiment directory."""

    lock_path = run / ".runner.lock"
    handle = lock_path.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:  # pragma: no cover - the formal experiment runs on Windows
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise RuntimeError(
            f"another runner is already writing this experiment directory: {run}"
        ) from error
    try:
        yield
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover - the formal experiment runs on Windows
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _configure_threads() -> None:
    for name in _THREAD_ENVIRONMENT:
        os.environ[name] = "1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hash(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _json_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _spec_payload(spec: ExperimentSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "contract": spec.contract,
        "data_root": str(spec.data_root),
        "split": str(spec.split),
        "expected_days": spec.expected_days,
        "expected_jobs": spec.expected_jobs,
        "first_day_seed": spec.first_day_seed,
        "last_day_seed": spec.last_day_seed,
        "machines_stage1": spec.machines_stage1,
        "machines_stage2": spec.machines_stage2,
        "day_length_minutes": spec.day_length_minutes,
        "seeds": list(spec.seeds),
        "workers": spec.workers,
        "protocol": asdict(spec.protocol),
        "algorithms": [
            {
                "name": item.name,
                "label": item.label,
                "parameters": dict(item.parameters),
            }
            for item in spec.algorithms
        ],
    }


def audit_instances(spec: ExperimentSpec) -> dict[str, object]:
    instances = [load_instance(path) for path in spec.instance_paths]
    counts = [instance.job_count for instance in instances]
    total_jobs = sum(counts)
    if spec.expected_jobs and total_jobs != spec.expected_jobs:
        raise ValueError(f"DBD split has {total_jobs} jobs, expected {spec.expected_jobs}")
    if {(instance.machines_stage1, instance.machines_stage2) for instance in instances} != {
        (spec.machines_stage1, spec.machines_stage2)
    }:
        raise ValueError(
            "daily instances do not use the configured "
            f"{spec.machines_stage1}/{spec.machines_stage2} machine setting"
        )
    if {(instance.capacity_stage1, instance.capacity_stage2) for instance in instances} != {
        (15, 11_000)
    }:
        raise ValueError("daily instances do not use the published 15/11000 capacities")
    dataset_sha256 = _tree_hash(spec.instance_paths, spec.data_root)
    if spec.is_formal and dataset_sha256 != FORMAL_DATASET_SHA256:
        raise ValueError(
            "formal proxy91 instance contents changed; start from the locked Real365 data"
        )
    sequence = [
        {
            "case_day_index": index,
            "relative_path": str(path.relative_to(spec.data_root)).replace("\\", "/"),
            "instance": instance.name,
            "instance_sha256": instance.source_sha256,
            "jobs": instance.job_count,
        }
        for index, (path, instance) in enumerate(zip(spec.instance_paths, instances))
    ]
    return {
        "days": len(instances),
        "total_jobs": total_jobs,
        "minimum_jobs": min(counts),
        "maximum_jobs": max(counts),
        "mean_jobs": total_jobs / len(instances),
        "dataset_sha256": dataset_sha256,
        "instance_sequence": sequence,
        "instance_sha256": {
            str(path.relative_to(spec.data_root)).replace("\\", "/"): instance.source_sha256
            for path, instance in zip(spec.instance_paths, instances)
        },
    }


def _task_id(task: Mapping[str, object]) -> str:
    stable = {key: value for key, value in task.items() if key != "task_id"}
    return _json_hash(stable)[:24]


def build_tasks(spec: ExperimentSpec) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for day_index, path in enumerate(spec.instance_paths):
        for algorithm in spec.algorithms:
            for seed in spec.seeds:
                task: dict[str, object] = {
                    "experiment": spec.name,
                    "contract": spec.contract,
                    "planning_mode": "DBD",
                    "case_day_index": day_index,
                    "day_length_minutes": spec.day_length_minutes,
                    "instance_path": str(path),
                    "algorithm": algorithm.name,
                    "algorithm_label": algorithm.label,
                    "parameters": dict(algorithm.parameters),
                    "seed": seed,
                    "protocol": asdict(spec.protocol),
                }
                task["task_id"] = _task_id(task)
                tasks.append(task)
    return tasks


def _budget_summary(
    spec: ExperimentSpec,
    audit: Mapping[str, object],
) -> dict[str, object]:
    sequence = audit.get("instance_sequence")
    job_counts = (
        [int(item["jobs"]) for item in sequence if isinstance(item, Mapping) and "jobs" in item]
        if isinstance(sequence, list)
        else []
    )
    task_multiplier = len(spec.algorithms) * len(spec.seeds)
    if spec.protocol.mode == "official_time":
        tau = float(spec.protocol.tau or 1.0)
        machine_sum = spec.machines_stage1 + spec.machines_stage2
        daily_seconds = [
            min((5000.0 + tau * jobs**2 * machine_sum / 2.0) / 1000.0, 1800.0)
            for jobs in job_counts
        ]
        per_algorithm_seed = sum(daily_seconds)
        all_tasks = per_algorithm_seed * task_multiplier
        return {
            "mode": spec.protocol.mode,
            "minimum_daily_seconds": round(min(daily_seconds, default=0.0), 6),
            "maximum_daily_seconds": round(max(daily_seconds, default=0.0), 6),
            "per_algorithm_seed_seconds": round(per_algorithm_seed, 6),
            "all_tasks_budget_seconds": round(all_tasks, 6),
            "all_tasks_budget_hours": round(all_tasks / 3600.0, 6),
            "ideal_parallel_hours_at_configured_workers": round(
                all_tasks / spec.workers / 3600.0,
                6,
            ),
        }
    if spec.protocol.mode == "evaluations":
        maximum = int(spec.protocol.max_evaluations or 0)
        return {
            "mode": spec.protocol.mode,
            "evaluations_per_task": maximum,
            "all_tasks_evaluation_limit": maximum * len(job_counts) * task_multiplier,
        }
    return {
        "mode": spec.protocol.mode,
        "seconds_per_task": float(spec.protocol.time_limit_seconds or 0.0),
    }


def _resolve_budget(
    protocol: Mapping[str, object],
    instance: ProblemInstance,
) -> tuple[int | None, float | None]:
    mode = str(protocol["mode"])
    if mode == "official_time":
        return None, published_time_limit(instance, float(protocol.get("tau", 1.0)))
    if mode == "evaluations":
        return int(protocol["max_evaluations"]), None
    if mode == "wall_clock":
        return None, float(protocol["time_limit_seconds"])
    raise ValueError(f"unsupported protocol mode: {mode!r}")


def _utilization_components(
    batches: tuple[Batch, ...],
    machines: int,
) -> dict[str, int | float]:
    processing_minutes = sum(batch.processing_time for batch in batches)
    if not batches:
        return {
            "processing_minutes": 0,
            "active_horizon_minutes": 0,
            "recomputed_utilization": 0.0,
        }
    active_horizon = max(batch.end for batch in batches) - min(batch.start for batch in batches)
    utilization = (
        processing_minutes / (machines * active_horizon)
        if machines > 0 and active_horizon > 0
        else 0.0
    )
    return {
        "processing_minutes": processing_minutes,
        "active_horizon_minutes": active_horizon,
        "recomputed_utilization": utilization,
    }


def run_one(task: Mapping[str, object]) -> dict[str, object]:
    """Run one independent day/algorithm/seed task."""

    _configure_threads()
    if str(task.get("planning_mode")) != "DBD":
        raise ValueError("this isolated runner accepts DBD tasks only")
    instance = load_instance(str(task["instance_path"]))
    raw_protocol = task["protocol"]
    if not isinstance(raw_protocol, Mapping):
        raise ValueError("task protocol must be a mapping")
    protocol = dict(raw_protocol)
    max_evaluations, time_limit_seconds = _resolve_budget(protocol, instance)
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    result = solve(
        str(task["algorithm"]),
        instance,
        seed=int(task["seed"]),
        max_evaluations=max_evaluations,
        time_limit_seconds=time_limit_seconds,
        parameters=dict(task.get("parameters", {})),
    )
    wall_seconds = time.perf_counter() - wall_started
    cpu_seconds = time.process_time() - cpu_started
    validate_schedule(instance, result.schedule)
    moments = {
        name: value.to_dict()
        for name, value in job_level_metric_moments(
            instance,
            result.schedule.batches_stage1,
            result.schedule.batches_stage2,
            result.schedule.completion_stage1,
            result.schedule.completion_stage2,
        ).items()
    }
    release_sum = sum(job.release for job in instance.jobs)
    flow_sum = float(moments["time_in_system"]["sum"])
    identity_rhs = result.schedule.objective - release_sum
    if abs(flow_sum - identity_rhs) > 1e-9:
        raise RuntimeError("sum(TIS) != sum(Cj2) - sum(rj)")
    bins = int(task["day_length_minutes"]) // 60
    arrivals = [0] * bins
    for job in instance.jobs:
        arrivals[(job.release % int(task["day_length_minutes"])) // 60] += 1
    day_index = int(task["case_day_index"])
    day_offset = day_index * int(task["day_length_minutes"])
    lower_bound = published_jobwise_lower_bound(instance)
    metrics = asdict(result.schedule.metrics)
    stage1_utilization = _utilization_components(
        result.schedule.batches_stage1,
        instance.machines_stage1,
    )
    stage2_utilization = _utilization_components(
        result.schedule.batches_stage2,
        instance.machines_stage2,
    )
    if not math.isclose(
        float(stage1_utilization["recomputed_utilization"]),
        result.schedule.metrics.utilization_stage1,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ) or not math.isclose(
        float(stage2_utilization["recomputed_utilization"]),
        result.schedule.metrics.utilization_stage2,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("persisted utilization components disagree with schedule metrics")
    return {
        "task_id": str(task["task_id"]),
        "experiment": str(task["experiment"]),
        "contract": str(task["contract"]),
        "planning_mode": "DBD",
        "case_day_index": day_index,
        "algorithm": str(task["algorithm"]),
        "algorithm_label": str(task["algorithm_label"]),
        "parameters": dict(task.get("parameters", {})),
        "seed": int(task["seed"]),
        "protocol": str(protocol["name"]),
        "protocol_mode": str(protocol["mode"]),
        "protocol_spec": protocol,
        "max_evaluations": max_evaluations,
        "time_limit_seconds": time_limit_seconds,
        "instance": instance.name,
        "instance_path": str(instance.source),
        "instance_sha256": instance.source_sha256,
        "jobs": instance.job_count,
        "families": len(instance.active_families),
        "machines_stage1": instance.machines_stage1,
        "machines_stage2": instance.machines_stage2,
        "capacity_stage1": instance.capacity_stage1,
        "capacity_stage2": instance.capacity_stage2,
        "day_length_minutes": int(task["day_length_minutes"]),
        "objective": result.schedule.objective,
        "calendar_objective": result.schedule.objective + day_offset * instance.job_count,
        "release_sum": release_sum,
        "lower_bound": lower_bound,
        "paper_gap_lb_percent": 100.0
        * (result.schedule.objective - lower_bound)
        / max(1, result.schedule.objective),
        "evaluations": result.evaluations,
        "iterations": result.iterations,
        "runtime_seconds": wall_seconds,
        "process_cpu_runtime_seconds": cpu_seconds,
        "budget_elapsed_seconds": result.runtime_seconds,
        "history": list(result.history),
        "history_evaluations": list(result.history_evaluations),
        "feasible": True,
        **metrics,
        "stage1_processing_minutes": stage1_utilization["processing_minutes"],
        "stage1_active_horizon_minutes": stage1_utilization["active_horizon_minutes"],
        "stage2_processing_minutes": stage2_utilization["processing_minutes"],
        "stage2_active_horizon_minutes": stage2_utilization["active_horizon_minutes"],
        "total_batch_cycles": result.schedule.metrics.total_batch_cycles,
        "metric_moments": moments,
        "arrivals_hourly": arrivals,
        "completed_stage1_hourly": list(
            hourly_completed_devices(
                result.schedule.batches_stage1,
                day_length_minutes=int(task["day_length_minutes"]),
            )
        ),
        "completed_stage2_hourly": list(
            hourly_completed_devices(
                result.schedule.batches_stage2,
                day_length_minutes=int(task["day_length_minutes"]),
            )
        ),
        "diagnostics": dict(result.diagnostics),
    }


def _read_successes(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    if not path.is_file():
        return rows
    payload = path.read_text(encoding="utf-8")
    lines = payload.splitlines(keepends=True)
    valid_character_count = 0
    repaired_tail = False
    for index, line in enumerate(lines):
        if line.strip():
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("JSONL records must be objects")
                task_id = str(value["task_id"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                if index != len(lines) - 1:
                    raise ValueError(f"invalid results JSONL line {index + 1}") from error
                # A forced process close can interrupt only the final append.
                # Preserve all fsynced complete rows and remove that partial tail.
                path.write_text(payload[:valid_character_count], encoding="utf-8")
                repaired_tail = True
                break
            if task_id in rows:
                raise ValueError(f"duplicate task_id in results JSONL: {task_id}")
            rows[task_id] = value
        valid_character_count += len(line)
    if not repaired_tail and payload and not payload.endswith(("\n", "\r")):
        with path.open("a", encoding="utf-8", newline="") as handle:
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    if repaired_tail:
        print(f"[resume] removed an incomplete final JSONL record from {path}", flush=True)
    return rows


def _validate_success_records(
    successes: Mapping[str, Mapping[str, object]],
    tasks: list[dict[str, object]],
    audit: Mapping[str, object],
) -> None:
    expected = {str(task["task_id"]): task for task in tasks}
    unknown = sorted(set(successes) - set(expected))
    if unknown:
        raise ValueError("results contain unknown task identifiers: " + ", ".join(unknown[:5]))
    sequence = audit.get("instance_sequence")
    if not isinstance(sequence, list):
        raise ValueError("dataset audit has no ordered instance sequence")
    for task_id, record in successes.items():
        task = expected[task_id]
        fields = (
            "experiment",
            "contract",
            "planning_mode",
            "case_day_index",
            "algorithm",
            "algorithm_label",
            "seed",
            "day_length_minutes",
        )
        mismatched = [field for field in fields if record.get(field) != task.get(field)]
        if record.get("parameters") != task.get("parameters"):
            mismatched.append("parameters")
        if record.get("protocol_spec") != task.get("protocol"):
            mismatched.append("protocol_spec")
        if (
            Path(str(record.get("instance_path", ""))).resolve()
            != Path(str(task["instance_path"])).resolve()
        ):
            mismatched.append("instance_path")
        day = int(task["case_day_index"])
        if not 0 <= day < len(sequence) or not isinstance(sequence[day], Mapping):
            mismatched.append("case_day_index")
        else:
            expected_instance = sequence[day]
            if record.get("instance_sha256") != expected_instance.get("instance_sha256"):
                mismatched.append("instance_sha256")
            if int(record.get("jobs", -1)) != int(expected_instance.get("jobs", -2)):
                mismatched.append("jobs")
        if mismatched:
            raise ValueError(
                f"stored result {task_id} does not match the current task contract: "
                + ", ".join(sorted(set(mismatched)))
            )


def _code_hashes() -> dict[str, str]:
    return {
        "dbd_runner": source_tree_sha256(PROJECT_ROOT / "src"),
        "ibscabc_dependency": source_tree_sha256(IBSCABC_SRC / "cssd_ibscabc"),
        "baselines_dependency": source_tree_sha256(BASELINES_SRC / "cssd_baselines"),
    }


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("PyYAML", "numpy", "scipy", "matplotlib", "ortools", "tqdm"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _new_run_directory(output_root: Path, name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = output_root.resolve() / f"{name}_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    (path / "raw").mkdir()
    return path


def _resolve_workers(spec: ExperimentSpec, workers: int | None) -> int:
    selected = spec.workers if workers is None else int(workers)
    if selected <= 0:
        raise ValueError("workers must be positive")
    if spec.is_formal and selected != spec.workers:
        raise ValueError(f"formal DBD execution requires exactly {spec.workers} external workers")
    return selected


def _progress_payload(
    total: int,
    successful: int,
    failures: int,
    *,
    state: str,
) -> dict[str, object]:
    return {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "state": state,
        "task_count_total": total,
        "task_count_successful": successful,
        "task_count_failed_attempts": failures,
        "task_count_pending": total - successful,
    }


def _terminate_pool(
    pool: ProcessPoolExecutor,
    futures: Iterable[Future[object]],
) -> None:
    """Stop only this runner's children after a parent-side interruption."""

    for future in futures:
        future.cancel()
    terminate_workers = getattr(pool, "terminate_workers", None)
    if callable(terminate_workers):  # Python 3.14+
        terminate_workers()
        return
    # Python 3.10 has no public terminate_workers(). Terminating these exact
    # executor-owned children is necessary because one formal task can run for
    # more than 25 minutes. Raw results are parent-written, so no JSONL write is
    # interrupted inside a terminated worker.
    process_map = getattr(pool, "_processes", {}) or {}
    processes = list(process_map.values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    pool.shutdown(wait=True, cancel_futures=True)


def _create_manifest(
    run: Path,
    spec: ExperimentSpec,
    tasks: list[dict[str, object]],
    audit: Mapping[str, object],
    workers: int,
) -> dict[str, object]:
    hashes = _code_hashes()
    task_manifest_sha256 = _json_hash(tasks)
    payload: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "finished_utc": None,
        "config_path": str(spec.source),
        "config_sha256": spec.source_sha256,
        "split_sha256": _sha256(spec.split),
        "task_manifest_sha256": task_manifest_sha256,
        "experiment": _spec_payload(spec),
        "audit": dict(audit),
        "budget_summary": _budget_summary(spec, audit),
        "workers": workers,
        "python": sys.version,
        "python_executable": sys.executable,
        "package_versions": _package_versions(),
        "platform": platform.platform(),
        "cpu_model": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "logical_cpu_count": os.cpu_count(),
        "code_sha256": hashes,
        "isolation": {
            "project_root": str(PROJECT_ROOT),
            "output_root": str(run),
            "sibling_projects_are_read_only_dependencies": True,
            "python_bytecode_writes_disabled": True,
            "planning_mode": "DBD_only",
        },
        "internal_thread_environment": {name: "1" for name in _THREAD_ENVIRONMENT},
        "task_count_total": len(tasks),
        "task_count_successful": 0,
        "task_count_failed_attempts": 0,
        "task_count_pending": len(tasks),
    }
    _atomic_json(
        run / "task_manifest.json",
        {"sha256": task_manifest_sha256, "tasks": tasks},
    )
    _atomic_json(run / "manifest.json", payload)
    _atomic_json(
        run / "progress.json",
        _progress_payload(len(tasks), 0, 0, state="created"),
    )
    return payload


def _resume_manifest(
    run: Path,
    spec: ExperimentSpec,
    tasks: list[dict[str, object]],
    audit: Mapping[str, object],
    workers: int,
) -> dict[str, object]:
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"resume directory has no manifest: {run}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_manifest_path = run / "task_manifest.json"
    stored_tasks: object = None
    if task_manifest_path.is_file():
        stored_task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
        if isinstance(stored_task_manifest, Mapping):
            stored_tasks = stored_task_manifest.get("tasks")
    checks = {
        "config": manifest.get("config_sha256") == spec.source_sha256,
        "split": manifest.get("split_sha256") == _sha256(spec.split),
        "tasks": manifest.get("task_manifest_sha256") == _json_hash(tasks),
        "task_manifest_file": stored_tasks == tasks,
        "dataset": manifest.get("audit") == dict(audit),
        "code": manifest.get("code_sha256") == _code_hashes(),
        "packages": manifest.get("package_versions") == _package_versions(),
        "workers": int(manifest.get("workers", -1)) == workers,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("resume contract changed: " + ", ".join(failed))
    manifest["resumed_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(manifest_path, manifest)
    return manifest


def _build_outputs(run: Path, *, require_pass: bool) -> None:
    from .reporting import build_report

    build_report(run)
    if not require_pass:
        return
    integrity_path = run / "report" / "integrity.json"
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if integrity.get("status") != "PASS":
        raise RuntimeError(f"completed tasks failed report integrity checks: {integrity_path}")


def _execute_tasks(
    run: Path,
    manifest: dict[str, object],
    tasks: list[dict[str, object]],
    audit: Mapping[str, object],
    selected_workers: int,
    build_outputs: bool,
) -> Path:
    results_path = run / "raw" / "results.jsonl"
    failures_path = run / "raw" / "failures.jsonl"
    successes = _read_successes(results_path)
    _validate_success_records(successes, tasks, audit)
    pending = [task for task in tasks if str(task["task_id"]) not in successes]
    failure_attempts = int(manifest.get("task_count_failed_attempts", 0))
    total = len(tasks)
    if not pending:
        manifest.update(
            {
                "finished_utc": manifest.get("finished_utc")
                or datetime.now(timezone.utc).isoformat(),
                "task_count_successful": len(successes),
                "task_count_failed_attempts": failure_attempts,
                "task_count_pending": 0,
            }
        )
        _atomic_json(run / "manifest.json", manifest)
        _atomic_json(
            run / "progress.json",
            _progress_payload(total, len(successes), failure_attempts, state="complete"),
        )
        if build_outputs:
            _build_outputs(run, require_pass=True)
        return run

    manifest["last_started_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["finished_utc"] = None
    _atomic_json(run / "manifest.json", manifest)
    _atomic_json(
        run / "progress.json",
        _progress_payload(total, len(successes), failure_attempts, state="running"),
    )
    printer = _AsyncProgressPrinter()

    def accept(task: Mapping[str, object], record: dict[str, object]) -> None:
        nonlocal failure_attempts
        _validate_success_records(
            {str(task["task_id"]): record},
            [dict(task)],
            audit,
        )
        _append_jsonl(results_path, record)
        successes[str(task["task_id"])] = record
        completed = len(successes)
        _atomic_json(
            run / "progress.json",
            _progress_payload(total, completed, failure_attempts, state="running"),
        )
        printer.emit(
            f"[{completed}/{total}] {record['algorithm_label']} "
            f"day={int(record['case_day_index']) + 1} J={record['jobs']} "
            f"objective={record['objective']}"
        )

    def reject(task: Mapping[str, object], error: Exception) -> None:
        nonlocal failure_attempts
        failure_attempts += 1
        payload = {
            "failed_utc": datetime.now(timezone.utc).isoformat(),
            "task_id": str(task["task_id"]),
            "algorithm": str(task["algorithm"]),
            "case_day_index": int(task["case_day_index"]),
            "instance_path": str(task["instance_path"]),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        }
        _append_jsonl(failures_path, payload)
        _atomic_json(
            run / "progress.json",
            _progress_payload(total, len(successes), failure_attempts, state="running"),
        )
        printer.emit(
            f"[failure] {task['algorithm']} day={int(task['case_day_index']) + 1}: {error}",
        )

    def checkpoint_parent_stop(state: str, error: BaseException | None = None) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        manifest.update(
            {
                f"last_{state}_utc": timestamp,
                "task_count_successful": len(successes),
                "task_count_failed_attempts": failure_attempts,
                "task_count_pending": total - len(successes),
            }
        )
        if error is not None:
            manifest["last_parent_error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        _atomic_json(run / "manifest.json", manifest)
        _atomic_json(
            run / "progress.json",
            _progress_payload(
                total,
                len(successes),
                failure_attempts,
                state=state,
            ),
        )

    try:
        if selected_workers == 1:
            try:
                for task in pending:
                    try:
                        record = run_one(task)
                    except Exception as error:  # keep independent remaining tasks alive
                        reject(task, error)
                    else:
                        accept(task, record)
            except KeyboardInterrupt:
                checkpoint_parent_stop("interrupted")
                raise
            except BaseException as error:
                checkpoint_parent_stop("aborted", error)
                raise
        else:
            pool: ProcessPoolExecutor | None = None
            futures: dict[Future[dict[str, object]], Mapping[str, object]] = {}
            try:
                pool = ProcessPoolExecutor(max_workers=selected_workers)
                task_iterator = iter(pending)
                for _ in range(min(selected_workers, len(pending))):
                    task = next(task_iterator)
                    futures[pool.submit(run_one, task)] = task
                while futures:
                    completed, _ = wait(
                        futures,
                        timeout=HEARTBEAT_SECONDS,
                        return_when=FIRST_COMPLETED,
                    )
                    if not completed:
                        _atomic_json(
                            run / "progress.json",
                            _progress_payload(
                                total,
                                len(successes),
                                failure_attempts,
                                state="running",
                            ),
                        )
                        continue
                    for future in completed:
                        task = futures.pop(future)
                        try:
                            record = future.result()
                        except Exception as error:
                            reject(task, error)
                        else:
                            accept(task, record)
                        try:
                            next_task = next(task_iterator)
                        except StopIteration:
                            continue
                        futures[pool.submit(run_one, next_task)] = next_task
            except KeyboardInterrupt:
                if pool is not None:
                    _terminate_pool(pool, futures)
                checkpoint_parent_stop("interrupted")
                raise
            except BaseException as error:
                if pool is not None:
                    _terminate_pool(pool, futures)
                checkpoint_parent_stop("aborted", error)
                raise
            else:
                pool.shutdown(wait=True)
    finally:
        printer.close()

    pending_count = total - len(successes)
    state = "complete" if pending_count == 0 else "incomplete"
    manifest.update(
        {
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "task_count_successful": len(successes),
            "task_count_failed_attempts": failure_attempts,
            "task_count_pending": pending_count,
        }
    )
    _atomic_json(run / "manifest.json", manifest)
    _atomic_json(
        run / "progress.json",
        _progress_payload(total, len(successes), failure_attempts, state=state),
    )
    if build_outputs:
        _build_outputs(run, require_pass=pending_count == 0)
    if pending_count:
        raise RuntimeError(
            f"{pending_count} task(s) remain incomplete; resume the same directory: {run}"
        )
    return run


def run_experiment(
    config: str | Path,
    output_root: str | Path | None = None,
    *,
    workers: int | None = None,
    resume: str | Path | None = None,
    build_outputs: bool = True,
) -> Path:
    _configure_threads()
    if resume is not None and output_root is not None:
        raise ValueError("output_root cannot be combined with resume")
    spec = load_experiment_spec(config)
    for algorithm in spec.algorithms:
        validate_parameters(algorithm.name, algorithm.parameters)
    audit = audit_instances(spec)
    tasks = build_tasks(spec)
    selected_workers = _resolve_workers(spec, workers)
    if resume is None:
        root = PROJECT_ROOT / "outputs" if output_root is None else Path(output_root)
        run = _new_run_directory(root, spec.name)
    else:
        run = Path(resume).resolve()
        if not run.is_dir():
            raise FileNotFoundError(f"resume directory does not exist: {run}")
    with _exclusive_run_lock(run):
        manifest = (
            _create_manifest(run, spec, tasks, audit, selected_workers)
            if resume is None
            else _resume_manifest(run, spec, tasks, audit, selected_workers)
        )
        print(f"[run] {run}", flush=True)
        return _execute_tasks(
            run,
            manifest,
            tasks,
            audit,
            selected_workers,
            build_outputs,
        )


def _dry_run(config: str | Path, workers: int | None) -> None:
    spec = load_experiment_spec(config)
    for algorithm in spec.algorithms:
        validate_parameters(algorithm.name, algorithm.parameters)
    audit = audit_instances(spec)
    tasks = build_tasks(spec)
    selected_workers = _resolve_workers(spec, workers)
    payload = {
        "experiment": spec.name,
        "contract": spec.contract,
        "planning_mode": "DBD",
        "algorithms": [item.label for item in spec.algorithms],
        "seeds": list(spec.seeds),
        "workers": selected_workers,
        "task_count": len(tasks),
        "audit": audit,
        "budget_summary": _budget_summary(spec, audit),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the isolated CSSD DBD case study")
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument("--output", help="output root; defaults to this project's outputs/")
    parser.add_argument("--workers", type=int, help="external process count")
    parser.add_argument("--resume", help="existing experiment directory to resume")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the task audit")
    args = parser.parse_args(argv)
    if args.dry_run:
        _dry_run(args.config, args.workers)
        return
    run = run_experiment(
        args.config,
        args.output,
        workers=args.workers,
        resume=args.resume,
    )
    print(run)


if __name__ == "__main__":
    main()
