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
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterator, Mapping

from .config import (
    FORMAL_DATASET_SHA256,
    ExperimentSpec,
    PROJECT_ROOT,
    load_experiment_spec,
)
from .model import load_instance, official_time_limit
from .solvers import solve_ffcbspt, solve_kraul_ga


HEARTBEAT_SECONDS = 15.0
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class _AsyncPrinter:
    """Keep result persistence independent from a paused Windows console."""

    def __init__(self) -> None:
        self.messages: queue.Queue[str | None] = queue.Queue(maxsize=512)
        self.thread = threading.Thread(target=self._print_loop, daemon=True)
        self.thread.start()

    def _print_loop(self) -> None:
        while True:
            message = self.messages.get()
            if message is None:
                return
            print(message, flush=True)

    def emit(self, message: str) -> None:
        try:
            self.messages.put_nowait(message)
        except queue.Full:
            pass

    def close(self) -> None:
        try:
            self.messages.put_nowait(None)
        except queue.Full:
            return
        self.thread.join(timeout=2.0)


@contextmanager
def _run_lock(root: Path) -> Iterator[None]:
    lock_path = root / ".runner.lock"
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
        else:  # pragma: no cover
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise RuntimeError(f"another process is writing this run: {root}") from error
    try:
        yield
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _configure_worker_threads() -> None:
    for variable in THREAD_ENVIRONMENT:
        os.environ[variable] = "1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _source_hash() -> str:
    digest = hashlib.sha256()
    candidates = sorted(
        [*PROJECT_ROOT.joinpath("src").rglob("*.py"), *PROJECT_ROOT.joinpath("scripts").glob("*.py")]
    )
    for path in candidates:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str]:
    output: dict[str, str] = {}
    for distribution in ("kraul-dbd-reproduction", "numpy", "PyYAML", "matplotlib"):
        try:
            output[distribution] = version(distribution)
        except PackageNotFoundError:
            output[distribution] = "not-installed"
    return output


def _dataset_hash(paths: tuple[Path, ...], data_root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(data_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
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
        handle.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _experiment_payload(spec: ExperimentSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "contract": spec.contract,
        "data_root": str(spec.data_root),
        "split": str(spec.split),
        "source_notebook": str(spec.source_notebook),
        "expected_days": spec.expected_days,
        "expected_jobs": spec.expected_jobs,
        "first_day_seed": spec.first_day_seed,
        "last_day_seed": spec.last_day_seed,
        "machines_stage1": spec.machines_stage1,
        "machines_stage2": spec.machines_stage2,
        "capacities": list(spec.capacities),
        "day_length_minutes": spec.day_length_minutes,
        "seeds": list(spec.seeds),
        "workers": spec.workers,
        "protocol": asdict(spec.protocol),
        "algorithms": [
            {
                "name": item.name,
                "label": item.label,
                "profile": item.profile,
                "overrides": dict(item.overrides),
            }
            for item in spec.algorithms
        ],
    }


def audit_instances(spec: ExperimentSpec) -> dict[str, object]:
    sequence: list[dict[str, object]] = []
    total_jobs = 0
    for day, path in enumerate(spec.instance_paths):
        instance = load_instance(path)
        total_jobs += instance.job_count
        if (instance.machines_stage1, instance.machines_stage2) != (
            spec.machines_stage1,
            spec.machines_stage2,
        ):
            raise ValueError(f"machine setting mismatch in {path.name}")
        if (instance.capacity_stage1, instance.capacity_stage2) != spec.capacities:
            raise ValueError(f"capacity setting mismatch in {path.name}")
        sequence.append(
            {
                "case_day_index": day,
                "case_day_number": day + 1,
                "relative_path": path.relative_to(spec.data_root).as_posix(),
                "instance": instance.name,
                "instance_sha256": instance.source_sha256,
                "jobs": instance.job_count,
                "families": instance.family_count,
                "official_time_limit_seconds": official_time_limit(
                    instance, spec.protocol.tau or 1.0
                ),
            }
        )
    if spec.expected_jobs and total_jobs != spec.expected_jobs:
        raise ValueError(f"split has {total_jobs} jobs, expected {spec.expected_jobs}")
    dataset_sha256 = _dataset_hash(spec.instance_paths, spec.data_root)
    if spec.is_formal and dataset_sha256 != FORMAL_DATASET_SHA256:
        raise ValueError("formal proxy91 dataset contents changed")
    return {
        "dataset_sha256": dataset_sha256,
        "total_jobs": total_jobs,
        "day_count": len(sequence),
        "sequence": sequence,
    }


def _tasks(spec: ExperimentSpec, audit: Mapping[str, object], run_contract: str) -> list[dict[str, object]]:
    sequence = audit.get("sequence")
    if not isinstance(sequence, list):
        raise ValueError("audit sequence is missing")
    output: list[dict[str, object]] = []
    for day_record, path in zip(sequence, spec.instance_paths):
        day = int(day_record["case_day_index"])
        for algorithm in spec.algorithms:
            seeds: tuple[int | None, ...] = (None,) if algorithm.name == "ffcbspt" else spec.seeds
            for seed in seeds:
                if spec.protocol.mode == "official_time":
                    limit = float(day_record["official_time_limit_seconds"])
                    max_evaluations = None
                else:
                    limit = float(spec.protocol.time_limit_seconds or 0.0)
                    max_evaluations = spec.protocol.max_evaluations
                task_id = f"day-{day + 1:03d}__{algorithm.name}"
                if seed is not None:
                    task_id += f"__seed-{seed}"
                task = {
                    "task_id": task_id,
                    "case_day_index": day,
                    "case_day_number": day + 1,
                    "instance_path": str(path),
                    "instance": str(day_record["instance"]),
                    "instance_sha256": str(day_record["instance_sha256"]),
                    "jobs": int(day_record["jobs"]),
                    "algorithm": algorithm.name,
                    "algorithm_label": algorithm.label,
                    "seed": seed,
                    "profile": algorithm.profile,
                    "overrides": dict(algorithm.overrides),
                    "time_limit_seconds": limit,
                    "max_evaluations": max_evaluations,
                    "day_length_minutes": spec.day_length_minutes,
                    "run_contract": run_contract,
                }
                task["task_hash"] = _json_hash(task)
                output.append(task)
    return output


def _worker(task: Mapping[str, object]) -> dict[str, object]:
    _configure_worker_threads()
    instance = load_instance(str(task["instance_path"]))
    if instance.source_sha256 != task["instance_sha256"]:
        raise ValueError(f"instance changed before task {task['task_id']}")
    algorithm = str(task["algorithm"])
    if algorithm == "ffcbspt":
        result = solve_ffcbspt(instance, int(task["day_length_minutes"]))
    elif algorithm == "kraul_ga":
        seed = task.get("seed")
        if seed is None:
            raise ValueError("Kraul-GA task has no random seed")
        raw_overrides = task.get("overrides")
        overrides = raw_overrides if isinstance(raw_overrides, Mapping) else {}
        result = solve_kraul_ga(
            instance,
            seed=int(seed),
            time_limit_seconds=float(task["time_limit_seconds"]),
            profile=str(task["profile"]),
            overrides=overrides,
            day_length_minutes=int(task["day_length_minutes"]),
            max_evaluations=(
                int(task["max_evaluations"]) if task.get("max_evaluations") is not None else None
            ),
        )
    else:
        raise ValueError(f"unsupported algorithm: {algorithm}")
    objective = float(result["objective"])
    release_sum = float(result["release_sum"])
    tis = result.get("time_in_system")
    if not isinstance(tis, Mapping) or abs(float(tis["sum"]) - (objective - release_sum)) > 1e-8:
        raise ValueError("sum(TIS) != sum(Cj2) - sum(release)")
    day = int(task["case_day_index"])
    calendar_offset = day * int(task["day_length_minutes"])
    return {
        "status": "success",
        **{key: value for key, value in task.items() if key != "instance_path"},
        "relative_instance_path": instance.path.name,
        "families": instance.family_count,
        "machines_stage1": instance.machines_stage1,
        "machines_stage2": instance.machines_stage2,
        "capacity_stage1": instance.capacity_stage1,
        "capacity_stage2": instance.capacity_stage2,
        "calendar_objective": objective + calendar_offset * instance.job_count,
        "feasible": True,
        **result,
    }


def _load_successes(path: Path, expected: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    if not path.is_file():
        return {}
    successes: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {line_number}: {path}") from error
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number} is not an object: {path}")
            task_id = str(record.get("task_id", ""))
            status = record.get("status")
            if status not in {"success", "error"}:
                raise ValueError(f"invalid result status at line {line_number}: {status!r}")
            if task_id not in expected:
                raise ValueError(f"result contains unknown task ID at line {line_number}: {task_id!r}")
            if status == "success":
                if record.get("task_hash") != expected[task_id].get("task_hash"):
                    raise ValueError(f"task contract changed for resumed result {task_id}")
                if task_id in successes:
                    raise ValueError(f"duplicate successful result for {task_id}")
                successes[task_id] = record
    return successes


def _terminate_pool(
    executor: ProcessPoolExecutor,
    active: Mapping[Future[dict[str, object]], Mapping[str, object]],
) -> None:
    """Stop only this run's workers so Ctrl+C returns promptly on Windows."""
    for future in active:
        future.cancel()
    terminate_workers = getattr(executor, "terminate_workers", None)
    if callable(terminate_workers):  # Python 3.14+
        try:
            terminate_workers()
            return
        except Exception:  # pragma: no cover - fall back to older implementation
            pass
    process_mapping = getattr(executor, "_processes", None) or {}
    processes = list(process_mapping.values())
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
        except (OSError, ValueError):
            pass
    for process in processes:
        try:
            process.join(timeout=2.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=2.0)
        except (OSError, ValueError):
            pass
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except (OSError, RuntimeError):
        pass


def _archive_resolved_errors(manifest: dict[str, object]) -> None:
    errors = manifest.pop("errors", None)
    if not errors:
        return
    history = manifest.setdefault("failed_attempt_history", [])
    if not isinstance(history, list):
        raise ValueError("manifest failed_attempt_history must be a list")
    history.append(
        {
            "resolved_utc": datetime.now(timezone.utc).isoformat(),
            "errors": errors,
        }
    )


def _progress_payload(
    *,
    total: int,
    completed: int,
    failed: int,
    active: Mapping[Future[dict[str, object]], Mapping[str, object]],
    started: float,
) -> dict[str, object]:
    elapsed = time.perf_counter() - started
    return {
        "status": "running" if completed + failed < total else ("failed" if failed else "complete"),
        "total": total,
        "completed": completed,
        "failed": failed,
        "remaining": max(0, total - completed - failed),
        "active_tasks": [str(task["task_id"]) for task in active.values()],
        "elapsed_seconds": elapsed,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }


def _progress_line(payload: Mapping[str, object]) -> str:
    elapsed = float(payload["elapsed_seconds"])
    active = list(payload.get("active_tasks", []))
    active_text = ", ".join(map(str, active[:3]))
    if len(active) > 3:
        active_text += f", +{len(active) - 3} more"
    return (
        f"[progress] {payload['completed']}/{payload['total']} complete, "
        f"{payload['failed']} failed, elapsed={elapsed / 60:.1f} min, active=[{active_text}]"
    )


def _prepare_run(
    spec: ExperimentSpec,
    audit: Mapping[str, object],
    output: Path,
    resume: bool,
    workers: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    source_sha = _source_hash()
    notebook_sha = _sha256(spec.source_notebook)
    contract = _json_hash(
        {
            "config_sha256": spec.source_sha256,
            "dataset_sha256": audit["dataset_sha256"],
            "source_sha256": source_sha,
            "notebook_sha256": notebook_sha,
        }
    )
    tasks = _tasks(spec, audit, contract)
    manifest_path = output / "manifest.json"
    if resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"resume directory has no manifest: {output}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparisons = {
            "run_contract": contract,
            "config_sha256": spec.source_sha256,
            "source_sha256": source_sha,
            "notebook_sha256": notebook_sha,
        }
        for key, value in comparisons.items():
            if manifest.get(key) != value:
                raise ValueError(f"cannot resume because {key} changed")
        if int(manifest.get("workers", -1)) != workers:
            raise ValueError("resume must use the original worker count")
    else:
        existing = [path for path in output.iterdir() if path.name != ".runner.lock"]
        if existing:
            raise FileExistsError(f"new run directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "status": "running",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_contract": contract,
            "config_sha256": spec.source_sha256,
            "source_sha256": source_sha,
            "notebook_sha256": notebook_sha,
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "dependency_versions": _dependency_versions(),
            "workers": workers,
            "task_count_total": len(tasks),
            "experiment": _experiment_payload(spec),
            "audit": audit,
            "known_source_differences": {
                "paper_final_vs_notebook": {
                    "paper_final": {"crossover_rate": 0.95, "mutation_type": "insert"},
                    "notebook_example": {"crossover_rate": 0.80, "mutation_type": "flip"},
                },
                "roulette_direction": (
                    "released code samples in direct proportion to the minimization objective; "
                    "formal reproduction preserves this behaviour"
                ),
                "time_limit_semantics": (
                    "released GA checks the wall-clock limit only at generation boundaries and "
                    "evaluates the final population after the loop; actual runtime can exceed "
                    "the nominal paper limit and the overrun is reported per task"
                ),
                "truncated_lower_bound": (
                    "public Notebook ends calculate_lower_bound with 'lower_bound = processin'; "
                    "the lower bound is unused by FFCBSPT/GA acceptance and is omitted"
                ),
                "diagnostic_re_evaluations": (
                    "the Notebook report() redundantly reevaluates the final population after "
                    "solution_time is frozen; these print-only evaluations do not affect the "
                    "selected schedule and are omitted from this refactor"
                ),
                "random_stream_isolation": (
                    "the Notebook uses NumPy's unseeded global RNG; each DBD task here uses its "
                    "own locked RandomState seed so multiprocessing order cannot change results "
                    "and the seed set matches the other three algorithms"
                ),
                "zero_release_padding": (
                    "released scheduler filters zero entries used for rectangular-array padding; "
                    "this refactor stores explicit batches so a real batch released at minute 0 "
                    "is scheduled instead of being mistaken for padding"
                ),
            },
        }
        _atomic_json(manifest_path, manifest)
        _atomic_json(output / "task_manifest.json", {"run_contract": contract, "tasks": tasks})
    return manifest, tasks


def run_experiment(
    config_path: str | Path,
    *,
    output: str | Path | None = None,
    resume: str | Path | None = None,
    workers_override: int | None = None,
    build_report: bool = True,
) -> Path:
    spec = load_experiment_spec(config_path)
    audit = audit_instances(spec)
    workers = spec.workers if workers_override is None else int(workers_override)
    if workers <= 0:
        raise ValueError("workers must be positive")
    if spec.is_formal and workers != spec.workers:
        raise ValueError("formal run must use the locked worker count")
    if resume is not None:
        root = Path(resume).resolve()
        is_resume = True
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        root = (
            Path(output).resolve()
            if output is not None
            else (PROJECT_ROOT / "outputs" / f"{spec.name}_{timestamp}").resolve()
        )
        is_resume = False
    root.mkdir(parents=True, exist_ok=True)

    with _run_lock(root):
        manifest, tasks = _prepare_run(spec, audit, root, is_resume, workers)
        for directory in (root / "raw", root / "tables", root / "figures", root / "report"):
            directory.mkdir(parents=True, exist_ok=True)
        expected = {str(task["task_id"]): task for task in tasks}
        results_path = root / "raw" / "results.jsonl"
        successes = _load_successes(results_path, expected)
        todo = [task for task in tasks if task["task_id"] not in successes]
        printer = _AsyncPrinter()
        printer.emit(f"[run] {root}")
        if successes:
            printer.emit(f"[resume] retained {len(successes)} verified successful tasks")
        if not todo:
            if manifest.get("status") == "complete":
                printer.emit(
                    "[resume] experiment is already complete; no task or runtime metadata changed"
                )
            else:
                _archive_resolved_errors(manifest)
                manifest["status"] = "complete"
                manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
                manifest["successful_tasks"] = len(successes)
                manifest["failed_tasks"] = 0
                _atomic_json(root / "manifest.json", manifest)
                elapsed = float(manifest.get("wall_runtime_seconds", 0.0))
                _atomic_json(
                    root / "progress.json",
                    {
                        "status": "complete",
                        "total": len(tasks),
                        "completed": len(successes),
                        "failed": 0,
                        "remaining": 0,
                        "active_tasks": [],
                        "elapsed_seconds": elapsed,
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
                printer.emit("[resume] all persisted tasks verified; finalized stale run metadata")
            if build_report:
                from .reporting import build_report as build_experiment_report

                build_experiment_report(root)
            printer.close()
            return root
        prior_runtime = float(manifest.get("wall_runtime_seconds", 0.0))
        if is_resume and prior_runtime == 0.0:
            previous_progress = root / "progress.json"
            if previous_progress.is_file():
                prior_runtime = float(
                    json.loads(previous_progress.read_text(encoding="utf-8")).get(
                        "elapsed_seconds", 0.0
                    )
                )
        started = time.perf_counter()
        failed = 0
        active: dict[Future[dict[str, object]], Mapping[str, object]] = {}
        iterator = iter(todo)
        last_heartbeat = 0.0
        completed = len(successes)
        errors: list[str] = []

        def submit_next(executor: ProcessPoolExecutor) -> bool:
            try:
                task = next(iterator)
            except StopIteration:
                return False
            active[executor.submit(_worker, task)] = task
            return True

        def checkpoint_abort(status: str) -> None:
            session_runtime = time.perf_counter() - started
            manifest["status"] = status
            manifest[f"{status}_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["successful_tasks"] = completed
            manifest["failed_tasks"] = failed
            manifest["last_session_wall_runtime_seconds"] = session_runtime
            manifest["wall_runtime_seconds"] = prior_runtime + session_runtime
            previous_errors = manifest.get("errors", [])
            combined_errors = list(previous_errors) if isinstance(previous_errors, list) else []
            combined_errors.extend(errors)
            if combined_errors:
                manifest["errors"] = combined_errors
            _atomic_json(root / "manifest.json", manifest)
            progress = _progress_payload(
                total=len(tasks),
                completed=completed,
                failed=failed,
                active=active,
                started=started,
            )
            progress["status"] = status
            _atomic_json(root / "progress.json", progress)

        _configure_worker_threads()
        executor: ProcessPoolExecutor | None = None
        try:
            executor = ProcessPoolExecutor(max_workers=workers)
            for _ in range(min(workers, len(todo))):
                submit_next(executor)
            while active:
                done, _ = wait(tuple(active), timeout=1.0, return_when=FIRST_COMPLETED)
                for future in done:
                    task = active.pop(future)
                    try:
                        record = future.result()
                    except Exception as error:  # noqa: BLE001 - persist task-level diagnostics
                        failed += 1
                        message = f"{task['task_id']}: {type(error).__name__}: {error}"
                        errors.append(message)
                        _append_jsonl(
                            results_path,
                            {
                                "status": "error",
                                **{key: value for key, value in task.items() if key != "instance_path"},
                                "error": message,
                                "traceback": traceback.format_exc(),
                            },
                        )
                        printer.emit(f"[error] {message}")
                    else:
                        _append_jsonl(results_path, record)
                        completed += 1
                        printer.emit(
                            f"[{completed + failed}/{len(tasks)}] {record['algorithm_label']} "
                            f"day={record['case_day_number']} J={record['jobs']} "
                            f"objective={record['objective']}"
                        )
                    submit_next(executor)
                now = time.perf_counter()
                if done or now - last_heartbeat >= HEARTBEAT_SECONDS:
                    progress = _progress_payload(
                        total=len(tasks),
                        completed=completed,
                        failed=failed,
                        active=active,
                        started=started,
                    )
                    _atomic_json(root / "progress.json", progress)
                    if now - last_heartbeat >= HEARTBEAT_SECONDS:
                        printer.emit(_progress_line(progress))
                        last_heartbeat = now
        except KeyboardInterrupt:
            if executor is not None:
                _terminate_pool(executor, active)
            checkpoint_abort("interrupted")
            printer.emit(f"[interrupted] resume with --resume {root}")
            raise
        except BaseException as error:
            if executor is not None:
                _terminate_pool(executor, active)
            errors.append(f"runner: {type(error).__name__}: {error}")
            checkpoint_abort("aborted")
            raise
        else:
            if executor is not None:
                executor.shutdown(wait=True)
        finally:
            printer.close()

        final_progress = _progress_payload(
            total=len(tasks), completed=completed, failed=failed, active={}, started=started
        )
        _atomic_json(root / "progress.json", final_progress)
        manifest["status"] = "failed" if failed else "complete"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["successful_tasks"] = completed
        manifest["failed_tasks"] = failed
        session_runtime = time.perf_counter() - started
        manifest["last_session_wall_runtime_seconds"] = session_runtime
        manifest["wall_runtime_seconds"] = prior_runtime + session_runtime
        if failed:
            previous_errors = manifest.get("errors", [])
            combined_errors = list(previous_errors) if isinstance(previous_errors, list) else []
            combined_errors.extend(errors)
            manifest["errors"] = combined_errors
        else:
            _archive_resolved_errors(manifest)
        _atomic_json(root / "manifest.json", manifest)
        if failed:
            raise RuntimeError(f"{failed} tasks failed; rerun with --resume {root}")
        if build_report:
            from .reporting import build_report as build_experiment_report

            build_experiment_report(root)
        return root


def dry_run(config_path: str | Path) -> dict[str, object]:
    spec = load_experiment_spec(config_path)
    audit = audit_instances(spec)
    tasks = _tasks(spec, audit, "dry-run")
    ga_seconds = sum(
        float(task["time_limit_seconds"]) for task in tasks if task["algorithm"] == "kraul_ga"
    )
    return {
        "config": str(spec.source),
        "contract": spec.contract,
        "dataset_sha256": audit["dataset_sha256"],
        "days": audit["day_count"],
        "jobs": audit["total_jobs"],
        "workers": spec.workers,
        "tasks": len(tasks),
        "ga_tasks": sum(task["algorithm"] == "kraul_ga" for task in tasks),
        "nominal_ga_task_seconds": ga_seconds,
        "ideal_parallel_ga_hours": ga_seconds / spec.workers / 3600.0,
        "algorithms": [item.name for item in spec.algorithms],
        "ga_profile": next(
            (item.profile for item in spec.algorithms if item.name == "kraul_ga"), None
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run isolated Kraul FFCBSPT/GA DBD experiments")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="new output directory")
    parser.add_argument("--resume", help="existing interrupted output directory")
    parser.add_argument("--workers", type=int, help="development-only worker override")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args(argv)
    if args.output and args.resume:
        parser.error("--output and --resume are mutually exclusive")
    if args.dry_run:
        print(json.dumps(dry_run(args.config), ensure_ascii=False, indent=2))
        return
    result = run_experiment(
        args.config,
        output=args.output,
        resume=args.resume,
        workers_override=args.workers,
        build_report=not args.no_report,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
