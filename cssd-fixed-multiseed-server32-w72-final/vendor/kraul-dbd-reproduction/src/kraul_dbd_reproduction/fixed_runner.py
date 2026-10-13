from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .fixed_config import (
    FORMAL_FIXED_DATASET_SHA256,
    FixedScaleSpec,
    load_fixed_scale_spec,
)
from .model import load_instance, official_time_limit
from .runner import (
    HEARTBEAT_SECONDS,
    PROJECT_ROOT,
    _AsyncPrinter,
    _append_jsonl,
    _archive_resolved_errors,
    _atomic_json,
    _configure_worker_threads,
    _dataset_hash,
    _dependency_versions,
    _json_hash,
    _load_successes,
    _progress_line,
    _progress_payload,
    _run_lock,
    _sha256,
    _source_hash,
    _terminate_pool,
)
from .solvers import solve_kraul_ga


def _experiment_payload(spec: FixedScaleSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "contract": spec.contract,
        "data_root": str(spec.data_root),
        "split": str(spec.split),
        "source_notebook": str(spec.source_notebook),
        "expected_instances": spec.expected_instances,
        "expected_jobs": spec.expected_jobs,
        "expected_by_jobs": dict(spec.expected_by_jobs),
        "day_length_minutes": spec.day_length_minutes,
        "seeds": list(spec.seeds),
        "workers": spec.workers,
        "protocol": asdict(spec.protocol),
        "algorithm": {
            "name": spec.algorithm_name,
            "label": spec.algorithm_label,
            "profile": spec.profile,
            "overrides": dict(spec.overrides),
        },
    }


def audit_fixed_instances(spec: FixedScaleSpec) -> dict[str, object]:
    sequence: list[dict[str, object]] = []
    counts: Counter[int] = Counter()
    total_jobs = 0
    expected_machines = {25: (3, 2), 100: (12, 5), 250: (12, 5), 500: (12, 5), 1000: (12, 5)}
    for index, path in enumerate(spec.instance_paths):
        instance = load_instance(path)
        counts[instance.job_count] += 1
        total_jobs += instance.job_count
        if instance.job_count not in spec.expected_by_jobs:
            raise ValueError(f"unexpected job size J={instance.job_count}: {path.name}")
        if spec.is_formal:
            if instance.family_count != 5:
                raise ValueError(f"formal fixed instance is not F=5: {path.name}")
            if (instance.machines_stage1, instance.machines_stage2) != expected_machines[
                instance.job_count
            ]:
                raise ValueError(f"formal fixed machine setting changed: {path.name}")
            if (instance.capacity_stage1, instance.capacity_stage2) != (15, 11000):
                raise ValueError(f"formal fixed capacities changed: {path.name}")
        sequence.append(
            {
                "instance_index": index,
                "relative_path": path.relative_to(spec.data_root).as_posix(),
                "instance": instance.name,
                "instance_sha256": instance.source_sha256,
                "dataset_group": f"fixed_j{instance.job_count}",
                "jobs": instance.job_count,
                "families": instance.family_count,
                "machines_stage1": instance.machines_stage1,
                "machines_stage2": instance.machines_stage2,
                "capacity_stage1": instance.capacity_stage1,
                "capacity_stage2": instance.capacity_stage2,
                "official_time_limit_seconds": official_time_limit(
                    instance,
                    spec.protocol.tau or 1.0,
                ),
            }
        )
    if dict(sorted(counts.items())) != dict(spec.expected_by_jobs):
        raise ValueError(
            f"split scale counts {dict(sorted(counts.items()))} differ from "
            f"expected {dict(spec.expected_by_jobs)}"
        )
    if total_jobs != spec.expected_jobs:
        raise ValueError(f"split has {total_jobs} jobs, expected {spec.expected_jobs}")
    dataset_sha256 = _dataset_hash(spec.instance_paths, spec.data_root)
    if spec.is_formal and dataset_sha256 != FORMAL_FIXED_DATASET_SHA256:
        raise ValueError("formal fixed-scale dataset contents changed")
    return {
        "dataset_sha256": dataset_sha256,
        "instance_count": len(sequence),
        "total_jobs": total_jobs,
        "counts_by_jobs": dict(sorted(counts.items())),
        "sequence": sequence,
    }


def _tasks(
    spec: FixedScaleSpec,
    audit: Mapping[str, object],
    run_contract: str,
) -> list[dict[str, object]]:
    sequence = audit.get("sequence")
    if not isinstance(sequence, list):
        raise ValueError("audit sequence is missing")
    output: list[dict[str, object]] = []
    for record, path in zip(sequence, spec.instance_paths):
        if not isinstance(record, Mapping):
            raise ValueError("invalid audit sequence record")
        for seed in spec.seeds:
            if spec.protocol.mode == "official_time":
                limit = float(record["official_time_limit_seconds"])
                max_evaluations = None
            else:
                limit = float(spec.protocol.time_limit_seconds or 0.0)
                max_evaluations = spec.protocol.max_evaluations
            index = int(record["instance_index"])
            task = {
                "task_id": f"instance-{index + 1:03d}__kraul_ga__seed-{seed}",
                "instance_index": index,
                "instance_path": str(path),
                "relative_instance_path": str(record["relative_path"]),
                "instance": str(record["instance"]),
                "instance_sha256": str(record["instance_sha256"]),
                "dataset_group": str(record["dataset_group"]),
                "jobs": int(record["jobs"]),
                "algorithm": spec.algorithm_name,
                "algorithm_label": spec.algorithm_label,
                "seed": seed,
                "profile": spec.profile,
                "overrides": dict(spec.overrides),
                "protocol_name": spec.protocol.name,
                "protocol_mode": spec.protocol.mode,
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
    raw_overrides = task.get("overrides")
    overrides = raw_overrides if isinstance(raw_overrides, Mapping) else {}
    result = solve_kraul_ga(
        instance,
        seed=int(task["seed"]),
        time_limit_seconds=float(task["time_limit_seconds"]),
        profile=str(task["profile"]),
        overrides=overrides,
        day_length_minutes=int(task["day_length_minutes"]),
        max_evaluations=(
            int(task["max_evaluations"]) if task.get("max_evaluations") is not None else None
        ),
    )
    objective = float(result["objective"])
    release_sum = float(result["release_sum"])
    time_in_system = result.get("time_in_system")
    if not isinstance(time_in_system, Mapping):
        raise ValueError("missing time-in-system moment")
    if abs(float(time_in_system["sum"]) - (objective - release_sum)) > 1e-8:
        raise ValueError("sum(TIS) != sum(Cj2) - sum(release)")
    schedule = result.get("schedule")
    if not isinstance(schedule, Mapping):
        raise ValueError("missing decoded schedule")
    completion_stage2 = schedule.get("completion_stage2")
    if not isinstance(completion_stage2, list) or len(completion_stage2) != instance.job_count:
        raise ValueError("stage-2 completion vector has the wrong length")
    makespan = max(float(value) for value in completion_stage2)
    return {
        "status": "success",
        **{key: value for key, value in task.items() if key != "instance_path"},
        "families": instance.family_count,
        "machines_stage1": instance.machines_stage1,
        "machines_stage2": instance.machines_stage2,
        "capacity_stage1": instance.capacity_stage1,
        "capacity_stage2": instance.capacity_stage2,
        "makespan": makespan,
        "feasible": True,
        **result,
    }


def _prepare_run(
    spec: FixedScaleSpec,
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
            "fixed_scale_contract": {
                "statistical_unit": "instance",
                "seed_aggregation": "mean within instance before paired comparison",
                "time_budget": "published formula, tau=1, capped at 1800 seconds",
                "machine_source": "six-value header of each public instance",
                "ga_profile": "paper_final",
                "selection_semantics": "released_objective_proportional",
                "convergence_x": "elapsed time divided by nominal instance budget",
                "convergence_y": "best-so-far improvement relative to first population best",
            },
        }
        _atomic_json(manifest_path, manifest)
        _atomic_json(output / "task_manifest.json", {"run_contract": contract, "tasks": tasks})
    return manifest, tasks


def run_fixed_scale_experiment(
    config_path: str | Path,
    *,
    output: str | Path | None = None,
    resume: str | Path | None = None,
    workers_override: int | None = None,
    build_report: bool = True,
) -> Path:
    spec = load_fixed_scale_spec(config_path)
    audit = audit_fixed_instances(spec)
    workers = spec.workers if workers_override is None else int(workers_override)
    if workers <= 0:
        raise ValueError("workers must be positive")
    if spec.is_formal and workers != spec.workers:
        raise ValueError("formal fixed-scale run must use the locked worker count")
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
        todo.sort(key=lambda task: (-float(task["time_limit_seconds"]), int(task["instance_index"])))
        printer = _AsyncPrinter()
        printer.emit(f"[run] {root}")
        if successes:
            printer.emit(f"[resume] retained {len(successes)} verified successful tasks")
        if not todo:
            if manifest.get("status") != "complete":
                _archive_resolved_errors(manifest)
                manifest["status"] = "complete"
                manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
                manifest["successful_tasks"] = len(successes)
                manifest["failed_tasks"] = 0
                _atomic_json(root / "manifest.json", manifest)
            if build_report:
                from .fixed_reporting import build_fixed_report

                build_fixed_report(root)
            printer.close()
            return root

        prior_runtime = float(manifest.get("wall_runtime_seconds", 0.0))
        started = time.perf_counter()
        completed = len(successes)
        failed = 0
        errors: list[str] = []
        active: dict[Future[dict[str, object]], Mapping[str, object]] = {}
        iterator = iter(todo)
        last_heartbeat = 0.0

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
            combined = list(previous_errors) if isinstance(previous_errors, list) else []
            combined.extend(errors)
            if combined:
                manifest["errors"] = combined
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
                    except Exception as error:  # noqa: BLE001
                        failed += 1
                        message = f"{task['task_id']}: {type(error).__name__}: {error}"
                        errors.append(message)
                        _append_jsonl(
                            results_path,
                            {
                                "status": "error",
                                **{
                                    key: value
                                    for key, value in task.items()
                                    if key != "instance_path"
                                },
                                "error": message,
                                "traceback": traceback.format_exc(),
                            },
                        )
                        printer.emit(f"[error] {message}")
                    else:
                        _append_jsonl(results_path, record)
                        completed += 1
                        printer.emit(
                            f"[{completed + failed}/{len(tasks)}] Kraul-GA "
                            f"J={record['jobs']} instance={record['instance']} "
                            f"objective={record['objective']} "
                            f"runtime={float(record['runtime_seconds']):.1f}s"
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
            total=len(tasks),
            completed=completed,
            failed=failed,
            active={},
            started=started,
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
            combined = list(previous_errors) if isinstance(previous_errors, list) else []
            combined.extend(errors)
            manifest["errors"] = combined
        else:
            _archive_resolved_errors(manifest)
        _atomic_json(root / "manifest.json", manifest)
        if failed:
            raise RuntimeError(f"{failed} tasks failed; rerun with --resume {root}")
        if build_report:
            from .fixed_reporting import build_fixed_report

            build_fixed_report(root)
        return root


def dry_run_fixed_scale(config_path: str | Path) -> dict[str, object]:
    spec = load_fixed_scale_spec(config_path)
    audit = audit_fixed_instances(spec)
    tasks = _tasks(spec, audit, "dry-run")
    budget_by_jobs: dict[int, float] = defaultdict(float)
    for task in tasks:
        budget_by_jobs[int(task["jobs"])] += float(task["time_limit_seconds"])
    total = sum(budget_by_jobs.values())
    return {
        "config": str(spec.source),
        "contract": spec.contract,
        "dataset_sha256": audit["dataset_sha256"],
        "instances": audit["instance_count"],
        "jobs_total": audit["total_jobs"],
        "counts_by_jobs": audit["counts_by_jobs"],
        "seeds": list(spec.seeds),
        "workers": spec.workers,
        "tasks": len(tasks),
        "ga_profile": spec.profile,
        "nominal_seconds_by_jobs": dict(sorted(budget_by_jobs.items())),
        "nominal_total_seconds": total,
        "ideal_parallel_hours": total / spec.workers / 3600.0,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run isolated fixed-scale Kraul-GA experiments")
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
        print(json.dumps(dry_run_fixed_scale(args.config), ensure_ascii=False, indent=2))
        return
    result = run_fixed_scale_experiment(
        args.config,
        output=args.output,
        resume=args.resume,
        workers_override=args.workers,
        build_report=not args.no_report,
    )
    print(result, flush=True)
